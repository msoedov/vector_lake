import datetime
import logging
import math
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from typing import Any
from json import dumps

import numpy as np
import pandas as pd
import pytz
from pydantic import BaseModel
from sklearn.metrics.pairwise import cosine_similarity

from vector_lake.core.hnsw import HNSW

# Optional dependencies
try:
    import boto3
except ImportError:
    boto3 = object()


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)


class LSH:
    def __init__(self, dim, num_hashes, bucket_size=100_000, random_seed=42):
        """Initialize LSH object.

        dim: Dimension of vectors
        num_hashes: Number of hash functions (hyperplanes) to use
        bucket_size: Size of hash buckets
        """
        self.dim = dim
        self.num_hashes = num_hashes
        self.bucket_size = bucket_size
        np.random.seed(random_seed)
        self.hyperplanes = np.random.randn(self.num_hashes, self.dim)
        self.buckets = {}

    @property
    def max_partitions(self):
        return 2**self.num_hashes

    def _hash(self, vector):
        """Hashes a vector using all hyperplanes.

        Returns a string of 0s and 1s.
        """
        return int(
            "".join(
                [
                    "1" if np.dot(hyperplane, vector) > 0 else "0"
                    for hyperplane in self.hyperplanes
                ]
            ),
            base=2,
        )

    route = _hash

    def insert(self, vector, label):
        """Inserts a vector into the LSH structure.

        vector: The vector to insert
        label: An identifier for the vector
        """
        h = self._hash(vector)
        if h not in self.buckets:
            self.buckets[h] = []
        if len(self.buckets[h]) < self.bucket_size:
            self.buckets[h].append((vector, label))

    def query(self, vector):
        """Queries the LSH structure for similar vectors to the given vector.

        vector: The vector to query
        Returns a list of similar vectors and their labels.
        """
        h = self._hash(vector)
        return self.buckets.get(h, [])


def make_granularity(D: int, M: int) -> list:
    if D < 0 or M < 0:
        raise ValueError("D and M must be positive integers")

    base = M ** (1 / D)  # Base level for each dimension
    # Initialize levels with the floor of base
    levels = [int(base) for _ in range(D)]
    # Assign the remaining splits (if any) to the first dimensions
    i = 0
    k = i
    while np.prod(levels) < M:
        levels[i] += 1
        k = i
        i = (i + 1) % D
    if np.prod(levels) > M:
        levels[k] -= 1
    return levels


def make_nodes(levels, num_shards):
    nodes = np.array(list(product(*[np.arange(_l) for _l in levels])))

    zeros_nodes = np.zeros_like(nodes).astype(float)
    for k, row in enumerate(nodes):
        for i, val in enumerate(row):
            zeros_nodes[k][i] = float(val) / float(levels[i])
    return zeros_nodes


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.debug(
            f"VPS for {func.__name__}: {1/elapsed_time:.1f} vectors per second"
        )
        return result

    return wrapper


executor = ThreadPoolExecutor(5)


def experimental_calculate_monthly_s3_costs(file_size_gb):
    """Calculates monthly S3 costs based on size of uploaded files.

    Args:
        file_size_gb (float): The size of the file in gigabytes.

    Returns:
        float: The monthly cost in USD.
    """
    cost_per_gb_per_month = 0.023  # as of cutoff in September 2021
    monthly_cost = file_size_gb * cost_per_gb_per_month

    return monthly_cost


# TODO: unused
def cosine_similarity_p(data_vectors, target, num_groups=5, split_if_more=50):
    if len(data_vectors) <= split_if_more:
        return cosine_similarity(data_vectors, target)

    data_groups = np.array_split(data_vectors, num_groups)

    most_similar_all = []

    for most_similar_indices in executor.map(
        lambda group: cosine_similarity(group, target), data_groups
    ):
        most_similar_all.append(most_similar_indices)

    return np.concatenate(most_similar_all)


class LazyBucket(BaseModel):
    """This class is a representation of a "lazy" data bucket for a distributed
    search system. The lazy loading technique is used to defer initialization
    of an object until the point at which it is needed.

    Attributes:
        db_location (str): The path to the location where the bucket data is stored.
        segment_index (int): The index of the current segment. This index is used to name the bucket file.
        bucket_name (str, optional): The name of the parquet file where the bucket data is stored.
                                     The segment index will be inserted in place of '{}'. Defaults to "segment-{}.parquet".
        metadata_name (str, optional): The name of the json file where the metadata associated with the bucket is stored.
                                        The segment index will be inserted in place of '{}'. Defaults to "segment-{}-metadata.json".
        loaded (bool, optional): A flag that indicates whether the bucket has been loaded. Defaults to False.
        dirty (bool, optional): A flag that indicates whether there are unsaved changes in the bucket. Defaults to False.
        frame (pandas.DataFrame | None, optional): The pandas DataFrame that contains the data of the bucket. Defaults to None.
        frame_schema (list[str], optional): The schema of the DataFrame. Defaults to ["id", "vector", "metadata", "document", "timestamp"].
        vectors (list): A list to store vectors from the DataFrame for easy access and manipulation. Defaults to an empty list.
        dirty_rows (list): A list to store new rows that haven't been added to the DataFrame yet. Defaults to an empty list.
    """

    db_location: str
    segment_index: str
    bucket_name: str = "segment-{}.parquet"
    metadata_name: str = "segment-{}-metadata.json"
    loaded: bool = False
    dirty: bool = False
    frame: Any | None = None
    frame_schema: str = ["id", "vector", "metadata", "document", "timestamp"]
    vectors = []
    dirty_rows = []
    hnsw: Any = None
    attrs: dict[str, Any] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hnsw = HNSW("cosine", m0=5, ef=10)

    def __repr__(self):
        return f"<{self.__class__.__name__}({self.segment_index=} {len(self.vectors)=} {self.dirty=} {self.loaded=} )>"

    @property
    def key(self):
        return self.bucket_name.format(self.segment_index)

    @property
    def frame_location(self):
        bucket_name = self.bucket_name.format(self.segment_index)
        return f"{self.db_location}/{bucket_name}"

    def _lazy_load(self):
        if self.loaded:
            return

        if os.path.exists(self.frame_location):
            self.frame = pd.read_parquet(self.frame_location)
        else:
            self.frame = pd.DataFrame(columns=self.frame_schema)
            self.attrs = self.frame.attrs
        if list(self.frame.columns) != self.frame_schema:
            raise ValueError(f"Invalid frame_schema {self.frame.columns=}")
        self.loaded = True
        self.vectors = self.frame["vector"].tolist()
        self.dirty_rows = self.frame.to_dict("records")
        for v in self.vectors:
            self.hnsw.add(v)

    def append(self, vector: np.ndarray, **attrs):
        if not self.loaded:
            self._lazy_load()
        uid = uuid.uuid1().urn
        document = {
            "id": uid,
            "vector": vector,
            "metadata": attrs.get("metadata", {"name": "unknown"}),
            "document": attrs.get("document", ""),
            "timestamp": attrs.get("timestamp", datetime.datetime.now(pytz.UTC)),
        }

        self.dirty_rows.append(document)
        self.dirty = True
        self.vectors.append(vector)
        self.hnsw.add(vector)
        return uid

    def search(self, vector: np.ndarray, k: int = 4):
        self._lazy_load()
        try:
            results = self.hnsw.search(vector, k)
        except ValueError:  # Empty graph
            return []
        return results

    def sync(self, **attrs):
        if not self.dirty:
            return

        self.frame = self.frame._append(self.dirty_rows, ignore_index=True)
        if self.frame.empty:
            return
        # TODO: eval last sync time
        now_dt = datetime.datetime.now(pytz.UTC)
        self.frame.attrs["last_update"] = dumps(now_dt, indent=4, sort_keys=True, default=str)
        for k, v in attrs.items():
            self.frame.attrs[k] = v

        os.makedirs(self.db_location, exist_ok=True)
        self.frame.to_parquet(self.frame_location, compression="gzip")
        self.dirty = False

    def delete(self):
        """This function deletes a file if it exists at a specified location.

        :return: If the file specified by `self.frame_location` does not exist, the function returns nothing
        (i.e., `None`). If the file exists and is successfully deleted, the function also returns nothing.
        If an exception occurs during the deletion process, the function catches the exception and does not
        re-raise it, so it also returns nothing.
        """
        if not os.path.exists(self.frame_location):
            return
        try:
            os.remove(self.frame_location)
        except Exception:
            ...

    def __len__(self):
        if not self.loaded:
            self._lazy_load()
        return len(self.vectors)

    def memory_footprint(self):
        return self.frame.memory_usage(deep=True).sum()

    def delete_local(self):
        return self.delete()

    def delete_remote(self):
        return ...


class S3Bucket(LazyBucket):
    remote_location: str = ""
    bytes_transferred: int = 0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.remote_location = self.db_location.strip("s3://")
        self.db_location = self.local_storage

    @property
    def local_storage(self):
        db_location = self.db_location.replace("://", "_")
        return f"/tmp/vector_lake_{db_location}"

    @property
    def s3_client(self):
        return boto3.client("s3", region_name=os.environ.get("AWS_DEFAULT_REGION"))
            #"s3", endpoint_url=os.environ.get("LOCALSTACK_ENDPOINT_URL")



    def _lazy_load(self):
        if self.loaded:
            return
        logger.info(f"Loading fragment {self.key} from S3")
        # Check if object exists in S3
        try:
            self.s3_client.head_object(Bucket=self.remote_location, Key=self.key)
        except Exception:
            logger.info("Fragment does not exist in S3")
            super()._lazy_load()
        except Exception as e:
            logger.exception(f"Unexpected error while checking for fragment in S3: {e}")
        else:
            logger.info("Fragment exists in S3, downloading...")
            os.makedirs(os.path.dirname(self.frame_location), exist_ok=True)
            self.s3_client.download_file(
                self.remote_location, self.key, Filename=self.frame_location
            )
            super()._lazy_load()

    def sync(self):
        if not self.dirty:
            return
        super().sync()
        if self.frame.empty:
            return
        logger.info(f"Uploading fragment {self.key} to S3")
        self.s3_client.upload_file(
            self.frame_location,
            self.remote_location,
            self.bucket_name.format(self.segment_index),
            Callback=self.upload_progress_callback(
                self.bucket_name.format(self.segment_index)
            ),
        )
        self.dirty = False

    def upload_progress_callback(self, key):
        def upload_progress_callback(bytes_transferred):
            self.bytes_transferred += bytes_transferred
            logger.debug(
                "\r{}: {} bytes have been transferred".format(
                    key, self.bytes_transferred
                ),
                end="",
            )

    def delete_local(self):
        super().delete()

    def delete_remote(self):
        try:
            logger.info(f"Deleting fragment {self.key} from S3")
            self.s3_client.delete_object(
                Bucket=self.remote_location,
                Key=self.key,
            )
        except Exception:
            logger.exception("Failed to delete object from S3")

    def delete(self):
        super().delete()
        self.delete_remote()


class Index(BaseModel):
    location: str
    dimension: int
    approx_shards: int

    metric_function: str = "cosine"  # dotproduct | euqlidean | cosine
    max_cache_mb: int = 1024
    ttl: int = 3600
    buckets: list = []

    # HNSW specific parameters
    granularity: list | None
    num_shards: int | None
    nodes: list | None
    max_node: list | None
    W: list | None
    nn_mapping: Any | None
    lsh: Any | None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        router = self
        router.lsh = LSH(self.dimension, int(math.log(self.approx_shards, 2) + 0.5))
        router.num_shards = router.lsh.max_partitions
        bucket_cls = S3Bucket if router.location.startswith("s3://") else LazyBucket
        self.buckets = [
            bucket_cls(segment_index=str(_), db_location=self.location)
            for _ in range(router.num_shards)
        ]

    def update_max_node(self, index, diff, vector):
        if self.max_node[index] is None:
            self.max_node[index] = (diff, vector)
        else:
            dx, _ = self.max_node[index]
            if diff < dx:
                self.max_node[index] = (diff, vector)

    def build_links(self):
        W = [-1] * len(self.nodes)
        for k, pair in enumerate(self.max_node):
            if not pair:
                continue
            (_, v) = pair
            idx = self.vector_router(v)
            W[k] = idx
        self.W = W
        return W

    def vector_router(self, vector: np.array) -> int:
        if isinstance(vector, list):
            vector = np.array(vector)
        closest_index = self.lsh.route(vector)
        return closest_index

    def adjacent_routing(self, vector) -> int:
        closest_index = self.lsh.route(vector)
        yield self.buckets[closest_index]

    def __repr__(self):
        return f"<Index({self.location=}, {self.metric_function=}, {self.max_cache_mb}, {self.num_shards=})>"

    def add(self, embedding, **attrs):
        shard_index = self.vector_router(embedding)
        self.buckets[shard_index].append(embedding, **attrs)
        return shard_index

    @timer_decorator
    def _query(self, vector, k: int = 4) -> list:
        results = []
        computed_distances = []
        rows = []
        vector = np.array(vector)
        ts = time.time()
        rows_append = rows.append
        for shard in self.adjacent_routing(vector):
            te = time.time() - ts
            logger.debug(f"adjacent routing took vps{1/te:.1f}")
            shard._lazy_load()
            shard_np = np.array(shard.vectors)

            closest_indices_d = shard.search(vector, k=k)
            closest_indices = [idx for idx, _ in closest_indices_d]
            shard_rows = shard.dirty_rows
            closest_vectors = shard_np[closest_indices]

            for idx in closest_indices[::-1]:
                rows_append(shard_rows[idx])
            shard_rows = []

            results.extend(list(closest_vectors))
            computed_distances.extend([d for _, d in closest_indices_d])
            if len(results) >= k:
                break
            logger.debug("next shard")

        result_vectors = np.array([vector for vector in results])
        if not result_vectors.any():
            return [], []

        combined = list(zip(computed_distances, result_vectors, rows))
        # Sort the combined list by distances
        sorted_combined = sorted(combined, key=lambda x: x[0])
        # Unzip the sorted list
        sorted_distances, sorted_vectors, sorted_rows = zip(*sorted_combined)
        return sorted_vectors[:k], sorted_rows[:k]

    def query(self, vector, k: int = 4) -> list:
        vectors, rows = self._query(vector, k)
        return vectors

    def persist(self):
        for bucket in self.buckets:
            if bucket.dirty:
                bucket.sync()

    def delete(self):
        """This function deletes all items in each bucket of a given object."""
        for bucket in self.buckets:
            bucket.delete()

    def delete_local(self):
        """This function delete_locals all items in each bucket of a given
        object."""
        for bucket in self.buckets:
            bucket.delete_local()

    def delete_remote(self):
        """This function delete_remotes all items in each bucket of a given
        object."""
        for bucket in self.buckets:
            bucket.delete_remote()

    @timer_decorator
    def load_local(self):
        for bucket in self.buckets:
            len(bucket)


class VectorLake(Index):
    def add(self, embedding: list, metadata: dict, document: str):
        if not metadata:
            # metadata can not be empty
            metadata = {"id": "1"}

        shard_index = self.vector_router(embedding)
        uid = self.buckets[shard_index].append(
            embedding, metadata=metadata, document=document
        )
        return uid

    def query(
        self,
        query_embeddings: list,
        n_results: int = 4,
    ):
        vectors, rows = super()._query(query_embeddings, k=n_results)
        return rows


class Partition(VectorLake):
    approx_shards: int = 0
    partition_key: str = "partition"

    def __init__(self, *args, **kwargs):
        BaseModel.__init__(self, *args, **kwargs)

        router = self
        router.granularity = []
        router.num_shards = 1
        router.nodes = np.random.rand(1, self.dimension)
        router.max_node = [None]
        bucket_cls = S3Bucket if router.location.startswith("s3://") else LazyBucket
        self.buckets = [
            bucket_cls(segment_index=self.partition_key, db_location=self.location)
        ]
