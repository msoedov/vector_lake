import os
from functools import partial

import numpy as np
import pytest

from vector_lake import VectorLake


def factory_fn(
    dimension=5,
    approx_shards=243,
    size=100,
    location="/tmp/cosine_x",
    force_clean=False,
):
    db = VectorLake(location=location, dimension=dimension, approx_shards=approx_shards)
    if force_clean:
        db.delete_local()
    N = size
    D = dimension
    embeddings = np.random.rand(N, D)
    for i, em in enumerate(embeddings):
        db.add(em, metadata={"id": i}, document=f"unit test: {i}")
    return db


@pytest.fixture(scope="session")
def index_factory():
    return factory_fn


@pytest.fixture(scope="session")
def s3_index_factory():
    os.environ["LOCALSTACK_ENDPOINT_URL"] = "http://134.209.75.14:4566"
    os.environ["AWS_ACCESS_KEY_ID"] = "foo"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "bar"
    os.environ["AWS_DEFAULT_REGION"] = "us-west-2"
    return partial(factory_fn, location="s3://unittest-vector-lake")
