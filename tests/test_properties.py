import math
import tempfile

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vector_lake import Index
from vector_lake.core.index import make_granularity


def index_factory(dimension=5, approx_shards=243, size=100, location="/tmp/cosine_x"):
    db = Index(location=location, dimension=dimension, approx_shards=approx_shards)
    N = size
    D = dimension
    embeddings = np.random.rand(N, D)
    for em in embeddings:
        db.add(em)
    return db


@given(
    D=st.integers(min_value=1, max_value=100), M=st.integers(min_value=1, max_value=100)
)
def test_make_granularity(D, M):
    levels = make_granularity(D, M)
    assert isinstance(levels, list)
    assert np.prod(levels) <= M
    assert all(not math.isnan(x) for x in levels)


@given(vector=st.lists(st.floats(min_value=-1, max_value=1), min_size=5, max_size=5))
def test_vector_router(vector):
    vector = np.array(vector)
    index = index_factory(dimension=5, approx_shards=243, size=100)
    node_index = index.vector_router(vector)
    assert 0 <= node_index < index.num_shards


@given(embedding=st.lists(st.floats(min_value=-1, max_value=1), min_size=5, max_size=5))
def test_add(embedding):
    embedding = np.array(embedding)
    index = index_factory(dimension=5, approx_shards=243, size=0)
    shard_index = index.add(embedding)
    assert 0 <= shard_index < index.num_shards


@given(
    vector=st.lists(st.floats(min_value=-1, max_value=1), min_size=5, max_size=5),
    k=st.integers(min_value=1, max_value=10),
)
def test_query(vector, k):
    vector = np.array(vector)
    index = index_factory(dimension=5, approx_shards=243, size=20)
    closest_vectors = index.query(vector, k)
    assert len(closest_vectors) <= k
    assert all(len(v) == 5 for v in closest_vectors)


@given(
    vector=st.lists(st.floats(min_value=-1, max_value=1), min_size=5, max_size=5),
    k=st.integers(min_value=1, max_value=10),
    size=st.integers(min_value=1, max_value=10),
)
def test_query_size_fuzzing(vector, k, size):
    vector = np.array(vector)
    index = index_factory(dimension=5, approx_shards=243, size=k)
    closest_vectors = index.query(vector, k)
    assert len(closest_vectors) <= k
    assert all(len(v) == 5 for v in closest_vectors)


@settings(deadline=500)
@given(
    vector=st.lists(st.floats(min_value=-1, max_value=1), min_size=5, max_size=5),
)
def test_query_empty(vector):
    vector = np.array(vector)
    index = index_factory(
        location="/tmp/cosine/empty", dimension=5, approx_shards=243, size=0
    )
    closest_vectors = index.query(vector, 4)
    assert len(closest_vectors) == 0
    index.delete_local()


# TODO: fix [0.0, 0.0, 0.0, 0.0, 0.0] case
@settings(deadline=500)
@pytest.mark.skip(reason="benchmarking")
@given(
    vector=st.lists(st.floats(min_value=0.1, max_value=1), min_size=5, max_size=5),
)
def test_query_persistent(vector):
    vector = np.array(vector)
    with tempfile.TemporaryDirectory() as temp_dir:
        index = index_factory(location=temp_dir, dimension=5, approx_shards=243, size=0)
        index.add(vector)

        results = index.query(vector, 4)
        assert len(results) == 1

        index.persist()
        results = index.query(vector, 4)
        assert len(results) == 1

        index = Index(location=index.location, dimension=5, approx_shards=243)
        results_after = index.query(vector, 4)
        assert len(results_after) == 1
        assert np.array_equal(results, results_after)


@pytest.mark.skip(reason="benchmarking")
@pytest.mark.parametrize(
    "db_size, shards",
    [(1_000, 243), (10_000, 243), (100_000, 243), (100_000, 1243), (1_000_000, 500)],
)
def test_benchmark(benchmark, db_size, shards):
    index = index_factory(dimension=5, approx_shards=shards, size=db_size)

    N = 1  # for example
    D = 5  # Dimensionality of each vector
    embeddings = np.random.rand(N, D)

    @benchmark
    def query_one():
        for em in embeddings:
            index.query(em, 4)
