import numpy as np

from vector_lake.core.index import make_granularity, make_nodes


class TestMakeGranularity:
    def test_D_is_1(self):
        D = 1
        M = 10
        levels = make_granularity(D, M)
        assert len(levels) == D
        assert np.prod(levels) == M
        assert all(isinstance(level, int) for level in levels)
        assert all(level > 0 for level in levels)
        assert all(level <= M for level in levels)
        assert sum(levels) == M

    def test_positive_integers(self):
        D = 3
        M = 10
        levels = make_granularity(D, M)
        assert isinstance(levels, list)
        assert all(isinstance(level, int) for level in levels)
        assert all(level > 0 for level in levels)


class TestMakeNodes:
    def test_levels_and_num_shards_as_1(self):
        levels = [1, 1]
        num_shards = 1
        nodes = make_nodes(levels, num_shards)
        assert nodes.shape == (1, 2)
        assert np.allclose(nodes, np.array([[0.0, 0.0]]))

    def test_default_values(self):
        levels = [2, 2]
        num_shards = 4
        nodes = make_nodes(levels, num_shards)
        assert nodes.shape == (4, 2)
        assert np.allclose(
            nodes, np.array([[0.0, 0.0], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]])
        )


class TestEndToEnd:
    def test_end_to_end(self, index_factory):
        index = index_factory(
            location="/tmp/cosine/empty",
            dimension=5,
            approx_shards=243,
            size=0,
            force_clean=True,
        )
        vector = np.random.rand(1, 5)[0]
        closest_vectors = index.query(vector, 4)
        assert len(closest_vectors) == 0
        index.add(vector, metadata={"id": 1}, document="unit test")
        closest_vectors = index.query(vector, 4)
        assert len(closest_vectors) == 1
        index.persist()
        # Reload the index
        index = index_factory(
            location="/tmp/cosine/empty",
            dimension=5,
            approx_shards=243,
            size=0,
        )
        closest_vectors = index.query(vector, 4)
        assert len(closest_vectors) == 1
        assert np.array_equal(closest_vectors[0]["vector"], vector)
