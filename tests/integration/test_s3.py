import numpy as np


class TestS3Integration:
    def test_end_to_end_s3(self, s3_index_factory):
        index = s3_index_factory(
            location="s3://unittest-vector-lake",
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
        index = s3_index_factory(
            location="s3://unittest-vector-lake",
            dimension=5,
            approx_shards=243,
            size=0,
        )
        closest_vectors = index.query(vector, 4)
        assert len(closest_vectors) == 1
        assert np.array_equal(closest_vectors[0]["vector"], vector)
        index.delete_remote()
