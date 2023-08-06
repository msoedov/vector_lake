import numpy as np

from examples.env import *  # noqa
from vector_lake import VectorLake

if __name__ == "__main__":
    db = VectorLake(location="s3://vector-lake", dimension=5, approx_shards=243)
    # db.delete()
    N = 100  # for example
    D = 5  # Dimensionality of each vector
    embeddings = np.random.rand(N, D)

    for em in embeddings:
        db.add(em, metadata={}, document="some document")
    # print(db.query([0.56325391, 0.1500543, 0.88579166, 0.73536349, 0.7719873]))
    db.persist()

    db = VectorLake(location="s3://vector-lake", dimension=5, approx_shards=243)
    # re-init test
    db.buckets
    db.query([0.56325391, 0.1500543, 0.88579166, 0.73536349, 0.7719873])
