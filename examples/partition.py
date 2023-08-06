import numpy as np

from examples.env import *  # noqa
from vector_lake.core.index import Partition

if __name__ == "__main__":
    db = Partition(location="s3://vector-lake", partition_key="feature", dimension=5)
    # db.delete()
    N = 100  # for example
    D = 5  # Dimensionality of each vector
    embeddings = np.random.rand(N, D)

    for em in embeddings:
        db.add(em, metadata={}, document="some document")
    # print(db.query([0.56325391, 0.1500543, 0.88579166, 0.73536349, 0.7719873]))
    db.persist()

    db = Partition(location="s3://vector-lake", key="feature", dimension=5)
    # re-init test
    db.buckets
    db.query([0.56325391, 0.1500543, 0.88579166, 0.73536349, 0.7719873])
