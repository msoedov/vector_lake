import numpy as np

from vector_lake.core import Index

if __name__ == "__main__":
    db = Index(location="cosine_db/db", dimension=5, approx_shards=243)

    N = 100  # for example
    D = 5  # Dimensionality of each vector
    embeddings = np.random.rand(N, D)

    for em in embeddings:
        db.add(em)

    db.query([0.56325391, 0.1500543, 0.88579166, 0.73536349, 0.7719873])
