# VectorLake

VectorLake is a robust, vector database designed for low maintenance, cost, efficient storage and ANN querying of any size vector data distributed across S3 files.

<p>
<img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/msoedov/vector_lake" />
<img alt="GitHub Last Commit" src="https://img.shields.io/github/last-commit/msoedov/vector_lake" />
<img alt="" src="https://img.shields.io/github/repo-size/msoedov/vector_lake" />
<img alt="GitHub Issues" src="https://img.shields.io/github/issues/msoedov/vector_lake" />
<img alt="GitHub Pull Requests" src="https://img.shields.io/github/issues-pr/msoedov/vector_lake" />
<img alt="Github License" src="https://img.shields.io/github/license/msoedov/vector_lake" />
</p>

## 🏷 Features

- Inspired by article [Which Vector Database Should I Use? A Comparison Cheatsheet](https://navidre.medium.com/which-vector-database-should-i-use-a-comparison-cheatsheet-cb330e55fca)

- VectorLake created with tradeoff to minimize db maintenance, cost and provide custom data partitioning strategies

- Native Big Data Support: Specifically designed to handle large datasets, making it ideal for big data projects.

- Vector Data Handling: Capable of storing and querying high-dimensional vectors, commonly used for embedding storage in machine learning projects.projects.

- Efficient Search: Efficient nearest neighbors search, ideal for querying similar vectors in high-dimensional spaces. This makes it especially useful for querying for similar vectors in a high-dimensional space.

- Data Persistence: Supports data persistence on disk, network volume and S3, enabling long-term storage and retrieval of indexed data.

- Customizable Partitioning: Trade-off design to minimize database maintenance, cost, and provide custom data partitioning strategies.

- Native support of LLM Agents.

- Feature store for experimental data.

## 📦 Installation

To get started with VectorLake, simply install the package using pip:

```shell
pip install vector_lake
```

## ⛓️ Quick Start

```python
import numpy as np
from vector_lake import VectorLake

db = VectorLake(location="s3://vector-lake", dimension=5, approx_shards=243)
N = 100  # for example
D = 5  # Dimensionality of each vector
embeddings = np.random.rand(N, D)

for em in embeddings:
    db.add(em, metadata={}, document="some document")
db.persist()

db = VectorLake(location="s3://vector-lake", dimension=5, approx_shards=243)
# re-init test
db.query([0.56325391, 0.1500543, 0.88579166, 0.73536349, 0.7719873])

```

### Custom feature partition

Custom partition to group features by custom category

```python
import numpy as np
from vector_lake.core.index import Partition

if __name__ == "__main__":
    db = Partition(location="s3://vector-lake", partition_key="feature", dimension=5)
    N = 100  # for example
    D = 5  # Dimensionality of each vector
    embeddings = np.random.rand(N, D)

    for em in embeddings:
        db.add(em, metadata={}, document="some document")
    db.persist()

    db = Partition(location="s3://vector-lake", key="feature", dimension=5)
    # re-init test
    db.buckets
    db.query([0.56325391, 0.1500543, 0.88579166, 0.73536349, 0.7719873])

```

### Local persistent volume

```python
import numpy as np
from vector_lake import VectorLake

db = VectorLake(location="/mnt/db", dimension=5, approx_shards=243)
N = 100  # for example
D = 5  # Dimensionality of each vector
embeddings = np.random.rand(N, D)

for em in embeddings:
    db.add(em, metadata={}, document="some document")
db.persist()

db = VectorLake(location="/mnt/db", dimension=5, approx_shards=243)
# re-init test
db.query([0.56325391, 0.1500543, 0.88579166, 0.73536349, 0.7719873])

```

## Langchain Retrieval

```python
from langchain.document_loaders import TextLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from vector_lake.langchain import VectorLakeStore

loader = TextLoader("Readme.md")
documents = loader.load()

# split it into chunks
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# create the open-source embedding function
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = VectorLakeStore.from_documents(documents=docs, embedding=embedding)

query = "What is Vector Lake?"
docs = db.similarity_search(query)

# print results
print(docs[0].page_content)

```

## Why VectorLake?

VectorLake gives you the functionality of a simple, resilient vector database, but with very easy setup and low operational overhead. With it you've got a lightweight and reliable distributed vector store.

VectorLake leverages Hierarchical Navigable Small World (HNSW) for data partitioning across all vector data shards. This ensures that each modification to the system aligns with vector distance. You can learn more about the design here.

### Limitations

TBD

## 🛠️ Roadmap

## 👋 Contributing

Contributions to VectorLake are welcome! If you'd like to contribute, please follow these steps:

- Fork the repository on GitHub
- Create a new branch for your changes
- Commit your changes to the new branch
- Push your changes to the forked repository
- Open a pull request to the main VectorLake repository

Before contributing, please read the contributing guidelines.

## License

VectorLake is released under the MIT License.
