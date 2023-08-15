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
