from langchain_neo4j import Neo4jGraph

# Other imports
from langchain_google_vertexai import VertexAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

neo4j_url = os.getenv("NEO4J_URL")
neo4j_username = os.getenv("NEO4J_USERNAME")
neo4j_password = os.getenv("NEO4J_PASSWORD")

llm = VertexAI(model_name="gemini-2.0-flash-001", temperature=0)

kg_transformer = LLMGraphTransformer(
    llm=llm,
    allowed_nodes=["Person", "Country", "Company"],
    allowed_relationships=["GENDER", "BORN_IN", "LOCATED_IN", "WORKED_AT"]
)

text_content = """
# Introduction
Marie Curie was a pioneering physicist and chemist.
## Early Life
Born in Warsaw, Poland, in 1867.
## Achievements
First woman to win a Nobel Prize.
"""

document = Document(page_content=text_content)

text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")])
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
recursive_sections = recursive_splitter.split_text(text_content)
sections = text_splitter.split_text(document.page_content)

graph_documents = []
for section in recursive_sections:
    metadata = section.metadata.copy()
    metadata.update({
        "source": "my_source",
        "author": "John Doe",
        "timestamp": "2025-02-28T12:00:00Z"
    })

    chunk = Document(page_content=section.page_content, metadata=metadata)

    graph_docs = kg_transformer.convert_to_graph_documents([chunk])
    graph_documents.extend(graph_docs)

graph = Neo4jGraph(url=neo4j_url, username=neo4j_username, password=neo4j_password)

print(graph_documents)
graph.add_graph_documents(graph_documents)