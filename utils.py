from langchain_ollama import OllamaEmbeddings
import argparse
import os
import shutil
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
# from langchain_community.vectorstores.chroma import Chroma
from langchain_chroma import Chroma


CHROMA_PATH = "chroma"
DATA_PATH = "data"


# Embeddings

def get_embedding_function():
    # embeddings = OllamaEmbeddings( model="qwen")
    embeddings = OllamaEmbeddings( model="bge-m3")
    return embeddings

# Database

def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, # number of characters
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        # db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/filename.md:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


# LLM

from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, ListOutputParser
from langchain.prompts import PromptTemplate

prompt_template = """
Answer the question based only on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

class RAGModel():
    def __init__(self, model: str, temperature: float = 3.5):
        self.model = OllamaLLM(model=model, temperature=temperature)
        self.prompt = PromptTemplate.from_template(prompt_template)
        self.chain = self.prompt | self.model | StrOutputParser()
        self.db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    def get_context(self, query_text: str):
        # Search the DB.
        # results = db.similarity_search_with_relevance_scores(query_text, k=5)
        results = self.db.similarity_search_with_score(query_text, k=5)

        # Store results as context
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        return context_text
    
    def query(self, query_text: str):
        # Get context from query
        context_text = self.get_context(query_text)

        print("Context:\n")
        print(context_text)
        print("\n#########")

        response = self.chain.invoke(
            {
                'context': context_text,
                'question': query_text
            }
        )
        print("\nResponse:")
        print(response)
        return response
    
    def set_prompt_template(self, template: str):
        self.prompt = PromptTemplate.from_template(template)

    