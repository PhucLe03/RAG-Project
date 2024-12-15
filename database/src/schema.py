import pandas as pd
from pymilvus import DataType, FieldSchema, CollectionSchema, Collection, connections, utility
import json
import numpy as np

DENSE_VECTOR_DIM = 1024  # Replace with the actual dimension of your dense vectors

# Define field schemas
def create_field_schemas():
    return [
        FieldSchema(
            name="entry_id",
            dtype=DataType.VARCHAR,
            is_primary=True,
            auto_id=False,
            max_length=255,
            description="Unique identifier for each entry (document, section, or code snippet).",
        ),
        FieldSchema(
            name="entry_type",
            dtype=DataType.VARCHAR,
            max_length=20,
            description="Type of entry: document, section, or code_snippet.",
        ),
        FieldSchema(
            name="parent_id",
            dtype=DataType.VARCHAR,
            max_length=255,
            description="The entry_id of the parent entry (if applicable).",
        ),
        FieldSchema(
            name="title",
            dtype=DataType.VARCHAR,
            max_length=255,
            description="Title of the document, section, or code snippet.",
        ),
        FieldSchema(
            name="description",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="Description or summary of the entry.",
        ),
        FieldSchema(
            name="metadata",
            dtype=DataType.JSON,
            description="JSON field for custom metadata (e.g., programming language, related links, or tags).",
        ),
        FieldSchema(
            name="version",
            dtype=DataType.VARCHAR,
            max_length=20,
            description="Framework version associated with the entry.",
            is_partition_key=True,
        ),
        FieldSchema(
            name="tag",
            dtype=DataType.VARCHAR,
            max_length=255,
            description="Tag derived from nav_title to indicate the subject of the entry.",
        ),
        FieldSchema(
            name="text_content",
            dtype=DataType.VARCHAR,
            max_length=65535,
            description="Raw text content of the entry (e.g., document or section body).",
        ),
        FieldSchema(
            name="code_content",
            dtype=DataType.VARCHAR,
            max_length=65535,
            description="Actual code snippet content (only for code_snippet entries).",
        ),
        FieldSchema(
            name="sparse_title_description",
            dtype=DataType.SPARSE_FLOAT_VECTOR,
            description="Sparse vector embedding for the title + description.",
        ),
        FieldSchema(
            name="dense_text_content",
            dtype=DataType.FLOAT_VECTOR,
            dim=DENSE_VECTOR_DIM,
            description="Dense vector embedding for the text content.",
        ),
        FieldSchema(
            name="dense_code_snippet",
            dtype=DataType.FLOAT_VECTOR,
            dim=DENSE_VECTOR_DIM,
            description="Dense vector embedding for the code content.",
        ),
    ]

# Connect to Milvus database
def connect_to_milvus(uri="http://localhost:19530"):
    connections.connect(uri=uri)

# Create or reset the collection
def create_collection(collection_name, schema):
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        collection.drop()
        print(f"Collection '{collection_name}' dropped.")

    collection = Collection(
        name=collection_name,
        schema=schema,
        consistency_level="Strong",
    )
    print(f"Collection '{collection_name}' created successfully.")
    return collection

# Create indices for the collection
def create_indices(collection):
    sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}
    collection.create_index("sparse_title_description", sparse_index)

    dense_text_index = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    }
    collection.create_index("dense_text_content", dense_text_index)

    dense_code_index = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200},
    }
    collection.create_index("dense_code_snippet", dense_code_index)

    collection.load()
    print("Collection loaded with indices.")

# Helper functions for data parsing
def safe_parse(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return {}

def parse_sparse_vector(value):
    if isinstance(value, str):
        try:
            sparse_data = eval(value)
            return {int(item[0]): float(item[1]) for item in sparse_data}
        except (SyntaxError, ValueError, TypeError):
            return {}
    return {}

def enforce_types(row):
    return {
        "entry_id": str(row["entry_id"]) if pd.notna(row["entry_id"]) else "",
        "entry_type": str(row["entry_type"]) if pd.notna(row["entry_type"]) else "",
        "parent_id": str(row["parent_id"]) if pd.notna(row["parent_id"]) else "",
        "title": str(row["title"]) if pd.notna(row["title"]) else "",
        "description": str(row["description"]) if pd.notna(row["description"]) else "",
        "metadata": safe_parse(row["metadata"]) if pd.notna(row["metadata"]) else {},
        "version": str(row["version"]) if pd.notna(row["version"]) else "",
        "tag": str(row["tag"]) if pd.notna(row["tag"]) else "",
        "text_content": str(row["text_content"])[:65535] if pd.notna(row["text_content"]) else "",
        "code_content": str(row["code_content"]) if pd.notna(row["code_content"]) else "",
        "sparse_title_description": parse_sparse_vector(row["sparse_title_description"]) if pd.notna(row["sparse_title_description"]) else {1: 0.01},
        "dense_text_content": json.loads(row["dense_text_content"]) if pd.notna(row["dense_text_content"]) and isinstance(row["dense_text_content"], str) else [0.0] * DENSE_VECTOR_DIM,
        "dense_code_snippet": json.loads(row["dense_code_snippet"]) if pd.notna(row["dense_code_snippet"]) and isinstance(row["dense_code_snippet"], str) else [0.0] * DENSE_VECTOR_DIM,
    }

# Insert data into the collection
def insert_data_in_batches(collection, file_path, batch_size=100):
    data = pd.read_csv(file_path)

    for start_idx in range(0, len(data), batch_size):
        end_idx = start_idx + batch_size
        batch = data.iloc[start_idx:end_idx]
        entities = [enforce_types(row) for _, row in batch.iterrows()]

        collection.insert(entities)
        print(f"Inserted batch {start_idx} to {end_idx}.")

# Main execution
def main():
    connect_to_milvus()

    collection_name = "nextjs_docs"
    schema = CollectionSchema(
        fields=create_field_schemas(),
        description="Schema for Next.js documentation, including documents, sections, and code snippets.",
        auto_id=False,
    )

    collection = create_collection(collection_name, schema)
    create_indices(collection)

    csv_file_path = "./preprocessed/merged_output.csv"  # Replace with the actual path to your CSV file
    insert_data_in_batches(collection, csv_file_path)

    collection.release()
    print(f"Collection '{collection_name}' is ready.")

if __name__ == "__main__":
    main()