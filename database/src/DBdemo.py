from pymilvus import connections, Collection, AnnSearchRequest, MilvusClient
from milvus_model.hybrid import BGEM3EmbeddingFunction
from scipy.sparse import csr_matrix

# Utility Functions
def connect_to_milvus(uri="http://localhost:19530"):
    """Connect to the Milvus server."""
    connections.connect(uri=uri)

def load_collection(collection_name):
    """Load a Milvus collection."""
    collection = Collection(collection_name)
    collection.load()
    return collection

def convert_sparse_vector(sparse_array):
    """Convert a sparse array to Milvus-compatible format."""
    coo = sparse_array.tocoo()
    return [[(int(i), float(v)) for i, v in zip(coo.col, coo.data)]]

# Specialized Search Functions
def sparse_search(client, collection_name, query_vector, limit=10):
    """Perform a sparse search on the collection using MilvusClient."""
    query_vector = convert_sparse_vector(query_vector)
    results = client.search(
        collection_name=collection_name,
        anns_field="sparse_title_description",
        data=query_vector,
        limit=limit,
        output_fields=["entry_id", "entry_type", "title", "description", "metadata", "text_content", "code_content"],
    )
    return results

def dense_text_search(collection, query_dense_text_embedding, limit=10):
    """Perform a dense text search on the collection."""
    return collection.search(
        data=[query_dense_text_embedding],
        anns_field="dense_text_content",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=limit,
        output_fields=["entry_id", "entry_type", "title", "description", "metadata", "text_content", "code_content"],
    )

def write_hybrid_results_to_file(filename, results):
    """Write hybrid search results to a file."""
    with open(filename, "w", encoding="utf-8") as file:
        for result in results:
            file.write(
                f"ID: {result['entry_id']}\n"
                f"Sparse Distance: {result['sparse_distance']}\n"
                f"Dense Distance: {result['dense_distance']}\n"
                f"Combined Score: {result['combined_score']}\n\n"
                f"Type: {result['entry_type']}\n"
                f"Title: {result['title']}\n"
                f"Description: {result['description']}\n"
                f"Metadata: {result['metadata']}\n"
                f"Text Content: {result['text_content']}\n"
                f"Code Content: {result['code_content']}\n"
            )

def hybrid_search(collection, query_sparse_embedding, query_dense_text_embedding, sparse_weight=0.5, dense_weight=1.0, limit=10):
    """Perform a hybrid search on the collection using sparse and dense text embeddings."""
    sparse_results = sparse_search(
        client=MilvusClient(uri="http://localhost:19530"),
        collection_name=collection.name,
        query_vector=query_sparse_embedding,
        limit=limit,
    )

    dense_results = dense_text_search(collection, query_dense_text_embedding, limit)

    # Combine results by entry_id
    combined_results_dict = {}

    # Process sparse results
    for hits in sparse_results:  # Iterate over batches of hits
        for hit in hits:  # Iterate over individual hits
            entity = hit.get("entity", {})
            distance = hit.get("distance")
            
            # Use hit.id or similar field if entity is empty
            entry_id = entity.get("entry_id") if entity else getattr(hit, "id", None)
            if not entry_id:
                continue  # Skip if no valid entry_id is found

            if entry_id not in combined_results_dict:
                combined_results_dict[entry_id] = {
                    "entry_id": entry_id,
                    "entry_type": entity.get("entry_type") if entity else "N/A",
                    "title": entity.get("title") if entity else "N/A",
                    "description": entity.get("description") if entity else "N/A",
                    "metadata": entity.get("metadata") if entity else {},
                    "text_content": entity.get("text_content") if entity else "",
                    "code_content": entity.get("code_content") if entity else "",
                    "sparse_distance": distance,
                    "dense_distance": 0  # Initialize dense distance as 0
                }
            else:
                combined_results_dict[entry_id]["sparse_distance"] = distance

    # Process dense results
    for hits in dense_results:  # Iterate over batches of hits
        for hit in hits:  # Iterate over individual hits
            distance = getattr(hit, "distance", 0)
            entity = getattr(hit, "entity", None) or {}
            
            # Use hit.id or similar field if entity is empty
            entry_id = entity.get("entry_id") if entity else getattr(hit, "id", None)
            if not entry_id:
                continue  # Skip if no valid entry_id is found

            if entry_id not in combined_results_dict:
                combined_results_dict[entry_id] = {
                    "entry_id": entry_id,
                    "entry_type": entity.get("entry_type") if entity else "N/A",
                    "title": entity.get("title") if entity else "N/A",
                    "description": entity.get("description") if entity else "N/A",
                    "metadata": entity.get("metadata") if entity else {},
                    "text_content": entity.get("text_content") if entity else "",
                    "code_content": entity.get("code_content") if entity else "",
                    "sparse_distance": 0,  # Initialize sparse distance as 0
                    "dense_distance": distance
                }
            else:
                combined_results_dict[entry_id]["dense_distance"] = distance

    # Combine sparse and dense distances
    combined_results = []
    for entry_id, data in combined_results_dict.items():
        combined_score = (
            data["sparse_distance"] * sparse_weight
            + data["dense_distance"] * dense_weight
        )
        data["combined_score"] = combined_score
        combined_results.append(data)

    # Sort results by combined score (highest to lowest)
    combined_results.sort(key=lambda x: x["combined_score"], reverse=True)

    # Limit results to the top `limit` entries
    return combined_results[:limit]

def code_search(collection, query_dense_code_embedding, limit=10):
    """Perform a dense code search on the collection."""
    return collection.search(
        data=[query_dense_code_embedding],
        anns_field="dense_code_snippet",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=limit,
        output_fields=["entry_id", "entry_type", "title", "description", "metadata", "text_content", "code_content"],
    )

def format_results(results):
    """Format the search results for better readability."""
    formatted = []
    for hits in results:
        for hit in hits:
            # Check if hit is a dictionary or an object
            if isinstance(hit, dict):  # For sparse search
                entity = hit.get("entity", {})
                distance = hit.get("distance")
            else:  # For dense or code search
                entity = getattr(hit, "entity", {})
                distance = getattr(hit, "distance")

            formatted.append(
                f"ID: {entity.get('entry_id')}\n"
                f"Distance: {distance}\n"
                f"Type: {entity.get('entry_type')}\n"
                f"Title: {entity.get('title')}\n"
                f"Description: {entity.get('description')}\n"
                f"Metadata: {entity.get('metadata')}\n"
                f"Content: {entity.get('text_content')}\n"
                f"Code: {entity.get('code_content')}\n"
            )
    return "\n\n".join(formatted)

def write_results_to_file(filename, results):
    """Write search results to a file."""
    formatted = format_results(results)
    with open(filename, "w", encoding="utf-8") as file:
        file.write(formatted)

# Main Search Loop
def main():
    connect_to_milvus()
    collection_name = "nextjs_docs"
    collection = load_collection(collection_name)

    ef = BGEM3EmbeddingFunction(use_fp16=False, device="cpu")

    while True:
        query = input("Enter your search query (or 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Exiting the search loop.")
            break

        query_embeddings = ef([query])
        query_sparse_embedding = csr_matrix(query_embeddings.get("sparse", None))
        query_dense_text_embedding = query_embeddings.get("dense", None)[0]
        query_dense_code_embedding = query_embeddings.get("dense", None)[0]

        search_type = input(
            "Select search type: (1) Sparse on documents, (2) Dense text, (3) Hybrid, (4) Query Code: "
        ).strip()

        try:
            if search_type == "1":
                if query_sparse_embedding is None:
                    raise ValueError("Sparse embedding is missing.")
                sparse_results = sparse_search(
                    client=MilvusClient(uri="http://localhost:19530"),
                    collection_name=collection_name,
                    query_vector=query_sparse_embedding,
                )
                write_results_to_file("sparse_results.txt", sparse_results)
                print("Sparse search results saved to 'sparse_results.txt'.")

            elif search_type == "2":
                if query_dense_text_embedding is None:
                    raise ValueError("Dense text embedding is missing.")
                dense_results = dense_text_search(collection, query_dense_text_embedding)
                write_results_to_file("dense_text_results.txt", dense_results)
                print("Dense text search results saved to 'dense_text_results.txt'.")

            elif search_type == "3":
                if query_sparse_embedding is None or query_dense_text_embedding is None:
                    raise ValueError("Both sparse and dense text embeddings are required for hybrid search.")
                hybrid_results = hybrid_search(
                    collection,
                    query_sparse_embedding=query_sparse_embedding,
                    query_dense_text_embedding=query_dense_text_embedding,
                    sparse_weight=1,
                    dense_weight=0.8,
                )
                write_hybrid_results_to_file("hybrid_results.txt", hybrid_results)
                print("Hybrid search results saved to 'hybrid_results.txt'.")

            elif search_type == "4":
                if query_dense_code_embedding is None:
                    raise ValueError("Dense code embedding is missing.")
                code_results = code_search(collection, query_dense_code_embedding)
                write_results_to_file("code_results.txt", code_results)
                print("Code search results saved to 'code_results.txt'.")

            else:
                print("Invalid option. Please try again.")

        except Exception as e:
            print(f"Error: {e}")

        cont = input("Perform another search? (yes/no): ").strip().lower()
        if cont != "yes":
            print("Exiting.")
            break

if __name__ == "__main__":
    main()
