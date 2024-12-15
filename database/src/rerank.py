from milvus_model.reranker import BGERerankFunction

bge_rf = BGERerankFunction(
    model_name="BAAI/bge-reranker-v2-m3",  # Specify the model name. Defaults to `BAAI/bge-reranker-v2-m3`.
    device="cpu" # Specify the device to use, e.g., 'cpu' or 'cuda:0'
)

def read_list_from_file(filename):
    with open(filename, 'r') as file:
        data_list = [line.strip() for line in file.readlines()]  # Read lines and remove newline characters
    return data_list

query = "The first artificial intelligence"

documents = read_list_from_file("hybrid_results")

results = bge_rf(
    query=query,
    documents=documents,
    top_k=3,
)

for result in results:
    print(f"Score: {result.score:.6f}")
    print(f"Text: {result.text}\n")
