import os
import numpy as np
from flask import Flask, request, jsonify
from modules.extraction.preprocessing import DocumentProcessing
from modules.extraction.embedding import Embedding
from modules.retrieval.index.bruteforce import FaissBruteForce
from modules.retrieval.search import FaissSearch
from modules.generator.question_answering import QA_Generator

app = Flask(__name__)

STORAGE_DIRECTORY = "storage/"
CHUNKING_STRATEGY = 'fixed-length' # or 'sentence'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
TOP_K = 5  # Number of top documents to retrieve
DISTANCE_METRIC = 'cosine'  # or 'euclidean', 'dot_product', 'minkowski'


def initialize_index():
    """
    1. Parse through all the documents contained in storage/corpus directory
    2. Chunk the documents using either a'sentence' and 'fixed-length' chunking strategies (indicated by the CHUNKING_STRATEGY value):
        - The CHUNKING_STRATEGY will configure either fixed chunk or sentence chunking
    3. Embed each chunk using Embedding class, using 'all-MiniLM-L6-v2' text embedding model as default.
    4. Store vector embeddings of these chunks in a BruteForace index, along with the chunks as metadata. 
    5. This function should return the FAISS index
    """
    faiss_index_path = os.path.join(STORAGE_DIRECTORY, "faiss_index", "faiss_index_bruteforce.pkl")

    # Check if the FAISS index already exists
    if os.path.exists(faiss_index_path):
        print("FAISS index already exists. Loading from disk...")
        faiss_index = FaissBruteForce.load(faiss_index_path)
        return faiss_index

    # If the index does not exist, create it
    print("Initializing FAISS index...")
    # Step 1: Parse through all the documents contained in storage/corpus directory
    documents = [os.path.join(STORAGE_DIRECTORY, f) for f in os.listdir(STORAGE_DIRECTORY) if f.endswith('.txt.clean')]
    print(f"Total number of documents: {len(documents)}")

    # Step 2: Chunk the documents
    chunks = []
    for document in documents:
        # Initialize DocumentProcessing class
        document_processing = DocumentProcessing()

        if CHUNKING_STRATEGY == 'sentence':
            chunks.extend(document_processing.sentence_chunking(document, num_sentences=5, overlap_size=0))
        elif CHUNKING_STRATEGY == 'fixed-length':
            chunks.extend(document_processing.fixed_length_chunking(document, chunk_size=256, overlap_size=0))

    print(f"Total number of chunks: {len(chunks)}")

    # Step 3: Generate embeddings for each chunk
    embedding_model = Embedding(model_name=EMBEDDING_MODEL)
    embeddings = []
    for chunk in chunks:
        embedding_vector = embedding_model.encode(chunk)
        embeddings.append(embedding_vector)

    print(f"Total number of embeddings: {len(embeddings)}")

    # Step 4: Store vector embeddings of these chunks in a BruteForace index
    faiss_index = FaissBruteForce(dim=len(embeddings[0]), metric=DISTANCE_METRIC)
    faiss_index.add_embeddings(np.array(embeddings), metadata=chunks)

    # Save the index to disk
    faiss_index.save(faiss_index_path)
    print("FAISS index initialized and saved successfully.")

    return faiss_index

@app.route("/generate", methods=["POST"])
def generate_answer():
    """
    Generate an answer to a given query by running the retrieval and reranking pipeline.

    This endpoint accepts a POST request with a JSON body containing the "query" field.
    It preprocesses and indexes the corpus if necessary, retrieves top-k relevant documents,
    and uses a language model to generate a final answer.

    Example curl command:
    curl -X POST http://localhost:5000/generate \
         -H "Content-Type: application/json" \
         -d '{"query": "What is the role of antioxidants in green tea?"}'

    :return: JSON response containing the generated answer.
    """
    #######################################
    # TODO: Implement generate_answer()
    #######################################
    query = request.json.get("query")
    if not query:
        return jsonify({"error": "Query is required"}), 400
    query_vector = Embedding(model_name=EMBEDDING_MODEL).encode(query)
    
    # Retrieve top-k relevant documents
    faiss_index = initialize_index()
    faiss_search = FaissSearch(faiss_index, metric=DISTANCE_METRIC)
    distances, indices, retrieved_chunks = faiss_search.search(query_vector, k=TOP_K)

    if distances is None or indices is None or retrieved_chunks is None:
        return jsonify({"error": "Search failed for question: {question}"}), 400

    # Initializing the answer generator
    os.environ["MISTRAL_API_KEY"] = "J39mhyh5SiYD5HAeZFZYIPF9zbozad5H"
    API_KEY = os.environ.get("MISTRAL_API_KEY")
    generator = QA_Generator(api_key=API_KEY)

    # Generate the answer using the retrieved chunks
    answer = generator.generate_answer(query, retrieved_chunks)

    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(port=5000, debug=True, host='0.0.0.0')
