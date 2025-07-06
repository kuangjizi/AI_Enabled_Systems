"""
Flask app for processing images.

This script provides two endpoints:
1. /identify: Processes an image and returns the top-k identities.
2. /add: Adds a provided image to the gallery with an associated name.

Usage:
    Run the app with: python app.py
    Sample curl command for /identify:
        curl -X POST -F "image=@/path/to/image.jpg" -F "k=3" http://localhost:5000/identify
        
    Sample curl command for /add:
        curl -X POST -F "image=@/path/to/image.jpg" -F "name=Firstname_Lastname" http://localhost:5000/add
"""

import os
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import pickle
import time

from modules.extraction.preprocessing import Preprocessing
from modules.extraction.embedding import Embedding 
from modules.retrieval.index.bruteforce import FaissBruteForce
from modules.retrieval.index.lsh import FaissLSH
from modules.retrieval.index.hnsw import FaissHNSW
from modules.retrieval.search import FaissSearch

app = Flask(__name__)

## List of designed parameters: 
# (Configure these parameters according to your design decisions)
DEFAULT_K = '3'
MODEL = 'vggface2'
INDEX = 'HNSW'
SIMILARITY_MEASURE = 'cosine'
FAISS_INDEX_DIR = 'faiss_index/'
GALLERY_IMAGE_DIR = 'storage/multi_image_gallery/'

# Helper function to generate embedding with give image path
def generate_embedding(model_name, image_path, target_image_size=160):
    # Execute preprocessing
    preprocessing = Preprocessing(image_size=target_image_size, device='mps')
    image = Image.open(image_path)
    image = preprocessing.process(image) # preprocessed image

    # Generate embedding for given image
    model = Embedding(pretrained=model_name, device='mps') # Use Metal (Apple GPU)
    embedding_vector = model.encode(image)

    return embedding_vector

# Helper function to dynamically initialize and save Faiss index 
def initialize_faiss_index(dim, index_type, similarity_measure, embeddings, metadata):
    if index_type == 'BruteForce':
        faiss_index = FaissBruteForce(dim=dim, metric=similarity_measure)
    elif index_type == 'LSH':
        faiss_index = FaissLSH(dim=dim, metric=similarity_measure)
    elif index_type == 'HNSW':
        faiss_index = FaissHNSW(dim=dim, metric=similarity_measure)
    else:
        raise ValueError(f"Unsupported index type: {index_type}")

    faiss_index.add_embeddings(embeddings, metadata)
    faiss_index_name = f"Faiss{index_type}_{similarity_measure}_index.pkl"
    faiss_index_filepath = os.path.join(FAISS_INDEX_DIR, faiss_index_name)
    faiss_index.save(faiss_index_filepath)

# Helper function to load a serialized FAISS index instance from a file.
def load_faiss_index(filepath):
    with open(filepath, 'rb') as f:
        instance = pickle.load(f)
    return instance

@app.route('/identify', methods=['POST'])
def identify():
    """
    Process the probe image to identify top-k identities in the gallery.

    Expects form-data with:
      - image: Image file to be processed.
      - k: (optional) Integer specifying the number of top identities 
           (default is 3).

    Returns:
      JSON response with a success message and the provided value of k.
      If errors occur, returns a JSON error message with the appropriate status code.
    """
    # Check if the request has the image file
    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    # Retrieve and validate the integer parameter "k"
    try:
        k = int(request.form.get('k', DEFAULT_K))
    except ValueError:
        return jsonify({"error": "Invalid integer for parameter 'k'"}), 400

    # Convert the image into a NumPy array
    try:
        image = np.array(Image.open(file))
        print(image)
    except Exception as e:
        return jsonify({
            "error": "Failed to convert image to numpy array",
            "details": str(e)
        }), 500

    ########################################
    # TASK 1a: Implement /identify endpoint
    #         to return the top-k identities
    #         of the provided probe.
    ########################################

    # Preprocess probe image
    probe_embedding = generate_embedding(MODEL, image_path=file, target_image_size=160)
    probe_vector = np.array(probe_embedding, dtype=np.float32).reshape(1, -1)

    # Load Index
    faiss_index_name = f"Faiss{INDEX}_{SIMILARITY_MEASURE}_index.pkl"
    faiss_index = load_faiss_index(os.path.join(FAISS_INDEX_DIR, faiss_index_name))
    faiss_search = FaissSearch(faiss_index, metric=SIMILARITY_MEASURE)

    # Start Retrieval
    distances, indices, meta_results = faiss_search.search(probe_vector, k)

    return jsonify({
        "message": f"Returned top-{k} identities",
        "ranked identities": [m for m in meta_results]
    }), 200


@app.route("/add", methods=['POST'])
def add():
    """
    Add a provided image to the gallery with an associated name.

    Expects form-data with:
      - image: Image file to be added.
      - name: String representing the identity associated with the image.

    Returns:
      JSON response confirming the image addition.
      If errors occur, returns a JSON error message with the appropriate status code.
    """
    # Check if the request has the image file
    if 'image' not in request.files:
        return jsonify({"Error": "No image part in the request"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"Error": "No file selected for uploading"}), 400

    # Convert the image into a NumPy array
    try:
        image = np.array(Image.open(file))
        print(image)
    except Exception as e:
        return jsonify({
            "error": "Failed to convert image to numpy array",
            "details": str(e)
        }), 500

    # Retrieve the 'name' parameter
    name = request.form.get('name')
    if not name:
        return jsonify({"Error": "Must have associated 'name'"}), 400

    ########################################
    # TASK 1b: Implement `/add` endpoint to
    #         add the provided image to the 
    #         catalog.
    ########################################
    # Save the image file
    image_path = os.path.join(GALLERY_IMAGE_DIR, name, file.filename) # e.g. storage/multi_image_gallery/Aaron_Sorkin/Aaron_Sorkin_0001.jpg
    try:
        Image.fromarray(image).save(image_path)
    except Exception as e:
        return jsonify({
            "error": "Failed to save image file",
            "details": str(e)
        }), 500

    # Preprocess the image and generate embedding
    image_embedding = generate_embedding(MODEL, image_path, target_image_size=160)
    image_vector = np.array(image_embedding, dtype=np.float32).reshape(1, -1)

    # Load Index
    faiss_index_name = f"Faiss{INDEX}_{SIMILARITY_MEASURE}_index.pkl"
    faiss_index = load_faiss_index(os.path.join(FAISS_INDEX_DIR, faiss_index_name))

    # Add the new image to the index
    faiss_index.add_embeddings(image_vector, [name])
    
    # Save the updated index
    with open(os.path.join(FAISS_INDEX_DIR, faiss_index_name), 'wb') as f:
        pickle.dump(faiss_index, f)

    return jsonify({
        "message": f"New image added to gallery (as {name}) and indexed into catalog {os.path.join(FAISS_INDEX_DIR, faiss_index_name)}."
    })


if __name__ == '__main__':
    # Create the directory for the FAISS index if it does not exist
    if not os.path.exists(FAISS_INDEX_DIR):
        os.makedirs(FAISS_INDEX_DIR)

    # Initialize the FAISS index if it does not exist
    faiss_index_name = f"Faiss{INDEX}_{SIMILARITY_MEASURE}_index.pkl"
    if not os.path.exists(os.path.join(FAISS_INDEX_DIR, faiss_index_name)):
        print(f"FAISS index file not found. Initializing new index with gallery images.\n")

        # Generate gallery embeddings from existing gallery images
        gallery_embeddings, gallery_metadata = [], []
        gallery_personnel = sorted(os.listdir(GALLERY_IMAGE_DIR))

        print(f"* Generating embeddings with {MODEL} model for {len(gallery_personnel)} identities in the gallery...")
        start_embedding_time = time.time()
        for identity in gallery_personnel:
            # ignore system files
            if identity == '.DS_Store': 
                continue 

            # Iterate subdirectories and generate embeddings
            gallery_image_files = os.listdir(os.path.join(GALLERY_IMAGE_DIR, identity))
            for img_file in gallery_image_files:
                gallery_image_path = os.path.join(GALLERY_IMAGE_DIR, identity, img_file)
                gallery_embeddings.append(generate_embedding(MODEL, gallery_image_path))
                gallery_metadata.append(identity)

        end_embedding_time = time.time()
        print(f"* Gallery_embeddings generated: {len(gallery_embeddings)} embeddings with {len(gallery_metadata)} metadata. Time spent: {(end_embedding_time - start_embedding_time):.2f} seconds\n")

        # Initialize the index with the gallery embeddings
        print(f"* Initializing FAISS {INDEX} index with {len(gallery_personnel)} identities and {len(gallery_embeddings)} images...")
        start_index_time = time.time()
        dim = len(gallery_embeddings[0])
        initialize_faiss_index(dim, INDEX, SIMILARITY_MEASURE, gallery_embeddings, gallery_metadata) 
        end_index_time = time.time()
        print(f"* FAISS {INDEX} index initialized. Time spent: {(end_index_time - start_index_time):.2f} seconds")

    else:
        print(f"FAISS index file found: {os.path.join(FAISS_INDEX_DIR, faiss_index_name)}\n")

    app.run(port=5000, debug=True, host='0.0.0.0')