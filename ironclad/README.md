# Quick Start
Instructions to start the FLASK service and the curl command to make detections and add new gallery images

### **Step 1: Go to the Root Directory**
Ensure you are at the project directory `ironclad`. 

```
cd ironclad
```

---

### **Step 2: Start the Flask Service and Initialize Index**
```
python app.py
```
It will scan the current gallery images under "storage/multi_image_gallery" and construct the FAISS index accordingly. 

**Key Attributes**
```
DEFAULT_K = '3'
MODEL = 'vggface2'
INDEX = 'HNSW'
SIMILARITY_MEASURE = 'cosine'
FAISS_INDEX_DIR = 'faiss_index/'
GALLERY_IMAGE_DIR = 'storage/multi_image_gallery/'
```
For more details about how the key attributes are determined, please check the [SystemReport.md](https://github.com/creating-ai-enabled-systems-spring-2025/kuang-jizi/blob/main/ironclad/SystemReport.md) file. 

In this step, a new index will be built with key attributes above, if there is no existing index file under "faiss_index" folder. The expected output for initializing index is as below. 

Output:

```
FAISS index file not found. Initializing new index with gallery images.

* Generating embeddings with vggface2 model for 1001 identities in the gallery...
* Gallery_embeddings generated: 2265 embeddings with 2265 metadata. Time spent: 574.86 seconds

* Initializing FAISS HNSW index with 1001 identities and 2265 images...
* FAISS HNSW index initialized. Time spent: 0.17 seconds

```
> Note that it may take about 10 mins for initializtion to complete


If the index exists, then the output message should be:
```
FAISS index file found: faiss_index/FaissHNSW_cosine_index.pkl
```

---

### **Step 3: Identify Personnel**

```
curl -X POST -F "image=@/Users/JK/Dev/JHU/Class4_AI_enabled_system/Lab/kuang-jizi/ironclad/storage/probe/Aaron_Sorkin/Aaron_Sorkin_0002.jpg" -F "k=5" http://localhost:5000/identify
```

* Absolute path may be needed for curl cmd. To find the path try `realpath {image_relative_path}` (e.g. `realpath storage/probe/Aaron_Sorkin/Aaron_Sorkin_0002.jpg`) to get the absolute path.
* In the example, top 5 identities will be retrieved for given probe image.

Output:

```
{
  "message": "Returned top-5 identities",
  "ranked identities": [
    "Kathleen_Kennedy_Townsend",
    "Donna_Shalala",
    "John_Negroponte",
    "Bud_Selig",
    "Paul_McCartney"
  ]
}
```

### **Step 4: Add New Image**

Add new image to the gallery directory and index into catalog.

```
curl -X POST -F "image=@/Users/JK/Dev/JHU/Class4_AI_enabled_system/Lab/kuang-jizi/ironclad/storage/probe/Aaron_Sorkin/Aaron_Sorkin_0002.jpg" -F "name=Aaron_Sorkin" http://localhost:5000/add
```
* In this example, the probe image of "Aaron_Sorkin" will be added to gallery.


Output:

```
{
  "message": "New image added to gallery (as Aaron_Sorkin) and indexed into catalog faiss_index/FaissHNSW_cosine_index.pkl."
}

```