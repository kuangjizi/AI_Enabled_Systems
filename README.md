# JHU spring 2025 - Creating AI-enabled System

## Assignment 1
### Objectives
* Create a Python class called RawDataHandler to handle data extraction, transformation, and descriptive analysis of raw data for machine learning tasks.
* Detailed instrunctions [here](https://jhu.instructure.com/courses/90236/assignments/944089?module_item_id=4401870)
### CodePointer
* securebank/modules/data/raw_data_handler.py

## Assignment 2
### Objectives
* Task 1: In a python script called securebank/app.py, implement a Flask server with the following endpoint:
* Task 2: Write a securebank/Dockerfile to build a Docker image and run a Docker container for users to interact with the system.
* Task 3: In a securebank/README.md file, write a "quick start" instructions you used to start the server (using Docker) and the curl command you used to predict if the transaction is fraudulent.
* Detailed instrunctions [here](https://jhu.instructure.com/courses/90236/assignments/944090?module_item_id=4401871)
### CodePointer
* securebank/modules/model.py
* securebank/app.py
* securebank/Dockerfile
* securebank/requirements.txt
* securebank/README.md
* securebank/test.json

## Assignment 3
### Objectives
* Task 1: Review the preprocessing.py to an external site. script. Implement the Preprocessing.capture_video() to an external site. method to yield every drop-rate'th frame. For the Inference Service to perform live detections, it must avoid causing a frame backlog.
Note: it is useful for you to use the yield keyword instead of the return keyword. You may read this reference to an external site. for more information on yield (i.e., generator method).
* Task 2: Review the model.py to an external site. script. Implement the Detector.predict() to an external site. method to output ALL the predictions of the YOLO model.
* Task 3: In the same model.py to an external site. script as Task 2, complete the Detector.post_process() to an external site. method to filter the predictions of the YOLO model based on a score_threshold.
* Task 4: Review the nms.py to an external site. script. Complete the NMS.filter() to an external site. method to apply Non-Maximum Suppression (NMS) to filter overlapping bounding boxes. Only use the numpy package for this task.
* Task 5: Review the app.py to an external site. script. Complete the InferenceService.run() to an external site. method to integrate all three modules in the inference service. This service must be able to:
    * Capture a stream via the UDP protocol (preprocess module)
    * Sequentially detect objects in a frame (model module)
    * Filter these detections by applying NMS (nms module)
    * Print per-frame detections (i.e, bounding box, class_id, object_score)
    * Save the frames in an output folder
    * Note: You will be evaluated using the app.py runner to an external site.. Do not modify this.

### CodePointer
* techtrack/modules/inference/model.py
* techtrack/modules/inference/nms.py
* techtrack/modules/inference/preprocessing.py
* techtrack/app.py
* techtrack/README.md


## Assignment 4
### Objectives
* Task 1: Review the augmentation.pyLinks to an external site. script. Implement the six static class methods. Follow the instructions embedded in the script labeled as Task 1.

* Task 2: Review the loss.pyLinks to an external site. script. Implement compute() class method which calculates the components of a (modified) YOLO loss. Follow instructions embedded in the script labeled as Task 2.

* Task 3: Review the demo_augmentation.ipynbLinks to an external site. notebook. Display and analyze the behavior of the Augmenter class and its impact on the Rectification Service and overall system. See "Task 3: Assignment Instructions" embedded in the notebook.

* Task 4: Review the hard_negative_mining.pyLinks to an external site. script and demo_hard_negatives.ipynbLinks to an external site. notebook. Display and analyze the behavior of the HardNegativeMiner class and its impact on the Rectification Service and overall system. See "Task 4: Assignment Instructions" embedded in the notebook. Follow instructions for Task A, B, and C.

### CodePointer
* techtrack/modules/rectification/augmentation.py
* techtrack/modules/rectification/hard_negative_mining.py
* techtrack/modules/utils/loss.py
* techtrack/notebooks/demo_augmentation.ipynb
* techtrack/notebooks/demo_hard_negatives.ipynb

## Assignment 5
### Objectives
* Task 1-2: Review the metrics.py script and the runner demonstrating the functions. Detailed implementation is provided in each function's docstrings. Complete the following tasks:

* Task 1: Evaluate model detections against the ground truth objects using the function evaluate_detections()
* Task 2: Compute the function calculate_precision_recall_curve()
Note: The function calculate_map_x_point_interpolated() is implemented for you. It may be helpful to complete the calculate_iou() function if you haven't already.

* Task 3: Review the demo_metrics.ipynb notebook. Complete the notebook. Demonstrate your metrics functions by calculating the Mean Average Precision (mAP) of YOLO Model 1.

* Task 4: Review the model_selection.ipynb notebook. Complete the notebook. Compare and contrast the predictive performance of the two yolo models provided. To argue for the optimal model, show results using visualizations (i.e., tables, graphs, etc.). Finally, indicate the reasons why you selected the optimal model.

### CodePointer
* techtrack/Dockerfile
* techtrack/README.md
* techtrack/modules/utils/metrics.py
* techtrack/modules/utils/tests.py
* techtrack/notebooks/demo_metrics.py
* techtrack/notebooks/model_selection.py


## Assignment 6
### Objectives
* Task 1: Review the extraction/preprocess.py and extraction/embedding.py. These scripts are already implemented. In a notebook named notebooks/demo_extraction.ipynb, demonstrate the embedding capability of the service for 'casia-webface' and 'vggface2'  by calculating the Euclidean distance of the following five probe images and their corresponding gallery images (use all available images for each individual) and note your observations:

    * Drew Barrymore
    * Warren Buffetf
    * Owen Wilson
    * Nelson Mandela
    * Ian Thorpe

* Task 2: In the notebook named ironclad/notebooks/demo_extraction.ipynb, precompute the embeddings of ALL images stored in storage/gallery/*. For each of the five probe images, calculate the following distance against all the images in the gallery. Sort the embeddings from shortest to longest distance and print the images of the ten nearest neighbors and the name associated with each image. Note your observations.

    * euclidean
    * dot_product
    * cosine
    * minkowski
    > Note: DO NOT print all the distances. Keep your notebook clean and organized!

* Task 3: In the notebook named ironclad/notebooks/demo_extraction.ipynb, report the rank positions of the five probe's associated gallery images. Note your observations.

If Person A is your probe, get the "rank position" of all A's images in the gallery.

* Task 4: In the notebook named ironclad/notebooks/demo_transformations.ipynb, transform the five probe images using the following transformations below. Repeat Tasks 2 and 3 using the transformed images and note your observations describing the performance impacts of various noise transformations and varying degrees of severity pertinent to the case.

    * horizontal flip
    * gaussian blur
    * increase brightness
    * decrease brightness
    > Note: You should not transform the images in the gallery. Only the transformations are applied to the probe images.

### CodePointer
* ironclad/modules/extraction/preprocssing.py
* ironclad/modules/extraction/embedding.py
* ironclad/notebooks/demo_extraction.ipynb
* ironclad/notebooks/demo_transformations.ipynb


## Assignment 9
### Objectives
* Task 1: Review the **textwave/modules/generator/question_answering.py** script which leverages the Mistral API. You can configure this class by specifying three class arguments:
    * The api_key argument should take in your unique API key (string) provided to you once you registered for a Mistral account.
        * Go to https://mistral.ai/Links to an external site. and register for a new account. You will need to follow the authentication process to complete this.
        * Once registered, log in to your account and create a new workspace.
        * Go to "La Plateforme" menu -> "Billing" -> "Go to billings plans page." Select "Experiment for free" and subscribe to the plan. You will need to complete the authentication process.
        * In the "La Plateforme" menu, select the "API Keys." You can view your API key here.
        * In a terminal, run the command: export MISTRAL_API_KEY=<YOUR_API_KEY>.  You must run this command each time you open a terminal to run this code. Optionally, add this line at the bottom of your ~/.bashrc file.
    * The temperature controls the randomness of the model's responses.
    * The generator_model specifies the model (e.g., mistral-{small|medium|large}-latest)
Review the notebook called **textwave/notebooks/demo_generator.ipynb**, show that your model can generate answers given the following context contained in the notebook.

> Note: Be mindful about your Mistral AI token limits!

* Task 2: Review textwave/modules/extraction/preprocessing.py. Add a method called fixed_length_chunking() using sentence chunking as a guide.

* Task 3: Review textwave/modules/extraction/embedding.py. In a script called textwave/app.py, Complete the method called initialize_index() which will perform the following:
    1. Parse through all the documents contained in storage/ directory
    2. Chunk the documents using either a 'sentence' and 'fixed-length' chunking strategies (default is 'fixed-length')
    3. Embed each chunk as a vector using Embedding class (default text embedding model is 'all-MiniLM-L6-v2')
    4. Store vector embeddings of these chunks in a BruteForce FAISS index, along with the chunks as metadata
    5. This function should return the FAISS index
    > Note: you may refer to your IronClad index and search implementations. Place them in textwave/modules/retrieval/. Follow IronClads files structure for index.

* Task 4: In a notebook called **textwave/notebooks/demo_retrieval.ipynb**, using all questions listed in **textwave/qa_resources/questions.tsv**, demonstrate your system's ability to retrieve the nearest neighbors. You will compare the retrieval performance of 'all-MiniLM-L6-v2' embedding model and another embedding model of your choosing from the available list(https://sbert.net/docs/sentence_transformer/pretrained_models.html).

Measure the retrieval (ranking) performance of the two embedding models based on retrieved chunks being in the proper target article (see ArticleFile column in textwave/qa_resources/questions.tsv).

For example, given a question with an associated ArticleFile as "S08_set3_a4" a true positive would be a chunk that is extracted from S08_set3_a4.txt.clean. You may choose to use default parameters/configuration for the other modules.

> Note: you may refer to your IronClad index and search implementations. Place them in textwave/modules/retrieval/. Follow IronClads files structure for index.

### CodePointer
* textwave/modules/generator/question_answering.py
* textwave/modules/extraction/preprocessing.py
* textwave/modules/extraction/embedding.py
* textwave/app.py
* textwave/notebooks/demo_generator.ipynb
* textwave/notebooks/demo_retrieval.ipynb

## Assignment 10
### Objectives
* Task 1: Review modules/utils/tfidf.py and modules/utils/bow.py. Without using external packages, implement TF-IDF and Bag-of-Words. Use the runners (if __name__ == "__main__") to guide your output.

* Task 2-4: In a notebook called notebooks/analyze_text_representations.ipynb, you will compare text representations using TF-IDF and Bag-of-Words, then evaluate their effectiveness in a downstream clustering task. For this task, answer the following questions using the data in storage/*.txt.clean. Use graphs, tables, and other visualizations to support your arguments.

    * Compare how different text pre-processing techniques (e.g., stemming vs. lemmatization vs. no processing) affect the performance of both vectorizers. You may use these nltk classes (see modules/utils/text_processing.py):
    ```
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
    ```

    * Compare how different chunking and overlap parameters affect the performance of both text representation approaches. You may use to start your analysis:
    ```
        chunking_strategy='sentence', overlap_size=2
        chunking_strategy='sentence', overlap_size=4
        chunking_strategy='fixed-length', fixed_length=50, overlap_size=1
        chunking_strategy='fixed-length', fixed_length=100, overlap_size=1
        chunking_strategy='fixed-length', fixed_length=150, overlap_size=1
    ```
    * Compare how varying vocabulary sizes (e.g., limiting to the "top-N" most frequent words) and observe trade-offs.

    > Hint: One way to compare representations is to assess how similar "neighbors" are compared to "non-neighbors." Can you think of ways to assess this algorithmically?

### CodePointer
* textwave/modules/utils/tfidf.py
* textwave/modules/utils/bow.py
* textwave/modules/utils/text_processing.py
* textwave/notebooks/analyze_text_representations.ipynb
