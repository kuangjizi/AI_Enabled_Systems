# AI-Enabled Systems

This repository contains five applied AI systems spanning retrieval-augmented generation, vector similarity search, recommendation systems, fraud detection, and computer vision. Each project is organized as an independent implementation with reusable modules and notebooks for experimentation and evaluation.

## Projects

| Project | Description | Key technologies |
| --- | --- | --- |
| [TextWave](./textwave) | Retrieval-augmented question-answering system that retrieves relevant document passages and uses an LLM to generate grounded answers. | Mistral, Sentence Transformers, FAISS, Hugging Face Transformers, NLTK, Flask |
| [Ironclad](./ironclad) | Face-identification service that converts images into embeddings and retrieves the closest identities from a searchable gallery. | FaceNet, PyTorch, FAISS HNSW/LSH/Flat, torchvision, Flask |
| [MovieMate](./moviemate) | Hybrid movie-recommendation system combining collaborative, content-based, and rule-based recommendations with diversification and drift detection. | scikit-surprise, scikit-learn, pandas, SciPy |
| [SecureBank](./securebank) | End-to-end fraud-detection service covering dataset preparation, feature engineering, model training, online prediction, and evaluation. | XGBoost, scikit-learn, pandas, Flask, Docker |
| [TechTrack](./techtrack) | Streaming object-detection system that processes video frames, detects logistics-related objects, and filters overlapping predictions. | YOLOv4-Tiny, OpenCV, FFmpeg, NumPy, Docker |

## TextWave

TextWave demonstrates a compact RAG pipeline. It cleans and chunks text documents, generates embeddings with `all-MiniLM-L6-v2`, stores them in a FAISS index, retrieves the most relevant passages, and supplies that context to a Mistral model for answer generation. The project also includes experiments with TF-IDF, bag-of-words, and cross-encoder reranking.

**Highlights:** document processing, semantic embeddings, vector retrieval, reranking, grounded generation, and answer-quality analysis.

## Ironclad

Ironclad is an embedding-based face-identification service. FaceNet generates image embeddings, while FAISS provides exact and approximate nearest-neighbor search. The implementation compares Flat, LSH, and HNSW indexes and supports both identifying a probe image and incrementally adding new gallery images.

**Highlights:** computer-vision embeddings, ANN index selection, cosine similarity, retrieval evaluation, and online index updates.

## MovieMate

MovieMate explores hybrid personalization using the MovieLens dataset. It combines SVD-based collaborative filtering, TF-IDF content similarity, and explicit rules. Additional modules measure recommendation diversity and detect distribution changes that may indicate a need for retraining.

**Highlights:** collaborative filtering, content-based ranking, cold-start analysis, diversification, and continuous-learning signals.

## SecureBank

SecureBank implements a fraud-detection workflow around an XGBoost classifier. It includes raw-data processing, reusable scikit-learn feature pipelines, model training, prediction, evaluation, request logging, and Flask endpoints. Docker and shell scripts package the workflow into a reproducible service.

**Highlights:** feature engineering, supervised learning, model evaluation, REST serving, logging, and containerization.

## TechTrack

TechTrack performs object detection on a live or prerecorded video stream. It loads a YOLOv4-Tiny model through OpenCV's DNN module, preprocesses frames, applies non-maximum suppression, draws detections, and saves the processed output. Supporting notebooks cover augmentation, hard-negative mining, and detection metrics.

**Highlights:** streaming inference, object detection, non-maximum suppression, precision/recall and mAP evaluation, and Docker deployment.

## Repository Structure

```text
AI_Enabled_Systems/
├── textwave/    # Retrieval-augmented question answering
├── ironclad/    # Face embedding and similarity search
├── moviemate/   # Hybrid movie recommendation
├── securebank/  # Fraud detection and model serving
└── techtrack/   # Streaming object detection
```

## Getting Started

Each project is self-contained. Open its directory and review the local `README.md`, `requirements.txt`, notebooks, and application entry point for project-specific setup and usage instructions.


