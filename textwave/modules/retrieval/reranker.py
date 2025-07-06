import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import pairwise_distances
import torch


class Reranker:
    """
    Perform reranking of documents based on their relevance to a given query.

    Supports multiple reranking strategies:
    - Cross-encoder: Uses a transformer model to compute pairwise relevance.
    - TF-IDF: Uses term frequency-inverse document frequency with similarity metrics.
    - BoW: Uses term Bag-of-Words with similarity metrics.
    - Hybrid: Combines TF-IDF and cross-encoder scores.
    - Sequential: Applies TF-IDF first, then cross-encoder for refined reranking.
    """

    def __init__(self, type, cross_encoder_model_name='cross-encoder/ms-marco-TinyBERT-L-2-v2', corpus_directory=''):
        """
        Initialize the Reranker with a specified reranking strategy and optional model and corpus.

        :param type: Type of reranking ('cross_encoder', 'tfidf', 'hybrid', or 'sequential').
        :param cross_encoder_model_name: HuggingFace model name for the cross-encoder (default: cross-encoder/ms-marco-TinyBERT-L-2-v2).
        :param corpus_directory: Directory containing .txt files for TF-IDF corpus (optional).
        """
        self.type = type
        self.cross_encoder_model_name = cross_encoder_model_name
        self.cross_encoder_model = AutoModelForSequenceClassification.from_pretrained(cross_encoder_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(cross_encoder_model_name)


    def rerank(self, query, context, distance_metric="cosine", seq_k1=None, seq_k2=None):
        """
        Dispatch the reranking process based on the initialized strategy.

        :param query: Input query string to evaluate relevance against.
        :param context: List of document strings to rerank.
        :param distance_metric: Distance metric used for TF-IDF reranking (default: "cosine").
        :param seq_k1: Number of top documents to select in the first phase (TF-IDF) of sequential rerank.
        :param seq_k2: Number of top documents to return from the second phase (cross-encoder) of sequential rerank.
        :return: Tuple of (ranked documents, ranked indices, corresponding scores).
        """
        if self.type == "cross_encoder":
            return self.cross_encoder_rerank(query, context)
        elif self.type == "tfidf":
            return self.tfidf_rerank(query, context, distance_metric=distance_metric)
        elif self.type == "bow":
            return self.bow_rerank(query, context, distance_metric=distance_metric)
        elif self.type == "hybrid":
            return self.hybrid_rerank(query, context, distance_metric=distance_metric)
        elif self.type == "sequential":
            return self.sequential_rerank(query, context, seq_k1, seq_k2, distance_metric=distance_metric)

    def cross_encoder_rerank(self, query, context):
        """
        Rerank documents using a cross-encoder transformer model.

        Computes relevance scores for each document-query pair, sorts them in
        descending order of relevance, and returns the ranked results.

        :param query: Query string.
        :param context: List of candidate document strings.
        :return: Tuple of (ranked documents, ranked indices, relevance scores).
        """
        query_document_pairs = [(query, doc) for doc in context]
        inputs = self.tokenizer(query_document_pairs, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            logits = self.cross_encoder_model(**inputs).logits
            relevance_scores = logits.squeeze(-1).tolist()

        # Get sorted indices (highest score = highest relevance)
        sorted_indices = np.argsort(relevance_scores)[::-1]  # Descending order

        # Sort context and distances accordingly
        ranked_docs = [context[i] for i in sorted_indices]
        ranked_scores = [relevance_scores[i] for i in sorted_indices]

        return ranked_docs, sorted_indices.tolist(), ranked_scores

    def tfidf_rerank(self, query, context, distance_metric="cosine"):
        """
        Rerank documents using TF-IDF vectorization and distance-based similarity.

        Creates a TF-IDF matrix from the query and context, computes pairwise distances,
        and sorts documents by similarity (lower distance implies higher relevance).

        :param query: Query string.
        :param context: List of document strings.
        :param distance_metric: Distance function to use (e.g., 'cosine', 'euclidean').
        :return: Tuple of (ranked documents, indices, similarity scores).
        """
        # Combine query and context for vectorization
        documents = [query] + context

        # Fit TF-IDF on the combined list
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)

        # Compute distance between query (row 0) and each document (rows 1:)
        distances = pairwise_distances(tfidf_matrix[0], tfidf_matrix[1:], metric=distance_metric).flatten()

        # Get sorted indices (lowest distance = highest relevance)
        sorted_indices = np.argsort(distances)

        # Sort context and distances accordingly
        ranked_docs = [context[i] for i in sorted_indices]
        ranked_scores = [distances[i] for i in sorted_indices]

        return ranked_docs, sorted_indices.tolist(), ranked_scores

    def bow_rerank(self, query, context, distance_metric="cosine"):
        """
        Rerank documents using TF-IDF vectorization and distance-based similarity.

        Creates a TF-IDF matrix from the query and context, computes pairwise distances,
        and sorts documents by similarity (lower distance implies higher relevance).

        :param query: Query string.
        :param context: List of document strings.
        :param distance_metric: Distance function to use (e.g., 'cosine', 'euclidean').
        :return: Tuple of (ranked documents, indices, similarity scores).
        """
        # Combine query and context for vectorization
        documents = [query] + context

        # Fit BoW vectorizer
        vectorizer = CountVectorizer()
        bow_matrix = vectorizer.fit_transform(documents)

        # Compute distance between query (row 0) and each document (rows 1:)
        distances = pairwise_distances(bow_matrix[0], bow_matrix[1:], metric=distance_metric).flatten()

        # Rank by increasing distance (i.e., more similar first)
        sorted_indices = np.argsort(distances)
        ranked_docs = [context[i] for i in sorted_indices]
        ranked_scores = [distances[i] for i in sorted_indices]

        return ranked_docs, sorted_indices.tolist(), ranked_scores

    def hybrid_rerank(self, query, context, distance_metric="cosine", tfidf_weight=0.3):
        """
        Combine TF-IDF and cross-encoder scores to produce a hybrid reranking.

        This approach balances fast lexical matching (TF-IDF) with deeper semantic understanding
        (cross-encoder) by computing a weighted average of both scores.

        :param query: Query string.
        :param context: List of document strings.
        :param distance_metric: Distance metric for the TF-IDF portion.
        :param tfidf_weight: Weight (0-1) assigned to TF-IDF score in final ranking.
        :return: Tuple of (ranked documents, indices, combined scores).
        """
        # Compute TF-IDF and cross-encoder scores
        documents = [query] + context
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(documents)
        tfidf_scores = pairwise_distances(tfidf_matrix[0], tfidf_matrix[1:], metric=distance_metric).flatten()

        query_document_pairs = [(query, doc) for doc in context]
        inputs = self.tokenizer(query_document_pairs, padding=True, truncation=True, return_tensors="pt")

        with torch.no_grad():
            logits = self.cross_encoder_model(**inputs).logits
            cross_encoder_scores = logits.squeeze(-1).tolist()

        # Normalize scores to [0, 1] range
        tfidf_scores_norm = [(x - min(tfidf_scores)) / (max(tfidf_scores) - min(tfidf_scores)) for x in tfidf_scores]
        cross_encoder_scores_norm = [(x - min(cross_encoder_scores)) / (max(cross_encoder_scores) - min(cross_encoder_scores)) for x in cross_encoder_scores]
        cross_encoder_scores_norm = [1 - x for x in cross_encoder_scores_norm]  # Invert scores for consistency

        # Combine scores using a weighted average
        combined_scores = [tfidf_weight * s1 + (1 - tfidf_weight) * s2 for s1, s2 in zip(tfidf_scores_norm, cross_encoder_scores_norm)]

        # Sort by combined scores
        sorted_indices = np.argsort(combined_scores)
        ranked_docs = [context[i] for i in sorted_indices]
        ranked_scores = [combined_scores[i] for i in sorted_indices]

        return ranked_docs, sorted_indices.tolist(), ranked_scores

    def sequential_rerank(self, query, context, seq_k1, seq_k2, distance_metric="cosine"):
        """
        Apply a two-stage reranking pipeline: TF-IDF followed by cross-encoder.

        This method narrows down the document pool using TF-IDF, then applies a
        cross-encoder to refine the top-k results for improved relevance accuracy.

        :param query: Query string.
        :param context: List of document strings.
        :param seq_k1: Top-k documents to retain after the first stage (TF-IDF).
        :param seq_k2: Final top-k documents to return after second stage (cross-encoder).
        :param distance_metric: Distance metric for TF-IDF.
        :return: Tuple of (ranked documents, indices, final relevance scores).
        """
        tfidf_docs, _, _ = self.tfidf_rerank(query, context, distance_metric=distance_metric)
        seq_k1_docs = tfidf_docs[:seq_k1]

        cross_encoder_docs, cross_encoder_indices, cross_encoder_scores = self.cross_encoder_rerank(query, seq_k1_docs)
        seq_k2_docs = cross_encoder_docs[:seq_k2]
        seq_k2_indices = cross_encoder_indices[:seq_k2]
        seq_k2_scores = cross_encoder_scores[:seq_k2]

        return seq_k2_docs, seq_k2_indices, seq_k2_scores


if __name__ == "__main__":
    query = "What are the health benefits of green tea?"
    documents = [
        "Green tea contains antioxidants that may help prevent cardiovascular disease.",
        "Coffee is also rich in antioxidants but can increase heart rate.",
        "Drinking water is essential for hydration.",
        "Green tea may also aid in weight loss and improve brain function."
    ]

    print("\nCross-Encoder Reranking:")
    reranker = Reranker(type="cross_encoder")
    docs, indices, scores = reranker.rerank(query, documents)
    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"Rank {i + 1}: Score={score:.4f} | {doc}")

    print("\nTF-IDF Reranking:")
    reranker = Reranker(type="tfidf")
    docs, indices, scores = reranker.rerank(query, documents)
    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"Rank {i + 1}: Score={score:.4f} | {doc}")

    print("\nBoW Reranking:")
    reranker = Reranker(type="bow")
    docs, indices, scores = reranker.rerank(query, documents)
    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"Rank {i + 1}: Score={score:.4f} | {doc}")

    print("\nHybrid Reranking:")
    reranker = Reranker(type="hybrid")
    docs, indices, scores = reranker.rerank(query, documents)
    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"Rank {i + 1}: Score={score:.4f} | {doc}")

    print("\nSequential Reranking:")
    reranker = Reranker(type="sequential")
    docs, indices, scores = reranker.rerank(query, documents, seq_k1=3, seq_k2=2)
    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"Rank {i + 1}: Score={score:.4f} | {doc}")
