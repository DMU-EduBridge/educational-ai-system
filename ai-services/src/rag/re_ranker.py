from typing import List
from sentence_transformers import CrossEncoder
from .document_processor import Document

class ReRanker:
    """
    Re-ranks a list of documents based on a query using a Cross-Encoder model.
    """

    def __init__(self, model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Initializes the ReRanker by loading a pre-trained Cross-Encoder model.

        Args:
            model_name: The name of the Cross-Encoder model to use.
        """
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Re-ranks the given documents based on their relevance to the query.

        Args:
            query: The search query.
            documents: A list of Document objects to be re-ranked.

        Returns:
            A new list of Document objects, sorted by relevance score in descending order.
        """
        if not documents:
            return []

        # Create pairs of [query, document_content] for scoring
        sentence_pairs = [[query, doc.content] for doc in documents]

        # Compute scores
        scores = self.model.predict(sentence_pairs)

        # Combine documents with their scores
        scored_documents = list(zip(scores, documents))

        # Sort documents by score in descending order
        scored_documents.sort(key=lambda x: x[0], reverse=True)

        # Return only the sorted documents
        return [doc for score, doc in scored_documents]
