"""
Motor Vectorial de RAG con SentenceTransformers y Similitud Coseno Normalizada.
"""
import numpy as np

class VectorRAGEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._embedder = None
        self.documents = []
        self.embeddings = None

    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder

    def index_documents(self, docs: list[str]):
        self.documents = docs
        self.embeddings = self.embedder.encode(docs, normalize_embeddings=True)

    def search(self, query: str, top_k=2, threshold=0.40):
        if self.embeddings is None or len(self.documents) == 0:
            return []
        q_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(self.embeddings, q_emb)
        indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in indices:
            if scores[idx] >= threshold:
                results.append({"text": self.documents[idx], "score": float(scores[idx])})
        return results

if __name__ == "__main__":
    docs = [
        "El plazo maximo para solicitar reembolsos es de 30 dias naturales con ticket de compra.",
        "Los envios estandar demoran de 3 a 5 dias habiles en territorio nacional.",
        "El soporte tecnico atiende de lunes a viernes de 09:00 a 18:00 hrs."
    ]
    engine = VectorRAGEngine()
    engine.index_documents(docs)
    res = engine.search("¿Como puedo pedir mi dinero de vuelta?")
    print("Resultados RAG:", res)
