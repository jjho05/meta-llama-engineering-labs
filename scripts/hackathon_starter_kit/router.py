"""
Router de Consultas para el Hackathon de IA.
Clasifica la complejidad e intencion de la consulta del usuario para optimizar latencia y costo.
"""
import re

class ModelRouter:
    def __init__(self):
        self.keywords_rag = ["politica", "reembolso", "garantia", "manual", "horario", "precio", "costo", "pedido", "envio"]
        self.keywords_lora = ["json", "esquema", "diagnostico", "sql", "codigo", "formato"]

    def route(self, query: str) -> str:
        text = query.lower()
        if any(k in text for k in self.keywords_rag):
            return "RAG_PIPELINE"
        if any(k in text for k in self.keywords_lora):
            return "LORA_ADAPTER"
        return "FAST_LLM"

if __name__ == "__main__":
    router = ModelRouter()
    queries = [
        "Hola, buenos dias",
        "¿Cual es la politica de reembolso de mi pedido?",
        "Genera un esquema JSON de diagnostico tecnico"
    ]
    for q in queries:
        print(f"Consulta: '{q}' -> Ruta seleccionada: {router.route(q)}")
