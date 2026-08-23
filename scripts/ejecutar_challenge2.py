#!/usr/bin/env python3
"""
Module: Applied Artificial Intelligence with Open Weights Models
Challenge 2: Institutional Policy Assistant with Semantic RAG Pipeline
Author: Ing. Jesús Javier Hernández Olvera
"""

import os
import re
import sys
import json
import time
import getpass
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------------------------------
# 1. Load Environment & Initialize Groq Client
# -----------------------------------------------------------------------------
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.environ.get("GROQ_API_KEY")

if not API_KEY:
    print("[Notice] GROQ_API_KEY not found in environment or .env file.")
    API_KEY = getpass.getpass("Enter your GROQ_API_KEY: ").strip()

if not API_KEY:
    print("[Error] A valid Groq API Key is required to run the RAG assistant.")
    sys.exit(1)

client = Groq(api_key=API_KEY)
print("=" * 100)
print("MODULE: APPLIED AI WITH OPEN WEIGHTS MODELS")
print("CHALLENGE 2: INSTITUTIONAL POLICY ASSISTANT WITH SEMANTIC RAG PIPELINE")
print("Author / Architect: Ing. Jesús Javier Hernández Olvera")
print("=" * 100)

# -----------------------------------------------------------------------------
# 2. Load Institutional Knowledge Base (Factual Policy Documents)
# -----------------------------------------------------------------------------
data_path = Path(__file__).parent.parent / "data" / "reglamento_academico_politicas.json"

if data_path.exists():
    with open(data_path, "r", encoding="utf-8") as f:
        documents = json.load(f)
else:
    documents = [
        {
            "id": "POL-001",
            "categoria": "Calificaciones",
            "titulo": "Escala de Calificaciones y Calificación Mínima Aprobatoria",
            "contenido": "La escala oficial de calificaciones es de 0 a 100 puntos. La calificación mínima aprobatoria en cualquier materia o módulo es de 70/100. Calificaciones inferiores a 70 requieren examen extraordinario o recursamiento."
        },
        {
            "id": "POL-002",
            "categoria": "Asistencia",
            "titulo": "Requisito Mínimo de Asistencia para Derecho a Examen",
            "contenido": "Todo estudiante debe cumplir con un mínimo del 80% de asistencia para mantener derecho a evaluación final ordinaria. Asistencias menores al 60% causan baja académica automática."
        },
        {
            "id": "POL-003",
            "categoria": "Bajas",
            "titulo": "Plazos Límites para la Baja Temporal de Materias",
            "contenido": "El período límite para solicitar la baja temporal formal de una asignatura sin penalización vence exactamente al término de la Semana 4 del período académico en curso."
        },
        {
            "id": "POL-004",
            "categoria": "Integridad",
            "titulo": "Política Institucional sobre Plagio y Uso de IA",
            "contenido": "El plagio o uso no acreditado de modelos generativos de IA en entregables individuales será sancionado con calificación automática de 0 (cero) y turno al Comité de Ética."
        },
        {
            "id": "POL-005",
            "categoria": "Prácticas",
            "titulo": "Requisitos para Prácticas Profesionales",
            "contenido": "El alumno únicamente puede iniciar el trámite de Prácticas Profesionales o Servicio Social tras haber acreditado formalmente al menos el 70% del total de créditos curriculares."
        }
    ]

print("Loaded " + str(len(documents)) + " factual policy documents into knowledge base.\n")

# -----------------------------------------------------------------------------
# 3. Vectorization with Sentence-Transformers (Dense Embeddings)
# -----------------------------------------------------------------------------
print("Initializing Sentence-Transformers multilingual embedding model...")
model_embed = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

corpus_texts = [f"{doc['titulo']}: {doc['contenido']}" for doc in documents]
corpus_embeddings = model_embed.encode(corpus_texts, convert_to_numpy=True, normalize_embeddings=True)
print("Generated " + str(corpus_embeddings.shape[0]) + " dense vectors of dimension " + str(corpus_embeddings.shape[1]) + ".\n")

# -----------------------------------------------------------------------------
# 4. Dense Retrieval & Cosine Similarity Ranking Function
# -----------------------------------------------------------------------------
def retrieve_relevant_chunks(query, top_k=2):
    query_vector = model_embed.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    similarities = np.dot(corpus_embeddings, query_vector)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    retrieved = []
    for idx in top_indices:
        retrieved.append({
            "doc": documents[idx],
            "score": float(similarities[idx]),
            "text": corpus_texts[idx]
        })
    return retrieved

# -----------------------------------------------------------------------------
# 5. RAG Augmented Generation Pipeline (Grounded in Facts)
# -----------------------------------------------------------------------------
def ask_policy_assistant(user_query, model_name="openai/gpt-oss-20b"):
    print("-" * 90)
    print(f"User Query: \"{user_query}\"")
    
    # Step 1: Semantic Retrieval
    retrieved = retrieve_relevant_chunks(user_query, top_k=2)
    context_text = "\n\n".join([f"[{r['doc']['id']} - {r['doc']['titulo']}]: {r['doc']['contenido']}" for r in retrieved])
    
    print(f"Top Retrieved Context (Score: {retrieved[0]['score']:.4f}): {retrieved[0]['doc']['titulo']}")
    
    # Step 2: System prompt conditioning against hallucination
    system_prompt = (
        "You are an official Institutional Policy Assistant. "
        "Answer the user query strictly based on the provided Context below. "
        "If the answer cannot be verified in the Context, respond: "
        "'Information not found in the official institutional policies.' "
        "Do not invent policies or extrapolate. Answer concisely and professionally without emojis.\n\n"
        f"CONTEXT:\n{context_text}"
    )
    
    # Step 3: LLM Inference on Groq LPU
    t0 = time.time()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0.1,
        max_tokens=400
    )
    elapsed = time.time() - t0
    answer = response.choices[0].message.content.strip()
    
    print(f"Response ({elapsed:.3f}s):\n{answer}\n")
    return answer

# -----------------------------------------------------------------------------
# 6. Execute Test Queries
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    test_questions = [
        "¿Cuál es la calificación mínima para pasar una materia y qué pasa si saco 65?",
        "¿Hasta qué semana puedo dar de baja una clase sin que afecte mi promedio?",
        "¿Cuántos créditos necesito para hacer el servicio social?",
        "¿Se permite estacionar motocicletas dentro del campus?"
    ]
    
    for q in test_questions:
        ask_policy_assistant(q)

print("=" * 100)
print("RAG Assistant test run completed successfully.")
