# 🧠 Módulo 1: Fundamentos de IA & Ecosistema de Modelos Abiertos

<div align="center">

**Arquitectura Transformer, Prompting Avanzado, Embeddings, RAG y Fine-Tuning con LoRA**

[🏠 Inicio](../../README.md) • [📚 Módulo 2](../modulo-2-automatizacion-agentes-whatsapp/README.md) • [🧪 Challenge 1](../../notebooks/01_Challenge1_MultiModel_Benchmark_Groq.ipynb) • [🧪 Challenge 2](../../notebooks/02_Challenge2_RAG_Politicas_SentenceTransformers.ipynb) • [🧪 Challenge 3](../../notebooks/03_Challenge3_FineTuning_LoRA_Llama.ipynb)

</div>

---

## 📋 Descripción del Módulo

En este primer módulo dominarás la física matemática y la ingeniería detrás de los modelos de pesos abiertos (*Open Weights*) de la familia **Meta Llama 3**. Comprenderás cómo viaja una cadena de texto desde su fragmentación en tokens BPE hasta las matrices de atención $Q, K, V$, el anclaje fáctico mediante sistemas RAG y la adaptación eficiente con matrices de bajo rango (LoRA/QLoRA).

---

## 🎯 Competencias Específicas

1. **Microarquitectura Transformer:** Entender el cálculo de auto-atención escalada, Rotary Position Embeddings (RoPE) y KV-Cache.
2. **Prompt Engineering de Precisión:** Aplicar In-Context Learning (Few-Shot), Chain-of-Thought (CoT) y delimitadores estructurados.
3. **Recuperación Semántica (RAG):** Construir espacios vectoriales densos con `sentence-transformers` y cálculo de similitud coseno.
4. **Optimización Paramétrica (PEFT):** Adaptar modelos de 8B/70B en GPUs accesibles mediante cuantización NF4 y matrices LoRA.

---

## 📚 Temario y Documentación

| Tema | Título del Contenido | Enfoque de Ingeniería | Cuaderno / Lab Asociado |
|---|---|---|---|
| **1.1** | [**Arquitectura Transformer & Llama 3**](01-arquitectura-transformer-llama3.md) | Tokenización BPE, proyecciones tensoriales, Grouped-Query Attention (GQA) y soberanía de pesos abiertos. | [Ver Tema](01-arquitectura-transformer-llama3.md) |
| **1.2** | [**Prompt Engineering & Sistemas RAG**](02-prompt-engineering-avanzado-rag.md) | Zero-Shot, Few-Shot, CoT y arquitectura de recuperación semántica contra alucinaciones. | [Ver Tema](02-prompt-engineering-avanzado-rag.md) |
| **1.3** | [**Fine-Tuning LoRA / QLoRA & Evaluación**](03-fine-tuning-lora-qlora-evaluacion.md) | Factorización matricial $\Delta W = B \cdot A$, cuantización en 4-bits y métricas Perplexity/BLEU. | [Ver Tema](03-fine-tuning-lora-qlora-evaluacion.md) |
| **1.4** | [**Del Prototipo al Pipeline Productivo**](04-del-prototipo-al-pipeline-productivo.md) | Microservicios con FastAPI, endpoints de inferencia, contenedores Docker y evaluación end-to-end. | [Ver Tema](04-del-prototipo-al-pipeline-productivo.md) |

---

## 🧪 Laboratorios Prácticos & Challenges

* **[Challenge 1 · Multi-Model Benchmark Groq LPU](../../notebooks/01_Challenge1_MultiModel_Benchmark_Groq.ipynb) ([Guía Markdown](challenge-1-benchmark-multi-modelo.md)):** Evaluación de latencia, throughput y calidad en modelos de 20B, 27B CoT y 120B.
* **[Challenge 2 · Asistente de Políticas con RAG Semántico](../../notebooks/02_Challenge2_RAG_Politicas_SentenceTransformers.ipynb) ([Guía Markdown](challenge-2-asistente-politicas-rag.md)):** Pipeline RAG con Sentence-Transformers, cálculo matricial en NumPy y síntesis condicionada.
* **[Challenge 3 · Fine-Tuning con LoRA & Evaluación](../../notebooks/03_Challenge3_FineTuning_LoRA_Llama.ipynb) ([Guía Markdown](challenge-3-fine-tuning-lora.md)):** Adaptación supervisada (PEFT) con SFTTrainer en GPU T4 y cuantificación de reducción de pérdida.
