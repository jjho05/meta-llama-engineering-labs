# 🦙 Meta Llama 3 Engineering Labs & Applied AI Specialization

<div align="center">

[![Meta AI](https://img.shields.io/badge/Meta_AI-Llama_3.1-0866FF?style=for-the-badge&logo=meta&logoColor=white)](https://ai.meta.com/llama/)
[![Groq LPU](https://img.shields.io/badge/Hardware-Groq_LPU_Inference-F55036?style=for-the-badge)](https://groq.com/)
[![RAG & Vectors](https://img.shields.io/badge/Retrieval-Sentence_Transformers-059669?style=for-the-badge)](https://sbert.net/)
[![WhatsApp API](https://img.shields.io/badge/Meta-WhatsApp_Cloud_API-25D366?style=for-the-badge&logo=whatsapp&logoColor=white)](https://developers.facebook.com/)
[![Google Colab](https://img.shields.io/badge/Notebooks-Google_Colab_Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

**Recurso educativo y de ingeniería completo para el desarrollo con modelos de pesos abiertos (*Open Weights*) de Meta Llama 3**

[📚 Contenido](#-contenido-del-programa) • [🚀 Inicio Rápido](#-inicio-rápido-en-local-cli) • [🧪 Cuadernos Google Colab](#-cuadernos-de-google-colab-y-laboratorios-en-1-clic) • [💻 Scripts](#-scripts-de-terminal) • [📖 Documentación](#-documentación-técnica-por-módulos)

</div>

---

## 📋 Descripción

Este repositorio contiene el material técnico, los manuales teóricos en Markdown, los cuadernos interactivos de Google Colab y los scripts de producción del programa de **Especialización en Inteligencia Artificial Aplicada con Meta Llama 3**, diseñado y construido por **Ing. Jesús Javier Hernández Olvera**. Incluye:

- ✅ **8 Temas Teóricos Detallados** organizados en 2 Módulos de Especialización
- ✅ **2 Cuadernos Jupyter (.ipynb)** listos para ejecutarse con 1 clic en Google Colab
- ✅ **2 Scripts Python de Producción** para benchmarking y RAG desde Terminal
- ✅ **100% Alineado con las Mejores Prácticas** de Meta AI, Hugging Face y Groq LPU
- ✅ **Arquitectura Libre de Alucinaciones** mediante Sentence-Transformers y RAG

---

## 🎓 Competencias Profesionales del Programa

Al completar este programa de ingeniería serás capaz de:

1. ⚡ **Dominar la Microarquitectura Transformer:** Comprender el cálculo de auto-atención escalada ($Q, K, V$), Grouped-Query Attention (GQA), Rotary Position Embeddings (RoPE) y KV-Cache.
2. 🔍 **Construir Sistemas RAG de Grado Industrial:** Generar representaciones densas con `sentence-transformers`, indexar espacios vectoriales y anclar respuestas en documentos verídicos.
3. 🛠️ **Optimizar con Fine-Tuning LoRA / QLoRA:** Adaptar modelos masivos mediante matrices de bajo rango en 4-bits sin requerir clusters inaccesibles de GPUs.
4. 🤖 **Desarrollar Agentes Autónomos Multi-Turno:** Gestionar estado conversacional persistente con Redis y orquestar llamadas a herramientas (*Function Calling*).
5. 🛡️ **Blindar y Desplegar en Producción:** Conectar la API oficial de WhatsApp Cloud con FastAPI, NGINX SSL y defensas activas contra inyecciones de prompts con **Llama Guard 3** y **Prompt Guard**.

---

## 🧪 Cuadernos de Google Colab y Laboratorios (En 1-Clic)

| # | Laboratorio / Challenge | Descripción de Ingeniería | Enlace Directo a Colab |
|---|---|---|---|
| **01** | **Challenge 1 · Multi-Model Benchmark** | Comparador empírico de latencia, throughput y calidad de respuesta entre modelos SLM (20B), CoT Reasoning (27B) y LLM masivo (120B) en chips Groq LPU con Google Colab Secrets. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjho05/meta-llama-engineering-labs/blob/main/notebooks/01_Challenge1_MultiModel_Benchmark_Groq.ipynb) |
| **02** | **Challenge 2 · Asistente de Políticas con RAG** | Pipeline RAG completo con embeddings multilingües de `sentence-transformers`, cálculo matricial con NumPy y síntesis fáctica condicionada anti-alucinación. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjho05/meta-llama-engineering-labs/blob/main/notebooks/02_Challenge2_RAG_Politicas_SentenceTransformers.ipynb) |

---

## 📚 Contenido del Programa

### [Módulo 1: Fundamentos de IA & Ecosistema de Modelos Abiertos](docs/modulo-1-fundamentos-ia/README.md)
**4 manuales de ingeniería • 2 challenges prácticos en Colab**

- [1.1 Arquitectura Transformer & Llama 3](docs/modulo-1-fundamentos-ia/01-arquitectura-transformer-llama3.md) — Tokenización BPE, tensores, GQA, RoPE y soberanía tecnológica.
- [1.2 Prompt Engineering & Sistemas RAG](docs/modulo-1-fundamentos-ia/02-prompt-engineering-avanzado-rag.md) — Zero-Shot, Few-Shot, Chain-of-Thought y mitigación de alucinaciones.
- [1.3 Fine-Tuning LoRA / QLoRA & Evaluación](docs/modulo-1-fundamentos-ia/03-fine-tuning-lora-qlora-evaluacion.md) — Matrices de bajo rango $\Delta W = B \cdot A$, cuantización NF4 y métricas Perplexity/BLEU.
- [1.4 Del Prototipo al Pipeline Productivo](docs/modulo-1-fundamentos-ia/04-del-prototipo-al-pipeline-productivo.md) — Microservicios FastAPI, endpoints de inferencia y Docker.

**Cuadernos Colab:** [Ver carpeta notebooks/](notebooks/)

---

### [Módulo 2: Automatización con Llama & WhatsApp Cloud API](docs/modulo-2-automatizacion-agentes-whatsapp/README.md)
**4 manuales de ingeniería • 2 scripts de producción en Python**

- [2.1 WhatsApp Cloud API & Webhooks](docs/modulo-2-automatizacion-agentes-whatsapp/01-whatsapp-cloud-api-arquitectura-webhooks.md) — Handshake criptográfico GET, parseo de eventos JSON y túneles ngrok.
- [2.2 Agentes Conversacionales & Memoria Redis](docs/modulo-2-automatizacion-agentes-whatsapp/02-agentes-conversacionales-memoria-redis.md) — Gestión de estado multi-turno, ventanas de contexto y Llama Stack.
- [2.3 Inferencia, Function Calling & Tools](docs/modulo-2-automatizacion-agentes-whatsapp/03-inferencia-function-calling-tools.md) — Inferencia en dos pasos, esquemas JSON Schema y validación en Pydantic.
- [2.4 Producción SRE & Seguridad Llama Guard](docs/modulo-2-automatizacion-agentes-whatsapp/04-produccion-seguridad-llama-guard.md) — Blindaje con Llama Guard 3, Prompt Guard, Docker Compose y NGINX SSL.

**Scripts de terminal:** [Ver carpeta scripts/](scripts/)

---

## 🚀 Inicio Rápido en Local (CLI)

### 1. Clonar el repositorio
```bash
git clone https://github.com/jjho05/meta-llama-engineering-labs.git
cd meta-llama-engineering-labs
```

### 2. Crear entorno virtual e instalar dependencias
```bash
python3 -m venv venv
source venv/bin/activate   # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configurar variables de entorno
```bash
cp .env.example .env
# Agrega tu clave secreta de Groq en .env
```

### 4. Ejecutar los scripts en terminal
```bash
# Challenge 1: Benchmarking Multi-Modelo en LPU
python3 scripts/ejecutar_challenge1.py --modelo openai/gpt-oss-20b --query "¿Qué es Grouped-Query Attention?"

# Challenge 2: Pipeline RAG de Políticas con Sentence-Transformers
python3 scripts/ejecutar_challenge2.py --modelo openai/gpt-oss-20b
```

---

## 👨‍💻 Dirección Técnica & Autoría

* **Creador & Arquitecto:** **Ing. Jesús Javier Hernández Olvera**
* **Especialización:** Arquitectura de Inteligencia Artificial, Modelos de Pesos Abiertos y Agentes Autónomos.
* **Plataforma Web Interactiva:** [https://github.com/jjho05/Inteligencia_Artificial_Aplicada_Llama](https://github.com/jjho05/Inteligencia_Artificial_Aplicada_Llama)

---

## 📄 Licencia

Distribuido bajo la **Licencia MIT**. Consulta el archivo [LICENSE](LICENSE) para más detalles.
