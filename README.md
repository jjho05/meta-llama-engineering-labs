# Meta Llama 3 Engineering Labs & Applied AI Specialization

[![Meta AI](https://img.shields.io/badge/Meta_AI-Llama_3.1-0866FF?style=for-the-badge&logo=meta&logoColor=white)](https://ai.meta.com/llama/)
[![Groq LPU](https://img.shields.io/badge/Hardware-Groq_LPU_Inference-F55036?style=for-the-badge)](https://groq.com/)
[![RAG & Vectors](https://img.shields.io/badge/Retrieval-Sentence_Transformers-059669?style=for-the-badge)](https://sbert.net/)
[![WhatsApp API](https://img.shields.io/badge/Meta-WhatsApp_Cloud_API-25D366?style=for-the-badge&logo=whatsapp&logoColor=white)](https://developers.facebook.com/)
[![Google Colab](https://img.shields.io/badge/Notebooks-Google_Colab_Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

Repositorio oficial de **cuadernos interactivos de Google Colab, laboratorios prácticos (*Hands-On*), scripts de terminal y documentación técnica en Markdown** del programa de Especialización en Inteligencia Artificial con Modelos de Pesos Abiertos (*Open Weights*), creado y mantenido por **Ing. Jesús Javier Hernández Olvera**.

---

## Cuadernos de Google Colab (Ejecución Inmediata en 1-Clic)

| # | Laboratorio / Challenge | Descripción de Ingeniería | Enlace Directo a Colab |
|---|---|---|---|
| **01** | **Challenge 1 · Multi-Model Benchmark** | Comparador empírico de latencia, throughput y calidad de respuesta entre modelos SLM (20B), CoT Reasoning (27B) y LLM masivo (120B) en chips Groq LPU con Google Colab Secrets. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjho05/meta-llama-engineering-labs/blob/main/notebooks/01_Challenge1_MultiModel_Benchmark_Groq.ipynb) |
| **02** | **Challenge 2 · Asistente de Políticas con RAG** | Pipeline RAG completo con embeddings multilingües de `sentence-transformers`, cálculo matricial con NumPy y síntesis fáctica condicionada anti-alucinación. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jjho05/meta-llama-engineering-labs/blob/main/notebooks/02_Challenge2_RAG_Politicas_SentenceTransformers.ipynb) |

---

## Estructura del Repositorio

```bash
meta-llama-engineering-labs/
├── docs/                                    # Manuales teóricos completos en Markdown (.md)
│   ├── modulo-1-fundamentos-ia/
│   │   ├── 01-arquitectura-transformer-llama3.md
│   │   ├── 02-prompt-engineering-avanzado-rag.md
│   │   ├── 03-fine-tuning-lora-qlora-evaluacion.md
│   │   └── 04-del-prototipo-al-pipeline-productivo.md
│   └── modulo-2-automatizacion-agentes-whatsapp/
│       ├── 01-whatsapp-cloud-api-arquitectura-webhooks.md
│       ├── 02-agentes-conversacionales-memoria-redis.md
│       ├── 03-inferencia-function-calling-tools.md
│       └── 04-produccion-seguridad-llama-guard.md
├── notebooks/                               # Cuadernos Jupyter listos para Google Colab
│   ├── 01_Challenge1_MultiModel_Benchmark_Groq.ipynb
│   └── 02_Challenge2_RAG_Politicas_SentenceTransformers.ipynb
├── scripts/                                 # Scripts Python ejecutables en Terminal
│   ├── ejecutar_challenge1.py
│   └── ejecutar_challenge2.py
├── data/                                    # Datasets y políticas de prueba para RAG
│   └── reglamento_academico_politicas.json
├── .env.example                             # Plantilla de variables de entorno
├── requirements.txt                         # Dependencias del entorno
├── LICENSE                                  # Licencia MIT
└── README.md                                # Documentación principal
```

---

## Contenido del Programa Formativo (Documentación Técnica)

### Módulo 1: Fundamentos de IA & Ecosistema de Modelos Abiertos
* [Tema 1.1 · Arquitectura Transformer & Llama 3](docs/modulo-1-fundamentos-ia/01-arquitectura-transformer-llama3.md): Mecanismo de auto-atención escalada ($Q, K, V$), Grouped-Query Attention (GQA), Rotary Position Embeddings (RoPE), KV-Cache y tokenizadores BPE de 128k.
* [Tema 1.2 · Prompt Engineering & RAG](docs/modulo-1-fundamentos-ia/02-prompt-engineering-avanzado-rag.md): In-Context Learning (Few-Shot), Chain-of-Thought (CoT), delimitadores especiales y delimitación contra alucinaciones probabilísticas.
* [Tema 1.3 · Fine-Tuning LoRA / QLoRA & Evaluación](docs/modulo-1-fundamentos-ia/03-fine-tuning-lora-qlora-evaluacion.md): Matrices de bajo rango ($\Delta W = B \cdot A$), cuantización NormalFloat4 (NF4), Unsloth y métricas Perplexity/BLEU.
* [Tema 1.4 · Del Prototipo al Pipeline Productivo](docs/modulo-1-fundamentos-ia/04-del-prototipo-al-pipeline-productivo.md): Microservicios con FastAPI, endpoints de inferencia, contenedores Docker y evaluación end-to-end.

### Módulo 2: Automatización con Llama & WhatsApp Cloud API
* [Tema 2.1 · WhatsApp Cloud API & Webhooks](docs/modulo-2-automatizacion-agentes-whatsapp/01-whatsapp-cloud-api-arquitectura-webhooks.md): Verificación criptográfica del handshake GET, parseo de eventos JSON y túneles con ngrok.
* [Tema 2.2 · Agentes Conversacionales & Memoria de Sesión](docs/modulo-2-automatizacion-agentes-whatsapp/02-agentes-conversacionales-memoria-redis.md): Gestión de estado multi-turno con Redis, ventanas deslizantes de contexto y Llama Stack.
* [Tema 2.3 · Function Calling & Herramientas](docs/modulo-2-automatizacion-agentes-whatsapp/03-inferencia-function-calling-tools.md): Inferencia en dos pasos, esquemas JSON Schema y validación estricta de payloads con Pydantic.
* [Tema 2.4 · Producción SRE & Seguridad con Llama Guard](docs/modulo-2-automatizacion-agentes-whatsapp/04-produccion-seguridad-llama-guard.md): Despliegue con Docker Compose, NGINX SSL con Let\x27s Encrypt, telemetría P95 y blindaje contra Jailbreaks con Llama Guard 3 y Prompt Guard.

---

## Inicio Rápido en Local (CLI)

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
# Edita .env y agrega tu GROQ_API_KEY
```

### 4. Ejecutar los scripts de producción
```bash
# Challenge 1: Benchmarking Multi-Modelo en LPU
python3 scripts/ejecutar_challenge1.py --modelo openai/gpt-oss-20b --query "¿Qué es Grouped-Query Attention?"

# Challenge 2: Pipeline RAG con Sentence-Transformers
python3 scripts/ejecutar_challenge2.py --modelo openai/gpt-oss-20b
```

---

## Autoría & Dirección Técnica

* **Creador & Arquitecto:** **Ing. Jesús Javier Hernández Olvera**
* **Especialización:** Arquitectura de Inteligencia Artificial, Modelos de Pesos Abiertos y Agentes Autónomos.
* **Plataforma Web Interactiva:** [https://github.com/jjho05/Inteligencia_Artificial_Aplicada_Llama](https://github.com/jjho05/Inteligencia_Artificial_Aplicada_Llama)

---

## Licencia

Distribuido bajo la **Licencia MIT**. Consulta el archivo [LICENSE](LICENSE) para más detalles.
