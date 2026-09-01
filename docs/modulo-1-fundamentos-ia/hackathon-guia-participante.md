<div align="center">

[Inicio](../../README.md) • [Módulo 1](README.md) • [Anterior: Challenge 3](challenge-3-fine-tuning-lora.md) • [Siguiente: Módulo 2](../modulo-2-automatizacion-agentes-whatsapp/01-whatsapp-cloud-api-arquitectura-webhooks.md)

</div>

---

MÓDULO 1 · PROYECTO INTEGRADOR & HACKATHON DE INGENIERÍA IA

# Guía Maestra de Construcción y Mentoría: Diseña y Despliega tu Sistema de IA con Meta Llama

**Centro de Acompañamiento, Plantillas y Mentoría para Participantes**. Esta guía técnica orienta a los desarrolladores y participantes en la concepción del problema, la curaduría de datos, las estrategias de segmentación (chunking), el dimensionamiento de hardware (VRAM), la integración de código modular y la resolución de bloqueos para construir su propio sistema de Inteligencia Artificial utilizando **Meta Llama 3**, **Búsqueda Vectorial RAG**, **Model Routing** y **FastAPI**.

---

## 1. ¿Cómo Concebir tu Proyecto? Qué SÍ Resuelve un LLM y Qué NO

El error más costoso en ingeniería de Inteligencia Artificial es seleccionar un problema que no se adapta a la naturaleza probabilística de los modelos de lenguaje:

### Problemas que un LLM + RAG Resuelve con Excelencia:
* **Asistente Normativo / Políticas Institucionales:** Consultas sobre reglamentos, estatutos o garantías donde se requiere anclaje fáctico estricto y cero alucinaciones.
* **Clasificación & Extracción Estructurada:** Transformación de correos, tickets de soporte o quejas libres en objetos JSON tipados (Pydantic).
* **Tutoría Socrática Adaptativa:** Desglose progresivo de conceptos técnicos con formulación de preguntas de validación al estudiante.
* **Enrutador de Intenciones (Router):** Clasificación en milisegundos para despachar consultas directas o derivar a la base vectorial.

### Problemas que NO Debes Intentar Resolver con un LLM Puro:
* **Cálculos Matemáticos / Contabilidad Exacta:** Los LLMs son modelos de lenguaje, no calculadoras deterministas. Use funciones Python estándar.
* **Almacenamiento Transaccional (CRUD):** Un LLM no reemplaza una base de datos relacional con propiedades ACID. Use PostgreSQL / SQLite.
* **Predicciones Numéricas de Series de Tiempo:** Para pronósticos cuantitativos utilice modelos dedicados como XGBoost o ARIMA.

---

## 2. Consideraciones Críticas de Ingeniería

1. **Calidad sobre Cantidad de Datos:** 15 fragmentos fácticos limpios, ordenados y sin contradicciones superan con creces a un PDF escaneado de 500 páginas lleno de tablas rotas o ruido OCR.
2. **Estrategia de Chunking (Segmentación):**
   * *Tamaño óptimo de fragmento:* $250 - 450$ caracteres ($\sim 60 - 90$ palabras).
   * *Solapamiento (Overlap):* $10\% - 15\%$ para preservar el contexto entre límites de párrafo.
3. **Calibración del Umbral Cosenoidal:** Fijar un umbral de corte mínimo en **$\ge 0.40$** en `rag_engine.py`. Si ningún documento supera este valor, el sistema debe admitir ignorancia de forma explícita.
4. **Métricas de Éxito del Proyecto:**
   * **Precisión fáctica:** $\ge 90\%$ en preguntas de prueba.
   * **Latencia de inferencia:** $\le 1.5\text{ s}$.
   * **Seguridad:** Cero credenciales expuestas en texto plano.

---

## 3. Buenas Prácticas vs. Antipatrones de la Industria

| Dimensión Técnica | Lo que SÍ Sirve (Buena Práctica) | Lo que NO Sirve (Antipatrón) |
| :--- | :--- | :--- |
| **Estructura del Prompt** | Delimitadores explícitos (ej. `[CONTEXTO]...[/CONTEXTO]`) y roles directos. | Párrafo desordenado esperando que el modelo adivine la instrucción. |
| **Control Fáctico** | Instruir: *"Si no está en el contexto, di 'No cuento con información oficial'."* | Dejar el modelo libre para que invente políticas o datos inexistentes. |
| **Temperatura** | `temperature = 0.1 - 0.2` para respuestas precisas y reproducibles. | `temperature = 0.8 - 1.0` provocando respuestas cambiantes y alucinación. |
| **Salida JSON** | Validar la salida con esquemas de `Pydantic` en FastAPI. | Asumir que el LLM siempre devolverá JSON parseable sin errores. |
| **Credenciales** | Cargar claves desde variables de entorno con `os.getenv()`. | Escribir tokens de Hugging Face o Groq directamente en el código fuente. |

---

## 4. Plan de Acción Operativo en Cinco Días

```mermaid
graph TD
    D1["Día 1: Curaduría Documental<br>Archivo documentos.txt con 10-15 fragmentos limpios"] --> D2["Día 2: Motor RAG e Indexación<br>SentenceTransformer y búsqueda cosenoidal >= 0.40"]
    D2 --> D3["Día 3: System Prompt Maestro<br>Delimitadores, rol y filtro de admisión de ignorancia"]
    D3 --> D4["Día 4: Enrutador Inteligente<br>Clase ModelRouter para clasificar intenciones"]
    D4 --> D5["Día 5: Servidor REST con FastAPI<br>Endpoint /v1/chat y validación en Swagger UI (/docs)"]
```

* **Día 1:** Definir la temática y redactar entre 10 y 15 párrafos fácticos en `documentos.txt`.
* **Día 2:** Implementar `rag_engine.py` y validar la recuperación con confianza $\ge 0.40$.
* **Día 3:** Redactar y auditar el System Prompt con pruebas fuera de dominio (cero alucinaciones).
* **Día 4:** Conectar `router.py` para responder directamente a saludos y derivar consultas complejas.
* **Día 5:** Exponer el servicio con `api_server.py`, probar en `/docs` y preparar el archivo `README.md`.

---

## 5. Catálogo de Cuatro Plantillas de Proyectos

### Plantilla 1: Asistente de Operaciones Comerciales, Envíos y Políticas de Devolución (E-Commerce)
* **Objetivo:** Resolver dudas sobre garantías, tiempos de entrega y procedimientos de cambio de productos sin incurrir en alucinaciones.
* **Módulos:** `rag_engine.py` + `api_server.py`.
* **Dataset de Ejemplo:**
  ```python
  politicas_tienda = [
      "Politica de Reembolsos (Art. 4): Las solicitudes aplican dentro de los primeros 30 dias naturales con ticket y empaque integro.",
      "Tiempos de Entrega (Art. 2): Los envios estandar demoran de 2 a 4 dias habiles en cobertura nacional.",
      "Garantia de Hardware (Art. 7): Los articulos cuentan con 12 meses de garantia directa ante defectos de manufactura."
  ]
  ```

### Plantilla 2: Asesor Normativo Institucional y Trámites Regulatorios (Legal / Académico)
* **Objetivo:** Resolver consultas sobre trámites de titulación, solicitudes laborales o normativas institucionales citando el artículo exacto.
* **Módulos:** `rag_engine.py` + `router.py`.

### Plantilla 3: Clasificador de Incidencias y Generador Estructurado JSON
* **Objetivo:** Transformar reportes de soporte en lenguaje natural en objetos JSON con tipado estricto (categoría, severidad, acción sugerida).
* **Módulos:** `lora_adapter.py` / Prompt Few-Shot + `api_server.py`.

### Plantilla 4: Tutor Educativo Adaptativo y Evaluación Socrática
* **Objetivo:** Desglosar conceptos de ingeniería de forma progresiva, aplicando preguntas formativas para evaluar el dominio del estudiante.
* **Módulos:** `rag_engine.py` + Directivas Pedagógicas en Prompt.

---

## 6. Presupuesto de Memoria VRAM y Google Colab

| Modelo Fundacional | Consumo VRAM Estimado | Viabilidad en GPU Tesla T4 (15 GB) | Recomendación Técnica |
| :--- | :--- | :--- | :--- |
| **TinyLlama 1.1B** | $\sim 4.2\text{ GB}$ | Óptimo (100% Viable) | Excelente para computadoras personales y pruebas rápidas. |
| **Meta Llama 3.2 1B** | $\sim 5.1\text{ GB}$ | Óptimo (Recomendado) | Modelo balanceado para el Hackathon. |
| **Meta Llama 3.2 3B** | $\sim 8.4\text{ GB}$ | Óptimo (Alta Capacidad) | Recomendado para razonamiento avanzado. |
| **Meta Llama 3.1 8B** | $\sim 11.8\text{ GB}$ | Viable con QLoRA 4-bit | Requiere `BitsAndBytes` 4-bit y Batch Size = 1. |

---

## 7. Código Modular del Starter Kit

### 1. Motor Vectorial RAG (`rag_engine.py`)
```python
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorRAGEngine:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.documentos = []
        self.embeddings = None

    def index_documents(self, docs: list[str]):
        self.documentos = docs
        self.embeddings = self.embedder.encode(docs, normalize_embeddings=True)

    def search(self, pregunta: str, top_k=2, umbral=0.40):
        q_emb = self.embedder.encode([pregunta], normalize_embeddings=True)[0]
        scores = np.dot(self.embeddings, q_emb)
        indices = np.argsort(scores)[::-1][:top_k]
        
        resultados = []
        for idx in indices:
            if scores[idx] >= umbral:
                resultados.append({"texto": self.documentos[idx], "confianza": float(scores[idx])})
        return resultados
```

### 2. Enrutador de Complejidad (`router.py`)
```python
class ModelRouter:
    def __init__(self):
        self.keywords_doc = ["politica", "reembolso", "devolucion", "garantia", "envio", "precio"]
        self.keywords_json = ["json", "esquema", "ticket", "ficha"]

    def route(self, query: str) -> str:
        text = query.lower()
        if any(k in text for k in self.keywords_doc):
            return "RAG_PIPELINE"
        if any(k in text for k in self.keywords_json):
            return "LORA_ADAPTER"
        return "FAST_LLM"
```

### 3. Servidor Web REST (`api_server.py`)
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

app = FastAPI(title="Motor de Asistencia IA - Hackathon Integrador", version="1.0.0")

class MensajeEntrada(BaseModel):
    mensaje: str
    usuario: str = "invitado"

class RespuestaSalida(BaseModel):
    respuesta: str
    tiempo_ms: float
    ruta_usada: str

@app.post("/v1/chat", response_model=RespuestaSalida)
async def chatear(entrada: MensajeEntrada):
    t0 = time.perf_counter()
    if not entrada.mensaje.strip():
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacio.")
    
    duracion = (time.perf_counter() - t0) * 1000
    return RespuestaSalida(
        respuesta="Respuesta fundamentada en la documentacion oficial.",
        tiempo_ms=round(duracion, 2),
        ruta_usada="RAG_PIPELINE"
    )
```

---

## 8. Diagnóstico de Incidencias Técnicas

1. **Erradicación de Alucinaciones:**
   * Incorporar directiva estricta: *"Responda únicamente usando la evidencia del Contexto. Si no está en el Contexto, admita que no cuenta con la información."*
   * Fijar `umbral = 0.40` en la búsqueda vectorial.
2. **Error `CUDA Out of Memory (OOM)`:**
   * Utilizar modelos de 1B/3B o activar cuantización en 4-bit (QLoRA) con `per_device_train_batch_size = 1`.
3. **Ingesta de Archivos Propios:**
   * Cargar archivos en el panel de Colab y procesar con `open("archivo.txt").readlines()`.
4. **Inspección de API en Navegador:**
   * Iniciar con `uvicorn api_server:app --reload` y acceder a `http://localhost:8000/docs` para interactuar con Swagger UI.

---

## 9. Rúbrica Oficial de Evaluación y Checklist de Calidad

- [ ] **Desacoplamiento:** El conocimiento proviene de una base documental externa y no del prompt.
- [ ] **RAG Operativo:** Los textos están indexados en `rag_engine.py` y la búsqueda responde con confianza cosenoidal.
- [ ] **Control Fáctico:** El modelo no alucina ante preguntas fuera de dominio.
- [ ] **Validación VRAM:** Ejecuta sin desbordamientos de memoria en Google Colab.
- [ ] **API Tipada:** `api_server.py` responde validando esquemas Pydantic.
- [ ] **Seguridad:** Las credenciales y claves se gestionan mediante variables de entorno.

---

<div align="center">

[Volver a Challenge 3](challenge-3-fine-tuning-lora.md) • [Continuar al Módulo 2: WhatsApp & Agentes](../modulo-2-automatizacion-agentes-whatsapp/01-whatsapp-cloud-api-arquitectura-webhooks.md)

</div>
