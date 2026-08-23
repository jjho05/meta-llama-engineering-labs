<div align="center">

[🏠 Inicio](../../README.md) • [📁 Módulo 1](README.md) • [⬅️ Anterior](03-fine-tuning-lora-qlora-evaluacion.md) • [Siguiente ➡️](challenge-1-benchmark-multi-modelo.md)

</div>

---

MÓDULO 1 TEMA 4 · PIPELINE COMPLETO & SERVICIO

# Pipeline Completo: De Notebook a Producción

**Un modelo en un notebook de Colab es un experimento; solo se convierte en un servicio cuando se despliega detrás de un endpoint consumible**. Domina las 4 etapas del pipeline industrial de IA, construye microservicios con FastAPI y Pydantic, y ejecuta pruebas end-to-end con tolerancia a fallos.

Guía de Inicio · Visión del Tema 1.4

### Resumen Ejecutivo & Visión: El Camino hacia la Producción

#### 1\. Resumen Ejecutivo Síntesis Rápida del Tema

Un modelo que funciona en un notebook de Google Colab o Jupyter sigue siendo un experimento aislado: nadie fuera de ese entorno puede consumirlo sin ejecutar celdas manualmente. El pipeline completo de IA abarca la **preparación rigurosa de datos** , el **fine-tuning con LoRA** , la **evaluación cuantitativa** y el **despliegue en microservicios FastAPI** con contratos de datos estrictos.

Etapa #1 Datos Crudos

##### 1\. Preparación de Datos

**Intuición:** Como cocinar un platillo: antes de encender la estufa, lavas, desinfectas y cortas los ingredientes según la receta. Si usas datos sucios, el comportamiento del modelo será caótico.

**Técnica:** Recolección, limpieza, anonimización de PII y formateo estricto en JSONL con delimitadores de Meta Llama 3.

Etapa #2 Adaptación

##### 2\. Fine-Tuning (LoRA)

**Intuición:** Como adaptar un traje a la medida: tomas un modelo base genérico y ajustas sus costuras para que responda con el tono, estilo y formato exacto que tu negocio necesita.

**Técnica:** Inyección de adaptadores de bajo rango ($W_0 + BA$) con Loss Masking y cuantización NF4 de 4 bits.

Etapa #3 Calidad

##### 3\. Evaluación Sistemática

**Intuición:** La cena de prueba antes de abrir el restaurante al público: no basta con que el chef pruebe cada salsa; necesitas verificar que toda la comida llegue a tiempo y en el término correcto.

**Técnica:** Medición automatizada en conjunto de prueba ciego con BLEU-4, ROUGE-L, Perplexity y Llama Guard 3.

Etapa #4 Producción

##### 4\. Despliegue (FastAPI)

**Intuición:** Abrir las puertas del restaurante: los comensales piden por la app (Request) y reciben su comida (Response) sin saber cómo opera la cocina internamente.

**Técnica:** Servicio HTTP asíncrono con Uvicorn, contratos de datos Pydantic v2 y streaming SSE.

Tema 1.4

## Pipeline Completo & Arquitectura de Servicio

Deconstrucción de extremo a extremo: contratos de API, streaming asíncrono, pruebas end-to-end y métricas de latencia MLOps.

¿No entendiste? Te lo explico fácil: El juez del concurso de cocina

Evaluar con métricas automáticas antiguas como BLEU o ROUGE es como calificar un platillo gourmet pesando los gramos de sal con una báscula. **LLM-as-a-Judge** es como tener a un **chef con estrella Michelin (Llama 3 70B)** que prueba el platillo, evalúa el sabor, la textura, la presentación y te da una rúbrica profesional del 1 al 5 justificando cada calificación. 

Consejo Pro: Mitigación del Sesgo de Posición en Evaluación

Cuando uses un LLM para comparar dos respuestas (Modelo A vs Modelo B), realiza siempre **dos pasadas intercambiando el orden** : en la primera pasada evalúa [A, B] y en la segunda [B, A]. Los modelos tienden a favorecer la primera opción que leen hasta en un 18% de los casos. 

Tema 1.4.1 · El Abismo del Prototipo

### Notebook vs Producción: El Abismo entre Prototipo y Servicio

#### 1\. Guía de Inicio El Síndrome de "En Mi Notebook Sí Funciona"

En ciencia de datos e IA, el entorno de notebook (Jupyter, Google Colab) es excelente para la exploración iterativa y el entrenamiento. Sin embargo, **un notebook no es un sistema de producción**. En un notebook, las variables residen en el estado volátil de la memoria global del kernel, las dependencias no están formalmente encapsuladas y la ejecución depende de la intervención manual del ingeniero. 

Para que una aplicación cliente (como WhatsApp, una aplicación web o un ERP) consuma la inteligencia de Llama 3, el modelo debe residir detrás de un **servidor web con contratos de entrada y salida deterministas** , control de concurrencia y manejo robusto de excepciones. 

#### 2\. Concepto Formal Matriz Comparativa: Notebook vs Microservicio Productivo

Dimensión de Ingeniería | En el Notebook (Experimento) | En Producción (FastAPI Microservicio)  
---|---|---  
Acceso & Conectividad | Solo el autor ejecuta celdas interactivamente | Cualquier cliente autorizado vía HTTP/REST o WebSocket  
Contrato de Datos | Variables sueltas en memoria global (sin tipado) | Esquema Pydantic v2 estricto con validación y serialización  
Manejo de Errores | Traceback crudo que detiene el kernel | Códigos HTTP estándar (422, 503, 504) y degradación suave  
Concurrencia & Throughput | 1 usuario secuencial (hilo bloqueante) | Cientos de usuarios simultáneos con ASGI (Uvicorn / vLLM)  
Documentación | Celdas de markdown desactualizadas | OpenAPI / Swagger interactivo generado automáticamente en `/docs`  
  
Principio Fundamental de Arquitectura

La calidad de un sistema de IA en producción no se mide solo por el loss del modelo, sino por la **robustez de su interfaz de servicio, su latencia bajo carga y su capacidad de recuperarse ante fallos**.

$$\text{Servicio Productivo} = \mathcal{M}_{\theta}(\text{Pesos}) + \mathcal{S}_{\text{API}}(\text{FastAPI}) + \mathcal{V}_{\text{Schema}}(\text{Pydantic}) + \mathcal{G}_{\text{SRE}}(\text{SLAs})$$

 

Desglose de los 4 Pilares del Servicio en Producción 4 componentes

$\mathcal{M}_{\theta}$ (Pesos)

**Modelo Optimizado:** Pesos de Meta Llama 3 con cuantización NF4 y KV-Cache configurado en memoria VRAM.

$\mathcal{S}_{\text{API}}$ (Endpoint)

**Capa de Servicio ASGI:** Servidor asíncrono FastAPI y Uvicorn para atender miles de llamadas HTTP concurrentes.

$\mathcal{V}_{\text{Schema}}$ (Pydantic)

**Contrato y Validación:** Validación en Rust de tipos, rangos y mensajes rechazando peticiones inválidas con error 422.

$\mathcal{G}_{\text{SRE}}$ (SLAs)

**Gobernanza y Resiliencia:** Middlewares de logging, rate limiting, circuit breakers y métricas Prometheus de latencia.

Autoevaluación 1.4.1

¿Por qué un modelo de Llama 3 empaquetado en un endpoint HTTP con FastAPI ofrece mayor confiabilidad que ejecutarlo en un notebook de Colab?

Advertencia Crítica: No Desactives la Moderación de Salida

Muchos desarrolladores cometen el error de moderar únicamente el mensaje de entrada del usuario para ahorrar tiempo. Sin embargo, un ataque de **inyección indirecta** (oculto en un PDF recuperado por RAG) puede forzar al modelo principal a emitir datos privados de otros clientes. **Llama Guard 3 debe auditar tanto la entrada como la salida**. 

Tema 1.4.2 · Ciclo de Vida MLOps

### Las 4 Etapas del Pipeline: De Datos Crudos a Inferencia Servida

#### 1\. Concepto Formal El Pipeline como Cadena de Transformación de Valor

Un pipeline de ingeniería de IA es una secuencia reproducible y auditable donde la salida de cada etapa constituye la entrada rigurosa de la siguiente. Si la etapa de datos contiene ruido o delimitadores incorrectos, la etapa de ajuste degradará los pesos, la etapa de evaluación fallará y el servicio en producción emitirá alucinaciones. 

Topología Visual del Pipeline de Ingeniería MLOps

Paso 1 · Ingesta

Datos Crudos → SFT

JSONL Sanitizado

Paso 2 · Ajuste

LoRA / QLoRA NF4

W_0 + (alpha/r)BA

Paso 3 · Evaluación

BLEU / ROUGE / Guard

Cero Regresiones

Paso 4 · Despliegue

FastAPI & Uvicorn

Streaming SSE & REST

Banco de Pruebas 1.4.1: Simulador de Pipeline Integral en 4 Fases 

Arquitectura MLOps End-to-End

#### ¿Cómo viaja la información a través de las 4 etapas del pipeline?

Haz clic en cada una de las 4 etapas para inspeccionar la herramienta de ingeniería utilizada, el artefacto producido, su métrica de control de calidad y el código de implementación correspondiente.

####  Etapa 1: Ingesta, Limpieza y Formateo de Datos SFT 

Pandas / HuggingFace Datasets

Los datos crudos de clientes y bases de datos se filtran, anonimizan (PII) y transforman a registros JSONL con delimitadores oficiales de Meta Llama 3. 

Artefacto Producido:

train_sft_clean.jsonl (10,000 pares) 

Criterio de Aceptación:

99.8% Integridad de Esquema Pydantic 

Implementación de la Etapa
    
    
    # 1. Validación de esquema JSONL con Pydantic
    from pydantic import BaseModel, Field
    
    class Turn(BaseModel):
        role: str
        content: str
    
    class SFTRecord(BaseModel):
        messages: list[Turn] = Field(min_items=2)

$$W_{\text{prod}} = \text{merge\_and\_unload}(W_0, B, A, \alpha, r) = W_0 + \frac{\alpha}{r}(B \cdot A)$$

 

Desglose de la Fusión de Adaptadores LoRA para Producción 5 variables

$W_{\text{prod}}$ (Matriz Final)

**Pesos Fusionados:** Matriz densa final en `bfloat16` lista para servir con vLLM o TensorRT-LLM sin sobrecarga.

$W_0$ (Base)

**Pesos Pre-entrenados:** Matriz original de Meta Llama 3 con dimensiones $(d \times k)$.

$B \cdot A$ (LoRA)

**Producto de Adaptadores:** Factorización de bajo rango ajustada durante el Supervised Fine-Tuning.

$\frac{\alpha}{r}$ (Escala)

**Factor de Ponderación:** Coeficiente constante que modula el impacto del conocimiento adaptado.

$\text{Overhead} \to 0$

**Cero Latencia Adicional:** Elimina la necesidad de bifurcar operaciones tensoriales en tiempo de inferencia.

Autoevaluación 1.4.2

Si los datos de entrenamiento contienen registros malformados o turnos de asistente vacíos, ¿en qué etapa del pipeline se manifiesta primero el problema de forma crítica?

Tema 1.4.3 · Contratos de API & FastAPI

### Empacar el Modelo: Endpoints, Contratos Pydantic v2 & OpenAPI

#### 1\. Concepto Formal El Endpoint como Mostrador de Pedidos

Un endpoint es una ruta de red (URI) combinada con un método HTTP (como `POST /v1/chat/completions`) que define un **contrato de comunicación bidireccional**. El cliente envía un _Request Payload_ con tipado estricto y el servidor garantiza una _Response_ en formato estándar. 

FastAPI utiliza **Pydantic v2** para validar tipos en tiempo de ejecución. Si el cliente envía una temperatura de `"alta"` en lugar de un número flotante `0.7`, FastAPI intercepta la petición en $O(1)$ y emite un código `422 Unprocessable Entity` antes de gastar un solo ciclo de GPU en inferencia. 

Banco de Pruebas 1.4.2: Cliente HTTP & Swagger UI Playground en Vivo 

OpenAPI 3.1 & Pydantic v2

#### Prueba de Validación de Esquemas HTTP en Tiempo Real

Envía peticiones con formato válido o inyecta errores de tipo para comprobar cómo Pydantic v2 y FastAPI protegen la GPU devolviendo respuestas `200 OK` o rechazando datos con `422 Unprocessable Entity`.

Seleccionar Caso de Envío HTTP: 

HTTP Request Payload (JSON): POST /v1/chat/completions

request_payload.json

JSON Válido

HTTP Server Response:

200 OK 92 ms

response_200.json

Cabeceras HTTP:
    
    
    content-type: application/json; charset=utf-8
    server: uvicorn / fastapi
    x-process-time: 92ms
    x-llama-engine: vLLM PagedAttention

#### 3\. Código de Producción Servidor FastAPI con Validación Pydantic v2

app_fastapi_llama3.py (Microservicio de Inferencia)
    
    
    from fastapi import FastAPI, HTTPException, status
    from pydantic import BaseModel, Field
    from typing import List, Optional
    import time
    
    app = FastAPI(
        title="Meta Llama 3 Production Microservice",
        version="1.0.0",
        description="API de alto throughput con contratos de datos estrictos"
    )
    
    class ChatMessage(BaseModel):
        role: str = Field(..., pattern="^(system|user|assistant)$")
        content: str = Field(..., min_length=1, max_length=16000)
    
    class ChatCompletionRequest(BaseModel):
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
        messages: List[ChatMessage] = Field(..., min_items=1)
        temperature: float = Field(0.7, ge=0.0, le=2.0)
        max_tokens: Optional[int] = Field(512, gt=0, le=4096)
    
    @app.post("/v1/chat/completions", status_code=status.HTTP_200_OK)
    async def create_chat_completion(req: ChatCompletionRequest):
        t0 = time.perf_counter()
        # Inferencia delegada al motor vLLM / Ollama
        output_text = "Respuesta generada con Llama 3."
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return {
            "model": req.model,
            "choices": [{"message": {"role": "assistant", "content": output_text}}],
            "latency_ms": round(elapsed_ms, 2)
        }

$$\text{Throughput}_{\text{cluster}} = \frac{\sum_{i=1}^{U} N_{\text{tokens}}^{(i)}}{\Delta t} = \frac{U \cdot \bar{N}_{\text{tokens}}}{T_{\text{servicio}}}$$

 

Desglose de la Capacidad y Throughput del Endpoint 4 métricas

$\text{Throughput}$

**Rendimiento Global:** Tasa total de tokens emitidos por segundo a través de todos los clientes conectados.

$U$ (Usuarios)

**Concurrencia Activa:** Número de peticiones simultáneas procesadas en el batch continuo de la GPU.

$\bar{N}_{\text{tokens}}$

**Tokens Promedio:** Longitud media de respuesta generada por solicitud ($S_{\text{gen}}$).

$T_{\text{servicio}}$

**Tiempo de Servicio:** Duración de ejecución en GPU incluyendo transferencia PCI-e y decodificación.

Autoevaluación 1.4.3

Si un cliente envía una petición HTTP con el campo temperature = -0.5, ¿qué acción ejecuta FastAPI antes de pasar la solicitud a la GPU?

Tema 1.4.4 · Experiencia de Usuario & Streaming

### Streaming de Inferencia: Server-Sent Events (SSE) & Time to First Token

#### 1\. Concepto Formal Latencia Percibida vs Latencia Total

En modelos de lenguaje grandes, generar una respuesta completa de 500 tokens puede tomar entre 3 y 8 segundos. Si el servidor espera a completar toda la secuencia antes de enviar la respuesta HTTP, el usuario experimenta una pantalla congelada. 

La solución estándar de la industria es el **Streaming asíncrono con Server-Sent Events (SSE)** bajo el tipo de contenido `text/event-stream`. El servidor emite cada token inmediatamente después de ser predicho por la función de muestreo ($y_t$), logrando un **Time to First Token (TTFT)** de menos de 100 ms. 

$$\text{Latencia Percibida} = \text{TTFT} = T_{\text{red}} + T_{\text{prefill}}(S_{\text{prompt}}) + T_{\text{decode}}(1)$$

 

Desglose de la Fórmula de Latencia Percibida (TTFT) 4 variables

$\text{TTFT}$

**Time to First Token:** Tiempo transcurrido desde que el usuario presiona Enviar hasta que la primera palabra aparece en pantalla.

$T_{\text{red}}$ (RTT)

**Latencia de Red:** Tiempo de transmisión del paquete HTTP y handshake TLS entre cliente y servidor.

$T_{\text{prefill}}$

**Fase de Prefill (Prompt Processing):** Tiempo de cálculo paralelo para procesar y proyectar los embeddings de todo el prompt de entrada en GPU.

$T_{\text{decode}}(1)$

**Primer Paso Autoregresivo:** Tiempo de ejecución de una pasada hacia adelante (forward pass) y muestreo Softmax del token #1.

Banco de Pruebas 1.4.5: Simulador de Inferencia en Streaming (SSE) vs Modo Bloqueante 

Server-Sent Events & TTFT

#### Demostración de Experiencia de Usuario: Streaming vs Bloqueante

Compara en vivo cómo se percibe la inferencia cuando el microservicio utiliza **Server-Sent Events (SSE)** emitiendo token por token con TTFT ultra-bajo (<80ms) frente al modo tradicional bloqueante (HTTP 1.1) donde el usuario espera inmóvil durante segundos.

\--

Time to First Token (TTFT)

\--

Inter-Token Latency (ITL)

\--

Duración Total de Respuesta

Terminal del Usuario (Tokens en Vivo): 

Presiona un botón arriba para iniciar la simulación de inferencia...

Flujo HTTP Crudo (text/event-stream): 
    
    
    Esperando conexión...

Autoevaluación 1.4.4

¿Cuál es el beneficio técnico principal de utilizar Server-Sent Events (SSE) en un endpoint de inferencia de Llama 3?

Tema 1.4.5 · Resiliencia & Testing E2E

### Probar Antes de Confiar: Testing End-to-End & Manejo de Fallos

#### 1\. Guía de Inicio El Ensayo General de la Orquesta

Probar cada función aislada con pruebas unitarias es como verificar que cada músico toque bien su instrumento en casa. El **Testing End-to-End (E2E)** es el ensayo general en el teatro: solo cuando toda la sinfonía suena junta detectas si el violín se tapa con la batería o si la entrada del coro está desfasada. 

Una suite de pruebas E2E profesional simula peticiones reales y valida tres requisitos obligatorios: 

1\. Exactitud de Negocio

La respuesta final cumple con el contrato JSON, los filtros de seguridad y el formato requerido por la aplicación cliente.

2\. Límites de Tiempo (SLA)

La petición completa se resuelve por debajo del umbral de timeout acordado (ej. < 2.0 segundos).

3\. Fallos Controlados (Graceful)

Si la base vectorial está caída o el documento no existe, el sistema emite un mensaje claro sin colapsar el servidor con error 500.

Banco de Pruebas 1.4.3: Inyector de Fallos Controlados & Auditor E2E 

Chaos Engineering & Resiliencia

#### Auditoría de Resiliencia ante Incidentes Reales en Producción

Selecciona un escenario de fallo para verificar cómo el microservicio ejecuta circuit breakers, timeouts y respuestas de degradación controlada evitando caídas del servicio.

Fallo Inyectado: Base Vectorial ChromaDB no responde (Connection Refused) 

El microservicio detectó la caída de la base vectorial en <15ms y ejecutó el fallback de degradación controlada respondiendo con la memoria paramétrica del modelo base sin lanzar 500 Internal Server Error no capturado. 

Manejado Gracefully (HTTP 503 con Fallback)

Logs del Servidor & Traza de Auditoría:
    
    
    [CRITICAL] 2026-08-18 21:14:02 - httpx.ConnectError: [Errno 111] Connection refused (chromadb:8000)
    [INFO] Activando Circuit Breaker: Fallback a Llama 3 Base sin contexto RAG.
    [200 OK] Response entregada al usuario con aviso de degradación en 98ms.

#### 2\. Código de Pruebas Suite de Testing E2E con PyTest & TestClient

test_pipeline_e2e.py (Pruebas Automatizadas)
    
    
    import pytest
    from fastapi.testclient import TestClient
    from app_fastapi_llama3 import app
    
    client = TestClient(app)
    
    def test_chat_completion_happy_path():
        # Caso feliz: contrato valido genera 200 OK y respuesta correcta
        payload = {
            "messages": [{"role": "user", "content": "Hola, ¿cómo estás?"}],
            "temperature": 0.7
        }
        res = client.post("/v1/chat/completions", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
    
    def test_chat_completion_invalid_type_422():
        # Robustez: temperatura invalida es rechazada con 422 sin colapsar
        payload = {
            "messages": [{"role": "user", "content": "Test"}],
            "temperature": "extremadamente alta"  # Tipo erroneo (str en vez de float)
        }
        res = client.post("/v1/chat/completions", json=payload)
        assert res.status_code == 422
        assert "detail" in res.json()

$$T_{\text{total\_E2E}} = T_{\text{red}} + T_{\text{retrieval}}(k) + T_{\text{prefill}} + \sum_{t=2}^{N} \text{ITL}_t$$

 

Desglose de la Latencia End-to-End en Pipelines RAG 5 términos

$T_{\text{total\_E2E}}$

**Latencia Completa:** Tiempo total de ejecución desde el envío de la consulta hasta la recepción del último token.

$T_{\text{retrieval}}(k)$

**Búsqueda Vectorial:** Tiempo de consulta de similitud coseno en ChromaDB recuperando los $k$ chunks más relevantes.

$T_{\text{prefill}}$

**Prefill del Prompt Aumentado:** Procesamiento en GPU del prompt enriquecido con el contexto recuperado.

$\text{ITL}_t$

**Latencia Inter-Token:** Tiempo medio entre la emisión de dos tokens consecutivos en la fase decodificadora.

$N$ (Longitud)

**Tokens Generados:** Cantidad total de tokens producidos por la respuesta del asistente.

Autoevaluación 1.4.5

¿Cuál es la diferencia fundamental entre probar una función en una celda de notebook y ejecutar una prueba End-to-End con PyTest?

Tema 1.4.6 · Dimensionamiento & MLOps

### Métricas de Latencia MLOps, Disponibilidad (SLA) & Concurrencia

#### 1\. Concepto Formal Formulación Matemática de Rendimiento en Inferencia

Para dimensionar clusters de inferencia en producción se aplican tres ecuaciones fundamentales de ingeniería de sistemas: 

Banco de Pruebas 1.4.4: Calculadora de Latencia MLOps, Concurrencia & SLA 

Dimensionamiento de Servidores

#### Simulación de Concurrencia y Experiencia de Usuario

Ajusta el número de usuarios simultáneos, los tokens por respuesta y el tipo de hardware para estimar en tiempo real el TTFT, la latencia total y el cumplimiento de SLA.

Usuarios Concurrentes: 20 usuarios

1 usuario 50 usuarios 100 usuarios

Tokens por Respuesta: 120 tokens

20 (Conciso) 250 (Párrafo) 500 (Extenso)

Hardware / Motor de Inferencia:

1x NVIDIA RTX 4090 (24GB) con vLLM PagedAttention 1x NVIDIA H100 (80GB) con TensorRT-LLM CPU Servidor / Apple Silicon con Ollama GGUF

Curva de Latencia vs Concurrencia de Usuarios (GPU Saturation Model) 

— Curva de Latencia (ms) \- - Límite SLA (2.0s)

111 ms

Time to First Token (TTFT)

1.25 s

Latencia Total E2E

1,840 tok/s

Throughput Total de Inferencia

99.4% SLA

Disponibilidad de Latencia (<2.0s)

**Régimen de Excelencia:** Latencia total de **1.25s** para 20 usuarios simultáneos. La experiencia conversacional es instantánea y fluida para canales como WhatsApp o Web. 

$$L = \lambda \cdot W \quad \iff \quad \text{Concurrencia} = \text{Tasa de Llegada (RPS)} \times \text{Tiempo de Respuesta (s)}$$

 

Desglose de la Ley de Little para Dimensionamiento de Clusters 4 variables

$L$ (Concurrencia)

**Solicitudes en Vuelo:** Número promedio de peticiones activas simultáneamente en el servidor.

$\lambda$ (RPS)

**Tasa de Llegada:** Flujo de peticiones nuevas por segundo recibidas por el balanceador de carga.

$W$ (Latencia)

**Tiempo en el Sistema:** Latencia promedio que tarda el microservicio en completar una petición.

$N_{\text{GPU}} = \lceil L / C \rceil$

**Cálculo de Hardware:** GPUs necesarias dividiendo la concurrencia $L$ entre la capacidad por GPU $C$.

Autoevaluación 1.4.6

Si tu sistema recibe 50 peticiones por segundo y cada una tarda en promedio 0.2 segundos en ser procesada, ¿cuántas solicitudes concurrentes están activas simultáneamente en el servidor según la Ley de Little?

Tema 1.4.7 · Contenerización & DevOps

### Empaquetado en Contenedores: Docker & Despliegue con Uvicorn

#### 1\. Concepto Formal Inmutabilidad y Aislamiento de Entorno

Para garantizar que el servicio corra de forma idéntica en local, en servidores on-premise o en la nube, el microservicio se empaqueta en un contenedor Docker con controladores NVIDIA CUDA y dependencias aisladas. 

Dockerfile (Producción con Soporte GPU)
    
    
    FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
    
    # 1. Instalamos Python 3.11 y dependencias de sistema
    RUN apt-get update && apt-get install -y python3.11 python3-pip && rm -rf /var/lib/apt/lists/*
    
    WORKDIR /app
    
    # 2. Copiamos requerimientos e instalamos paquetes
    COPY requirements.txt .
    RUN pip3 install --no-cache-dir -r requirements.txt
    
    # 3. Copiamos el código del microservicio y pesos fusionados
    COPY app_fastapi_llama3.py .
    COPY ./llama3-merged /app/llama3-merged
    
    EXPOSE 8000
    
    # 4. Servidor Uvicorn ASGI con múltiples workers
    CMD ["uvicorn", "app_fastapi_llama3:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]

$$\text{Disponibilidad (SLA)} = \frac{\text{MTBF}}{\text{MTBF} + \text{MTTR}} \times 100\% \ge 99.9\%$$

 

Desglose de la Ecuación de Disponibilidad y Resiliencia SRE 4 parámetros

$\text{SLA}$ (Uptime)

**Disponibilidad Comprometida:** Porcentaje de tiempo que el endpoint responde exitosamente (200 OK) bajo el umbral de latencia acordado.

$\text{MTBF}$ (Estabilidad)

**Mean Time Between Failures:** Tiempo medio operativo sin fallos entre dos incidentes de producción.

$\text{MTTR}$ (Recuperación)

**Mean Time To Recovery:** Tiempo promedio requerido para reiniciar el contenedor Docker o reasignar tráfico al nodo secundario.

$\text{Error Budget}$

**Presupuesto de Error:** Margen de tolerancia de fallos ($0.1\% \approx 43.8\text{ min/año}$) antes de congelar despliegues.

Autoevaluación 1.4.7

¿Cuál es el beneficio de compilar una imagen Docker multi-stage para un microservicio de Llama 3?

Tema 1.4.8 · Gobernanza & Documentación

### Documentación del Pipeline como Entregable Profesional

#### 1\. Concepto Formal La Documentación no es un Trámite, es el Sistema

En ingeniería de software profesional, un pipeline sin documentación de contratos, dependencias y esquemas de datos no puede ser mantenido por nadie más que su creador. Documentar el pipeline implica tres componentes indispensables: 

1

##### Contrato OpenAPI 3.1 (/docs)

Especificación viva interactiva que permite a otros equipos de frontend, backend y QA probar las llamadas HTTP con ejemplos reales.

2

##### Configuración Declarativa (pipeline_config.yaml)

Desacopla hiperparámetros, rutas de adaptadores LoRA y umbrales de similitud del código fuente en archivos versionables.

3

##### Endpoint de Salud (/health)

Health check que valida el estado de la GPU, la memoria VRAM y la conexión con bases vectoriales para balanceadores de carga.

Síntesis del Módulo 1 Completo

Has completado el recorrido completo de la arquitectura de LLMs: desde el cálculo de probabilidades autoregresivas y tokenización BPE en el **Tema 1.1** , el prompting avanzado y RAG en el **Tema 1.2** , el fine-tuning matemático con LoRA en el **Tema 1.3** , hasta el empaquetado en microservicios FastAPI en el **Tema 1.4**. Estás listo para integrar este agente con la WhatsApp Cloud API en el **Módulo 2**.

Terminología & MLOps

## Glosario Técnico de Producción

Conceptos clave de arquitectura de microservicios, inferencia asíncrona y despliegue industrial de Meta Llama 3.

Arquitectura #01

Pipeline de Producción

Flujo de ingeniería estructurado y automatizado que transforma datos crudos en un servicio consumible mediante ingesta, fine-tuning, evaluación y despliegue en microservicios.

Principio: Un modelo en notebook es un experimento; detrás de un endpoint es un servicio.

FastAPI #02

ASGI (Asynchronous Server Gateway Interface)

Estándar de interfaz asíncrona de Python que permite a servidores como Uvicorn manejar miles de conexiones concurrentes I/O sin bloquear hilos del sistema operativo.

Implementación: Base de FastAPI y Starlette para concurrencia de alto rendimiento.

FastAPI #03

Pydantic v2 Core

Biblioteca de validación de datos y tipado estricto en Python basada en un motor compilado en Rust. Valida tipos, rangos y esquemas JSON con rendimiento hasta 20x superior a v1.

Seguridad: Emite error 422 Unprocessable Entity ante payloads no válidos.

FastAPI #04

Server-Sent Events (SSE)

Protocolo estándar de transporte unidireccional sobre HTTP (Content-Type: text/event-stream) que transmite tokens generados por Llama 3 al cliente de forma incremental e interactiva.

Impacto UX: Reduce el Time to First Token (TTFT) percibido de segundos a milisegundos.

MLOps #05

Time to First Token (TTFT)

Métrica de latencia que mide el tiempo transcurrido desde que el usuario envía su request hasta que el primer token es generado y emitido por el modelo. Abarca el procesamiento del prompt y KV-Cache prefill.

Objetivo SLA: TTFT < 100 ms en aplicaciones conversacionales en vivo.

MLOps #06

Inter-Token Latency (ITL)

Tiempo promedio transcurrido entre la emisión de dos tokens consecutivos durante la fase de decodificación autorregresiva de Llama 3.

Fluidez: Un ITL < 30 ms/tok garantiza una lectura natural sin pausas perceptibles.

MLOps #07

Ley de Little ($L = \lambda W$)

Teorema de teoría de colas que establece que el número promedio de solicitudes concurrentes en el servidor ($L$) equivale a la tasa de llegada ($\lambda$) multiplicada por el tiempo de servicio ($W$).

Dimensionamiento: Permite calcular la capacidad de workers y GPUs necesarias para un SLA.

Arquitectura #08

merge_and_unload()

Método de la librería PEFT de Hugging Face que fusiona permanentemente las matrices LoRA ($BA$) con los pesos base ($W_0$) obteniendo $W_{\text{final}}$, eliminando la sobrecarga computacional de bifurcación.

Producción: Imprescindible para servir con motores optimizados como vLLM o TensorRT-LLM.

MLOps #09

Prueba End-to-End (E2E)

Estrategia de verificación integral que prueba el flujo completo desde la petición HTTP del cliente, validación de esquemas, consulta RAG, inferencia de Llama 3 y serialización de respuesta.

Contraste: Supera los falsos positivos de las celdas aisladas de Jupyter Notebook.

FastAPI #10

OpenAPI 3.1 & Swagger UI

Especificación estándar para describir contratos de APIs RESTful. FastAPI genera automáticamente una interfaz interactiva en `/docs` para pruebas en vivo y generación de SDKs.

Entregable: Elimina fricción entre equipos de frontend, backend y QA.

MLOps #11

Contenerización Docker Multi-Stage

Técnica de empaquetado que separa la etapa de compilación e instalación de dependencias de la imagen final de ejecución, descartando compiladores temporales y reduciendo el peso de la imagen.

Seguridad: Minimiza vulnerabilidades CVE y acelera despliegues en Kubernetes.

Arquitectura #12

Loss Masking (label = -100)

Técnica de ingeniería de datos SFT que asigna el valor -100 a los tokens de las instrucciones del sistema y del usuario en el tensor de etiquetas, instruyendo a PyTorch a calcular gradientes solo en las respuestas del modelo.

Eficiencia: Evita que el modelo gaste capacidad aprendiendo a predecir las preguntas del usuario.

MLOps #13

Throughput de Inferencia

Volumen total de tokens generados por unidad de tiempo (tokens/segundo) procesados colectivamente por el cluster de inferencia a través de múltiples usuarios concurrentes.

Escalabilidad: Aumenta con técnicas de batching continuo como PagedAttention en vLLM.

Arquitectura #14

Llama Guard 3 Shield

Modelo especializado de clasificación de seguridad de Meta AI que evalúa prompts y respuestas de Llama 3 contra 14 categorías de riesgo ético y de seguridad informática en milisegundos.

Gobernanza: Actúa como cortafuegos antes de entregar contenido al usuario final.

MLOps #15

Circuit Breaker & Graceful Fallback

Patrón de resiliencia que detecta fallos repetidos o latencias excesivas en subsistemas externos (bases vectoriales o GPU) e interrumpe las llamadas para entregar respuestas alternativas estructuradas.

Estabilidad: Previene el colapso en cascada de los workers ASGI ante fallos de infraestructura.

Práctica & Aplicación de Ingeniería

## Ejercicios Prácticos del Tema 1.4

Consolida tus competencias resolviendo los 4 ejercicios del temario oficial con soluciones detalladas paso a paso, análisis arquitectónico y consideraciones MLOps.

Ejercicio 1

#### Diagrama de Flujo y Cadena de Transformación del Pipeline

Enunciado Oficial 

Dibuja un diagrama de flujo con las cuatro etapas del pipeline que llevan datos crudos a un modelo servido. Escribe al menos una herramienta o técnica mencionada en la lectura para cada etapa (preparación de datos, ajuste, evaluación y despliegue). 

Ver Solución de Ingeniería Paso a Paso & Análisis MLOps

1

#####  Etapa 1: Ingesta, Limpieza y Formateo de Datos SFT 

**Herramientas:** `Pandas`, `HuggingFace Datasets`, `Pydantic v2`.  
**Técnica:** Se filtran caracteres de control corruptos, se anonimizan datos personales (PII) y se construye el esquema JSONL con los delimitadores oficiales de Meta Llama 3 (`<|start_header_id|>system<|end_header_id|>`, `<|start_header_id|>user...`).  
**Artefacto Producido:** Archivo `train_sft_clean.jsonl` con 10,000 pares validados.  
**Criterio de Calidad:** 100% de cumplimiento del esquema Pydantic y cero turnos de asistente vacíos. 

2

#####  Etapa 2: Ajuste Eficiente (Fine-Tuning LoRA / QLoRA) 

**Herramientas:** `TRL (SFTTrainer)`, `PEFT`, `bitsandbytes`, `Unsloth`.  
**Técnica:** Cuantización NF4 en 4 bits de los pesos base $W_0$ combinada con adaptadores de bajo rango ($r=16, \alpha=32$) y _Loss Masking_ (etiqueta `label = -100` en tokens de usuario) para que el optimizador solo calcule gradientes sobre las respuestas del asistente.  
**Artefacto Producido:** Pesos de adaptador `adapter_model.safetensors` (~16 MB).  
**Criterio de Calidad:** Curva de pérdida convergente sin sobreajuste (Train Loss $\approx 0.8$, Validation Loss $\approx 0.95$). 

3

#####  Etapa 3: Evaluación Sistemática & Blindaje de Seguridad 

**Herramientas:** `Evaluate`, `Llama Guard 3`, `PyTest`.  
**Técnica:** Evaluación en conjunto ciego de prueba calculando Perplejidad ($PPL$), $BLEU\text{-}4$, $ROUGE\text{-}L$ y auditoría de alineación ética contra 14 categorías de riesgo con Llama Guard 3.  
**Artefacto Producido:** Reporte formal de métricas `eval_report_v1.json`.  
**Criterio de Calidad:** $ROUGE\text{-}L > 0.85$ y 0% de violaciones en guardrails de seguridad. 

4

#####  Etapa 4: Despliegue en Microservicios con FastAPI 

**Herramientas:** `FastAPI`, `Uvicorn`, `Docker`, `vLLM / Ollama`.  
**Técnica:** Fusión de pesos con `merge_and_unload()`, empaquetado en contenedor Docker multi-stage, exposición de endpoint REST `POST /v1/chat/completions` con streaming Server-Sent Events (SSE) y contrato OpenAPI.  
**Artefacto Producido:** Imagen Docker de producción `enterprise/llama3-service:v1.0`.  
**Criterio de Calidad:** $TTFT < 100\text{ ms}$ y disponibilidad $\ge 99.9\%$ bajo pruebas de carga. 

Ejercicio 2

#### Analogía del Mostrador de Pedidos: Deconstrucción de FastAPI

Enunciado Oficial 

Explica con tus propias palabras por qué un endpoint de FastAPI se parece al mostrador de pedidos de un restaurante. Identifica qué parte de esa analogía representa la request, la response y la cocina interna del modelo. 

Ver Solución de Ingeniería Paso a Paso & Mapeo Arquitectónico

A

#####  El Mostrador de Pedidos (El Endpoint FastAPI) 

Representa la **interfaz pública estandarizada** del servicio. Así como un comensal no entra a la cocina a manipular las ollas ni necesita saber a qué temperatura exacta está el horno, el cliente de software (como WhatsApp o un frontend web) no interactúa directamente con los tensores ni con la memoria VRAM de la GPU: solo se comunica con la URL del endpoint (`POST /v1/chat/completions`). 

B

#####  La Request (La Orden del Cliente con Menú Validado) 

Es el **payload JSON** que envía el cliente con los parámetros de su consulta (lista de mensajes, temperatura, tokens máximos). Pydantic v2 actúa como el cajero del mostrador: si pides un platillo que no existe o especificas una temperatura negativa, el cajero rechaza la orden de inmediato con código `422 Unprocessable Entity` antes de enviar la comanda a la cocina, protegiendo los recursos. 

C

#####  La Cocina Interna (Inferencia en GPU con Llama 3) 

Es la maquinaria pesada de ejecución: los kernels CUDA, la memoria KV-Cache, las matrices de auto-atención QKV y la base vectorial RAG. La cocina recibe la comanda validada, procesa los tensores y prepara el resultado sin que el cliente conozca la complejidad interna. 

D

#####  La Response (El Platillo Empaquetado o la Entrega en Streaming) 

Es la **respuesta HTTP final** entregada al cliente con código `200 OK`, el texto generado por Llama 3 y el conteo de tokens consumidos. En modo *Streaming (SSE)*, equivale a servir los platillos uno a uno a la mesa a medida que van saliendo del fuego, reduciendo la espera percibida por el comensal. 

Ejercicio 3

#### Caso de Prueba End-to-End con Manejo de Fallos Controlados

Enunciado Oficial 

Escribe un caso de prueba end-to-end para el asistente RAG del caso práctico. Define una pregunta realista, el resultado esperado y un escenario de fallo controlado (por ejemplo, cuando el documento buscado no existe). 

Ver Solución de Ingeniería Paso a Paso & Script PyTest

1

#####  Caso 1: Escenario Feliz (Happy Path con Recuperación RAG) 

**Pregunta del Usuario:** "¿Cuál es la política oficial de viáticos para viajes internacionales de soporte técnico?"  
**Flujo Técnico:** La consulta se vectoriza → ChromaDB encuentra 3 chunks con similitud coseno $> 0.82$ → Llama 3 sintetiza la respuesta citando la Sección 5.1 del Manual de Políticas Corporativas.  
**Resultado Esperado:** Código HTTP `200 OK`, latencia total $< 1.2\text{ s}$ y mención explícita del artículo correspondiente. 

2

#####  Caso 2: Escenario de Fallo Controlado (Documento Inexistente) 

**Pregunta del Usuario:** "¿Cuál es el procedimiento para solicitar reembolso de boletos para el viaje a Marte 2029?"  
**Comportamiento del Pipeline:** La búsqueda vectorial en ChromaDB arroja un score de similitud máximo de $0.14$ (por debajo del umbral de corte de $0.35$).  
**Respuesta Controlada (*Graceful Fallback*):** El sistema intercepta el score bajo y devuelve: _"No se encontró documentación oficial sobre este tema en el repositorio corporativo. Por favor contacta a RRHH."_  
**Aserción de Confianza:** Se evita la alucinación de datos inexistentes y el servidor entrega código HTTP `200 OK` estructurado en vez de un fallo no controlado. 

3

#####  Implementación de la Prueba con PyTest 

test_rag_pipeline_fallbacks.py
    
    
    def test_rag_controlled_fallback_empty_docs():
        client = TestClient(app)
        payload = {"messages": [{"role": "user", "content": "¿Viaje a Marte 2029?"}]}
        res = client.post("/v1/chat/completions", json=payload)
        
        assert res.status_code == 200
        answer = res.json()["choices"][0]["message"]["content"]
        assert "No se encontró documentación oficial" in answer

Ejercicio 4

#### Análisis Comparativo de Confianza: Celda de Notebook vs Flujo E2E

Enunciado Oficial 

Enumera tres diferencias concretas entre probar una celda de notebook que ejecuta el modelo, y probar el flujo completo que incluye la búsqueda RAG, la generación con Llama y la respuesta del endpoint. ¿Por qué la segunda opción da mayor confianza? 

Ver Solución de Ingeniería Paso a Paso & Análisis Comparativo

1

#####  Diferencia 1: Dependencias Ocultas y Estado Global de Memoria 

En un notebook, una celda puede funcionar porque depende de una variable o función auxiliar ejecutada en una celda previa hace 3 horas. En una prueba End-to-End, el entorno arranca desde cero en un proceso aislado, garantizando que el microservicio sea 100% autónomo y reproducible sin variables fantasmas. 

2

#####  Diferencia 2: Serialización, Validación de Tipos y Protocolo de Red 

El notebook ejecuta objetos nativos de Python directamente en memoria (`dict`, tensores PyTorch). La prueba E2E somete la petición a la serialización JSON real, cabeceras HTTP, validación estricta de esquemas Pydantic v2 y des-serialización de respuesta, detectando incompatibilidades de tipos que en el notebook pasan desapercibidas. 

3

#####  Diferencia 3: Manejo de Concurrencia, Timeouts y Resiliencia 

Si la base vectorial tarda 8 segundos o la GPU se satura, el notebook simplemente se congela de manera indefinida. La prueba E2E valida que los middlewares de FastAPI apliquen timeouts (`asyncio.timeout`), rate limiting y circuit breakers para liberar los workers ASGI y mantener el SLA sin colapsar. 

#####  ¿Por qué la prueba End-to-End otorga máxima confianza profesional? 

Porque **emula exactamente la experiencia del usuario final y el comportamiento del sistema bajo condiciones reales de producción**. Demuestra que todos los subsistemas (validación, embeddings, búsqueda vectorial, inferencia autoregresiva y red) colaboran armónicamente como un servicio industrial robusto. 

Gobernanza & Bibliografía

## Fuentes Oficiales & Referencias MLOps

Documentación oficial, estándares de la industria y guías de arquitectura de Meta AI, FastAPI y MLOps.

FastAPI · 2024 Framework ASGI

#### FastAPI Framework Documentation

Guía oficial de diseño de APIs asíncronas con Starlette, inyección de dependencias, middlewares de métricas y validación con Pydantic v2. 

[ Consultar en FastAPI Docs ](https://fastapi.tiangolo.com/)

Meta AI · 2024 Guía de Inferencia

#### Meta Llama 3 Inference & Serving Guide

Especificación de Meta sobre hardware recomendado, optimización de KV-Cache, Grouped-Query Attention y serving distribuido. 

[ Consultar en Meta Llama Docs ](https://llama.meta.com/docs/)

Encode / Uvicorn · 2024 Servidor ASGI

#### Uvicorn: The Lightning-Fast ASGI Server

Arquitectura de servidor web asíncrono basada en uvloop (bindings C de libuv) y httptools para concurrencia masiva en microservicios Python. 

[ Consultar en Uvicorn.org ](https://www.uvicorn.org/)

Pydantic · 2023 Validación en Rust

#### Pydantic v2 Core: Data Validation Speed

Samuel Colvin et al. presentan el motor de validación reescrito en Rust (`pydantic-core`) que acelera la serialización JSON hasta 20x. 

[ Consultar en Pydantic Docs ](https://docs.pydantic.dev/latest/)

PyTest · 2024 Testing Automatizado

#### PyTest & Async Testing Best Practices

Metodología de testing para microservicios asíncronos con fixtures, parametrización de casos límite y validación de endpoints HTTP con TestClient. 

[ Consultar en PyTest.org ](https://docs.pytest.org/)

W3C · 2015 Estándar Web

#### Server-Sent Events (SSE) Specification

Estándar de la W3C para transporte unidireccional de flujos de eventos sobre HTTP (`text/event-stream`), núcleo de la inferencia en streaming. 

[ Consultar en W3C / WHATWG ](https://html.spec.whatwg.org/multipage/server-sent-events.html)

UC Berkeley · SOSP 2023 Paper SOTA

#### vLLM: Efficient Memory Management with PagedAttention

Kwon et al. introducen PagedAttention para eliminar la fragmentación del KV-Cache en memoria GPU, multiplicando el throughput de inferencia por 2–4x. 

[ Consultar en arXiv: 2309.06180 ](https://arxiv.org/abs/2309.06180)

Docker · 2024 Contenerización

#### Docker Multi-Stage Builds for AI & CUDA

Guía de ingeniería para compilar imágenes ligeras con NVIDIA Container Toolkit, aislando dependencias y minimizando la superficie de ataque. 

[ Consultar en Docker Docs ](https://docs.docker.com/build/building/multi-stage/)

Heroku / Open Source Metodología

#### The Twelve-Factor App Methodology

Principios arquitectónicos para construir microservicios nativos en la nube: configuración en entorno, procesos sin estado y desacoplamiento de logs. 

[ Consultar en 12factor.net ](https://12factor.net/)

MIT Press · 1961 Teoría de Colas

#### Little's Law: A Proof for the Queuing Formula

John Little formula $L = \lambda W$, ecuación fundamental de sistemas computacionales que vincula concurrencia de requests, tasa de llegada y latencia. 

[ Consultar en INFORMS ](https://www.informs.org/)

Meta AI · 2024 Seguridad en Producción

#### Llama Guard 3: Guardrail Safety Classifier

Modelo especializado de clasificación de seguridad para interceptar prompts y salidas no seguras en microservicios antes de llegar al usuario. 

[ Consultar en Meta AI Docs ](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama-guard-3/)

Linux Foundation · 2024 Estándar OpenAPI

#### OpenAPI 3.1 Specification Standard

Especificación oficial para contratos de API RESTful con JSON Schema 2020-12, base de la documentación viva generada automáticamente por FastAPI. 

[ Consultar en OpenAPI Initiative ](https://spec.openapis.org/oas/latest.html)

OpenAccess AI Collective Framework de Entrenamiento

#### Axolotl: Streamlined Post-Training for Large Models

Herramienta de configuración YAML para entrenar adaptadores LoRA con empaquetado de secuencias FlashAttention y optimizadores 8-bit. 

[ Consultar Axolotl GitHub ](https://github.com/axolotl-ai-cloud/axolotl)

Lin et al. · MLSys Paper Científico

#### AWQ: Activation-aware Weight Quantization for LLM Compression

Técnica de cuantización en 4 bits que protege el 1% de los pesos más importantes reduciendo el uso de VRAM a la mitad sin pérdida de calidad. 

[ Consultar Paper AWQ ](https://arxiv.org/abs/2306.00978)

Hugging Face Core Librería de Alineación

#### TRL: Transformer Reinforcement Learning with DPO & SFT

Librería estándar para Direct Preference Optimization (DPO) y Supervised Fine-Tuning (SFT) sobre modelos de lenguaje abiertos. 

[ Consultar HF TRL Docs ](https://huggingface.co/docs/trl/)

Dao et al. · ICLR Aceleración de Memoria

#### FlashAttention-2: Faster Attention with Better Work Partitioning

Algoritmo de paralelización por bloques de SRAM que acelera el cálculo de atención hasta 2.5x respecto a implementaciones estándar. 

[ Consultar FlashAttention-2 ](https://github.com/Dao-AILab/flash-attention)

---

<div align="center">

[⬅️ Anterior](03-fine-tuning-lora-qlora-evaluacion.md) • [🏠 Inicio](../../README.md) • [📁 Módulo 1](README.md) • [Siguiente ➡️](challenge-1-benchmark-multi-modelo.md)

</div>
