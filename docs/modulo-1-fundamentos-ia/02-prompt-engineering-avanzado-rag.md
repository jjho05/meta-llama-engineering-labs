<div align="center">

[🏠 Inicio](../../README.md) • [📁 Módulo 1](README.md) • [⬅️ Anterior](01-arquitectura-transformer-llama3.md) • [Siguiente ➡️](03-fine-tuning-lora-qlora-evaluacion.md)

</div>

---

MÓDULO 1 TEMA 2 · PROMPT ENGINEERING Y RAG CON LLAMA

# Prompt Engineering y RAG con Llama

**Cuándo pedir mejor y cuándo darle documentos reales**. Domina las estrategias de instrucción Zero-shot, Few-shot y Chain-of-Thought, comprende las barreras de la memoria estática y construye arquitecturas de Generación Aumentada por Recuperación (RAG) con búsqueda semántica y embeddings vectoriales.

Resumen Ejecutivo Estrategias de Prompting Arquitectura RAG Ejercicios & Soluciones

Guía de Inicio · Visión del Tema 1.2

### Resumen Ejecutivo & Visión: Prompt Engineering y RAG con Llama

#### 1\. Resumen Ejecutivo Síntesis Rápida del Tema

El prompt engineering —mediante técnicas como zero-shot, few-shot y chain-of-thought— optimiza cómo un modelo de lenguaje aplica su conocimiento previo, pero nunca le inyecta datos nuevos. Cuando la respuesta depende de información específica, cambiante o posterior a su entrenamiento, RAG ofrece la solución al recuperar documentos relevantes mediante embeddings y búsqueda semántica, para luego generar una respuesta anclada en hechos actuales y no en memorias estáticas que puedan derivar en alucinaciones.

¿Qué vas a aprender en este tema?

• Las tres estrategias básicas de prompting — **zero-shot** , **few-shot** y **chain-of-thought** — y cuándo conviene cada una.

• Por qué un modelo nunca sabe algo que no vio en su entrenamiento, sin importar cuán elaborado sea el prompt.

• Qué es **RAG (Retrieval-Augmented Generation)** y cómo resuelve el problema de _“el modelo no lo sabe”_.

• Cómo funcionan los **embeddings** y la **búsqueda semántica** para recuperar conocimiento externo.

Tema 1.2

## Prompt Engineering y RAG con Llama

Deconstrucción técnica de las técnicas de instrucción en contexto, vectores semánticos y arquitecturas de recuperación de información fáctica para modelos de lenguaje abiertos.

¿No entendiste? Te lo explico fácil: Las coordenadas GPS de las ideas

Un embedding vectorial es como las **coordenadas de latitud y longitud en un mapa mental**. Las palabras _"rey"_ y _"reina"_ están a pocos metros de distancia en el mapa, mientras que _"computadora cuántica"_ está en otro continente. Cuando buscas información, el sistema simplemente mide la distancia en línea recta (distancia coseno) entre tu pregunta y los fragmentos de tus documentos. 

Consejo Pro: Chunking con Solapamiento (Overlap) del 15%

Al fragmentar documentos largos, aplica siempre un **solapamiento del 10% al 20%** entre fragmentos consecutivos (ej. 512 tokens con 64 tokens de overlap). Esto evita que una frase o definición clave quede partida a la mitad entre dos bloques distintos. 

Tema 1.2.1 · Técnicas de Instrucción en Contexto

### Tres Formas de Pedir Mejor: Zero-Shot, Few-Shot y Chain-of-Thought

#### 1\. Guía de Inicio El Chef Maestro y la Despensa

Piensa en el modelo como un **chef con años de entrenamiento**. El prompt engineering no le agrega ingredientes nuevos a su despensa; solo le indica cómo cocinar mejor con lo que ya tiene. Estas son las tres formas más probadas de darle esa instrucción: 

Estrategia #1 Directo

##### Zero-Shot

**Intuición:** Es como pedirle a un mecánico experto que cambie una llanta sin mostrarle cómo lo hiciste antes; confías en que su entrenamiento le indica el procedimiento correcto. 

**Técnica:** Solicitud directa de una tarea sin proporcionar ejemplos previos de entrada-salida. El modelo depende únicamente de los patrones aprendidos durante su entrenamiento para inferir lo que se le pide. 

Estrategia #2 Demostraciones

##### Few-Shot

**Intuición:** Es como mostrarle a alguien dos o tres fotos de cómo doblar una camisa antes de pedirle que doble toda la pila; calibra el resultado sin necesidad de explicar reglas abstractas. 

**Técnica:** Prompt que incluye una pequeña muestra de ejemplos (entrada-salida) antes de la consulta real, alineando al modelo con un formato o tono específico y reduciendo la ambigüedad de la respuesta. 

Estrategia #3 Razonamiento Paso a Paso

##### Chain-of-Thought (CoT)

**Intuición:** Es como exigirle a un estudiante que muestre el desarrollo de una ecuación en el cuaderno en lugar de solo escribir la respuesta final; si el razonamiento falla en el paso 3, detectas el error antes de aceptar el resultado. 

**Técnica:** Instrucción explícita para que el modelo desglose su razonamiento intermedio paso a paso antes de emitir la respuesta final, mejorando drásticamente el desempeño en problemas de lógica, matemáticas y razonamiento multi-paso. 

#### 2\. Concepto Formal Acondicionamiento Probabilístico en Contexto

En términos formales, el Prompt Engineering no modifica los pesos neuronales estáticos $\theta$ de Llama 3, sino que modifica la distribución condicional prefijando una secuencia de tokens de contexto $\mathcal{I} = (i_1, \dots, i_k)$: 

$$P(Y \mid X, \mathcal{I}) = \prod_{t=1}^{m} P(y_t \mid X, \mathcal{I}, y_1, \dots, y_{t-1})$$

 

Desglose de la Fórmula de Inferencia en Contexto 6 elementos

$P(Y \mid X, \mathcal{I})$

**Probabilidad Condicionada por el Prompt:** La probabilidad de que el modelo genere la respuesta $Y$ condicionada tanto por la consulta del usuario ($X$) como por la instrucción o ejemplos provistos ($\mathcal{I}$). 

$\mathcal{I}$ (Instrucción)

**Instrucción / Demostraciones en Contexto:** El conjunto de tokens del prompt de sistema, roles de Few-Shot o la orden explícita de CoT ("Piensa paso a paso"). 

$X$ (Query)

**Consulta de Entrada (Query):** El problema puntual, texto a clasificar o pregunta formulada por el usuario final. 

$y_t$ (Token)

**Token Generado en el Paso $t$:** Cada subpalabra que el modelo predice sucesivamente durante la emisión de la respuesta. 

$\prod_{t=1}^m$

**Productorio Secuencial:** Multiplicación de las probabilidades calculadas token por token a lo largo de toda la longitud $m$ de la respuesta generada. 

$y_1, \dots, y_{t-1}$

**Historial Autoregresivo Generado:** Los tokens intermedios que el modelo ya emitió (por ejemplo, los pasos 1 y 2 de una deducción CoT) que guían la predicción del paso 3. 

Límite Infranqueable del Prompting

Estas tres técnicas comparten un límite fundamental: **todas trabajan exclusivamente con lo que el modelo ya aprendió durante su entrenamiento**. Ninguna le da información nueva; solo mejoran cómo accede a su memoria interna y cómo formatea el resultado. 

Banco de Pruebas 1.2.1: Comparador Interactivo Zero-Shot vs Few-Shot vs Chain-of-Thought 

Llama 3 8B Instruct Simulator

Seleccionar Tarea: 1\. Problema Aritmético Multi-Paso 2\. Clasificación de Reseñas de Clientes 3\. Extracción Estructurada de Datos JSON 4\. Diagnóstico Lógico & Reglas de Negocio

Prompt Enviado al Modelo:

Respuesta Generada por Llama 3:

Autoevaluación 1.2.1

¿Cuál es la principal ventaja de utilizar Chain-of-Thought (CoT) frente a Zero-Shot en problemas que exigen deducción lógica o cálculo aritmético?

Advertencia Crítica: La Trampa de 'Lost in the Middle'

Los LLMs recuerdan mejor la información al principio y al final del prompt, pero tienden a ignorar fragmentos colocados en el centro (_Lost in the Middle_). Si recuperas 10 fragmentos en RAG, coloca los **2 más relevantes al inicio** y el **tercero al final** justo antes de la pregunta del usuario. 

Tema 1.2.2 · Límites de la Memoria Paramétrica

### El Límite del Prompt: Lo que el Modelo Nunca Vio & Alucinaciones

#### 1\. Guía de Inicio El Bibliotecario en el Sótano

Imagina a un **bibliotecario que pasó décadas leyendo libros hasta una fecha de corte y luego quedó encerrado en el sótano**. Puedes redactar la pregunta más elegante del mundo, pero si el libro se publicó ayer, él jamás lo habrá visto. Los LLMs como Llama funcionan igual: su conocimiento se congela en la fecha límite de sus datos de entrenamiento. 

Si le preguntas por una política que se actualizó la semana pasada, no importa qué tan elaborado sea tu prompt: el modelo no tiene esa información. Puede inventar una respuesta que suena convincente; este fenómeno se llama **alucinación** , y representa el riesgo central de confiar en memoria paramétrica para hechos actualizables o muy específicos. 

#### 2\. Concepto Formal Memoria Paramétrica ($\theta$) vs Memoria No Paramétrica

Los pesos neuronales $\theta$ almacenan _patrones lingüísticos, gramática, razonamiento abstracto y hechos generales_. Sin embargo, no son una base de datos relacional auditable: 

Memoria Paramétrica (Pesos del LLM)

Estática, opaca y congelada en la fecha de preentrenamiento. Modificarla requiere un proceso costoso de reentrenamiento o fine-tuning. Proclive a alucinar ante hechos específicos.

Memoria No Paramétrica (Base Vectorial RAG)

Dinámica, auditable y actualizable al segundo. Permite agregar, modificar o revocar documentos empresariales sin tocar un solo parámetro del modelo base.

Banco de Pruebas 1.2.2: Comparador en Vivo — Memoria Paramétrica (Alucinación) vs RAG (Hecho Real) 

Auditoría Fáctica & Línea de Tiempo

Fecha de Corte Llama 3: Dic 2023 | Cambio de Política: 1 de Enero 2024

Solo Modelo (Sin RAG) Memoria Congelada

Modelo + RAG (Con Base Vectorial) Base Vectorial ChromaDB

Autoevaluación 1.2.2

¿Por qué redactar un prompt extremadamente detallado no evita que un modelo alucine sobre un evento ocurrido ayer?

Tema 1.2.3 · Arquitectura de Generación Aumentada

### RAG y Embeddings: Buscar Antes de Responder

#### 1\. Guía de Inicio El Archivero Eficiente

En lugar de confiar en la memoria del bibliotecario, **RAG es como si, antes de responder, le pidieras a un archivero que corra a la biblioteca y traiga los tres libros más relevantes**. El experto entonces responde leyendo esos libros en tiempo real. La arquitectura separa el trabajo en dos fases: **recuperar** y **generar**. 

Componente #1

##### Retriever (Recuperador)

**Intuición:** Es el archivero que, ante una pregunta, revisa los estantes y te entrega solo los documentos que tratan el tema, descartando todo lo irrelevante. 

**Técnica:** Componente de búsqueda que explota una base de conocimiento externa para extraer los fragmentos de texto más relevantes a la consulta del usuario, antes de que el modelo genere la respuesta. 

Componente #2

##### Embeddings (Incrustaciones)

**Intuición:** Es como traducir cada documento a un idioma numérico universal donde las ideas parecidas quedan cerca unas de otras en un mapa geométrico. 

**Técnica:** Representaciones vectoriales continuas de alta dimensión (ej. 1,536 dimensiones) que capturan las relaciones semánticas del lenguaje natural. 

Componente #3

##### Similitud Coseno

**Intuición:** Es medir con un transportador el ángulo entre dos flechas: si apuntan en la misma dirección, hablan exactamente de lo mismo. 

**Técnica:** Métrica matemática que cuantifica la proximidad angular entre dos vectores en el espacio latente, normalizada en el rango $[-1, 1]$. 

#### 2\. Concepto Formal Formulación Matemática Bimodal de RAG

RAG modela la probabilidad de generar una secuencia de respuesta $y$ a partir de la consulta $x$ marginalizando sobre los fragmentos documentales recuperados $z \in \text{Top-}k$ (Paper Fundacional de Lewis et al., Meta AI FAIR): 

$$P(y \mid x) = \sum_{z \in \text{Top-}k} P(z \mid x) \cdot P(y \mid x, z)$$

 

Desglose de la Ecuación RAG 5 variables

$P(y \mid x)$

**Probabilidad de la Respuesta Final:** La verosimilitud de que el texto emitido por el sistema resuelva la consulta del usuario de forma certera y sin alucinaciones. 

$P(z \mid x)$

**Probabilidad del Retriever (Recuperación):** Puntuación de similitud semántica calculada por el modelo de embeddings entre la pregunta $x$ y el fragmento documental $z$. 

$P(y \mid x, z)$

**Probabilidad del Generator (Llama 3):** La probabilidad con la que el modelo autorregresivo genera la respuesta $y$ anclada obligatoriamente en el fragmento recuperado $z$. 

$z \in \text{Top-}k$

**Conjunto de Fragmentos Seleccionados:** Los $k$ pasajes con mayor puntuación vectorial devueltos por el índice HNSW de la base vectorial. 

$\sum_{z}$

**Combinación de Evidencias:** Integración y síntesis de todos los fragmentos relevantes para redactar una respuesta cohesiva y unificada. 

Banco de Pruebas 1.2.3: Visualizador del Flujo RAG Paso a Paso 

Arquitectura en 4 Fases

Autoevaluación 1.2.3

En la arquitectura RAG, ¿cuál es la responsabilidad principal del componente 'Retriever'?

Tema 1.2.4 · Espacios Semánticos & Álgebra Vectorial

### Embeddings y Similitud Coseno: Medir Distancias de Significado

#### 1\. Concepto Formal Vectores Densos y Similitud Coseno

Un modelo de embeddings transforma un fragmento de texto en un vector denso $\mathbf{u} \in \mathbb{R}^{d}$. La afinidad conceptual entre dos textos no se mide por coincidencia de palabras exactas, sino por el **coseno del ángulo $\theta$** entre sus vectores multidimensionales: 

$$\text{Similitud Coseno}(\mathbf{u}, \mathbf{v}) = \cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \frac{\sum_{i=1}^d u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \sqrt{\sum_{i=1}^d v_i^2}}$$

 

Desglose de la Fórmula de Similitud Coseno 6 elementos

$\mathbf{u} \cdot \mathbf{v}$

**Producto Punto (Escalar):** Suma de las multiplicaciones elemento a elemento ($\sum u_i v_i$). Mide cuánto apuntan ambos vectores en la misma dirección. 

$\|\mathbf{u}\|_2$

**Norma Euclidiana ($L_2$):** La longitud geométrica del vector ($\sqrt{\sum u_i^2}$). Normaliza la métrica para que textos largos no dominen artificialmente sobre textos cortos. 

$\cos(\theta) = 1.0$

**Significado Idéntico ($\theta = 0^\circ$):** Vectores paralelos; los textos expresan exactamente la misma idea semántica. 

$\cos(\theta) = 0.0$

**Ortogonales / Sin Relación ($\theta = 90^\circ$):** Los textos pertenecen a dominios completamente desconectados (ej. repostería y física de partículas). 

$\cos(\theta) = -1.0$

**Opuestos Diametrales ($\theta = 180^\circ$):** Vectores en direcciones exactamente inversas. 

Banco de Pruebas 1.2.4: Calculadora Vectorial 2D & Similitud Coseno (Gráficos Retina HiDPI) 

Álgebra Lineal en Tiempo Real

Seleccionar Par de Frases: 1\. Sinónimos ("Coche rojo en avenida" vs "Auto carmesí en calzada") 2\. Dominio Relacionado ("Meta Llama 3 LLM" vs "Python & PyTorch") 3\. No Relacionados ("Pizza napolitana" vs "Cálculo cuántico y cúbits")

Similitud Coseno: 0.9842

Ángulo θ entre Vectores: 10.2°

Producto Punto (u · v): 0.9691

Autoevaluación 1.2.4

Si dos frases no comparten ninguna palabra idéntica pero expresan la misma idea (ej. 'auto veloz' y 'carro rápido'), ¿qué valor de Similitud Coseno producirán sus embeddings?

Tema 1.2.5 · Indexación & Bases de Datos Vectoriales

### Búsqueda Semántica, Chunking y Bases de Datos Vectoriales

#### 1\. Concepto Formal Fragmentación con Solapamiento & Algoritmo HNSW

Para que un libro o manual empresarial pueda indexarse con precisión, se divide en **fragmentos (chunks)** con un margen de **solapamiento (overlap)** para evitar que oraciones importantes queden partidas a la mitad: 

$$\text{Paso (Stride)} = S - O, \quad N_{\text{chunks}} = \left\lceil \frac{|D| - O}{S - O} \right\rceil$$

 

Desglose de Fragmentación (Chunking) 5 variables

$|D|$ (Doc Size)

**Longitud Total del Documento:** Cantidad neta de tokens o palabras que componen el archivo original antes de fragmentar. 

$S$ (Chunk Size)

**Tamaño del Fragmento:** Longitud máxima de cada bloque (ej. 512 tokens), diseñada para preservar la coherencia contextual sin saturar la ventana de atención. 

$O$ (Overlap)

**Margen de Solapamiento ($O < S$):** Cantidad de tokens compartidos entre bloques continuos (ej. 64 tokens) para evitar cortes abruptos de oraciones clave. 

$\text{Paso} = S - O$

**Desplazamiento de Ventana:** Intervalo de avance tras el cual comienza a indexarse el siguiente fragmento del documento. 

$N_{\text{chunks}}$

**Total de Pasajes Indexados:** Número exacto de vectores que se almacenarán en la base de datos vectorial (ChromaDB / FAISS). 

Banco de Pruebas 1.2.5: Explorador de Chunking & Búsqueda Vectorial Top-k 

ChromaDB / FAISS Engine

Tamaño de Chunk (Words): 25 palabras

Solapamiento / Overlap: 5 palabras

Fragmentos Generados en Memoria:

Probar Búsqueda Semántica Vectorial:

Autoevaluación 1.2.5

¿Por qué es una buena práctica incluir un porcentaje de solapamiento (overlap) al fragmentar documentos para RAG?

Tema 1.2.6 · Caso Práctico Empresarial

### Caso Práctico: Políticas que Cambian Cada Trimestre

#### 1\. El Escenario Real Asistente Corporativo de Atención al Cliente

Un equipo de atención a clientes necesita que su asistente responda con las **políticas de devolución vigentes, que cambian cada trimestre**. En lugar de afinar el prompt esperando que el modelo adivine, construyen un pipeline RAG: 

1

**Indexación:**

Fragmentan el documento de políticas más reciente y generan los embeddings de cada fragmento, almacenándolos en una base vectorial.

2

**Consulta:**

Cuando llega una pregunta sobre devoluciones, el sistema busca los párrafos relevantes mediante similitud semántica.

3

**Generación:**

Inyecta esos párrafos como contexto junto a la pregunta en el prompt; Llama genera la respuesta basándose únicamente en ese texto real.

4

**Actualización:**

Cuando la política cambia, el equipo solo reemplaza el documento indexado. No reentrena el modelo, ni reescribe el prompt, ni toca la lógica del asistente. Así evitan que el sistema responda con una política vieja que ya no aplica.

#### 2\. Código de Producción Pipeline RAG con Llama 3 & ChromaDB

rag_pipeline_llama3.py (FastAPI + ChromaDB)
    
    
    import chromadb
    from sentence_transformers import SentenceTransformer
    from groq import Groq
    
    # 1. Inicializar base vectorial en local y modelo de embeddings
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="politicas_empresa_q1_2024")
    embed_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    
    # 2. Indexar documento de políticas actualizado
    politicas_texto = [
     "Las devoluciones en 2024 deben tramitarse dentro de 15 días naturales desde la entrega.",
     "El reembolso se emite al método de pago original en un plazo de 3 a 5 días hábiles.",
     "Artículos en liquidación final no son elegibles para cambio ni devolución."
    ]
    collection.add(
     documents=politicas_texto,
     ids=["p1", "p2", "p3"]
    )
    
    # 3. Función de Consulta RAG
    def consultar_asistente(pregunta: str) -> str:
     # Recuperar fragmentos relevantes (Top-2)
     resultados = collection.query(query_texts=[pregunta], n_results=2)
     contexto = "\n".join(resultados["documents"][0])
     
     # Generación anclada con Meta Llama 3
     client = Groq()
     prompt = f"""Contexto Oficial:\n{contexto}\n\nPregunta: {pregunta}\nResponde estrictamente con el contexto:"""
     
     response = client.chat.completions.create(
     model="llama3-8b-8192",
     messages=[{"role": "user", "content": prompt}],
     temperature=0.1
     )
     return response.choices[0].message.content

Autoevaluación 1.2.6

Cuando una empresa actualiza sus políticas de devolución cada trimestre, ¿cuál es el procedimiento correcto en una arquitectura RAG?

Tema 1.2.7 · RAG Avanzado de Grado Industrial

### RAG Avanzado: Hybrid Search, Reranking & Lost in the Middle

#### 1\. Técnicas Avanzadas Optimizando la Precisión en Ventanas de Contexto Largas

En entornos de producción masiva con Llama 3 (ventana de contexto de 128k tokens), tres optimizaciones marcan la diferencia entre un prototipo básico y un motor de búsqueda empresarial: 

$$\text{RRF\_Score}(d) = \sum_{m \in \\{\text{BM25}, \text{Dense}\\}} \frac{1}{k + r_m(d)}$$

 

Desglose de Reciprocal Rank Fusion (RRF) 5 variables

$\text{RRF\_Score}(d)$

**Puntuación Fusionada del Documento:** Métrica unificada no paramétrica que combina los ordenamientos de múltiples recuperadores sin sesgos de escala. 

$m \in \\{\text{BM25}, \text{Dense}\\}$

**Métodos de Búsqueda Híbrida:** Combina búsqueda léxica exacta (BM25 para códigos/nombres) con búsqueda semántica vectorial (Embeddings para significado). 

$r_m(d)$ (Rango)

**Posición Ordinal:** El lugar en el ranking ($1, 2, 3, \dots$) asignado al documento $d$ por el método de recuperación $m$. 

$k = 60$ (Constante)

**Factor de Regularización:** Constante estándar (propuesta por Cormack et al.) que suaviza la caída de puntuación y evita que el Top-1 domine desproporcionadamente. 

$\sum_{m}$ (Fusión)

**Suma de Inversos de Rangos:** Integra las evidencias de todos los motores de búsqueda para construir el ranking final antes del modelo Reranker. 

1\. Búsqueda Híbrida (Hybrid Search)

Combina búsqueda por coincidencia exacta BM25 (ideal para códigos de producto, SKUs y números de serie) con búsqueda semántica densa de embeddings.

2\. Cross-Encoder Reranking

Reordena los Top-25 fragmentos recuperados con un modelo clasificador de alta precisión (como BGE-Reranker) antes de entregar los Top-3 definitivos a Llama 3.

3\. Mitigación de 'Lost in the Middle'

Los Transformers prestan mayor atención al inicio y final del contexto. Posicionar los fragmentos más críticos en los extremos del prompt maximiza la precisión.

Autoevaluación 1.2.7

¿Por qué la Búsqueda Híbrida (Hybrid Search = BM25 + Embeddings) supera a la búsqueda puramente vectorial cuando se consultan catálogos de productos técnicos?

Tema 1.2.8 · Agentes de Razonamiento & Prompting Autónomo

### Patrón ReAct & Auto-Consistencia: De Prompts a Sistemas Autónomos

#### 1\. Concepto Formal El Ciclo ReAct: Pensamiento $\to$ Acción $\to$ Observación

El patrón **ReAct (Reason + Act)** combina la generación de razonamiento interno de Chain-of-Thought con la capacidad de ejecutar llamadas a herramientas externas (APIs, calculadoras o bases de datos vectoriales). Para problemas de alta incertidumbre, la técnica de **Auto-Consistencia (Self-Consistency)** muestrea múltiples rutas de deducción y selecciona la respuesta por consenso mayoritario: 

$$\hat{y} = \arg\max_{y} \sum_{i=1}^{N} \mathbb{I}\left( \text{ExtractAnswer}(c_i) = y \right) \quad \text{donde } c_i \sim P_{\text{CoT}}(C \mid X, \mathcal{I})$$

 

Desglose de Auto-Consistencia (Self-Consistency) 6 elementos

$\hat{y}$ (Respuesta Final)

**Resultado por Consenso:** La respuesta final seleccionada como la más verosímil y robusta tras evaluar múltiples cadenas de razonamiento. 

$N \in [5, 20]$

**Número de Muestreos en Paralelo:** Cantidad de rutas CoT independientes generadas por el modelo configurando temperatura $T > 0$ (ej. $T = 0.7$). 

$c_i \sim P_{\text{CoT}}$

**Cadena de Pensamiento $i$-ésima:** Secuencia de pasos deductivos generada estocásticamente para resolver el problema $X$. 

$\text{ExtractAnswer}(c_i)$

**Extractor de Conclusión:** Función de parsing o formateo que aísla el valor numérico, opción o etiqueta final emitida al término de la cadena $c_i$. 

$\mathbb{I}(\dots)$ (Indicatriz)

**Conteo de Votos:** Devuelve $1$ si la cadena $c_i$ llegó a la respuesta $y$, y $0$ si concluyó un resultado diferente. 

$\arg\max_y$ (Moda)

**Votación Mayoritaria:** Selecciona el valor de $y$ que obtuvo la mayor frecuencia acumulada entre todas las $N$ rutas exploradas. 

Pensamiento (Thought)

El modelo deduce qué información necesita: _"No conozco el tipo de cambio de hoy. Debo invocar la API financiera."_

Acción (Action)

El LLM emite un llamado estructurado: `get_exchange_rate(currency="USD_MXN")`.

Observación (Observation)

El runtime ejecuta la función y devuelve el valor real: `{"rate": 18.25}`, con el cual Llama 3 finaliza la respuesta.

Autoevaluación 1.2.8

¿Cuál es el beneficio de la técnica de Self-Consistency aplicada sobre Chain-of-Thought?

Terminología Oficial del Curso

## Glosario Técnico del Tema 1.2

Definiciones formales y rigurosas de los conceptos clave de Prompt Engineering y arquitecturas RAG. 

Oficial #01 Temario

Alucinación

Respuesta generada por un modelo que suena coherente y convincente, pero contiene hechos inexactos o inventados. Es el principal riesgo cuando se consulta sobre información posterior al entrenamiento o muy específica.

Riesgo Central: Confiar en memoria paramétrica para hechos cambiantes o específicos.

Oficial #02 Temario

Chain-of-thought (CoT)

Técnica de prompting que obliga al modelo a exhibir su razonamiento intermedio antes de entregar la respuesta final. Resulta esencial para problemas de lógica o cálculo donde un salto directo aumenta la probabilidad de error.

Efecto: Reduce drásticamente el error al condicionar cada paso en deducciones previas.

Oficial #03 Temario

Embedding

Vector numérico de alta dimensionalidad que representa el significado semántico de un texto. Permite que la máquina mida cercanía conceptual entre frases, incluso si no comparten vocabulario idéntico.

Espacio Geométrico: Vectores densos en $\mathbb{R}^{d}$ (ej. 1536 dimensiones).

Oficial #04 Temario

Few-shot

Técnica de prompting en la que se proveen al modelo unos pocos ejemplos de entrada-salida antes de la tarea definitiva. Calibra el formato y reduce la ambigüedad sin requerir ajuste de parámetros internos.

Calibración: Estandariza estructuras JSON y tonos sin modificar pesos neuronales.

Oficial #05 Temario

RAG (Retrieval-Augmented Generation)

Arquitectura que separa la generación de texto en dos fases: primero recupera documentos relevantes de una fuente externa y luego instruye al modelo para responder basándose en ese contexto recuperado.

Ventaja: Cero alucinaciones en datos dinámicos sin costo de reentrenamiento.

Oficial #06 Temario

Zero-shot

Técnica de prompting en la que se solicita una tarea directamente, sin ejemplos previos. El modelo debe resolverla confiando exclusivamente en el conocimiento adquirido durante su entrenamiento.

Uso: Tareas simples y directas donde no se requiere formato rígido.

Recuperación #07

Retriever (Recuperador)

Componente de software que consulta un índice vectorial y extrae los $k$ pasajes con mayor similitud a la pregunta del usuario antes de la inferencia.

Velocidad: Consultas en índices HNSW con latencias inferiores a 15 ms.

Vectores #08

Similitud Coseno (Cosine Similarity)

Métrica de álgebra lineal que evalúa el coseno del ángulo entre dos vectores normalizados $\frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$, variando entre -1.0 y 1.0.

Métrica Estándar: Invariante a la longitud del texto.

Vectores #09

Base de Datos Vectorial (Vector DB)

Sistema de almacenamiento optimizado para indexar y buscar representaciones densas multidimensionales mediante algoritmos de vecinos más cercanos (ANN).

Ecosistema: ChromaDB, FAISS, Milvus, Qdrant y pgvector.

Preprocesamiento #10

Chunking (Fragmentación)

Proceso de división de documentos extensos en bloques de texto más pequeños con un porcentaje de solapamiento para preservar continuidad semántica.

Parámetros Clave: Tamaño de chunk (tokens) y solapamiento (overlap).

Optimización #11

Cross-Encoder Reranker

Modelo neuronal que evalúa conjuntamente la pregunta y cada fragmento candidato para recalcular una puntuación de relevancia de máxima precisión.

Beneficio: Filtra falsos positivos devueltos por la búsqueda bi-encoder inicial.

Búsqueda #12

Búsqueda Híbrida (Hybrid Search)

Fusión ponderada entre búsqueda léxica basada en palabras clave (BM25) y búsqueda semántica vectorial densa.

Fórmula: $\text{Score} = \alpha \cdot \text{Dense} + (1-\alpha) \cdot \text{BM25}$.

RAG #13

Reciprocal Rank Fusion (RRF)

Algoritmo no paramétrico para combinar listas clasificadas de diferentes motores de búsqueda sumando las inversas de sus posiciones: $\text{RRF}(d) = \sum \frac{1}{60 + r(d)}$.

Aplicación: Fusión óptima de BM25 y búsqueda vectorial sin requerir calibración de pesos.

Chunking #14

Solapamiento de Chunks (Overlap)

Cantidad de tokens compartidos entre fragmentos de texto adyacentes durante la partición documental para evitar la pérdida de información semántica en los bordes de corte.

Regla de oro: Configurar entre 10% y 20% del tamaño total del chunk (ej. 50-100 tokens).

Prompting #15

Auto-Consistencia (Self-Consistency)

Estrategia de inferencia que genera múltiples rutas de razonamiento Chain-of-Thought con temperatura $T > 0$ y selecciona la respuesta con mayor consenso mediante votación mayoritaria.

Impacto: Incrementa la precisión en tareas de lógica y matemáticas en más de un 15%.

Práctica & Aplicación de Ingeniería

## Ejercicios Prácticos del Tema 1.2

Consolida tus competencias en Prompt Engineering y RAG resolviendo los 4 ejercicios del temario oficial con soluciones paso a paso.

Ejercicio 1

#### Diseño de Prompt Few-Shot para Clasificación de Reseñas

Enunciado Oficial 

Escribe un prompt few-shot para que un modelo clasifique reseñas de productos como Positiva, Negativa o Neutra. Incluye tres ejemplos con su formato de entrada-salida, y termina con una reseña nueva para que clasifique. 

Ver Solución de Ingeniería Paso a Paso & Template Oficial

1

##### Estructura del Prompt Few-Shot con Delimitadores

prompt_few_shot_clasificacion.txt
    
    
    # Instrucción del Sistema
    Eres un clasificador de sentimiento estricto para comercio electrónico.
    Tu tarea es clasificar la reseña del cliente en exactamente una de estas tres categorías: [Positiva, Negativa, Neutra].
    Responde únicamente con el formato JSON: {"sentimiento": "<categoría>", "confianza": <0.0-1.0>}
    
    # Ejemplos Few-Shot (Calibración de Tono y Formato)
    Reseña: "El paquete llegó un día antes de lo programado y el producto superó mis expectativas en calidad."
    Salida: {"sentimiento": "Positiva", "confianza": 0.98}
    
    Reseña: "El color no coincide con las fotografías del anuncio y el material se siente quebradizo."
    Salida: {"sentimiento": "Negativa", "confianza": 0.95}
    
    Reseña: "El pedido llegó en tiempo normal. El empaque es estándar y cumple con la descripción básica."
    Salida: {"sentimiento": "Neutra", "confianza": 0.88}
    
    # Nueva Entrada para Inferencia
    Reseña: "Funciona correctamente para lo básico, aunque la batería dura menos de lo que afirma la caja."
    Salida:

2

##### Justificación de Ingeniería

Al incluir un ejemplo para cada una de las 3 clases, el espacio latente del transformer queda restringido: el modelo no emitirá explicaciones conversacionales largas, sino la estructura JSON exacta solicitada, permitiendo su parseo directo por microservicios en producción. 

Ejercicio 2

#### Desglose de Razonamiento: Chain-of-Thought en Problemas Matemáticos

Enunciado Oficial 

Explica por qué una técnica de chain-of-thought podría mejorar la respuesta de un modelo ante la operación matemática: “Un tren lleva 120 pasajeros. Bajan 15 en la primera parada y suben el doble de los que bajaron en la segunda. ¿Cuántos pasajeros hay al final?”. Describe qué pasaría si solo se usara zero-shot. 

Ver Solución de Ingeniería Paso a Paso & Comparativa Zero vs CoT

A

##### Fallo en Modo Zero-Shot Directo

En Zero-Shot, el modelo intenta predecir el número final en un solo paso hacia adelante ($t+1$). Dado que los LLMs son modelos autorregresivos sin memoria de cálculo fuera de los tokens emitidos, forzar una respuesta inmediata causa que el transformer confunda los operadores o duplique la resta inicial, emitiendo cifras incorrectas como 135 o 105. 

B

##### Mecanismo de Desglose con Chain-of-Thought (CoT)

Al forzar la directiva _"Razona paso a paso antes de emitir la respuesta"_ , el modelo genera tokens intermedios en el contexto que sirven como memoria de trabajo para los siguientes pasos de atención:  
• **Paso 1:** Pasajeros iniciales = $120$.  
• **Paso 2:** Bajan $15$ en la parada 1 → $120 - 15 = 105$ pasajeros restantes.  
• **Paso 3:** Suben el doble de los que bajaron en la parada 2 → $2 \times 15 = 30$ pasajeros nuevos.  
• **Paso 4:** Total acumulado → $105 + 30 = 135$ pasajeros al final.  
**Resultado:** La autoatención sobre los cálculos intermedios garantiza 100% de precisión lógica. 

Ejercicio 3

#### Arquitectura RAG para Información Médica Dinámica

Enunciado Oficial 

Imagina que trabajas en una clínica y necesitas que un asistente con Llama responda sobre los efectos secundarios de medicamentos que la autoridad sanitaria actualiza mensualmente. Diseña un esquema de RAG en tres pasos (indexación, recuperación y generación) y justifica por qué los embeddings son más útiles que una búsqueda por palabra clave exacta. 

Ver Solución de Ingeniería Paso a Paso & Pipeline RAG

1

##### Fase 1: Indexación y Chunking

Cada mes, las circulares de la autoridad sanitaria se dividen en fragmentos de $512\text{ tokens}$ con solapamiento de $64\text{ tokens}$. Un modelo como `bge-base-en-v1.5` genera vectores densos de $768\text{ dimensiones}$ que se indexan en ChromaDB con metadatos de medicamento y fecha. 

2

##### Fase 2: Recuperación Vectorial por Similitud Coseno

Cuando el médico o paciente pregunta: _"¿Qué pasa si me duele la cabeza tras tomar fármaco X?"_ , la consulta se vectoriza y se extraen los $k=3$ chunks con mayor similitud semántica. 

3

##### Fase 3: Generación Anclada (Grounded Generation)

Se inyecta el contexto recuperado en el prompt de Llama 3 con instrucción estricta de no alucinar. Llama 3 sintetiza la respuesta citando la sección oficial de farmacovigilancia. 

##### ¿Por qué Embeddings superan a la Búsqueda por Palabra Clave Exacta?

La búsqueda por palabras clave falla si el usuario escribe _"vértigo y mareos"_ pero el documento médico oficial utiliza el término técnico _"hipotensión ortostática o trastorno vestibular"_. Los embeddings capturan la proximidad semántica en el espacio vectorial, uniendo el lenguaje coloquial del paciente con la nomenclatura médica oficial. 

Ejercicio 4

#### Análisis de Decisión Arquitectónica: RAG vs Fine-Tuning

Enunciado de Aplicación 

Un banco necesita actualizar las tasas de interés y comisiones de sus productos crediticios cada lunes. Analiza por qué reentrenar o hacer fine-tuning al modelo semanalmente es inviable y por qué RAG es la única solución arquitectónicamente viable. 

Ver Solución de Ingeniería Paso a Paso & Matriz de Decisión

1

##### Inviabilidad de Fine-Tuning Semanal

1\. **Costo y Latencia de Reentrenamiento:** Compilar adaptadores LoRA semanalmente requiere pipelines de cómputo GPU continuos.  
2\. **Riesgo de Olvido Catastrófico:** El fine-tuning no garantiza que el modelo recuerde con 100% de precisión números exactos (tasas de interés como 14.85%), ya que los LLMs parametrizan probabilidades y sufren de alucinación con valores numéricos aislados.  
3\. **Imposibilidad de Borrado Inmediato:** Si una tasa se revoca por orden regulatoria, un modelo con fine-tuning no permite eliminar un dato puntual sin volver a reentrenar. 

2

##### Ventajas Deterministas de RAG

Con RAG, actualizar las tasas toma menos de 2 segundos: se reemplaza el documento en la base vectorial ChromaDB. Llama 3 lee el valor exacto del contexto inyectado y cita la circular vigente con trazabilidad absoluta y sin riesgo de alucinación paramétrica. 

Evidencia Científica & Recursos Oficiales

## Fuentes de Información Reales & Referencias Académicas

Todo el contenido técnico, algoritmos vectoriales y arquitecturas presentados en este tema están fundamentados en investigaciones científicas publicadas y repositorios de código abierto. 

Meta AI (FAIR) · 2020 Paper Fundacional RAG

#### Retrieval-Augmented Generation for NLP Tasks

El artículo pionero que propuso la formulación matemática bimodal de RAG, combinando modelos generativos autorregresivos con un índice no paramétrico para responder con cero alucinaciones. 

[ Consultar en arXiv: 2005.11401 ](https://arxiv.org/abs/2005.11401)

Google Brain · 2022 Chain-of-Thought (CoT)

#### Chain-of-Thought Prompting Elicits Reasoning

Investigación seminal de Jason Wei et al. que demostró que instruir al modelo a generar deducciones paso a paso desencadena razonamiento aritmético y simbólico emergente. 

[ Consultar en arXiv: 2201.11903 ](https://arxiv.org/abs/2201.11903)

OpenAI · 2020 In-Context Learning

#### Language Models are Few-Shot Learners

Estudio formal que fundamentó el aprendizaje en contexto (Few-Shot), probando cómo $K$ ejemplos demostrativos calibran el formato y las etiquetas sin alterar gradientes ni parámetros. 

[ Consultar en arXiv: 2005.14165 ](https://arxiv.org/abs/2005.14165)

Princeton & Google · 2022 Patrón ReAct

#### ReAct: Synergizing Reasoning and Acting

Introduce el bucle interactivo de razonamiento y acción (Thought → Action → Observation), permitiendo a los LLMs invocar APIs, herramientas y bases de datos vectoriales en tiempo real. 

[ Consultar en arXiv: 2210.03629 ](https://arxiv.org/abs/2210.03629)

Google Research · 2022 Auto-Consistencia

#### Self-Consistency in Chain-of-Thought

Propone muestrear $N$ rutas estocásticas de razonamiento CoT y seleccionar la conclusión final por votación mayoritaria ($\arg\max$), incrementando notablemente la precisión matemática. 

[ Consultar en arXiv: 2203.11171 ](https://arxiv.org/abs/2203.11171)

Malkov & Yashunin · 2018 Algoritmo HNSW

#### Efficient ANN Search Using HNSW Graphs

El algoritmo matemático detrás de ChromaDB, FAISS y Pinecone para buscar vecinos más cercanos en grafos navegables jerárquicos con complejidad logarítmica $\mathcal{O}(\log N)$. 

[ Consultar en arXiv: 1603.09320 ](https://arxiv.org/abs/1603.09320)

Meta AI (FAIR) · 2020 Dense Passage Retrieval

#### Dense Passage Retrieval (DPR) for Open-QA

Demuestra cómo los bi-encoders siameses entrenados con pérdida contrastiva superan a los métodos léxicos BM25 comparando vectores de embeddings continuos en espacio latente. 

[ Consultar en arXiv: 2004.04906 ](https://arxiv.org/abs/2004.04906)

Univ. of Waterloo · 2009 Fusión Híbrida RRF

#### Reciprocal Rank Fusion in Information Retrieval

Formulación matemática estándar de la industria (Cormack et al.) que fusiona listas de ranking de búsqueda léxica (BM25) y búsqueda vectorial mediante la suma de inversos de rangos normalizados. 

[ Consultar en ACM SIGIR: 1572114 ](https://dl.acm.org/doi/10.1145/1571941.1572114)

Stanford & Berkeley · 2023 Atención en Contexto

#### Lost in the Middle: LLMs in Long Contexts

Evidencia empírica fundamental que demuestra cómo la posición de los pasajes dentro del prompt influye en la capacidad de recuperación, motivando el ordenamiento en extremos y el reranking. 

[ Consultar en arXiv: 2307.03172 ](https://arxiv.org/abs/2307.03172)

BAAI · 2023 Embeddings & Reranker

#### BGE: General Multilingual Embeddings & Reranker

Arquitectura de modelos de embeddings y cross-encoder rerankers de alta precisión multilingüe (inglés/español), evaluados en el benchmark estandarizado MTEB. 

[ Consultar en arXiv: 2309.07597 ](https://arxiv.org/abs/2309.07597)

Carnegie Mellon (CMU) · 2022 Búsqueda HyDE

#### Precise Zero-Shot Dense Retrieval (HyDE)

Propone Hypothetical Document Embeddings (HyDE), técnica donde el LLM genera una respuesta preliminar hipotética para consultar el espacio vectorial con mayor alineación semántica. 

[ Consultar en arXiv: 2212.10496 ](https://arxiv.org/abs/2212.10496)

Univ. of Washington & Allen AI · 2023 RAG con Auto-Reflexión

#### Self-RAG: Learning to Retrieve, Generate & Critique

Framework avanzado que entrena tokens especiales de autorreflexión para que el LLM decida autónomamente cuándo recuperar documentos y verifique la veracidad de sus citas. 

[ Consultar en arXiv: 2310.11511 ](https://arxiv.org/abs/2310.11511)

Wang et al. · ICLR Paper Científico

#### Self-Consistency Improves Chain of Thought Reasoning

Técnica de muestreo con votación mayoritaria para elevar la precisión en tareas matemáticas y lógicas complejas con Llama 3. 

[ Consultar Paper Self-Consistency ](https://arxiv.org/abs/2203.11171)

DAIR.AI · 2024 Guía de Referencia

#### Prompt Engineering Guide: Techniques & Benchmarks

Compendio exhaustivo de patrones de diseño de prompts, metaprompting y evaluación de solidez contra inyecciones de contexto. 

[ Consultar DAIR.AI Guide ](https://www.promptingguide.ai/)

Anthropic Research Mejores Prácticas

#### Using XML Tags for Structured Prompt Framing

Guía metodológica sobre el uso de etiquetas delimitadoras XML para prevenir la confusión de instrucciones y estructurar entradas complejas. 

[ Consultar Guía XML Tags ](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags)

Stanford NLP · 2024 Framework de Optimización

#### DSPy: Programming—not Prompting—Foundation Models

Compilador de prompts declarativo que optimiza automáticamente las instrucciones y los ejemplos Few-Shot mediante algoritmos de teleprompter. 

[ Consultar DSPy Stanford ](https://github.com/stanfordnlp/dspy)

---

<div align="center">

[⬅️ Anterior](01-arquitectura-transformer-llama3.md) • [🏠 Inicio](../../README.md) • [📁 Módulo 1](README.md) • [Siguiente ➡️](03-fine-tuning-lora-qlora-evaluacion.md)

</div>
