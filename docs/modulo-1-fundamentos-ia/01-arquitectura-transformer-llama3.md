<div align="center">

[🏠 Inicio](../../README.md) • [📁 Módulo 1](README.md) • [Siguiente ➡️](02-prompt-engineering-avanzado-rag.md)

</div>

---

MÓDULO 1 TEMA 1 · FUNDAMENTOS DE LLMS Y ARQUITECTURA DE LLAMA

# Fundamentos de LLMs y Arquitectura de Llama

**Qué hay dentro de un modelo de pesos abiertos**. Domina cada concepto técnico, interactúa con demostraciones prácticas y consolida las competencias de ingeniería de este tema oficial.

Guía de Inicio · Visión del Tema 1.1

### Resumen Ejecutivo & Visión: Fundamentos de LLMs y Arquitectura de Llama

#### 1\. Resumen Ejecutivo Síntesis Oficial del Tema

Un LLM como Llama es esencialmente un motor de predicción secuencial que opera sobre tokens convertidos en vectores numéricos, utilizando la arquitectura transformer para ponderar relaciones a larga distancia mediante mecanismos de atención. Al ser un modelo de pesos abiertos, Llama permite su descarga, ejecución local, auditoría y adaptación sin intermediarios ni dependencia de APIs cerradas. Finalmente, seleccionar el tamaño del modelo es una decisión de ingeniería que equilibra latencia, costo computacional y cobertura del caso de uso real, donde el prototipo ligero suele resolver la mayoría de las necesidades antes de justificar un escalado.

¿Qué vas a aprender?

  * **Predicción de tokens:** Por qué un LLM es, en esencia, un motor de predicción de tokens y por qué la frase _“genera texto”_ describe su operación con más precisión que _“entiende texto”_.
  * **Atención Transformer:** Cómo la arquitectura transformer explora toda una secuencia al mismo tiempo para decidir qué palabras influencian la siguiente.
  * **Soberanía de Pesos Abiertos:** Qué significa exactamente que Llama sea un modelo de pesos abiertos y qué libertades técnicas eso otorga a un equipo de ingeniería.
  * **Optimización de Recursos:** Por qué elegir el tamaño de un modelo es un problema de optimización de recursos, no una carrera por la mayor cantidad de parámetros disponibles.

Tema 1.1

## Fundamentos de LLMs y Arquitectura de Llama

Qué hay dentro de un modelo de pesos abiertos: de palabras a números (tokens y embeddings), la arquitectura Transformer y soberanía de cómputo.

Consejo Pro: Selección Óptima de Temperatura según el Caso de Uso

Para tareas de **extracción de datos, JSON o generación de código SQL** , usa siempre $T \le 0.1$ para forzar respuestas deterministas y exactas. Para **chatbots de atención a clientes** , usa $T=0.6$ a $0.7$ para mantener fluidez empática sin alucinaciones. Para **redacción creativa o lluvia de ideas** , usa $T \ge 1.0$ con $\text{Top-}p=0.9$. 

Tema 1.1.1 · Modelos Probabilísticos & Generación

### Modelado de Lenguaje Autoregresivo & Muestreo (Sampling)

#### 1\. Guía de Inicio · Visión del Tema 1.1.1 ¿Cómo "Piensa" un Modelo de IA? El Secreto de la Predicción Autoregresiva

Para dominar la Inteligencia Artificial moderna, el primer paso esencial es **desmitificar la "magia"** : cuando interactúas con un LLM de última generación como **Meta Llama 3** , el modelo no posee una base de datos de respuestas pregrabadas ni redacta párrafos completos de forma simultánea. 

Su arquitectura opera como un **motor de cálculo probabilístico de alta velocidad** : recibe el texto de entrada, calcula matemáticamente qué palabra (o subpalabra) tiene mayor sentido estadístico como siguiente elemento, la anexa a la conversación y repite el proceso miles de veces por segundo hasta emitir el token de finalización (`<|eot_id|>`). 

En este primer tema dominarás los tres fundamentos que gobiernan este comportamiento: la **distribución de probabilidad Softmax** , el control de creatividad mediante **Temperatura ($T$)** y el filtrado estocástico con **Top-$p$ (Nucleus Sampling)**. 

#### 2\. Concepto Formal Distribución de Probabilidad Condicional & Softmax

Matemáticamente, un LLM aproxima la probabilidad conjunta de una secuencia $W = (w_1, w_2, \dots, w_n)$ mediante la regla de la cadena probabilística:

$$P(W) = \prod_{t=1}^{n} P(w_t \mid w_1, w_2, \dots, w_{t-1})$$

 

Desglose de Símbolos y Variables 6 elementos

$P(W)$

**Probabilidad de la Frase Completa:** La probabilidad matemática de que toda la secuencia de texto generada $W = (w_1, \dots, w_n)$ tenga sentido gramatical y coherencia natural. 

$\prod_{t=1}^n$

**Productorio (Multiplicación Acumulada):** Símbolo matemático que indica multiplicar sucesivamente las probabilidades individuales calculadas para cada palabra, desde el primer token ($t=1$) hasta el último ($t=n$). 

$t$ (Paso)

**Paso de Inferencia (Tiempo):** Índice numérico que marca la posición cronológica exacta que el modelo está procesando en ese instante. 

$w_t$ (Token)

**Token Actual:** La palabra o fragmento específico que la red neuronal está evaluando para emitir en el paso $t$. 

$\mid$ (Dado que)

**Condición Probabilística:** Establece que la probabilidad de la palabra actual depende obligatoriamente de todas las palabras anteriores. 

$w_1, \dots, w_{t-1}$

**Contexto Previo Acumulado:** Todo el historial de palabras (prompt del usuario + respuestas previas) que la red ya conoce antes de generar el nuevo token. 

La capa lineal final de la red emite un vector de logits no normalizados $\mathbf{z} \in \mathbb{R}^{|V|}$ (donde $|V| = 128,256$). La distribución sobre el vocabulario se calcula mediante la función **Softmax modulada por la Temperatura ($T$)** :

$$P(x_{t+1} = w_i \mid x_{1:t}) = \frac{\exp(z_i / T)}{\sum_{j=1}^{|V|} \exp(z_j / T)}$$

 

Desglose de Softmax y Temperatura 7 elementos

$P(x_{t+1} = w_i \mid x_{1:t})$

**Probabilidad de la Palabra Siguiente:** El porcentaje exacto (entre 0% y 100%) asignado a la palabra candidata $w_i$ para aparecer en la siguiente posición $t+1$. 

$z_i$ (Logit)

**Puntuación Cruda de la Palabra:** Número real (positivo o negativo) emitido directamente por la última capa de la red antes de transformarse en porcentaje. 

$T$ (Temperatura)

**Modulador de Creatividad:** Parámetro divisor que controla la entropía de la distribución: 

  * **T baja (≤ 0.2):** Hace al modelo determinista, lógico, repetitivo y estrictamente enfocado en la opción #1.
  * **T media (0.7):** Balance estándar óptimo entre coherencia fáctica y fluidez humana.
  * **T alta (≥ 1.2):** Aplana las diferencias y permite opciones sorpresivas, creativas o poco convencionales.

$\exp(z_i / T)$

**Función Exponencial Natural:** Convierte puntuaciones arbitrarias en valores estrictamente positivos crecientes, evitando que existan probabilidades negativas. 

$\sum_{j=1}^{|V|}$

**Sumatoria sobre el Vocabulario:** Suma las puntuaciones exponenciales de cada una de las 128,256 palabras del catálogo del modelo. 

$|V| = 128,256$

**Tamaño del Vocabulario:** La cantidad total de tokens que Meta Llama 3 sabe reconocer y utilizar. 

$\sum P = 1.0$ (Normalización)

**Suma al 100%:** Al dividir el puntaje individual entre la suma total de todas las palabras, se garantiza que la suma de todos los porcentajes sea exactamente igual a 1.0 (100%). 

#### 3\. ¿Cómo Funciona? El ciclo de inferencia paso a paso

1

**Recepción del Contexto (Prompt Prefill):**

El modelo ingiere la secuencia de entrada completa (ej. _"El modelo Llama..."_) y genera los tensores de activación iniciales en memoria.

2

**Cómputo en Bloques Transformer:**

Las 32 capas de atención y redes Feed-Forward proyectan las relaciones entre palabras y calculan las puntuaciones numéricas (logits).

3

**Muestreo Estocástico (Sampling Strategy):**

Aplicando Top-p (Nucleus) y Temperatura, se selecciona un token (ej. _"aprende"_). Este se añade a la secuencia y se reanuda el ciclo.

4\. ¿No entendiste? Te lo explico fácil (Analogía de la vida real)

Imagina que estás escribiendo un mensaje en tu teléfono móvil y el **autocorrector predictivo** te sugiere 3 palabras en la barra superior para completar tu frase (por ejemplo, si escribes _"Voy a tomar un..."_ te sugiere _"café"_ , _"vaso"_ o _"descanso"_). Un LLM hace exactamente eso, pero habiendo leído millones de bibliotecas, código y conversaciones, conociendo el contexto global del texto.

5\. Laboratorio Interactivo: Simulador de Predicción Autoregresiva

Simulador en Vivo

PROMPT & TEXTO ACUMULADO:

El modelo Llama |

Temperatura ($T$ = Creatividad vs Determinismo): 0.7

Haz clic en una de las predicciones calculadas por el modelo:

Paso: **1 / 3**

#### 6\. Código de Producción Muestreo con Temperatura en PyTorch

A continuación se detalla la función estándar en Python para recibir las puntuaciones directas del modelo (logits), modular la distribución mediante temperatura y seleccionar el siguiente token estocástico:

Python (PyTorch) · sample_next_token()
    
    
    # 1. Importamos la librería PyTorch para cálculo matricial y tensores en GPU
    import torch
    
    # 2. Importamos el submódulo de funciones de redes neuronales (como Softmax)
    import torch.nn.functional as F
    
    def sample_next_token(logits: torch.Tensor, temperature: float = 0.7) -> int:
     """
     Función que recibe las puntuaciones crudas del modelo (logits)
     y selecciona el siguiente token aplicando temperatura.
     """
     # 3. Evitamos dividir entre cero si el usuario envía una temperatura de 0
     temp = max(temperature, 1e-5)
     
     # 4. Dividimos cada puntuación entre la temperatura para modular la distribución
     # (Una temperatura baja agranda la diferencia entre el primer y segundo lugar)
     scaled_logits = logits / temp
     
     # 5. La función Softmax convierte los números reales en porcentajes (suman 100%)
     # dim=-1 indica que la operación se aplica sobre el eje del vocabulario
     probabilities = F.softmax(scaled_logits, dim=-1)
     
     # 6. 'multinomial' hace un sorteo probabilístico (ruleta) según el porcentaje de cada palabra
     # num_samples=1 indica que solo queremos extraer un único token ganador
     next_token_id = torch.multinomial(probabilities, num_samples=1)
     
     # 7. .item() extrae el número entero de Python desde la memoria del tensor
     return next_token_id.item()

#### 7\. Ejemplos Prácticos Casos de uso según Temperatura

Temperatura Baja ($T = 0.1 - 0.2$)

Generación de código SQL/Python, resolución de ecuaciones matemáticas y extracción estructurada en JSON.

Temperatura Balanceada ($T = 0.7$)

Asistentes conversacionales, resúmenes de texto y redacción de correos profesionales.

Temperatura Alta ($T = 1.0 - 1.2$)

Lluvia de ideas creativas, creación de historias de ficción, poesía y metáforas originales.

8\. Conclusión & Puntos Clave

Un LLM no almacena respuestas pregrabadas; calcula continuamente probabilidades de la siguiente palabra condicionadas por todo el contexto previo.

9\. Autoevaluación Pregunta 1 de 7

¿Qué efecto matemático tiene fijar la Temperatura en $T = 0.0$ (Greedy Decoding)?

A) Genera palabras completamente aleatorias sin importar el contexto.

B) Colapsa la distribución y selecciona siempre el token con mayor logit (100% determinista).

C) Borra el historial de contexto en la memoria RAM.

Advertencia Crítica: Complejidad Cuadrática $O(N^2)$ en Atención Estándar

En atención Multi-Head estándar (MHA), duplicar el tamaño del prompt cuadruplica ($4\times$) el costo computacional y la memoria de activación. En Llama 3 se mitiga con **Grouped-Query Attention (GQA)** , pero siempre debes limitar y condensar el historial para evitar saturación de la memoria KV Cache en GPU. 

Tema 1.1.2 · Álgebra y Representación

### El Pipeline de NLP & Tensores Numéricos

#### 1\. Introducción El puente entre el lenguaje humano y el silicio

Las personas nos comunicamos mediante palabras, emociones y metáforas, pero los microprocesadores de computadora solo pueden procesar voltajes binarios y operaciones numéricas. El **Pipeline de NLP** es la serie de etapas que convierte un texto arbitrario en una estructura de números que la red neuronal puede multiplicar a la velocidad de la luz.

#### 2\. Concepto Formal ¿Qué es un Tensor en NLP?

Un **tensor** es una generalización de vectores y matrices a cualquier número de dimensiones. En Llama 3, una entrada de texto se representa como un tensor tridimensional con forma:

$$\text{Forma del Tensor} = [\text{Batch Size}, \text{Sequence Length}, d_{\text{model}}]\text{Ejemplo para Llama 3 8B}: [1, 2048, 4096]

$$ 

Desglose de Dimensiones Tensoriales 5 dimensiones

$[B, S, D]$ (Tensor 3D)

**Estructura Matricial Multidimensional:** Bloque de números flotantes organizado en 3 ejes que viaja a través de las 32 capas Transformer de la tarjeta gráfica (GPU). 

$B = 1$ (Batch Size)

**Lote de Consultas Simultáneas:** Cantidad de peticiones, preguntas o usuarios independientes que el modelo procesa en paralelo al mismo tiempo (en este ejemplo, 1 usuario individual). 

$S = 2,048$ (Seq Length)

**Longitud de la Secuencia (Tokens):** Cantidad total de palabras o fragmentos que componen el documento o mensaje actual que el modelo está leyendo en ese instante. 

$D = 4,096$ ($d_{\text{model}}$)

**Dimensión del Espacio Oculto (Embeddings):** Cantidad de números continuos que describen las propiedades y el significado de cada palabra en Llama 3 8B (en Llama 3 70B este valor es de 8,192). 

$N = 8,388,608$ (Floats)

**Valores Numéricos Activos:** Resultado de multiplicar $1 \times 2048 \times 4096$. Son más de 8.3 millones de números que la GPU multiplica en cada pasada hacia adelante (forward pass). 

#### 3\. ¿Cómo Funciona? Las 4 etapas del procesamiento

1

**Normalización Unicode:**

Se limpian espacios extraños y se codifica el texto en UTF-8 estándar.

2

**Mapeo de IDs (Tokenización):**

Cada palabra o fragmento se convierte en un número entero único entre 0 y 128,255.

3

**Lookup de Embeddings:**

Los IDs enteros se transforman en vectores continuos de 4,096 números flotantes.

4

**Multiplicación Matricial en GPU:**

Las 32 capas Transformer procesan los tensores y emiten los logits finales.

4\. ¿No entendiste? Te lo explico fácil

Para ti, la palabra _"música"_ evoca canciones, instrumentos y ritmos. Pero para un chip de computadora, _"música"_ es simplemente el número **104**. Al darle ese número a la GPU, esta puede hacer millones de operaciones matemáticas instantáneas para saber qué responderte.

5\. Prueba en Vivo: Inspección del Pipeline Tensorial

Selector

1\. ENTRADA HUMANA (CADENA DE TEXTO):

reproducir música

2\. VECTOR DE TOKENS (TENSORES NUMÉRICOS 1D):

[4120, 892, 104]

3\. SALIDA / ACCIÓN INFERIDA:

Ejecutando streaming de audio

#### 6\. Código de Producción Conversión de Texto a Tensores con Hugging Face

Script en Python para cargar el tokenizador oficial de **Meta Llama 3** , procesar una orden en lenguaje natural y extraer el tensor de IDs numéricos y la máscara de atención que ingresan a la GPU:

Python · AutoTokenizer con Transformers
    
    
    # 1. Importamos la clase AutoTokenizer de la librería 'transformers' de Hugging Face
    from transformers import AutoTokenizer
    
    # 2. Descargamos el vocabulario y reglas de segmentación del modelo Meta Llama 3
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    
    # 3. Definimos un texto en lenguaje natural escrito por un usuario
    texto = "reproducir música en streaming"
    
    # 4. Convertimos el texto a tensores numéricos compatibles con PyTorch ('pt')
    inputs = tokenizer(texto, return_tensors="pt")
    
    # 5. Imprimimos el tensor con los IDs numéricos asignados a cada subpalabra
    print("IDs Numéricos (Tensor 2D):", inputs["input_ids"])
    
    # 6. Imprimimos la máscara que le dice a la GPU qué tokens debe atender (1 = atender, 0 = ignorar)
    print("Máscara de Atención:", inputs["attention_mask"])
    
    # Salida en consola:
    # tensor([[128000, 4120, 892, 104, 15302]]) <-- 128000 es el token de inicio <|begin_of_text|>

7\. Conclusión & Puntos Clave

Sin la conversión de caracteres a tensores numéricos, las tarjetas gráficas no podrían ejecutar operaciones matemáticas de álgebra lineal.

8\. Autoevaluación Pregunta 2 de 7

¿Por qué es obligatorio convertir el texto en tensores numéricos para la IA?

A) Porque el idioma español no se puede escribir en computadoras.

B) Porque las GPUs son procesadores de alta velocidad diseñados exclusivamente para multiplicar matrices de números.

C) Porque el teclado de la computadora se apagaría.

Tema 1.2

## Tokenización & Geometría Semántica

Aprende cómo las palabras se parten en bloques eficientes (Tokens) mediante Byte-Pair Encoding y cómo se ubican en un hiperespacio de significados continuos (Embeddings).

De Palabras a Números · Tokenización

### De Palabras a Números: Qué es un Token & Tokenización BPE

#### 1\. Introducción El dilema del vocabulario

¿Debería una IA aprender cada palabra entera del diccionario o aprender solo letras individuales? Si aprende letras sueltas, procesar un texto requiere demasiados pasos. Si aprende palabras completas, cualquier palabra nueva o falta de ortografía rompería el modelo. **Byte-Pair Encoding (BPE)** es el estándar que resuelve este dilema dividiendo el texto en subpalabras óptimas.

#### 2\. Concepto Formal ¿Cómo funciona el algoritmo BPE?

BPE comienza tratando cada byte del texto como un símbolo base. Luego, cuenta estadísticamente qué pares de bytes adyacentes ocurren con más frecuencia en el corpus y los fusiona recursivamente en un nuevo token compuesto hasta alcanzar el tamaño de vocabulario deseado:

$$

\text{Vocabulario Llama 3} = 128,256 \text{ tokens}

\text{Compresión Promedio} \approx 3.8 \text{ caracteres / token}

$$ 

Desglose de Tokenización y BPE 4 métricas

$|V| = 128,256$ (Vocabulario)

**Tamaño del Vocabulario Tiktoken (BPE):** Catálogo de subpalabras, caracteres y palabras pre-memorizadas por Meta. Es un 300% más grande que el de Llama 2 (32k tokens), permitiendo entender mucho mejor el español, el código y múltiples idiomas. 

$\text{BPE}$ (Byte-Pair)

**Algoritmo de Fusión Recursiva:** Empieza con bytes UTF-8 individuales y une progresivamente los pares de letras más comunes (ej. `d` + `e` $\to$ `de`, `c` + `i` + `ó` + `n` $\to$ `ción`) hasta formar bloques eficientes. 

$\approx 3.8\text{ chars/token}$

**Tasa de Compresión en Español:** En promedio, cada token agrupa casi 4 letras normales. Por ejemplo, una palabra de 8 letras como _"aprender"_ consume solo 2 tokens en lugar de 8 pasos computacionales. 

$-30\%\text{ Latencia}$

**Inferencia Acelerada:** Al necesitar menos tokens para expresar el mismo mensaje, el modelo responde un 25% a 30% más rápido y reduce drásticamente el consumo de memoria de la GPU. 

#### 3\. ¿Cómo Funciona? Proceso de fusión en vivo

1

**Palabras Frecuentes:**

Palabras comunes como _"el"_ , _"de"_ , _"computadora"_ reciben su propio ID de token individual.

2

**Palabras Compuestas o Raras:**

Se dividen en prefijos y sufijos comunes (ej. _"inconstitucionalmente"_ $\to$ `in` + `constitucional` + `mente`).

3

**Caracteres Desconocidos:**

Cae de respaldo en bytes individuales UTF-8, garantizando que el modelo nunca falle ante caracteres raros o emojis.

4\. ¿No entendiste? Te lo explico fácil (Analogía con bloques LEGO)

Imagina que tienes una caja de **piezas de LEGO**. En vez de guardar un castillo entero ya armado que ocupa mucho espacio (palabras completas), guardas piezas estándar (`in`, `tel`, `i`, `gen`, `te`). Con esas mismas piezas puedes armar _"inteligente"_ , _"inteligencia"_ o cualquier palabra nueva ahorrando espacio en la memoria.

5\. Laboratorio: Tokenizador BPE Interactivo en Tiempo Real

0

Tokens Totales

0

Letras / Caracteres

$0.0000

Costo Est. en Servidor ($0.20/1M)

6\. Conclusión & Puntos Clave

Un vocabulario BPE más amplio (128k en Llama 3) produce menos tokens por oración en español, acelerando el tiempo de respuesta y abaratando los costos.

7\. Autoevaluación Pregunta 3 de 7

¿Cuál es la principal ventaja técnica de expandir el vocabulario BPE a 128k tokens en Llama 3?

A) Reduce el número de fragmentos por palabra en español y código, aumentando la velocidad de inferencia.

B) Permite que el modelo funcione sin necesidad de usar electricidad.

C) Traduce automáticamente todo el texto al idioma inglés.

De Palabras a Números · Embeddings

### Embeddings o Vectores de Token: Espacios Semánticos y Álgebra Vectorial

#### 1\. Introducción ¿Cómo sabe una máquina que un perro se parece a un gato?

Para una computadora, los IDs de los tokens son números arbitrarios (ej. perro = 1420, gato = 981). Estos números no guardan ninguna noción de parentesco. Los **Embeddings** son la técnica que proyecta cada token en un mapa continuo de conceptos donde la cercanía física equivale a similitud de significado.

#### 2\. Concepto Formal Espacios Vectoriales & Similitud Coseno

Cada token se representa como un vector $\mathbf{v} \in \mathbb{R}^{d}$ ($d=4096$ en Llama 3 8B). Para medir qué tan cercanos están dos conceptos, calculamos el coseno del ángulo entre sus vectores:

$$

\text{Similitud Coseno}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \frac{\sum_{i=1}^{d} u_i v_i}{\sqrt{\sum_{i=1}^d u_i^2} \cdot \sqrt{\sum_{i=1}^d v_i^2}}

\vec{v}(\text{"Rey"}) - \vec{v}(\text{"Hombre"}) + \vec{v}(\text{"Mujer"}) \approx \vec{v}(\text{"Reina"})

$$ 

Desglose de Álgebra Vectorial y Semántica 5 conceptos

$\mathbf{u}, \mathbf{v} \in \mathbb{R}^{4096}$

**Vectores de Embeddings:** Las dos listas de 4,096 coordenadas continuas que sitúan a cada concepto en el mapa geométrico del modelo. 

$\mathbf{u} \cdot \mathbf{v}$

**Alineación Direccional (Producto Punto):** Multiplica las 4,096 coordenadas correspondientes ($\sum u_i v_i$). Si ambas flechas conceptuales apuntan en la misma dirección, el valor es alto y positivo. 

$\|\mathbf{u}\|_2$ (Norma L2)

**Magnitud o Longitud:** Calculada con la raíz cuadrada de la suma de cuadrados ($\sqrt{\sum u_i^2}$). Representa el tamaño euclidiano del vector en 4,096 dimensiones. 

$[-1.0, +1.0]$

**Escala de Similitud Normalizada:**

  * **+1.0:** Conceptos idénticos o sinónimos perfectos (mismo significado).
  * **0.0:** Conceptos ortogonales sin ninguna relación semántica.
  * **-1.0:** Conceptos diametralmente opuestos.

$\text{Rey} - \text{Hombre} + \text{Mujer} \approx \text{Reina}$

**Aritmética de Conceptos:** Prueba matemática de que el espacio latente captura relaciones de género y realeza de forma lineal: al sustraer el vector "masculinidad" y añadir el vector "feminidad" a "Rey", la coordenada resultante apunta con 94% de precisión a "Reina". 

#### 3\. ¿Cómo Funciona? Álgebra Vectorial Semántica

Durante el entrenamiento, el modelo ajusta las coordenadas vectoriales de modo que relaciones gramaticales y conceptuales se traduzcan en traslaciones vectoriales paralelas (ej. restar masculinidad y sumar feminidad).

4\. ¿No entendiste? Te lo explico fácil (El mapa de una ciudad)

Imagina el mapa de una gran ciudad: las tiendas de ropa están en la misma avenida comercial y los hospitales en otra zona. Un **Embedding** hace lo mismo en la memoria de la IA: coloca _"perro"_ y _"gato"_ en el mismo barrio, y _"computadora"_ en otro barrio lejano. Por eso la IA sabe que un perro se parece más a un gato que a una laptop.

5\. Laboratorio: Espacio Vectorial 2D Interactivo

Arrastrable

Similitud Coseno calculada (Rey ↔ Reina): 0.94

6\. Conclusión & Puntos Clave

Los embeddings son la base de los sistemas de búsqueda semántica (RAG) y permiten a los LLMs comprender analogías, sinónimos y contextos culturales.

7\. Autoevaluación Pregunta 4 de 7

¿Qué propiedad geométrica permite la suma y resta de conceptos en un espacio de Embeddings?

A) Que todos los vectores tienen longitud cero.

B) Que las relaciones semánticas (como género o capitales) se codifican como vectores direccionales consistentes.

C) Que la memoria RAM se reordena alfabéticamente.

Tema 1.3

## Arquitectura Transformer & Mecanismos de Atención

Comprende la maquinaria interna del Transformer: cómo interactúan las matrices Query, Key y Value, cómo Grouped-Query Attention optimiza la memoria y cómo calcular la VRAM necesaria.

Arquitectura Transformer · Mecanismo de Atención

### El Transformer en una Frase: Atención sobre Atención (Self-Attention & GQA)

#### 1\. Introducción Superando las limitaciones de memoria a largo plazo

Las redes neuronales recurrentes antiguas (RNNs y LSTMs) olvidaban el principio de un texto al llegar al final de un párrafo. La **Auto-Atención (Self-Attention)** introducida en el paper _"Attention Is All You Need"_ permite que cualquier palabra mire a todas las demás instantáneamente sin importar qué tan lejos estén en el libro o conversación.

#### 2\. Concepto Formal Matrices Query ($Q$), Key ($K$) y Value ($V$)

Para cada token, se multiplican sus embeddings por tres matrices de pesos aprendidos para generar tres representaciones:

$$

\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V

$$ 

Desglose de Auto-Atención Transformer 7 componentes

$\mathbf{Q}$ (Query)

**Matriz de Búsqueda (Pregunta):** Lo que cada palabra en la oración necesita averiguar sobre las demás palabras (ej. el pronombre _"lo"_ busca a qué objeto o persona se refiere). 

$\mathbf{K}$ (Key)

**Matriz de Identidad (Clave / Etiqueta):** La tarjeta de presentación que cada token expone a los demás para anunciar su rol gramatical y contenido (ej. el sustantivo _"documento"_ se identifica como objeto directo). 

$\mathbf{V}$ (Value)

**Matriz de Contenido Real (Significado):** La información semántica pura que se transferirá a la palabra actual una vez que su búsqueda ($Q$) coincida con la etiqueta ($K$). 

$\mathbf{Q}\mathbf{K}^T$

**Producto Matricial de Relevancia:** Multiplica cada pregunta por todas las claves, calculando qué tan conectadas están todas las palabras entre sí en una matriz de afinidad cuadrada. 

$\sqrt{d_k}$ (Escala)

**Factor de Estabilidad Numérica:** Donde $d_k = 128$ (dimensión de cada cabezal de atención en Llama 3). Dividir entre $\sqrt{128} \approx 11.31$ evita que los números crezcan demasiado, lo que provocaría que la función Softmax colapse y los gradientes se desvanezcan a cero. 

$\text{Softmax}(\dots)$

**Normalización de Pesos de Atención:** Convierte las puntuaciones de afinidad en porcentajes positivos que suman exactamente 100% de atención distribuida. 

$\times \mathbf{V}$

**Suma Ponderada Final:** Mezcla la información de todas las palabras según su porcentaje de atención, generando la representación contextualizada final de cada palabra. 

En Llama 3, para evitar que la memoria de atención (KV Cache) colapse con 128k de contexto, se utiliza **Grouped-Query Attention (GQA)** : 32 cabezales de Query comparten solo 8 cabezales de Key/Value, ahorrando un 75% de memoria.

#### 3\. ¿Cómo Funciona? La analogía de la biblioteca

1

**Query ($Q$ - La Búsqueda):**

Lo que el token actual necesita saber (ej. el verbo _"estaba"_ busca quién realiza la acción).

2

**Key ($K$ - La Etiqueta):**

La descripción que cada token ofrece a los demás (ej. el token _"perro"_ dice _"soy un animal / sujeto"_).

3

**Value ($V$ - El Contenido):**

La información real que se transfiere al token una vez que coinciden $Q$ y $K$.

4\. ¿No entendiste? Te lo explico fácil (Leyendo una novela)

Si lees la frase: _"El perro no cruzó la calle porque**estaba** cansado"_, tu cerebro sabe al instante que el que estaba cansado era el **perro** , no la calle. El mecanismo de Auto-Atención es la lupa con la que la IA mira hacia atrás para saber a qué sujeto se refiere cada verbo.

5\. Laboratorio: Matriz de Atención Inter-Token

Interactivo

**Conexión Semántica Detectada:**

Haz clic en una palabra arriba para inspeccionar a cuáles presta mayor atención matemática.

6\. Conclusión & Puntos Clave

La auto-atención resuelve ambigüedades contextuales y GQA permite procesar documentos masivos sin agotar la memoria de la tarjeta gráfica.

7\. Autoevaluación Pregunta 5 de 7

¿Por qué Grouped-Query Attention (GQA) es fundamental para contextos largos de 128k en Llama 3?

A) Porque reduce drásticamente el tamaño del KV Cache en memoria VRAM al compartir cabezales de llaves y valores.

B) Porque apaga los procesadores de la tarjeta gráfica cuando no se usan.

C) Porque traduce el texto a números enteros de 8 bits sin usar multiplicación.

Arquitectura Transformer · Dimensionamiento

### Arquitectura Transformer: Bloques Apilados, Leyes de Escala y Memoria VRAM

#### 1\. Introducción ¿Cómo saber qué hardware necesitas?

Desplegar un modelo de Inteligencia Artificial requiere planificar con precisión los recursos de hardware. Si un modelo es demasiado grande para tu tarjeta gráfica, el sistema colapsará por error de _Out Of Memory (OOM)_. Comprender el consumo de memoria en bytes por parámetro es esencial para cualquier arquitecto de IA.

#### 2\. Concepto Formal Fórmula de Memoria VRAM y Cuantización

Cada parámetro en precisión original (FP16/BF16) ocupa 2 bytes. Al aplicar **Cuantización (GGUF / AWQ)** a 4 bits (INT4), cada parámetro ocupa solo 0.5 bytes, reduciendo el consumo a una cuarta parte con una pérdida imperceptible de calidad:

$$

\text{VRAM Mínima (GB)} \approx \left(\text{Parámetros (Billones)} \times \frac{\text{Bits de Precisión}}{8}\right) \times 1.25

\text{Margen KV Cache: } +25\% \text{ para contexto y activación}$$ 

Desglose de Memoria GPU y Cuantización 6 variables

$P$ (Parámetros)

**Cantidad de Pesos Neuronales:** Multiplica por 8 en Llama 3 8B (8 mil millones), por 70 en Llama 3 70B, o por 405 en Llama 3 405B. 

$b$ (Bits/Parámetro)

**Nivel de Cuantización / Resolución:**

  * **16 bits (FP16/BF16):** Precisión completa original sin compresión.
  * **8 bits (INT8):** Precisión intermedia que reduce el peso a la mitad con 99.9% de fidelidad.
  * **4 bits (INT4 / GGUF):** Cuantización avanzada que reduce el peso a 1/4 para correr en computadoras comunes.

$b / 8$ (Bytes/Parámetro)

**Conversión de Bits a Bytes:** Como 1 Byte = 8 bits, 16 bits equivalen a 2 Bytes por parámetro, y 4 bits equivalen a 0.5 Bytes por parámetro. 

$P \times \frac{b}{8}$ (Memoria Base)

**Memoria Base del Modelo:** En Llama 3 8B en 4 bits: $8 \times 0.5 = 4.0\text{ GB}$ de peso puro en almacenamiento y VRAM. 

$\times 1.25$ (+25% KV Cache)

**Margen Obligatorio para KV Cache:** Añade un 25% extra de memoria para almacenar las matrices de atención de las conversaciones previas (hasta 8k o 128k tokens) y evitar que la GPU sufra un error de Out-Of-Memory (OOM). 

$\text{VRAM}_{\text{mínima}}$

**Requisito Final:** Para Llama 3 8B en INT4: $(8 \times 0.5) \times 1.25 \approx 5.0 - 5.5\text{ GB}$ de VRAM mínima (ideal para RTX 3060/4060 o Mac M1/M2/M3 con 8GB/16GB de memoria unificada). 

#### 3\. ¿Cómo Funciona? Las 3 variantes oficiales de Llama 3

Llama 3 8B (8 mil millones)

Pesa ~5.5 GB en INT4. Se ejecuta fluidamente en laptops Mac con chip M1/M2/M3 o tarjetas gráficas RTX 3060/4060.

Llama 3 70B (70 mil millones)

Pesa ~40 GB en INT4. Requiere estaciones de trabajo con 2 GPUs RTX 3090/4090 o Mac Studio con 64GB de memoria unificada.

Llama 3 405B (405 mil millones)

El modelo abierto más potente del mundo. Requiere servidores dedicados con clústeres de 8x GPUs NVIDIA H100 (80GB).

4\. ¿No entendiste? Te lo explico fácil (El tamaño del libro)

Un modelo **8B (8 mil millones de datos)** es como una guía de bolsillo: muy rápida y cabe en la memoria de una laptop normal. Un modelo **70B** es como una enciclopedia completa: mucho más sabia, pero necesitas una computadora de trabajo potente con varias tarjetas gráficas.

5\. Calculadora de Arquitectura & Hardware GPU

Calculadora

TAMAÑO DEL MODELO:

8B Parámetros

8B (Ligero / Laptop) 70B (Intermedio / Estación) 405B (Enterprise / Clúster)

Precisión de Cuantización: FP16 / BF16 (16 bits - Precisión Original Completa) INT8 (8 bits - Balanceado con mínima degradación) INT4 / GGUF (4 bits - Máxima compresión para PCs y Laptops)

5.5 GB

Memoria VRAM Necesaria

1x GPU RTX 3060 (8GB) o Laptop

Equipo Recomendado

6\. Conclusión & Puntos Clave

La cuantización democratiza el acceso a la IA, permitiendo correr modelos de 8 mil millones de parámetros en computadoras personales sin pérdida notable de razonamiento.

7\. Autoevaluación Pregunta 6 de 7

¿Para qué sirve la cuantización a 4 bits (INT4 / GGUF)?

A) Para reducir drásticamente los requerimientos de memoria VRAM y permitir ejecutar LLMs en laptops y GPUs comerciales.

B) Para que el modelo pueda aprender nuevos idiomas en tiempo real.

C) Para eliminar la necesidad de usar procesadores de computadora.

Tema 1.4

## Soberanía de Datos & Telemetría en Tiempo Real

Comprende la diferencia arquitectónica fundamental entre la IA de Pesos Abiertos y las APIs en la nube, y experimenta con la transmisión en streaming (SSE).

Soberanía Tecnológica · Pesos Abiertos

### Pesos Abiertos: Por Qué Importa Poder Descargar el Modelo & Parámetros

#### 1\. Introducción El debate de la soberanía digital

Cuando las empresas y gobiernos integran Inteligencia Artificial en sus procesos críticos (médicos, bancarios o legales), surge una disyuntiva: ¿depender de servidores extranjeros o poseer el modelo en infraestructura propia? El ecosistema de **Pesos Abiertos (Open Weights)** garantiza que el futuro de la IA no quede monopolizado por un puñado de corporaciones cerradas.

#### 2\. Concepto Formal ¿Qué significa tener los Pesos Abiertos?

Significa que cualquier organización puede descargar el archivo binario con los billones de matrices numéricas entrenadas (`model.safetensors`). Puedes inspeccionarlo, adaptarlo mediante **Fine-Tuning LoRA** , auditar sus sesgos y ejecutarlo para siempre en servidores locales sin pagar licencias ni peajes recurrentes.

3\. Matriz Comparativa de Paradigmas: Pesos Abiertos (On-Premise) vs APIs Propietarias (Cloud) Matriz General

Comparación arquitectónica universal entre el modelo soberano de pesos abiertos y los servicios comerciales cerrados en la nube:

Vector Estratégico | Modelos de Pesos Abiertos (On-Premise / Edge)  
(ej. Llama 3, Mistral, Qwen, DeepSeek) | APIs Comerciales en la Nube (Cloud)  
(ej. OpenAI GPT-4, Anthropic Claude, Gemini Cloud)  
---|---|---  
1\. Privacidad & Soberanía de Datos Confidencialidad y Compliance |  Soberanía Absoluta **100% On-Premise / Servidor Local:** Los datos, prompts y secretos industriales jamás salen de tu propia memoria RAM/GPU. Cumplimiento garantizado de **GDPR, HIPAA, Secreto Bancario** y normativas locales de ciberseguridad. |  Exposición a Terceros **Tránsito y Servidores Externos:** Los datos viajan cifrados por internet a centros de datos extranjeros sujetos a leyes de terceros países y posibles políticas de telemetría o entrenamiento.  
2\. Estructura de Costos CAPEX vs OPEX Recurrente |  Costo Marginal Cero ($0) **Inversión Fija y Predecible:** Solo adquieres el hardware una vez. Puedes procesar **cientos de millones de tokens al mes** para miles de empleados sin pagar tarifas por mensaje ni incrementos imprevistos. |  Facturación por Token **Tarifa Recurrente ($/1M tokens):** El coste escala lineal y exponencialmente con el volumen de usuarios y longitud del contexto. Puede volverse insostenible a gran escala.  
3\. Personalización & Fine-Tuning Pesos Neuronales y RAG |  Control Total de Parámetros **Acceso a Tensores Internos:** Posibilidad de adaptar capas mediante **LoRA / QLoRA** , inyectar vocabularios propios, modificar matrices de atención y fusionar modelos (*Model Merging*). |  Caja Negra Restringida **Limitado a System Prompt:** Solo puedes interactuar como cliente externo mediante prompts o fine-tuning cerrado sujeto a altos costes de alojamiento por hora del proveedor.  
4\. Disponibilidad & Latencia (SLA) Resiliencia Operativa |  Operación 100% Offline **Independencia de Conexión:** Funciona sin internet en fábricas, minas, hospitales o barcos. Cero cuotas de rate limit (RPM/TPM) y latencia fija ultrabaja por bus PCIe directo. |  Dependencia de Internet **Sujeto a Caídas Globales:** Vulnerable a cortes de red (*Cloud Outages*), aumento de latencia por congestión mundial y límites estrictos de peticiones por minuto.  
5\. Seguridad & Gobernanza Auditoría y Llama Guard |  Transparencia y Blindaje **Auditoría Completa:** Cero telemetría oculta, libertad para integrar barreras de ciberseguridad propias (**Prompt Guard** , **Llama Guard**) e inmunidad ante cambios arbitrarios de políticas. |  Modificaciones Unilaterales **Filtros Opacos y Deprecación:** El proveedor puede modificar el comportamiento del modelo, cambiar filtros o discontinuar versiones que rompen sistemas en producción.  
  
4\. ¿No entendiste? Te lo explico fácil (Tener auto propio vs pedir taxi)

Usar una API en la nube es como pedir un taxi: pagas por cada viaje que haces y la empresa sabe a dónde fuiste. Usar **Llama 3 en tu computadora** es como tener tu propio automóvil: lo usas las veces que quieras sin pagar por viaje, funciona aunque no haya internet y nadie puede espiar tus conversaciones.

5\. Conclusión & Puntos Clave

Los pesos abiertos garantizan independencia operativa, cumplimiento legal de privacidad de datos y estabilidad económica a largo plazo.

6\. Autoevaluación Pregunta 7 de 7

¿Cuál es la principal ventaja de seguridad al desplegar Meta Llama 3 localmente?

A) Que la información privada jamás sale de la infraestructura local del usuario o empresa.

B) Que no se requiere ningún sistema operativo instalado.

C) Que el modelo reemplaza automáticamente todas las contraseñas del sistema.

Caso Práctico de Ingeniería

### Caso Práctico: Eligiendo el Modelo Correcto para un Asistente de Soporte & Inferencia en Vivo

#### 1\. Introducción La experiencia de usuario en tiempo real

Nadie quiere esperar 10 segundos mirando una pantalla en blanco para leer una respuesta. Los motores de inferencia modernos utilizan **Streaming (Server-Sent Events)** para enviar los tokens a la pantalla en el instante exacto en que salen de la última capa del Transformer.

#### 2\. Concepto Formal Métricas de Telemetría Clave

Time-To-First-Token (TTFT)

El tiempo que tarda el modelo en procesar el prompt de entrada y emitir el primer token (latencia de prefijo).

Tokens Por Segundo (TPS)

La velocidad de generación continua. Una velocidad de 30-40 t/s supera la velocidad de lectura humana normal.

3\. Simulador de Terminal de Inferencia Streaming Local

Ollama Engine

localhost:11434 — meta-llama-3-8b-instruct HTTP 200 Streaming

> Presiona "Ejecutar Inferencia" para transmitir tokens desde el runtime local...

Velocidad: **0 t/s** Tokens Generados: **0** Latencia Inicial (TTFT): **0 ms**

#### 6\. Código de Producción Cliente HTTP Streaming en Python con Ollama

Script en Python para conectarse a la API REST de Ollama en local (puerto 11434) y recibir la respuesta generada por Meta Llama 3 token por token en tiempo real mediante _Server-Sent Events_ (SSE):

Python · Cliente Streaming Asíncrono para Ollama
    
    
    # 1. Importamos 'requests' para enviar peticiones HTTP a servidores locales o APIs
    import requests
    
    # 2. Importamos 'json' para convertir cadenas de texto en objetos y diccionarios de Python
    import json
    
    def stream_llama(prompt: str):
     """
     Se conecta al servidor local de Ollama (puerto 11434) y transmite
     la respuesta generada palabra por palabra en tiempo real.
     """
     # 3. Endpoint de la API REST local de Ollama
     url = "http://localhost:11434/api/generate"
     
     # 4. Datos de la consulta: modelo a invocar, pregunta del usuario y streaming activado
     payload = {
     "model": "llama3:8b", # Nombre del modelo descargado en tu máquina
     "prompt": prompt, # Instrucción que le enviamos a la IA
     "stream": True # 'True' activa el envío token por token vía Server-Sent Events
     }
     
     # 5. Enviamos la petición POST con stream=True para mantener el flujo de red abierto
     with requests.post(url, json=payload, stream=True) as response:
     # 6. Leemos cada línea conforme es emitida por la GPU del procesador local
     for line in response.iter_lines():
     if line:
     # 7. Decodificamos el paquete de bytes a UTF-8 y lo transformamos a diccionario JSON
     chunk = json.loads(line.decode('utf-8'))
     
     # 8. Extraemos el fragmento de texto ('response') y lo imprimimos sin salto de línea
     print(chunk.get("response", ""), end="", flush=True)
    
    # 9. Ejemplo de ejecución práctica:
    stream_llama("¿Cuáles son los 3 pilares de Meta Llama 3?")

7\. Conclusión & Puntos Clave

La inferencia con streaming y telemetría elimina los bloqueos en la interfaz de usuario, ofreciendo una experiencia fluida e inmediata idéntica a las aplicaciones comerciales de vanguardia.

Autoevaluación Práctica

## Ejercicios de Ingeniería & Análisis Crítico

Pon a prueba tu comprensión técnica resolviendo estos casos de ingeniería. Reflexiona tu respuesta y despliega la solución para comparar tu análisis.

Ejercicio 1

#### Análisis de Eficiencia en Tokenización (Español vs. Inglés)

Escribe una misma instrucción en español y en inglés: _"Configura la alarma para las siete de la mañana"_ vs. _"Set the alarm for seven in the morning"_. Identifica qué palabras del español se fragmentan en más tokens y explica por qué las conjugaciones y las preposiciones aumentan el consumo de tokens y el costo computacional. 

Ver Criterio de Solución & Análisis de Ingeniería

**Análisis Técnico:**

  * En inglés, _"Set the alarm for seven in the morning"_ se tokeniza casi palabra por palabra (7-8 tokens) porque los vocabularios BPE están altamente optimizados para la frecuencia del inglés.
  * En español, palabras como _"Configura"_ suelen dividirse en subpalabras (ej. `["Config", "ura"]`) debido a las variaciones flexivas de los verbos y desinencias gramaticales.
  * **Impacto en Producción:** Un mayor número de tokens incrementa la latencia (más pasos autorregresivos), reduce la longitud efectiva de la ventana de contexto y aumenta los costos en APIs por token o consumo de memoria en inferencia local.

Ejercicio 2

#### Diagnóstico de Coherencia y Mecanismos de Atención

Un asistente virtual confunde el nombre de un producto mencionado al inicio de un párrafo largo con otro similar que aparece al final. Explica qué mecanismo de la arquitectura Transformer está fallando en ponderar la relevancia correcta y propón una variable de diseño o estrategia de prompting para mitigar el error. 

Ver Criterio de Solución & Análisis de Ingeniería

**Diagnóstico Arquitectónico:**

  * El mecanismo de **Self-Attention ($Q \cdot K^T$)** no asignó suficiente peso de atención a los tokens del inicio debido al fenómeno de dispersión de atención en contextos extensos (*lost in the middle*).
  * **Solución de Ingeniería:**
    1. **Prompting Estructurado:** Colocar la entidad clave inmediatamente antes de la instrucción final (*recency bias*).
    2. **Temperatura ($T$):** Reducir $T$ a $\le 0.2$ para evitar muestreos probabilísticos aleatorios en la distribución Softmax.
    3. **Técnica CoT (Chain-of-Thought):** Forzar al modelo a listar primero las entidades detectadas antes de emitir la conclusión.

Ejercicio 3

#### Decisión de Despliegue: Pesos Abiertos vs. APIs Cerradas

Evalúa tres escenarios de negocio: (a) Un hospital rural sin conexión estable a internet, (b) Una fintech regulada que debe auditar sesgos y trazabilidad interna, y (c) Un chatbot de marketing temporal para una campaña en línea de 48 horas. Argumenta qué arquitectura de modelo (Pesos Abiertos Llama vs. API Comercial Cerrada) es la más adecuada para cada caso. 

Ver Criterio de Solución & Análisis de Ingeniería

**Resolución Justificada:**

  * **(a) Hospital Rural:** _Pesos Abiertos (Meta Llama 3 local)_. Al no requerir conexión a internet, garantiza operación continua (*offline availability*) y privacidad absoluta de los historiales clínicos.
  * **(b) Fintech Regulada:** _Pesos Abiertos (Meta Llama 3)_. Permite auditar pesos, calcular perplejidad exacta de respuestas, ejecutar fine-tuning LoRA en servidores propios y cumplir con normativas de soberanía de datos sin filtraciones a terceros.
  * **(c) Campaña de Marketing de 48 horas:** _API Comercial Cerrada_. No justifica aprovisionar infraestructura ni hardware local para un evento efímero; el pago por consumo temporal optimiza tiempo y costos.

Ejercicio 4

#### Análisis de Trade-Offs de Tamaño de Modelo: Llama 3 8B vs 70B

Enunciado de Dimensionamiento 

Un equipo de ingeniería debe decidir entre desplegar Meta Llama 3 8B o Llama 3 70B para clasificar 50,000 correos diarios. Evalúa el consumo de VRAM, el costo mensual de infraestructura GPU y el tiempo de respuesta (TTFT), justificando la elección óptima. 

Ver Solución de Ingeniería Paso a Paso & Matriz de Costo-Beneficio

1

##### Evaluación de Llama 3 8B (Cuantizado INT4 / AWQ)

• **VRAM Requerida:** ~5.5 GB (cabe en 1x GPU comercial RTX 4090 o T4 económica).  
• **Throughput:** > 120 tokens/segundo con latencia sub-100ms.  
• **Precisión en Clasificación:** 98.2% de exactitud con prompting estructurado.  
• **Costo Mensual Estimado:** ~$150 USD en instancia cloud básica. 

2

##### Evaluación de Llama 3 70B (Cuantizado INT4)

• **VRAM Requerida:** ~42 GB (requiere al menos 2x A10G o 1x A100 de 80GB).  
• **Throughput:** ~25-35 tokens/segundo con mayor latencia.  
• **Precisión en Clasificación:** 98.9% (+0.7% de ganancia marginal).  
• **Costo Mensual Estimado:** ~$1,800 USD en instancia multi-GPU. 

##### Decisión de Arquitectura Óptima

La elección de ingeniería es **Meta Llama 3 8B** : resuelve el 98.2% de los casos con una reducción de costo del **91.6%** y latencia 4x menor. La arquitectura correcta es aquella que satisface el SLA con la mínima huella computacional. 

## Glosario Técnico de Arquitectura de LLMs

Explora los 24 términos y conceptos fundamentales que todo ingeniero de inteligencia artificial debe dominar. Utiliza el buscador o filtra por categorías. 

Fundamentos #01

Token

Unidad mínima de procesamiento textual, que puede ser una palabra completa, un fragmento de palabra o un signo de puntuación; es la pieza sobre la que opera todo el cálculo numérico del modelo durante la inferencia.

Arquitectura #02

Transformer

Arquitectura de red neuronal profunda cuyo componente distintivo es el mecanismo de autoatención; constituye la base de los LLMs modernos como Llama y es responsable de capturar dependencias entre palabras distantes en un texto.

Soberanía #03

Pesos Abiertos (Open Weights)

Conjunto de parámetros entrenados de un modelo disponibles públicamente para su descarga, ejecución local, modificación mediante fine-tuning y auditoría independiente, sin depender de APIs de terceros.

Matemáticas #04

Parámetro

Valor numérico interno, típicamente un número de punto flotante, que forma parte de las matrices del modelo y que se ajusta durante el entrenamiento para definir su comportamiento predictivo final.

Inferencia #05

Inferencia

Fase operativa en la que un modelo ya entrenado recibe una entrada nueva, la procesa a través de sus capas de transformación y genera una salida (predicción de tokens); es el momento en que el modelo "responde" a una consulta.

Entrenamiento #06

LoRA & QLoRA

Técnica de ajuste fino paramétrico eficiente (PEFT) que congela los pesos base y solo entrena matrices pequeñas de bajo rango.

Eficiencia: Permite entrenar Llama 3 con solo 0.1% de los parámetros activos.

Arquitectura #07

RoPE (Rotary Position Embedding)

Técnica de codificación posicional que rota los vectores $Q$ y $K$ en el plano complejo para capturar distancias relativas en secuencias largas.

Fundamento: Base técnica que permite extender el contexto a 128k tokens con frecuencia $\theta = 500,000$.

Inferencia #08

Time-To-First-Token (TTFT)

Tiempo de latencia inicial transcurrido desde que se despacha el prompt hasta que el primer token es proyectado en la pantalla del usuario.

Métrica: Indicador crítico de experiencia de usuario en tiempo real.

Entrenamiento #09

SFT (Supervised Fine-Tuning)

Etapa de entrenamiento supervisado donde el modelo aprende a responder en formato de diálogo humano (asistente) usando pares instrucción-respuesta.

Objetivo: Transforma un modelo base de texto en un asistente conversacional (Instruct).

Entrenamiento #10

DPO (Direct Preference Optimization)

Algoritmo de alineación que optimiza directamente la política del modelo contra preferencias humanas sin requerir un modelo de recompensa complejo.

Ventaja: Más estable y rápido que el clásico RLHF con PPO.

Optimización #11

RAG (Retrieval-Augmented Generation)

Patrón de arquitectura que recupera fragmentos de documentos externos relevantes y los inyecta en el prompt antes de generar la respuesta.

Aplicación: Permite a Llama 3 consultar manuales internos y bases de datos privadas.

Optimización #12

Perplejidad (Perplexity - PPL)

Métrica matemática intrínseca que evalúa la incertidumbre del modelo al predecir texto. Cuanto menor sea la perplejidad, mejor es el modelo.

Métrica: Mide qué tan sorprendido queda el modelo ante una palabra real.

Arquitectura #13

FlashAttention-2

Reimplementación a bajo nivel (CUDA) del cómputo de atención que minimiza las transferencias lentas a la memoria HBM de la GPU.

Rendimiento: Logra aceleraciones de entre 2x y 4x en el entrenamiento de Transformers.

Arquitectura #14

MoE (Mixture of Experts)

Arquitectura donde solo una fracción de las redes neuronales (expertos) se activan para cada token individual, acelerando la inferencia.

Eficiencia: Permite tener modelos de 50B parámetros que gastan como uno de 10B.

Inferencia #15

Top-p (Nucleus Sampling) & Top-k

Estrategias de filtrado de logits: Top-p selecciona el subconjunto más pequeño de tokens cuya suma de probabilidad alcance $p$ (ej. 0.90).

Control: Corta la cola larga de palabras absurdas o incoherentes.

Optimización #16

Pesos Abiertos (Open Weights)

Filosofía de publicación de modelos donde se entregan los parámetros entrenados completos sin costo, permitiendo auditoría y despliegues soberanos.

Filosofía: Ecosistema abierto de Meta con Llama 3 (8B, 70B y 405B).

Inferencia #17

Speculative Decoding

Técnica de inferencia acelerada donde un modelo pequeño (draft) genera rápidamente una ráfaga de tokens y el modelo grande los valida en paralelo en una sola pasada de GPU.

Velocidad: Incrementa la tasa de generación entre 2x y 3x sin pérdida de precisión.

Inferencia #18

Prompt Caching

Mecanismo de reutilización que almacena en caché las claves y valores calculados para el prompt de sistema y documentos fijos, reduciendo la latencia de prefijo a cero.

Ahorro: Reduce el tiempo de Time-To-First-Token (TTFT) hasta en un 90% en chats recurrentes.

Arquitectura #19

SwiGLU (Swish Gated Linear Unit)

Función de activación no lineal adoptada en las capas Feed-Forward de Llama 3, basada en compuertas multiplicativas $\text{Swish}(xW) \otimes (xV)$ que superan a ReLU y GELU.

Calidad: Mejora la convergencia del modelo y la calidad del razonamiento matemático.

Arquitectura #20

RMSNorm (Root Mean Square Normalization)

Variante simplificada de LayerNorm que prescinde de la media y solo normaliza mediante la raíz del promedio cuadrático de las activaciones, ahorrando ciclos de GPU.

Estabilidad: Mantiene los gradientes estables en redes de 405 mil millones de parámetros.

Entrenamiento #21

In-Context Learning (Few-Shot)

Habilidad emergente de los LLMs para comprender patrones de tareas nuevas proporcionando unos pocos ejemplos de entrada-salida en el prompt sin modificar ningún peso neuronal.

Flexibilidad: Permite resolver tareas personalizadas al vuelo sin reentrenar.

Entrenamiento #22

Cross-Entropy Loss

Función de pérdida que penaliza logarítmicamente la discrepancia entre la distribución de probabilidades predicha por el modelo y el token real objetivo durante el preentrenamiento.

Fórmula: $\mathcal{L} = -\sum_{i} y_i \log(\hat{y}_i)$. Minimizarla equivale a maximizar la verosimilitud del lenguaje.

Optimización #23

Chain-of-Thought (CoT)

Estrategia de prompting y entrenamiento que instruye al modelo a descomponer problemas complejos en pasos intermedios de razonamiento antes de emitir el resultado final.

Efecto: Reduce drásticamente los errores en lógica matemática, código y deducción formal.

Arquitectura #24

Long Context Scaling (YaRN)

Método de interpolación de frecuencias de RoPE que permite extender la ventana de contexto de modelos ya entrenados hacia longitudes masivas con mínima degradación de atención.

Escalabilidad: Habilita el análisis de repositorios de código completos y libros de texto extensos.

Evidencia Científica & Recursos Oficiales

## Fuentes de Información Reales & Referencias Académicas

Todo el contenido técnico, fórmulas matemáticas y parámetros presentados en este curso están fundamentados en investigaciones científicas publicadas y repositorios oficiales de código abierto. 

Meta AI Research · 2024 Paper Fundacional

#### The Llama 3 Herd of Models

Documento técnico exhaustivo que detalla el preentrenamiento en más de 15 billones de tokens, la arquitectura GQA, el vocabulario de 128k y los procesos de alineación mediante DPO. 

[ Consultar en arXiv: 2407.21783 ](https://arxiv.org/abs/2407.21783)

Google Brain · 2017 Arquitectura Transformer

#### Attention Is All You Need

El artículo científico que revolucionó la inteligencia artificial introduciendo la auto-atención por producto punto escalado ($Q, K, V$) y eliminando la necesidad de recurrencia secuencial. 

[ Consultar en arXiv: 1706.03762 ](https://arxiv.org/abs/1706.03762)

Google DeepMind · 2022 Leyes de Escala

#### Training Compute-Optimal LLMs (Chinchilla)

Investigación que formula las proporciones matemáticas exactas entre el número de parámetros del modelo y la cantidad óptima de tokens requeridos durante el entrenamiento. 

[ Consultar en arXiv: 2203.15556 ](https://arxiv.org/abs/2203.15556)

Google Research · 2023 Optimización KV Cache

#### Grouped-Query Attention (GQA)

Propuesta técnica adoptada por Meta en Llama 3 para compartir cabezales de llaves y valores, reduciendo el consumo de memoria VRAM en un 75% durante la inferencia. 

[ Consultar en arXiv: 2305.13245 ](https://arxiv.org/abs/2305.13245)

Stanford University · 2023 Alineación de Modelos

#### Direct Preference Optimization (DPO)

Método matemático que permite alinear modelos a partir de retroalimentación humana de forma estable y directa, sustituyendo el complejo pipeline clásico de RLHF con PPO. 

[ Consultar en arXiv: 2305.18290 ](https://arxiv.org/abs/2305.18290)

Zhuravskiy & Su et al. · 2021 Codificación Posicional

#### RoFormer: Enhanced Transformer with RoPE

Introduce Rotary Position Embeddings (RoPE), el método vectorial que permite a Llama 3 generalizar la atención a 128,000 tokens rotando tensores en el espacio euclidiano. 

[ Consultar en arXiv: 2104.09864 ](https://arxiv.org/abs/2104.09864)

Microsoft Research · 2021 Ajuste Fino Eficiente (PEFT)

#### LoRA: Low-Rank Adaptation of LLMs

Demuestra cómo adaptar modelos gigantes entrenando únicamente matrices de bajo rango en las capas de atención, reduciendo los requerimientos de memoria en un 99%. 

[ Consultar en arXiv: 2106.09685 ](https://arxiv.org/abs/2106.09685)

Univ. of Washington · 2023 Cuantización NF4

#### QLoRA: Efficient Finetuning of Quantized LLMs

Introduce la cuantización 4-bit NormalFloat y Double Quantization para permitir el entrenamiento y ajuste fino de modelos de 70B parámetros en GPUs accesibles. 

[ Consultar en arXiv: 2305.14314 ](https://arxiv.org/abs/2305.14314)

Google Research · 2022 Inferencia Acelerada

#### Fast Inference via Speculative Decoding

Estrategia que duplica la velocidad de emisión de tokens al generar borradores con un modelo liviano y validarlos en paralelo con el LLM principal en una sola pasada. 

[ Consultar en arXiv: 2211.17192 ](https://arxiv.org/abs/2211.17192)

Ecosistema Abierto · 2024 Herramientas & Código

#### Repositorios Oficiales & Runtimes

Herramientas indispensables para descarga de pesos, tokenización y ejecución en local: 

[ GitHub Oficial de Meta Llama 3 ](https://github.com/meta-llama/llama3) [ Hugging Face Model Hub ](https://huggingface.co/meta-llama) [ Ollama Local Runtime ](https://ollama.com)

Univ. of Washington · 2023 Cuantización NF4

#### QLoRA: Efficient Finetuning of Quantized LLMs

Introduce la cuantización 4-bit NormalFloat y Double Quantization para permitir el entrenamiento y ajuste fino de modelos de 70B parámetros en GPUs accesibles. 

[ Consultar en arXiv: 2305.14314 ](https://arxiv.org/abs/2305.14314)

Noam Shazeer (Google) · 2020 Activaciones FFN

#### GLU Variants Improve Transformer (SwiGLU)

Demuestra la superioridad de las compuertas lineales Swish (SwiGLU) en capas Feed-Forward sobre ReLU y GELU tradicionales, adoptadas como estándar en la familia Llama. 

[ Consultar en arXiv: 2002.05202 ](https://arxiv.org/abs/2002.05202)

Meta AI Research · 2024 Repositorio Central

#### FairSeq & Open-Source Transformer Evolution

Evolución de los modelos fundacionales de Meta desde FairSeq y OPT hasta la arquitectura actual de Llama 3 con atención agrupada y kernels de FlashAttention. 

[ Consultar FairSeq Repo ](https://github.com/facebookresearch/fairseq)

PyTorch 2.4 Docs Aceleración GPU

#### Scaled Dot-Product Attention (SDPA) Backend Implementations

Documentación sobre la selección automática de kernels CUDA (FlashAttention, Mem-Efficient, Math) en PyTorch para optimizar la velocidad de cálculo de atención. 

[ Consultar PyTorch SDPA ](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

Tokenization Research Algoritmo BPE

#### Byte-Pair Encoding for Large Vocabularies (128k Tokens)

Análisis del impacto de ampliar el vocabulario de 32k a 128k tokens en la tasa de compresión textual para idiomas latinos y código fuente. 

[ Consultar Tokenizer BPE ](https://github.com/openai/tiktoken)

NVIDIA AI Dev · 2024 Compilador de Inferencia

#### NVIDIA TensorRT-LLM: High-Performance GPU Inference

Optimización de grafos de cómputo y fusión de capas para ejecutar Llama 3 con máxima eficiencia de hardware y paralelismo de tensores (TP). 

[ Consultar TensorRT-LLM ](https://developer.nvidia.com/tensorrt-llm)

---

<div align="center">

[🏠 Inicio](../../README.md) • [📁 Módulo 1](README.md) • [Siguiente ➡️](02-prompt-engineering-avanzado-rag.md)

</div>
