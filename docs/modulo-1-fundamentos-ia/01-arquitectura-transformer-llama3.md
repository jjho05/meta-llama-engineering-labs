# Tema 1.1 · Fundamentos de LLMs y Arquitectura de Llama 3

Módulo 1

IA Aplicada con Modelos Abiertos

Fundamentos de LLMs y Arquitectura de Llama: qué hay dentro de un modelo de pesos abiertos
TL;DR
Resumen ejecutivo: Un LLM como Llama es esencialmente un motor de predicción secuencial que opera sobre tokens convertidos en vectores numéricos, utilizando la arquitectura transformer para ponderar relaciones a larga distancia mediante mecanismos de atención. Al ser un modelo de pesos abiertos, Llama permite su descarga, ejecución local, auditoría y adaptación sin intermediarios ni dependencia de APIs cerradas. Finalmente, seleccionar el tamaño del modelo es una decisión de ingeniería que equilibra latencia, costo computacional y cobertura del caso de uso real, donde el prototipo ligero suele resolver la mayoría de las necesidades antes de justificar un escalado.
¿Qué vas a aprender?
Por qué un LLM es, en esencia, un motor de predicción de tokens y por qué la frase “genera texto” describe su operación con más precisión que “entiende texto”.
Cómo la arquitectura transformer explora toda una secuencia al mismo tiempo para decidir qué palabras influencian la siguiente.
Qué significa exactamente que Llama sea un modelo de pesos abiertos y qué libertades técnicas eso otorga a un equipo de ingeniería.
Por qué elegir el tamaño de un modelo es un problema de optimización de recursos, no una carrera por la mayor cantidad de parámetros disponibles.
De palabras a números: qué es un token
Imagina que intentas enviar un mensaje a un extraterrestre que no conoce el alfabeto. En lugar de letras, le envías códigos numéricos únicos para cada fragmento de tu mensaje: una palabra común tiene su propio número, una palabra rara se parte en sílabas con códigos propios, y cada signo de puntuación es una pieza independiente. Esa traducción es el primer paso de todo modelo de lenguaje.

Tokenización

Intuición: Es como cortar un texto en piezas de un juego de mesa de formas irregulares: a veces una pieza contiene una palabra entera como “casa”, otras veces una palabra larga o técnica se fragmenta en pedazos más pequeños, y los espacios o comas son fichas individuales.
Técnica: Proceso previo a la inferencia en el que un algoritmo divide el texto en unidades discretas pertenecientes a un vocabulario predefinido, asignando a cada una un identificador numérico único que el modelo puede procesar matemáticamente.
Embeddings o vectores de token

Intuición: Es como ubicar cada pieza de ese juego en un mapa gigante de cientos de dimensiones: palabras con significados similares quedan naturalmente cerca, de modo que el modelo “siente” la cercanía semántica sin haber visto jamás una letra.
Técnica: Representación densa de alta dimensión, compuesta por números de punto flotante, que codifica propiedades semánticas y sintácticas de un token; estas coordenadas se aprenden durante el entrenamiento y son la única realidad que el modelo procesa internamente.
El modelo nunca ve letras, solo números. Para un LLM, una pregunta en español no es texto: es una secuencia de vectores que viaja por capas de transformaciones matemáticas.

Esto tiene consecuencias directas en la práctica. Cuando configuras un límite de tokens en una llamada a una API, no estás limitando palabras, sino esas piezas numéricas resultantes del corte del texto. Por la estructura de los vocabularios actuales, un mismo párrafo en español suele consumir más tokens que su equivalente en inglés, porque las conjugaciones verbales, las tildes y los géneros generan fragmentos adicionales que en inglés quedan comprimidos en formas más simples. Ese detalle de ingeniería afecta el costo, la latencia y la cantidad de contexto disponible en cada consulta.

El transformer en una frase: atención sobre atención
La arquitectura que hizo posible a Llama se llama transformer, y su pieza central es el mecanismo de atención. Para entender su ruptura con modelos anteriores, piensa en una orquesta donde cada músico, en lugar de leer una partitura secuencial, escucha simultáneamente a todos los demás instrumentos y decide, en tiempo real, a quién prestar atención para tocar su siguiente nota. No hay un director que imponga un orden rígido; la armonía emerge de esas conexiones dinámicas.

Mecanismo de atención (Self-Attention)

Intuición: Imagina leer un párrafo largo y, justo antes de pronunciar la siguiente palabra, hacer un barrido mental de todo lo escrito para ponderar qué fragmentos son útiles en este instante exacto. Una palabra al inicio del texto puede ser determinante para elegir una palabra al final, sin importar la distancia que las separe.
Técnica: Capacidad computacional de la arquitectura transformer para calcular dinámicamente pesos de relevancia entre todos los pares de tokens de una secuencia, permitiendo capturar dependencias a larga distancia sin procesar la información en orden estricto.
Arquitectura Transformer

Intuición: Es como una refinería de petróleo que opera en paralelo: en lugar de ensamblar una frase gota a gota, docenas de capas procesan toda la secuencia al mismo tiempo, purificando el significado capa tras capa hasta que surge la predicción final del siguiente token.
Técnica: Arquitectura de red neuronal profunda basada en mecanismos de autoatención y redes feed-forward organizadas en bloques apilados, capaz de transformar representaciones vectoriales de entrada en representaciones de salida aptas para la generación secuencial de texto.
Esa capacidad de pesar relaciones entre palabras distantes es lo que distingue a un transformer de arquitecturas previas basadas en recurrencia secuencial, y es la razón por la que estos modelos mantienen coherencia temática en textos de miles de tokens.

Pesos abiertos: por qué importa poder descargar el modelo
Cuando usas un modelo cerrado al que solo se accede por API, interactúas con una caja negra: envías texto, recibes texto, pero nunca tienes acceso a los valores internos que dictan esas respuestas. Llama es distinto: Meta publica los pesos del modelo, los números exactos que definen su comportamiento. Es la diferencia entre recibir un plato terminado en un restaurante y recibir la receta completa, los ingredientes medidos y el horno para cocinarla donde tú prefieras.

Parámetros y pesos

Intuición: Son como los tornillos de ajuste fino de un motor de competición: cada uno está calibrado en una posición exacta tras millones de iteraciones de prueba, y en conjunto determinan cómo responde el acelerador ante cada toque del pie.
Técnica: Valores numéricos de punto flotante, típicamente organizados en matrices y tensores, que constituyen la memoria aprendida del modelo; se ajustan durante el entrenamiento mediante descenso del gradiente y dictan la transformación exacta de cada entrada vectorial en una salida probabilística.
Modelo de pesos abiertos

Intuición: Es comparable a recibir los planos completos de un edificio en lugar de solo alquilar un departamento: puedes revisar la tubería, hacer modificaciones estruturales o trasladar la construcción a otro terreno según tus necesidades locales.
Técnica: Publicación de los parámetros entrenados que permite la descarga directa, la ejecución en infraestructura propia, el fine-tuning y la auditoría independiente del comportamiento del modelo sin depender de un proveedor externo.
Esta apertura técnica hace viables proyectos que de otro modo serían imposibles. Para una clínica que necesita procesar historiales médicos sin conexión a internet, para una startup bajo regulaciones estrictas de privacidad que no puede enviar datos a una API externa, o para un investigador que necesita auditar sesgos en las respuestas, tener los pesos abiertos elimina la dependencia de un proveedor. Incluso permite ejecutar versiones de Llama en entornos de GPU gratuita como Google Colab para prototipos, reduciendo la barrera de entrada en fases de experimentación.

Caso práctico: eligiendo el modelo correcto para un asistente de soporte
Un equipo de soporte técnico quiere construir un asistente que responda preguntas frecuentes de clientes. En lugar de desplegar inmediatamente la versión más grande disponible, elige una estrategia de validación incremental.

Primero, define un conjunto de 20 preguntas reales extraídas de tickets históricos. Luego, prototipa con la versión más ligera de Llama ejecutada en Groq, midiendo dos métricas críticas: el tiempo de respuesta (latencia percibida por el usuario) y la tasa de resolución correcta. El resultado es revelador: el modelo pequeño resuelve el 90 % de los casos con una latencia mucho menor. Solo para el 10 % restante —preguntas técnicas con dependencias complejas y contexto extenso— consideran escalar a un modelo con mayor capacidad paramétrica.

La lección no es meramente económica; es de diseño: la arquitectura correcta es la que resuelve tu caso de uso con el menor costo computacional posible.

Ejercicios
1.
Análisis de tokenización: Escribe una misma instrucción en español y en inglés (por ejemplo: "Configura la alarma para las siete de la mañana" / "Set the alarm for seven in the morning"). Identifica visualmente qué palabras del español probablemente se partan en más tokens y explica por qué las conjugaciones y las preposiciones aumentan la cuenta.
2.
Diagnóstico de coherencia: Un asistente virtual confunde el nombre de un producto mencionado al inicio de un párrafo largo con otro similar que aparece al final. Explica qué mecanismo de la arquitectura transformer está siendo insuficiente y propón una variable de diseño (tamaño de contexto, número de capas de atención o estrategia de prompting) que podría mitigar el error.
3.
Decisión de despliegue: Presenta tres escenarios: (a) un hospital rural sin conexión estable, (b) una fintech que audita sesgos en respuestas y (c) un chatbot de prueba en línea para una campaña de marketing. Argumenta para cada uno si un modelo de pesos abiertos como Llama o un modelo cerrado por API es más adecuado, justificando con control de infraestructura, privacidad de datos y capacidad de auditoría.
Glosario
Token: Unidad mínima de procesamiento textual, que puede ser una palabra completa, un fragmento de palabra o un signo de puntuación; es la pieza sobre la que opera todo el cálculo numérico del modelo durante la inferencia.
Transformer: Arquitectura de red neuronal profunda cuyo componente distintivo es el mecanismo de autoatención; constituye la base de los LLMs modernos como Llama y es responsable de capturar dependencias entre palabras distantes en un texto.
Pesos abiertos (open weights): Conjunto de parámetros entrenados de un modelo disponibles públicamente para su descarga, ejecución local, modificación mediante fine-tuning y auditoría independiente, sin depender de APIs de terceros.
Parámetro: Valor numérico interno, típicamente un número de punto flotante, que forma parte de las matrices del modelo y que se ajusta durante el entrenamiento para definir su comportamiento predictivo final.
Inferencia: Fase operativa en la que un modelo ya entrenado recibe una entrada nueva, la procesa a través de sus capas de transformación y genera una salida (predicción de tokens); es el momento en que el modelo "responde" a una consulta.