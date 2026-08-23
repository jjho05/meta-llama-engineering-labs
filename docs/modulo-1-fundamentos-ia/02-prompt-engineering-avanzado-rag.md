<div align="center">

[🏠 Inicio](../../README.md) • [📁 Módulo 1](README.md) • [⬅️ Anterior](01-arquitectura-transformer-llama3.md) • [Siguiente ➡️](03-fine-tuning-lora-qlora-evaluacion.md)

</div>

---

# Tema 1.2 · Prompt Engineering y Sistemas RAG

Prompt Engineering y RAG con Llama
TL;DR
Resumen ejecutivo: El prompt engineering —mediante técnicas como zero-shot, few-shot y chain-of-thought— optimiza cómo un modelo de lenguaje aplica su conocimiento previo, pero nunca le inyecta datos nuevos. Cuando la respuesta depende de información específica, cambiante o posterior a su entrenamiento, RAG ofrece la solución al recuperar documentos relevantes mediante embeddings y búsqueda semántica, para luego generar una respuesta anclada en hechos actuales y no en memorias estáticas que puedan derivar en alucinaciones.
Prompt Engineering y RAG con Llama: cuándo pedir mejor y cuándo darle documentos reales

¿Qué vas a aprender?
Las tres estrategias básicas de prompting — zero-shot, few-shot y chain-of-thought — y cuándo conviene cada una.
Por qué un modelo nunca sabe algo que no vio en su entrenamiento, sin importar cuán elaborado sea el prompt.
Qué es RAG (Retrieval-Augmented Generation) y cómo resuelve el problema de “el modelo no lo sabe”.
Cómo funcionan los embeddings y la búsqueda semántica para recuperar conocimiento externo.
Tres formas de pedir mejor: zero-shot, few-shot y chain-of-thought
Piensa en el modelo como un chef con años de entrenamiento. El prompt engineering no le agrega ingredientes nuevos a su despensa; solo le indica cómo cocinar mejor con lo que ya tiene. Estas son las tres formas más probadas de darle esa instrucción:

Zero-shot

Intuición: Es como pedirle a un mecánico experto que cambie una llanta sin mostrarle cómo lo hiciste antes; confías en que su entrenamiento le indica el procedimiento correcto.
Técnica: Solicitud directa de una tarea sin proporcionar ejemplos previos de entrada-salida. El modelo depende únicamente de los patrones aprendidos durante su entrenamiento para inferir lo que se le pide.
Few-shot

Intuición: Es como mostrarle a alguien dos o tres fotos de cómo doblar una camisa antes de pedirle que doble toda la pila; calibra el resultado sin necesidad de explicar reglas abstractas.
Técnica: Prompt que incluye una pequeña muestra de ejemplos (entrada-salida) antes de la consulta real, alineando al modelo con un formato o tono específico y reduciendo la ambigüedad de la respuesta.
Chain-of-thought (CoT)

Intuición: Es como exigirle a un estudiante que muestre el desarrollo de una ecuación en el cuaderno en lugar de solo escribir la respuesta final; si el razonamiento falla en el paso 3, detectas el error antes de aceptar el resultado.
Técnica: Instrucción explícita para que el modelo desglose su razonamiento intermedio paso a paso antes de emitir la respuesta final, mejorando drásticamente el desempeño en problemas de lógica, matemáticas y razonamiento multi-paso.
Estas tres técnicas comparten un límite infranqueable: todas trabajan con lo que el modelo ya aprendió durante su entrenamiento. Ninguna le da información nueva; solo mejoran cómo accede a su memoria interna.

El límite del prompt: lo que el modelo nunca vio
Imagina a un bibliotecario que pasó décadas leyendo libros hasta una fecha de corte y luego quedó encerrado en el sótano. Puedes redactar la pregunta más elegante del mundo, pero si el libro se publicó ayer, él jamás lo habrá visto. Los LLMs como Llama funcionan igual: su conocimiento se congela en la fecha límite de sus datos de entrenamiento.

Si le preguntas por una política que se actualizó la semana pasada, no importa qué tan elaborado sea tu prompt: el modelo no tiene esa información. Puede inventar una respuesta que suena convincente; este fenómeno se llama alucinación, y representa el riesgo central de confiar en memoria paramétrica para hechos actualizables o muy específicos.

RAG y embeddings: buscar antes de responder
En lugar de confiar en la memoria del bibliotecario, RAG es como si, antes de responder, le pidieras a un archivero que corra a la biblioteca y traiga los tres libros más relevantes. El experto entonces responde leyendo esos libros en tiempo real. La arquitectura separa el trabajo en dos fases: recuperar y generar.

Retriever (Recuperador)

Intuición: Es el archivero que, ante una pregunta, revisa los estantes y te entrega solo los documentos que tratan el tema, descartando todo lo irrelevante.
Técnica: Componente de búsqueda que explota una base de conocimiento externa para extraer los fragmentos de texto más relevantes a la consulta del usuario, antes de que el modelo genere la respuesta.
Embeddings

Intuición: Es como traducir cada frase a un idioma numérico universal donde “el coche rojo” y “el auto carmesí” suenan casi idéntico, aunque usen palabras distintas.
Técnica: Representación densa de texto en un vector de alta dimensionalidad, generada por un modelo de lenguaje, que codifica significado semántico y permite comparar similitud conceptual más allá de la coincidencia léxica exacta.
Búsqueda semántica

Intuición: Es la diferencia entre buscar una palabra exacta en el índice de un libro y buscar el tema que te interesa aunque el autor use sinónimos o expresiones coloquiales.
Técnica: Método de recuperación que compara la proximidad geométrica entre vectores de embeddings para encontrar contenido relacionado por significado, no por correspondencia literal de términos.
Caso práctico: políticas que cambian cada trimestre
Un equipo de atención a clientes necesita que su asistente responda con las políticas de devolución vigentes, que cambian cada trimestre. En lugar de afinar el prompt esperando que el modelo adivine, construyen un pipeline RAG:

Indexación: Fragmentan el documento de políticas más reciente y generan los embeddings de cada fragmento, almacenándolos en una base vectorial.
Consulta: Cuando llega una pregunta sobre devoluciones, el sistema busca los párrafos relevantes mediante similitud semántica.
Generación: Inyecta esos párrafos como contexto junto a la pregunta en el prompt; Llama genera la respuesta basándose únicamente en ese texto real.
Actualización: Cuando la política cambia, el equipo solo reemplaza el documento indexado. No reentrena el modelo, ni reescribe el prompt, ni toca la lógica del asistente. Así evitan que el sistema responda con una política vieja que ya no aplica.
Ejercicios
1.
Escribe un prompt few-shot para que un modelo clasifique reseñas de productos como Positiva, Negativa o Neutra. Incluye tres ejemplos con su formato de entrada-salida, y termina con una reseña nueva para que clasifique.
2.
Explica por qué una técnica de chain-of-thought podría mejorar la respuesta de un modelo ante la operación matemática: “Un tren lleva 120 pasajeros. Bajan 15 en la primera parada y suben el doble de los que bajaron en la segunda. ¿Cuántos pasajeros hay al final?”. Describe qué pasaría si solo se usara zero-shot.
3.
Imagina que trabajas en una clínica y necesitas que un asistente con Llama responda sobre los efectos secundarios de medicamentos que la autoridad sanitaria actualiza mensualmente. Diseña un esquema de RAG en tres pasos (indexación, recuperación y generación) y justifica por qué los embeddings son más útiles que una búsqueda por palabra clave exacta.
Glosario
Alucinación: Respuesta generada por un modelo que suena coherente y convincente, pero contiene hechos inexactos o inventados. Es el principal riesgo cuando se consulta sobre información posterior al entrenamiento o muy específica.
Chain-of-thought (CoT): Técnica de prompting que obliga al modelo a exhibir su razonamiento intermedio antes de entregar la respuesta final. Resulta esencial para problemas de lógica o cálculo donde un salto directo aumenta la probabilidad de error.
Embedding: Vector numérico de alta dimensionalidad que representa el significado semántico de un texto. Permite que la máquina mida cercanía conceptual entre frases, incluso si no comparten vocabulario idéntico.
Few-shot: Técnica de prompting en la que se proveen al modelo unos pocos ejemplos de entrada-salida antes de la tarea definitiva. Calibra el formato y reduce la ambigüedad sin requerir ajuste de parámetros internos.
RAG (Retrieval-Augmented Generation): Arquitectura que separa la generación de texto en dos fases: primero recupera documentos relevantes de una fuente externa y luego instruye al modelo para responder basándose en ese contexto recuperado.
Zero-shot: Técnica de prompting en la que se solicita una tarea directamente, sin ejemplos previos. El modelo debe resolverla confiando exclusivamente en el conocimiento adquirido durante su entrenamiento.

Fine-tuning y Evaluación de Modelos: ajustar Llama sin reentrenarlo desde cero

---

<div align="center">

[⬅️ Tema Anterior](01-arquitectura-transformer-llama3.md) • [🏠 Inicio](../../README.md) • [📁 Módulo 1](README.md) • [Tema Siguiente ➡️](03-fine-tuning-lora-qlora-evaluacion.md)

</div>
