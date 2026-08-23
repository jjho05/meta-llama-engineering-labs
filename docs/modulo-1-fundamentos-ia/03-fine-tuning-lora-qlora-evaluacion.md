<div align="center">

[🏠 Inicio](../../README.md) • [📁 Módulo 1](README.md) • [⬅️ Anterior](02-prompt-engineering-avanzado-rag.md) • [Siguiente ➡️](04-del-prototipo-al-pipeline-productivo.md)

</div>

---

# Tema 1.3 · Fine-Tuning con LoRA / QLoRA y Métricas de Evaluación

TL;DR
Resumen ejecutivo: El fine-tuning completo de modelos como Llama exige recursos inaccesibles para la mayoría de los equipos, por lo que LoRA y QLoRA permiten ajustar el comportamiento entrenando solo una fracción de parámetros nuevos, incluso en una GPU gratuita de Colab. Para validar el resultado, combina métricas automáticas como perplexity y BLEU con un benchmark propio evaluado por humanos o por un modelo auxiliar. Finalmente, recuerda que Llama Guard actúa como una capa de seguridad independiente, no como reemplazo del ajuste del modelo.
¿Qué vas a aprender?
Qué es el fine-tuning y en qué se diferencia de RAG o el prompt engineering.
Por qué reentrenar todos los parámetros de un modelo grande no es viable para la mayoría de equipos.
Cómo LoRA y QLoRA ajustan un modelo entrenando solo una fracción de sus parámetros.
Qué métricas usar para confirmar si un fine-tuning realmente mejoró el modelo.
Fine-tuning completo vs. LoRA: mover 2 parámetros en vez de 2 mil millones
Imagina que compras un edificio de ladrillos ya construido —el modelo preentrenado— y necesitas adaptarlo a una nueva industria. El fine-tuning completo sería como mover paredes de carga, cambiar tuberías y reconfigurar cada espacio: funciona, pero requiere maquinaria pesada, semanas de obra y un presupuesto enorme. Un modelo como Llama tiene miles de millones de parámetros; actualizarlos todos con nuevos datos consume memoria y cómputo fuera del alcance de la mayoría de los proyectos.

LoRA resuelve el problema de otra forma: en lugar de modificar la estructura original, instala módulos ligeros —como muebles modulares inteligentes que redirigen el flujo de trabajo sin tocar los cimientos— en puntos específicos de la arquitectura. Solo esas matrices nuevas se entrenan; el modelo base se queda intacto y congelado.

LoRA (Low-Rank Adaptation)

Intuición: En lugar de reconstruir un edificio, añades muebles modulares que alteran cómo se usa el espacio sin mover una sola pared.
Técnica: Técnica de fine-tuning que inserta y entrena matrices pequeñas de rango reducido en capas específicas de la arquitectura, dejando los pesos originales del modelo completamente congelados.
LoRA no reescribe la obra; añade capas de ajuste que el modelo consulta como notas al margen en un manual técnico.

Fine-tuning completo	LoRA
Parámetros actualizados	Todos los pesos del modelo	Solo las matrices adaptadoras nuevas
Memoria necesaria	Clúster de GPUs de alto rendimiento	Una GPU comercial o gratuita de Colab
Modelo base	Se sobrescribe; difícil de revertir	Se conserva intacto; los adapters son intercambiables
QLoRA: hacerlo caber en una GPU normal
QLoRA lleva la idea un paso más allá. Si LoRA es usar muebles modulares, QLoRA es conseguir que todo el edificio sea portátil comprimiendo los planos y materiales a una versión ligera que ocupa menos camiones de mudanza, sin perder la integridad estructural. Además de usar matrices LoRA, cuantiza el modelo base —reduce la precisión numérica con la que se almacenan sus pesos— para que ocupe mucha menos memoria.

QLoRA

Intuición: Es como mudar una biblioteca entera en vehículos compactos comprimiendo los libros, sin perder la información esencial para reconstruirlos.
Técnica: Variante de LoRA que cuantiza el modelo base para reducir drásticamente el uso de memoria durante el entrenamiento, permitiendo el ajuste en hardware con VRAM limitada.
QLoRA demuestra que no necesitas un supercomputador para enseñarle a un modelo gigante tu jerga específica: basta con precisión reducida y matemáticas de bajo rango.

Esto es lo que hace posible hacer fine-tuning de un modelo grande en una sola GPU gratuita de Colab, en vez de necesitar un clúster.

Cómo saber si el fine-tuning funcionó: métricas
No basta con probarlo y ver si suena bien. Necesitas señales cuantificables que te digan si el modelo realmente aprendió el patrón o simplemente repite frases que parecen convincentes.

Perplexity mide qué tan sorprendido o “confundido” está el modelo al predecir el siguiente token de un texto de referencia. Piensa en un lector experto: si anticipa la siguiente palabra sin esfuerzo, tiene baja perplejidad; si la frase lo desconcierta, la perplejidad se dispara. Técnicamente, cuantifica la incertidumbre promedio del modelo ante un texto; un valor más bajo indica mayor alineación con el dominio de referencia.
BLEU compara la salida generada contra una referencia humana. Es como un examen de traducción donde el profesor contrasta tu frase con una respuesta modelo de un experto, contando coincidencias de secuencias de palabras consecutivas. Originalmente diseñada para traducción automática, resulta útil cuando existe una respuesta canónica esperada.
Benchmarks propios son el estándar de oro para tareas abiertas. Consisten en un conjunto de preguntas representativas de tu caso de uso, evaluadas por humanos o por un segundo modelo. Es equivalente a un examen final diseñado por tu empresa: nadie externo sabe qué conocimiento específico se exige, pero tu equipo sí.
Caso práctico: el tono exacto de un despacho contable
Un despacho contable quiere que Llama responda usando el tono y la terminología exacta de sus dictámenes. Hacen fine-tuning con LoRA usando 200 ejemplos de dictámenes previos, entrenando el ajuste en una sola sesión de Colab con GPU gratuita. El modelo base permanece inmutable; solo crecen las matrices adaptadoras que capturan el estilo profesional del despacho.

Miden el resultado con un benchmark propio: 30 preguntas típicas evaluadas por un contador senior, comparando el modelo ajustado contra el modelo base. El modelo ajustado usa la terminología correcta en el 93% de los casos, contra el 61% del modelo sin ajustar. Esa diferencia no es subjetiva: es una ganancia medida en precisión terminológica que el prompting solo no lograba.

Seguridad sin interferencia: Llama Guard
Es tentador pensar que un modelo ajustado ya es seguro, pero el fine-tuning orientado al estilo no garantiza la seguridad del contenido. Aquí entra Llama Guard, un modelo de clasificación entrenado por separado para detectar contenido problemático —violento, sexual, ilegal o tóxico— tanto en las entradas del usuario como en las salidas generadas.

Piensa en él como un guardia de seguridad en la entrada de un edificio: no modifica la tubería interna ni los grifos, pero intercepta lo que no debe pasar. Funciona de forma independiente al fine-tuning, por lo que puedes ajustar el estilo de tu Llama con LoRA y, al mismo tiempo, mantener a Llama Guard revisando cada interacción en la puerta.

Ejercicios
1.
Comparación de estrategias: Explica con tus palabras por qué el prompt engineering y RAG no modifican los parámetros internos del modelo, mientras que LoRA sí lo hace. ¿En qué escenario preferirías cada enfoque?
2.
Diseño de métrica: Imagina que necesitas evaluar un modelo ajustado para redactar correos formales de atención al cliente. ¿Por qué BLEU podría ser insuficiente si los correos válidos admiten variaciones de tono? ¿Qué rol jugaría un benchmark propio en este caso?
3.
Arquitectura de seguridad: Describe cómo Llama Guard y un modelo ajustado con LoRA pueden coexistir en el mismo sistema sin interferir. ¿Qué ventaja tiene mantener esa capa de seguridad separada del proceso de ajuste?
4.
Análisis de viabilidad: El despacho contable usó 200 ejemplos y una GPU gratuita. Basándote en la diferencia entre 93% y 61% de precisión terminológica, argumenta por qué QLoRA fue decisivo para que un equipo pequeño pudiera obtener esa mejora sin infraestructura empresarial.
Glosario
Fine-tuning: Proceso de ajustar los parámetros de un modelo ya entrenado usando datos nuevos y específicos para una tarea concreta. Es la diferencia entre un modelo generalista y un asesor especializado en tu dominio.
LoRA (Low-Rank Adaptation): Técnica de fine-tuning que entrena matrices pequeñas adicionales sin modificar los pesos originales del modelo. Permite tener múltiples especializaciones en un mismo modelo base.
QLoRA: Variante de LoRA que además cuantiza el modelo base para reducir el uso de memoria. Hace accesible el ajuste de modelos grandes desde hardware modesto.
Perplexity: Métrica que mide qué tan bien predice un modelo el siguiente token de un texto de referencia. Valores menores indican mayor fluidez y alineación con el dominio.
BLEU: Métrica algorítmica que calcula la similitud entre un texto generado y una referencia humana comparando secuencias de palabras consecutivas. Es útil cuando existe una respuesta correcta canónica.
Llama Guard: Modelo separado entrenado para clasificar contenido problemático en entradas o salidas de otro modelo. Actúa como control de calidad independiente de la especialización del modelo principal.


Pipeline Completo: por qué un modelo que funciona en el notebook no es lo mismo que un modelo en producción

---

<div align="center">

[⬅️ Tema Anterior](02-prompt-engineering-avanzado-rag.md) • [🏠 Inicio](../../README.md) • [📁 Módulo 1](README.md) • [Tema Siguiente ➡️](04-del-prototipo-al-pipeline-productivo.md)

</div>
