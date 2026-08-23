<div align="center">

[🏠 Inicio](../../README.md) • [📁 Módulo 1](README.md) • [⬅️ Anterior](03-fine-tuning-lora-qlora-evaluacion.md) • [Siguiente ➡️](../modulo-2-automatizacion-agentes-whatsapp/01-whatsapp-cloud-api-arquitectura-webhooks.md)

</div>

---

# Tema 1.4 · Del Prototipo al Pipeline Productivo con FastAPI y Docker

TL;DR
Resumen ejecutivo: Un modelo que funciona en un notebook de Colab sigue siendo un experimento; solo se convierte en un servicio cuando se despliega detrás de un endpoint consumible. El pipeline completo abarca la preparación de datos, el fine-tuning —por ejemplo con LoRA—, la evaluación y el despliegue mediante una API frecuentemente construida con FastAPI, para que otras aplicaciones puedan consumirlo. Las pruebas end-to-end y la documentación del pipeline son parte del entregable, no pasos opcionales, porque garantizan que todo el flujo funcione junto y que alguien más pueda operarlo sin depender de su creador.
Un modelo en un notebook es un experimento; un modelo detrás de un endpoint es un servicio.

¿Qué vas a aprender?
Las etapas completas de un pipeline de IA, desde los datos crudos hasta un modelo desplegado en un ambiente productivo.
Qué hace un endpoint (por ejemplo construido con FastAPI) y por qué representa la forma estándar de servir modelos en la industria.
Qué significa probar un pipeline end-to-end, y por qué validar el modelo aislado no garantiza que todo el sistema funcione en conjunto.
Por qué documentar el pipeline es parte del entregable profesional, no un paso opcional al final del proyecto.
De datos crudos a modelo servido: las etapas del pipeline
Un pipeline de IA rara vez es solo el modelo. Es una cadena completa que empieza con datos: recolectarlos, limpiarlos y darles exactamente el formato que el modelo espera. Sigue con el ajuste, como el fine-tuning con LoRA. Continúa con la evaluación de salidas, y termina con el despliegue, el momento en que el modelo deja de ser un experimento y se convierte en algo que otra persona o sistema puede usar sin tu supervisión directa. Y a lo largo de todo este recorrido, documentar cada decisión, transformación y dependencia es parte del entregable profesional, no un apéndice que se escribe al final como un trámite.

Cada etapa puede entenderse como un eslabón que transforma el trabajo crudo en valor entregable:

Preparación de datos

Intuición: Imagina que vas a cocinar un platillo complicado. Antes de encender la estufa, compras ingredientes, los lavas, quitas lo dañado y los cortas según la receta. Si saltas este paso, el sabor final será impredecible.
Técnica: Es la etapa de recolección, limpieza y formateo de los datos crudos para que cumplan con el esquema de entrada que el modelo requiere durante el entrenamiento y la inferencia.
Ajuste (fine-tuning)

Intuición: Es como adaptar un traje hecho a la medida: tomas un modelo base genérico y lo ajustas con datos propios para que responda con el estilo, tono o conocimiento específico que tu negocio necesita.
Técnica: Proceso de entrenamiento adicional sobre un modelo preentrenado —por ejemplo usando LoRA (Low-Rank Adaptation)— para especializarlo en una tarea o dominio particular sin reentrenar todos los parámetros desde cero.
Evaluación

Intuición: Antes de abrir un restaurante, haces una cena de prueba con amigos para detectar si algún plato sale salado o frío. No basta con que el chef pruebe cada salsa por separado.
Técnica: Fase de validación sistemática en la que se miden métricas de calidad, precisión o comportamiento del modelo usando un conjunto de datos de prueba que no participó en el entrenamiento.
Despliegue (deployment)

Intuición: Es el momento de abrir las puertas del restaurante al público. La cocina ya funciona, pero ahora necesitas que los comensales puedan hacer pedidos desde afuera sin que tú les expliques cómo se prepara cada platillo.
Técnica: Proceso de poner el modelo o sistema en un ambiente productivo —servidores, contenedores o funciones serverless— donde otros usuarios o aplicaciones puedan consumirlo de manera autónoma y continua.
Empacar el modelo: qué hace un endpoint
Un endpoint es un punto de acceso: una dirección específica a la que otra aplicación puede enviar una solicitud y recibir una respuesta, sin necesidad de saber cómo funciona el modelo por dentro. Piensa en él como el mostrador de pedidos de un restaurante a domicilio: tú pides una hamburguesa por teléfono o app, desconoces totalmente cómo se cocina la carne o se calienta el pan, y recibes el platillo empaquetado en tu puerta. En el mundo del software, tu request es la orden y la response es el platillo terminado.

FastAPI es una de las herramientas más usadas en Python para construir estos endpoints, porque aprovecha las type hints nativas del lenguaje para manejar automáticamente la validación de datos y generar documentación interactiva de la API. Gracias a esto, quien consume tu modelo no tiene que adivinar qué campos enviar ni en qué formato.

En este módulo vas a construir un endpoint que recibe una pregunta, la pasa por tu pipeline de RAG y fine-tuning, y devuelve la respuesta generada por Llama. Ese mismo patrón arquitectónico lo vas a reutilizar cuando conectes tu agente a WhatsApp en el Módulo 2, demostrando que una buena abstracción de endpoint sirve para múltiples canales de comunicación.

Probar antes de confiar: testing del pipeline
Probar cada pieza por separado es como revisar que cada músico de una orquesta toque bien su instrumento en casa. El testing end-to-end es el ensayo general en el teatro: solo ahí descubres que el violín se tapa con la batería, o que la entrada del coro está desfasada. Hasta que no suena todo junto frente a una solicitud real, no sabes si la sinfonía funciona.

Esta prueba simula una petición de principio a fin y verifica tres cosas esenciales: que la respuesta final sea correcta según los criterios de negocio, que el tiempo de respuesta sea aceptable para el usuario, y que el sistema maneje bien los casos donde algo falla en lugar de colapsar en silencio.

Caso práctico: del notebook a la primera prueba real
Un equipo construyó un asistente de Llama con RAG que funcionaba perfecto en su notebook de Colab. Cuando quisieron conectarlo a su primera aplicación de prueba, nadie fuera del notebook podía usarlo: el modelo era un monolito local, sin interfaz externa y con dependencias ocultas.

Construyeron un endpoint con FastAPI que recibe una pregunta, ejecuta la búsqueda RAG, genera la respuesta y la devuelve en un formato estándar. Antes de darlo por terminado, corrieron 15 pruebas end-to-end simulando preguntas reales, incluyendo casos donde el documento buscado no existía, y ajustaron el sistema para responder con un mensaje claro en vez de fallar en silencio.

La diferencia entre ambos mundos se resume fácilmente:

Aspecto	En el notebook	En producción
Acceso	Solo el creador ejecuta celdas	Cualquier aplicación autorizada vía HTTP
Interfaz	Código interactivo	Endpoint con contrato de entrada/salida
Robustez	Depuración manual caso a caso	Manejo automático de errores y tiempos de espera
Ejercicios
1.
Dibuja un diagrama de flujo con las cuatro etapas del pipeline que llevan datos crudos a un modelo servido. Escribe al menos una herramienta o técnica mencionada en la lectura para cada etapa. — Recuerda incluir la preparación de datos, el ajuste, la evaluación y el despliegue.
2.
Explica con tus propias palabras por qué un endpoint de FastAPI se parece al mostrador de pedidos de un restaurante. Identifica qué parte de esa analogía representa la request, la response y la cocina interna del modelo.
3.
Escribe un caso de prueba end-to-end para el asistente RAG del caso práctico. Define una pregunta realista, el resultado esperado y un escenario de fallo controlado (por ejemplo, cuando el documento buscado no existe).
4.
Enumera tres diferencias concretas entre probar una celda de notebook que ejecuta el modelo, y probar el flujo completo que incluye la búsqueda RAG, la generación con Llama y la respuesta del endpoint. ¿Por qué la segunda opción da mayor confianza?
Glosario
Pipeline: Secuencia de etapas (datos, ajuste, evaluación, despliegue) que convierte datos crudos en un modelo funcional y operativo en producción.
Endpoint: Punto de acceso al que una aplicación externa envía una solicitud y recibe una respuesta de un servicio, sin conocer su lógica interna.
FastAPI: Framework de Python usado para construir endpoints con validación automática de datos y generación de documentación interactiva, basado en estándares como OpenAPI.
Testing end-to-end: Prueba que simula una solicitud real de principio a fin, verificando que todo el flujo —datos, modelo y salida— funcione de manera conjunta y robusta.
Despliegue (deployment): Proceso de poner un modelo o sistema en un ambiente donde otros pueden usarlo sin supervisión directa.

---

<div align="center">

[⬅️ Tema Anterior](03-fine-tuning-lora-qlora-evaluacion.md) • [🏠 Inicio](../../README.md) • [📁 Módulo 1](README.md) • [Tema Siguiente ➡️](../modulo-2-automatizacion-agentes-whatsapp/01-whatsapp-cloud-api-arquitectura-webhooks.md)

</div>
