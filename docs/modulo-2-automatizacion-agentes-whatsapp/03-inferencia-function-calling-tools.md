# Tema 2.3 · Inferencia en Dos Pasos, Function Calling y Validación Pydantic

TL;DR
Resumen ejecutivo: Conectar Llama a WhatsApp significa que el modelo actúa como el motor de razonamiento detrás del canal: recibe el mensaje, decide la respuesta y, cuando hace falta, usa function calling para ejecutar una acción real (como consultar un pedido) en vez de solo generar texto. El ciclo completo, desde el webhook hasta la respuesta enviada, debe medirse en conjunto para que la experiencia se sienta rápida para el usuario.
¿Qué vas a aprender?
Cómo conectar Llama como el motor que decide qué responder a cada mensaje de WhatsApp, sin que el usuario perciba que hay dos sistemas trabajando.
Qué es function calling y por qué permite que el modelo ejecute acciones reales (consultar bases de datos, activar procesos) en lugar de limitarse a inventar texto.
Cómo estructurar el código de tu servidor para que reciba un mensaje, lo enriquezca con contexto, se comunique con el modelo y regrese una respuesta al teléfono.
Qué límites tiene esta integración y cómo diseñarla con validaciones, timeouts y manejo de errores para que sea confiable en producción.
WhatsApp como canal, Llama como motor
Imagina un restaurante de alta cocina: el cliente solo interactúa con el mesero, pero todas las decisiones importantes —qué ingredientes usar, en qué orden, cómo emplatar— las toma el chef en la cocina. WhatsApp cumple el rol del mesero: es el canal por donde entra la pregunta y por donde sale la respuesta. Llama es el chef, el motor de razonamiento: recibe la orden en su lenguaje, interpreta la intención del cliente y decide si puede responder directamente o necesita pedir algo a la bodega (tu código) antes de servir el plato final.

Esta separación es clave: el canal no entiende de contextos ni de lógica de negocio; su trabajo es entregar bits. El modelo, por su parte, no tiene acceso directo a tus sistemas internos a menos que tú se lo otorgues de forma controlada.

Function calling: cuando el modelo decide actuar
Hasta ahora, el flujo básico con Llama era texto entrante, texto saliente. Function calling agrega una capa de inteligencia operativa: le presentas al modelo un catálogo de acciones disponibles, cada una con su nombre, su utilidad y los parámetros que requiere. Cuando el modelo detecta que la intención del usuario exige datos o efectos que no tiene en su memoria, no inventa: emite una señal estructurada pidiendo que se ejecute una función específica con argumentos concretos.

Tu código es quien corre esa función en el mundo real, captura su resultado y se lo devuelve al modelo. Solo entonces Llama genera la respuesta final en lenguaje natural, como si hubiera entendido todo el proceso. Mientras más clara sea la descripción de cada herramienta, mejor decidirá el modelo cuándo usarla y con qué argumentos.

Function calling

Intuición: Es como darle a un asesor un listado de extensiones telefónicas; en lugar de inventar la respuesta, marca la que corresponde, recibe el dato y luego te lo explica con palabras.
Técnica: Capacidad del modelo para generar una llamada estructurada —nombre de función más argumentos tipados— en lugar de texto libre, delegando la ejecución al código externo.
Herramienta expuesta (tool)

Intuición: Es el teléfono de la extensión: tiene una etiqueta que dice a qué departamento pertenece y qué datos necesita para atenderte.
Técnica: Función nativa del sistema, descrita mediante un esquema con nombre, descripción y parámetros, que el modelo puede solicitar invocar.
Motor de razonamiento

Intuición: Es el jefe de piso que decide si puede resolver la duda directamente o necesita pasar al depósito a verificar existencias antes de responder.
Técnica: Rol asignado a Llama dentro de la arquitectura del agente: interpretar la entrada, seleccionar la estrategia de respuesta —texto directo o invocación de herramienta— y sintetizar la salida final.
Definir herramientas en Python: el contrato entre el modelo y tu código
Una herramienta no es magia: es una función común y corriente de Python, pero acompañada de un esquema descriptivo que el modelo puede leer. Ese esquema debe decir, con precisión quirúrgica, qué hace la función, qué parámetros espera, de qué tipo son y cuáles son obligatorios.

Mientras más clara sea esa descripción, mejor decidirá el modelo cuándo usar la herramienta y con qué argumentos. Si describes una función de consulta de pedidos como "obtiene información de compras", el modelo puede confundirla con una de historial de facturas. Si la describes como "recibe un numero_pedido entero y devuelve el estado actual de envío desde el sistema de logística", el modelo tiene suficiente contexto para invocarla solo cuando cuente con ese dato y en esa situación específica.

En la práctica, esto significa que debes mantener separada la definición del contrato (lo que lees al modelo) de la implementación (el código que corre en tu servidor). El primero es un mapa; el segundo es el territorio.

De la pregunta al mensaje de respuesta: el ciclo completo
El ciclo de un agente conectado a WhatsApp es una cadena de pasos donde cada eslabón suma latencia y posibles puntos de fallo. Diseñarlo bien implica verlo como un todo, no como silos independientes.

1.
El webhook recibe el mensaje — WhatsApp Cloud API envía un evento HTTP a tu servidor cada vez que un usuario escribe. Tu código valida la firma del webhook, extrae el número de teléfono, el texto y el ID del mensaje.
2.
Tu servidor arma el contexto — Recupera el historial reciente de la conversación desde tu base de datos o caché. Construye el prompt de sistema junto con las definiciones de las herramientas disponibles.
3.
Llama procesa y decide — El modelo analiza la intención del usuario. Si detecta que necesita una herramienta, devuelve una estructura de llamada en lugar de texto conversacional. Si decide responder directamente, salta al paso 5.
4.
Ejecución de la función y devolución del resultado — Tu servidor valida los argumentos que pidió el modelo, ejecuta la función real (por ejemplo, una consulta SQL o una API interna) y encapsula el resultado. Ese resultado se envía de vuelta a Llama como un mensaje de tipo tool, para que genere la respuesta final.
5.
Envío de la respuesta al usuario — Tu servidor llama a la API de envío de WhatsApp con el texto generado. El mensaje llega al teléfono del usuario, completando el ciclo.
Cada uno de estos pasos consume milisegundos o segundos. Por eso, medir el tiempo total del ciclo es lo único que determina si la experiencia se siente rápida o lenta para quien está escribiendo desde su teléfono.

Midiendo lo que realmente importa: latencia del ciclo
Es tentador optimizar solo el tiempo que tarda Llama en generar tokens, pero eso ignora la realidad del usuario. Para quien espera en el chat, la única métrica que existe es el tiempo que transcurre entre que envió el mensaje y recibió la respuesta.

Fase	Qué incluye	Por qué importa
Recepción (webhook)	Validación, parseo y filtrado del mensaje de WhatsApp	Determina si tu servidor recibió la señal de forma íntegra y segura
Razonamiento (Llama)	Procesamiento del prompt, decisión de herramienta y generación	Suele ser el cuello de botella en tiempo bruto
Acción y respuesta	Ejecución de funciones, espera de bases de datos y envío por Cloud API	Define si la experiencia se siente instantánea o robótica
La latencia del ciclo es la suma de todas estas etapas. Optimizar solo el tiempo que tarda Llama en generar tokens ignora la realidad del usuario: cada fase agrega latencia, y el tiempo total del ciclo es lo único que determina si la experiencia se siente rápida o lenta para quien está escribiendo desde su teléfono. Diseñar para confiabilidad implica poner timeouts en cada etapa y considerar mensajes de cortesía cuando el proceso excede cierto umbral.

Caso práctico: consultar un pedido real desde WhatsApp
Un equipo de e-commerce define una herramienta consultar_pedido(numero_pedido) que consulta directamente el sistema real de pedidos. Cuando un usuario escribe “¿dónde está mi pedido 4521?”, Llama identifica que debe llamar a esa función con el número correcto. El sistema ejecuta la consulta real y regresa el estatus. Llama redacta la respuesta final en un tono natural en vez de simplemente devolver el dato crudo del sistema.

Diseñando para que no falle: límites y confiabilidad
Conectar un modelo de lenguaje a un canal de mensajería y a sistemas reales no es solo un ejercicio de código: es un diseño de sistemas. Estos son los riesgos que debes mitigar desde el primer día:

Validación de argumentos: El modelo puede pedir ejecutar una función con un parámetro vacío, mal tipado o fuera de rango. Tu código debe rechazar la ejecución antes de tocar la base de datos.
Idempotencia del webhook: Si la confirmación de entrega de WhatsApp falla, su sistema puede reintentar el mismo mensaje. Diseña tus herramientas para que ejecutarlas dos veces con la misma entrada no genere efectos secundarios duplicados (por ejemplo, no cobrar dos veces).
Manejo de timeouts: Si la herramienta externa tarda demasiado, el usuario verá el indicador de “escribiendo…” indefinidamente. Establece límites máximos de espera y define respuestas de fallback.
Errores en la API de envío: La WhatsApp Cloud API tiene rate limits y reglas de formato. Un mensaje demasiado largo o con caracteres no permitidos puede rechazarse después de todo el procesamiento. Valida la salida antes de llamar al endpoint de envío.
Ejercicios
1.
Escribe la descripción técnica que le darías al modelo para una herramienta consultar_clima(ciudad, fecha). Define el tipo de cada parámetro y escribe una descripción de una sola oración que evite que el modelo la confunda con una de reserva de hoteles.
2.
Traza el ciclo completo de principio a fin para la frase del usuario: “Quiero mi factura del pedido 9988”. Señala en qué momento exacto ocurre el function calling, qué validación debería hacer tu servidor antes de ejecutar la función y qué tipo de respuesta esperarías de regreso para que Llama redacte el cierre de la conversación.
3.
Sin cambiar el hardware donde corre Llama, propón dos cambios de arquitectura o lógica que reduzcan la latencia del ciclo que percibe el usuario final.
Glosario
Function calling: capacidad de un modelo de indicar que se debe ejecutar una función específica con ciertos argumentos, en vez de responder solo con texto.
Herramienta (tool): función real del código que se expone al modelo, junto con una descripción de qué hace y qué parámetros recibe.
Motor de razonamiento: el rol que cumple Llama dentro de un agente: decidir qué responder o qué acción tomar a partir de una entrada.
Latencia del ciclo: el tiempo total desde que llega un mensaje hasta que se envía la respuesta, incluyendo webhook, modelo y funciones ejecutadas.
API de envío de WhatsApp: la llamada que usa tu servidor para enviar un mensaje de vuelta al usuario a través de la WhatsApp Cloud API.

Proyecto Integrador: qué necesita un agente para pasar de prototipo a producción