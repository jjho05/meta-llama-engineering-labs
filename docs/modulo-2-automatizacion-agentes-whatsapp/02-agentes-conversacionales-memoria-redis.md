# Tema 2.2 · Agentes Conversacionales, Memoria de Sesión y Llama Stack

TL;DR
Resumen ejecutivo: Un agente conversacional debe gestionar el estado de la conversación fuera del modelo, ya que Llama no retiene memoria entre llamadas a la API. Este estado se guarda asociado a un identificador de usuario —típicamente su número de teléfono— y se reenvía como contexto dentro de la ventana de contexto en cada nueva llamada. Llama Stack estandariza los componentes comunes de construcción —memoria, herramientas y seguridad— para evitar reinventar la arquitectura en cada proyecto, mientras que un flujo conversacional bien diseñado combina reglas fijas para momentos críticos con la flexibilidad del modelo para interpretar y generar lenguaje natural.
Lo que vas a aprender	La intuición cotidiana	El concepto técnico
Por qué un agente necesita recordar el hilo conversacional	Como un mesero que olvida el pedido al entrar a la cocina y necesita que le recuerden qué mesa pidió qué	Estado de la conversación
Cómo se gestiona el estado entre mensajes de WhatsApp	Como guardar la ficha de un cliente en un archivo etiquetado con su nombre para retomarla después	Identificador de usuario + historial persistente
Qué es Llama Stack y cómo estandariza la construcción de agentes	Como usar un kit de herramientas en vez de forjar cada destornillador a mano	Componentes estándar de arquitectura
Cómo diseñar un flujo que no se sienta rígido	Como un semáforo inteligente que deja fluir el tráfico pero activa una barrera en emergencias	Flujo conversacional híbrido
Un agente conversacional no es solo un modelo que responde mensajes: es un sistema que debe recordar quién habla, qué se dijo antes y hacia dónde va la conversación.

De responder mensajes a sostener una conversación: qué es el estado
Imagina un mesero que, cada vez que entra a la cocina, pierde la memoria por completo. Cuando vuelve al salón, no recuerda qué mesa pidió la cuenta ni quién había pedido sin sal. Eso es, en esencia, cómo opera un LLM por diseño: entre una llamada a la API y la siguiente, el modelo no conserva ningún recuerdo previo. Si un usuario escribe “quiero agendar una cita” y, más tarde, “mejor el jueves”, el modelo necesita que tú le reenvíes explícitamente el mensaje anterior para entender a qué se refiere “el jueves”. Esa memoria de corto plazo, que se construye y mantiene fuera del modelo, es lo que en ingeniería de agentes se denomina estado de la conversación.

Guardar el estado significa almacenar, para cada usuario —identificado de forma única, por ejemplo mediante su número de teléfono en WhatsApp—, el historial reciente de mensajes junto con cualquier dato relevante que se haya recolectado, como el servicio solicitado o la fecha mencionada. Este paquete de información se prepara y se reinyecta en cada nueva llamada al modelo, siempre dentro de los límites de su ventana de contexto. Sin este mecanismo, el agente trataría cada mensaje como si fuera el primero de la conversación, repitiendo saludos, pidiendo datos ya entregados y frustrando al usuario.

Llama Stack: piezas estándar para no reinventar el agente
Construir un agente desde cero obliga a resolver los mismos problemas una y otra vez: cómo gestionar la memoria, cómo definir herramientas que el modelo pueda invocar y cómo aplicar capas de seguridad. Es como si cada vez que quisieras colgar un cuadro, tuvieras que fundir el clavo en un horno en lugar de abrir una caja de herramientas ya probadas. Llama Stack funciona exactamente como esa caja de herramientas: ofrece estos componentes de forma estandarizada para que no tengas que construir cada pieza desde cero cada vez que armas un nuevo agente.

El Stack organiza la arquitectura del agente en bloques reutilizables. Así se desglosan los principales:

Gestión de memoria

Intuición: Como el bloc de notas del mesero que le permite recordar, mesa por mesa, qué se pidió en turnos anteriores sin depender de su memoria biológica.
Técnica: Componente que almacena, recupera y actualiza el historial de mensajes y variables de sesión para inyectarlas como contexto en cada llamada al modelo.
Definición de herramientas

Intuición: Como darle al mesero un teléfono para pedir existencias en vez de obligarlo a fabricar los ingredientes en la cocina.
Técnica: Interfaz estandarizada que le permite al modelo solicitar acciones externas —consultar bases de datos, llamar APIs o ejecutar funciones— y recibir los resultados para integrarlos en su respuesta.
Capas de seguridad

Intuición: Como las reglas de un restaurante que impiden que el mesero revele la receta secreta o acepte propinas inadecuadas.
Técnica: Módulos de control que filtran entradas y salidas para aplicar políticas de uso, reducir riesgos de inyección de instrucciones y garantizar límites éticos en la interacción.
Diseñar el flujo: cuándo dejar que el modelo decida y cuándo poner una regla fija
Un flujo conversacional efectivo no otorga al modelo control absoluto sobre toda la interacción, pero tampoco lo encorseta en un menú rígido de opciones del tipo “presione 1 para facturación”. Piensa en un semáforo inteligente: la mayor parte del tiempo deja que el tráfico fluya interpretando el entorno, pero en una emergencia activa una barrera física sin consultar a nadie. Del mismo modo, ciertas decisiones críticas conviene fijarlas por regla inquebrantable —por ejemplo, qué hacer si el usuario pide hablar con una persona real, si solicita eliminar datos sensibles o si detectas un intento de acceso no autorizado—, mientras que la interpretación del lenguaje natural, la tonada de la respuesta y la navegación por temas generales se dejan a la flexibilidad del modelo generativo.

La clave del diseño está en separar el andamiaje de la conversación de su contenido. Las reglas actúan como rieles de tren que evitan descarrilamientos; el modelo actúa como el conductor que acelera, frena y elige la mejor ruta dentro de esos rieles.

Caso práctico: recordar el servicio correcto en una clínica
Una clínica quiere desplegar un agente en WhatsApp que ayude a agendar citas. Un paciente inicia la conversación escribiendo: “quiero una cita dental”. Dos mensajes después, pregunta: “¿tienen horario el sábado?”, sin repetir que el servicio es dental.

El equipo de desarrollo guarda el estado de la conversación asociado al número de teléfono del paciente. Ese estado incluye el historial reciente y la variable servicio = dental. Antes de cada nueva llamada a Llama, el sistema reenvía ese contexto comprimido pero completo dentro de la ventana de contexto. Gracias a esto, cuando el modelo recibe la pregunta sobre el sábado, ya sabe que se trata de una cita dental y puede responder con precisión sobre la disponibilidad de ese servicio específico, en lugar de pedir aclaraciones o asumir un servicio genérico.

Este patrón ilustra cómo la persistencia del estado transforma una secuencia de mensajes aislados en una verdadera conversación continua.

Ejercicios
1.
Rastrea el estado paso a paso — Revisa la conversación de la clínica e indica qué datos debería contener el estado después de cada mensaje. Identifica qué información nueva se añade y qué se reenviaría en la siguiente llamada al modelo.
2.
Distingue regla fija de flexibilidad del modelo — En cada situación siguiente, decide si debería implementarse como una regla inquebrantable o como una decisión delegada al modelo generativo: Un usuario escribe “quiero hablar con un humano”. Un usuario pregunta “¿qué síntomas debo tener para una consulta de cardiología?”. Un usuario intenta modificar la fecha de una cita ya confirmada.
3.
Analiza el costo de la memoria — Explica en tus propias palabras por qué reenviar todo el historial completo en cada llamada afecta directamente el uso de la ventana de contexto. ¿Qué estrategia podrías diseñar para mantener la coherencia conversacional sin saturar el contexto? Conecta tu respuesta con los principios de prompting eficiente que explorarás más adelante en el curso.
Glosario
Estado de la conversación: Información activa —historial de mensajes, variables recolectadas y metadatos de sesión— que se mantiene entre turnos de una conversación con un usuario. Sin este estado, el agente reiniciaría su comprensión en cada mensaje.
Llama Stack: Conjunto estandarizado de componentes de arquitectura para construir agentes basados en Llama, que agrupa memoria, herramientas y seguridad en módulos reutilizables para evitar construir cada solución desde cero.
Flujo conversacional: Diseño de la conducción de una conversación automatizada, que equilibra reglas fijas para decisiones críticas con la libertad del modelo para interpretar lenguaje natural y generar respuestas contextualizadas.
Ventana de contexto: Cantidad total de texto, medida en tokens, que un modelo puede procesar en una sola llamada; incluye el historial reenviado y la nueva instrucción. Define el límite físico de cuánta memoria de conversación puedes inyectar en cada interacción.
Identificador de usuario: Dato único —como un número de teléfono en WhatsApp o un ID de sesión— que permite asociar el estado de una conversación con la persona correcta y recuperarla entre mensajes asíncronos.

Integración Llama + WhatsApp: cómo conectar el modelo como motor de razonamiento detrás del canal