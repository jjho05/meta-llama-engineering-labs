<div align="center">

[🏠 Inicio](../../README.md) • [📁 Módulo 2](README.md) • [⬅️ Anterior](../modulo-1-fundamentos-ia/04-del-prototipo-al-pipeline-productivo.md) • [Siguiente ➡️](02-agentes-conversacionales-memoria-redis.md)

</div>

---

# Tema 2.1 · WhatsApp Cloud API: Arquitectura, Handshake GET y Webhooks

Módulo 2

Automatización Inteligente con WhatsApp

WhatsApp Cloud API
TL;DR
Resumen ejecutivo: La WhatsApp Cloud API traslada la infraestructura de mensajería a los servidores de Meta, liberándote del mantenimiento de un servidor propio. El flujo conversacional se sostiene sobre webhooks, que notifican a tu sistema cuando llega un mensaje, y llamadas a la API de envío, que devuelven la respuesta al usuario. Herramientas como ngrok permiten validar este ciclo bidireccional desde tu máquina local antes de desplegar en producción, mientras que el caso práctico demuestra que siempre conviene asegurar el pipeline con respuestas fijas antes de integrar la inteligencia generativa de Llama.
WhatsApp Cloud API: arquitectura y casos de uso para llevar tu agente de Llama a producción

¿Qué vas a aprender?
Arquitectura alojada: Qué es la WhatsApp Cloud API y en qué se diferencia de la API de Business tradicional.
Ciclo conversacional: Cómo fluye un mensaje entre un usuario de WhatsApp y tu servidor, y por qué ese recorrido es obligatoriamente de dos direcciones.
Webhooks como núcleo: Qué es un webhook y por qué es la pieza central de cualquier integración con WhatsApp.
Estrategia de canal: Qué casos de uso tienen sentido para un agente basado en Llama sobre este canal, y cuándo es mejor buscar otra vía de comunicación.
De la API de negocio al Cloud API: qué cambia
Antes, integrar WhatsApp a nivel de negocio era como tener que construir y mantener tu propia central telefónica en el sótano de la empresa: necesitabas un servidor de WhatsApp Business propio, con el mantenimiento que eso implica. La WhatsApp Cloud API cambia esa ecuación. Es el equivalente a contratar un servicio de telefonía gestionada: Meta aloja la infraestructura directamente y tú te conectas a ella por medio de una API, sin tener que correr ni mantener ese servidor.

Esto reduce drásticamente la complejidad operativa del lado de WhatsApp, pero el trabajo de construir la lógica del agente —recibir el mensaje, decidir qué responder, procesarlo con Llama y contestar— sigue siendo responsabilidad de tu equipo.

La diferencia es clara:

API de Business tradicional: tu equipo opera su propio servidor de WhatsApp Business, asume el mantenimiento y garantiza la disponibilidad de la instancia.
WhatsApp Cloud API: Meta aloja el servicio; tú solo consumes la API para enviar y recibir mensajes.
El flujo de un mensaje: de WhatsApp a tu servidor y de regreso
Imagina una partida de ping-pong. El usuario saca el servido: escribe un mensaje a tu número de WhatsApp. Meta recibe esa bola en su infraestructura de la Cloud API y la devuelve hacia tu campo a través de un webhook. Tu servidor recibe el envío, lee el contenido, lo procesa con la lógica de tu agente (en este curso, usando Llama) y vuelve a lanzar la bola: realiza una llamada a la API de envío de WhatsApp para que Meta entregue la respuesta al usuario.

Es un ciclo bidireccional y obligatorio:

1.
WhatsApp te avisa → Meta envía una notificación HTTP al webhook que configuraste.
2.
Tu servidor procesa → Extrae el texto y los metadatos del payload, ejecuta la lógica del agente y genera una respuesta.
3.
Tú respondes → Tu sistema llama a la API de envío de Meta con el texto generado y el identificador del usuario.
Sin el webhook no hay entrada; sin la llamada a la API no hay respuesta.

Webhooks: cómo tu servidor se entera de que llegó un mensaje
Configurar un webhook no es solo pegar una URL en un panel. Implica un intercambio inicial de verificación: Meta envía una señal de prueba a tu endpoint para confirmar que tu servidor es quien dice ser. Una vez superada esa validación, los eventos reales comienzan a llegar como notificaciones HTTP que tu servidor debe interpretar.

Webhook

Intuición: Es como el timbre de tu casa conectado a la portería de un edificio. No necesitas vigilar la calle todo el día; el portero te avisa solo cuando alguien llega por ti.
Técnica: Es un endpoint HTTP, una URL específica en tu servidor, configurada en Meta for Developers para recibir notificaciones de eventos generados por WhatsApp, como la llegada de un mensaje.
Payload

Intuición: Es como el contenido del sobre que el portero te entrega junto con el aviso. El timbre solo dice "tienes visita", pero dentro del sobre está el nombre del remitente, la hora y el mensaje escrito.
Técnica: Es el cuerpo de datos que Meta envía dentro de la notificación al webhook. Incluye la estructura del mensaje, el número de quien escribió y metadatos adicionales.
ngrok

Intuición: Es como pedirle a un conocido que vive en una dirección pública que te reenvíe el correo mientras terminas de construir tu buzón oficial. Tu casa (tu laptop) no aparece en el mapa público, pero él sí, así que todo pasa por él de forma temporal.
Técnica: Es una herramienta que crea un túnel temporal desde tu máquina local hacia una URL pública, permitiendo que servicios externos como Meta entreguen webhooks a un servidor que aún no está desplegado en un entorno productivo.
Caso práctico: un asistente de estatus de pedidos
Una tienda en línea quiere que sus clientes pregunten “¿dónde está mi pedido?” directamente por WhatsApp y reciban una respuesta inteligente generada por su agente. El equipo sigue una secuencia que prioriza la estabilidad del pipeline sobre la sofisticación del modelo:

1.
Registro en Meta for Developers Crean su cuenta de negocio, verifican el número de WhatsApp y obtienen las credenciales necesarias para usar la API.
2.
Configuración del webhook en desarrollo Levantan un servidor local y usan ngrok para exponerlo a internet con una URL pública temporal. Esa dirección la registran como el endpoint del webhook en el panel de Meta.
3.
Verificación inicial Meta inicia el intercambio de verificación; el servidor responde correctamente para activar la suscripción a eventos.
4.
Validación del ciclo con respuesta fija Un cliente escribe preguntando por su pedido. El servidor recibe el webhook, lee el payload, ignora por completo la lógica de Llama y responde con una respuesta fija. El objetivo aquí es confirmar que el puente bidireccional funciona sin ruido.
5.
Integración de Llama Una vez certificado que el flujo webhook → procesamiento → API de envío es estable y robusto, el equipo conecta el paso intermedio a su instancia de Llama para generar la respuesta real.
Esta disciplina de validar primero la infraestructura y después la inteligencia es lo que distingue un prototipo de una integración lista para producción.

Ejercicios
1.
Compara arquitecturas Dibuja un diagrama mental o en papel que contraste la API de negocio tradicional con la Cloud API. Identifica quién —tu equipo o Meta— asume la responsabilidad del servidor, las actualizaciones de seguridad y la disponibilidad del servicio en cada escenario. — En tu diagrama, señala dónde reside la lógica del agente basado en Llama en ambos modelos. ¿Qué tipo de empresa se beneficia más de eliminar el servidor propio?
2.
Rastrea el payload Imagina que recibes un webhook cuyo payload contiene la información del mensaje. Enumera los tres datos mínimos que tu servidor necesitaría extraer para: — Saber quién envió el mensaje. Saber qué texto contiene. Tener la referencia necesaria para responderle al mismo número usando la API de envío.
3.
Simula el ciclo de desarrollo Ordena correctamente las siguientes acciones del caso práctico y justifica por qué conectar Llama al final reduce el riesgo de errores en la integración. — (A) Exponer el servidor con ngrok. (B) Responder con texto fijo validando el ciclo completo. (C) Configurar el webhook en Meta for Developers. (D) Integrar la generación de respuestas con Llama.
4.
Filtra el canal Describe dos escenarios donde WhatsApp sea un canal natural para un agente conversacional y dos escenarios donde forzar la conversación por este medio sume fricción innecesaria. Piensa en dónde ya vive el usuario y en si el caso requiere notificaciones rápidas, ubicaciones compartidas o mensajes asíncronos que WhatsApp resuelve bien.
5.
Depura el webhook Supón que Meta reporta que no puede entregar eventos a tu URL. Sin inventar herramientas externas, enumera tres causas técnicas probables relacionadas con ngrok o con tu servidor local que deberías verificar antes de revisar tu código de procesamiento.
Glosario
WhatsApp Cloud API: versión de la API de WhatsApp Business alojada por Meta, que elimina la necesidad de operar y mantener tu propio servidor de mensajería.
Webhook: mecanismo por el cual un servicio externo (como WhatsApp) notifica a tu servidor que ocurrió un evento, mediante una petición a un endpoint previamente configurado.
Meta for Developers: plataforma oficial donde se configuran la cuenta de negocio, las credenciales de acceso y las suscripciones a webhooks necesarias para usar la WhatsApp Cloud API.
ngrok: herramienta que expone un servidor que corre en tu máquina local mediante una URL pública temporal, útil para recibir webhooks durante la fase de desarrollo.
Payload: el conjunto de datos que se envía dentro de una solicitud o notificación; en este contexto, el cuerpo que contiene el mensaje entrante y sus metadatos asociados.

Diseño de Agentes Conversacionales: cómo darle memoria y estructura a una conversación con Llama

---

<div align="center">

[⬅️ Tema Anterior](../modulo-1-fundamentos-ia/04-del-prototipo-al-pipeline-productivo.md) • [🏠 Inicio](../../README.md) • [📁 Módulo 2](README.md) • [Tema Siguiente ➡️](02-agentes-conversacionales-memoria-redis.md)

</div>
