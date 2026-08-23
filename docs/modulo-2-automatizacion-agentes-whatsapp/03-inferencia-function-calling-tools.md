<div align="center">

[🏠 Inicio](../../README.md) • [📁 Módulo 2](README.md) • [⬅️ Anterior](02-agentes-conversacionales-memoria-redis.md) • [Siguiente ➡️](04-produccion-seguridad-llama-guard.md)

</div>

---

MÓDULO 2 TEMA 3 · FUNCTION CALLING & HERRAMIENTAS OPERATIVAS

# Integración Llama + WhatsApp

**Del texto pasivo a las acciones en el mundo real**. Convierte a tu modelo en un agente ejecutor mediante Function Calling, define esquemas rigurosos de herramientas en JSON Schema, controla la latencia del ciclo completo por debajo de 3 segundos y blinda tu backend con filtros de idempotencia y reintentos tolerantes a fallos.

Guía de Inicio · Visión del Tema 2.3

### Resumen Ejecutivo (TL;DR) & Filosofía de Acción

#### 1\. Resumen Ejecutivo El Modelo no Ejecuta Código: Devuelve Intenciones Estructuradas

Un LLM jamás debe tener acceso directo a la consola del servidor ni a credenciales de bases de datos. **Llamadas a funciones (Function Calling / Tool Calling)** es el mecanismo estándar donde Llama 3 analiza el mensaje de WhatsApp y, si detecta una intención operativa (ej. reservar mesa o verificar stock), devuelve una estructura estructurada en **JSON con el nombre de la función y sus argumentos tipados**. 

Tu backend en Python/FastAPI recibe ese JSON, ejecuta la consulta real contra la base de datos y le reenvía el resultado al modelo para que redacte la respuesta final. Para que esta experiencia en WhatsApp sea impecable, la latencia de todo el ciclo debe mantenerse **por debajo de 3-4 segundos** , aplicando **idempotencia sobre el`wamid`** para descartar reintentos duplicados de Meta y evitar cobros o reservas duplicadas. 

¿No entendiste? Te lo explico fácil: La comanda del mesero y el cocinero

Imagina que vas a un restaurante. Tú eres el cliente en WhatsApp, el mesero es Llama 3 y el cocinero con la estufa es tu base de datos en Python. **El mesero no cocina la comida** ; toma tu orden en lenguaje natural (_"quiero unos chilaquiles verdes con huevo bien cocido"_), la traduce a una **comanda estandarizada** (`{ "platillo": "chilaquiles", "salsa": "verde", "extra": "huevo_bien_cocido" }`) y se la entrega al cocinero. El cocinero prepara el platillo real y se lo devuelve al mesero, quien te lo sirve con una sonrisa y te explica los detalles amablemente. 

Consejo Pro de Producción: Seguridad Zero-Trust con Tool Calling

Jamás construyas consultas SQL dinámicas concatenando los argumentos del JSON de Llama (ej. `f"SELECT * FROM users WHERE id={args['id']}"`). Valida siempre los argumentos con **Pydantic V2** y ejecuta sentencias parametrizadas (Prepared Statements) con ORMs o `asyncpg` para blindar tu base de datos contra inyecciones SQL indirectas. 

Tema 2.3.1 · Evolución del Agente

### Del Chat Informativo a las Acciones en el Mundo Real

#### 1\. El Límite del Chat Pasivo Responder Texto no Resuelve Problemas

Un chatbot tradicional que únicamente responde _"Nuestro horario de restaurante es de 13:00 a 23:00 y puedes reservar en nuestro sitio web"_ traslada toda la fricción al usuario. 

Un **agente conversacional moderno basado en Llama 3** opera como un recepcionista humano con acceso a la libreta de reservas: interpreta la intención en lenguaje natural, consulta la base de datos en tiempo real, bloquea la mesa disponible y entrega un número de confirmación directamente en WhatsApp. 

$$\Delta_{\text{action}}: (S_t, u_{t+1}) \xrightarrow{\text{NLU / Llama 3}} \mathcal{T}_{\text{call}}(\text{fn\\_name}, \mathbf{x}_{\text{args}}) \xrightarrow{\text{Backend SQL}} \text{Resultado}(\mathcal{R}) \longrightarrow S_{t+1}$$ 

Desglose Matemático de la Transición Operativa 5 fases

$(S_t, u_{t+1})$

**Par Estado Actual y Mensaje Entrante:** Vector de contexto previo más el nuevo requerimiento expresado en lenguaje natural por el usuario en WhatsApp. 

$\mathcal{T}_{\text{call}}$

**Estructura Tipada de Tool Call:** Objeto JSON generado por Llama 3 conteniendo el nombre exacto de la función registrada y el mapa de argumentos clave-valor. 

$\text{Resultado}(\mathcal{R})$

**Respuesta del Backend:** Datos crudos devueltos por la base de datos (ej. `{"mesa_id": 14, "confirmada": true, "folio": "M-8891"}`). 

$S_{t+1}$

**Nuevo Estado Consolidado:** Memoria de sesión enriquecida con la acción completada para que turnos posteriores recuerden la reserva. 

¿No entendiste? Te lo explico fácil: La diferencia entre un folleto y un mayordomo

Un chatbot pasivo es como un **folleto de papel** : solo te muestra texto estático y te dice _"llama al número o ve a la página"_. Un agente con Function Calling es como un **asistente personal ejecutivo** : le dices _"cancela mi vuelo y cámbialo para el martes temprano"_ , y él mismo entra al sistema de la aerolínea, valida si hay asientos, ejecuta el cambio y te envía el nuevo boleto de abordaje. 

Consejo Pro: Nombres de Funciones Autodescriptivos

Elige nombres de funciones con verbos claros en infinitivo y contexto explícito (ej. `consultar_saldo_bancario_cuenta` en lugar de `get_data` o `check`). Llama 3 utiliza la semántica del nombre para decidir si invoca la herramienta. 

Autoevaluación 2.3.1

¿Cuál es la diferencia fundamental entre un chatbot puramente informativo y un agente con capacidad de Function Calling?

#### 2\. Muestreo Guiado por Gramáticas BNF Garantía Matemática de Conformidad JSON

Tradicionalmente, para obtener JSON de un LLM se recurría a pedirle en el prompt _"Responde solo en formato JSON"_ y luego usar `json.loads()` en Python, cruzando los dedos para que el modelo no agregara texto explicativo como _"Aquí tienes tu JSON:"_. Este enfoque probabilístico produce una tasa de error del 2% al 8% en producción. 

La solución moderna de grado industrial es el **Muestreo Guiado (Constrained Decoding)** implementado en motores como _vLLM_ y _Outlines_. Durante cada paso de generación $t$, se aplica una máscara binaria $M_t \in \\{0, -\infty\\}$ sobre los logits del vocabulario: 

$$P(w_t \mid w_{1:t-1}) = ext{softmax}\left( \mathbf{z}_t + \mathbf{M}_t( ext{FSM}( ext{Grammar})) ight)$$ 

Desglose de Máscara de Logits por Autómata Finito 3 conceptos

$\mathbf{M}_t(v) = -\infty$

**Tokens Prohibidos:** Si el autómata espera un número de teléfono tras `"telefono": "`, cualquier token que no sea un dígito o comilla de cierre recibe $-\infty$ de probabilidad, haciendo matemáticamente imposible generar un JSON corrupto. 

$ ext{Cero Penalización}$

**Velocidad Nativa:** La indexación de la FSM se compila una sola vez en el inicio del servidor, manteniendo una velocidad de inferencia superior a 80 tokens por segundo en GPU. 

Tema 2.3.2 · Arquitectura de Dos Pasos

### Llamadas a Funciones en Profundidad: El Ciclo de Dos Invocaciones

#### 1\. El Protocolo Estandarizado Por qué el LLM no Ejecuta Código Directo

Si permitiéramos que un LLM ejecute código en el servidor, un atacante podría manipular el modelo con inyecciones de prompt para borrar tablas completas (`DROP TABLE usuarios`). 

El protocolo de **Tool Calling** separa el razonamiento de la ejecución mediante un ciclo en dos pasos: 

$$\text{Inferencia 1: } \mathcal{M}(\text{Mensaje}, \text{ToolsSchema}) \longrightarrow \text{tool\\_calls: } [\\{\text{name}, \text{args}\\}]$$ $$\text{Ejecución Backend: } \text{Resultado} = \text{Funciones}[\text{name}](**\text{args})$$ $$\text{Inferencia 2: } \mathcal{M}(\text{Historial} \oplus \text{Resultado}) \longrightarrow \text{Respuesta Conversacional WhatsApp}$$ 

Desglose Matemático del Pipeline de Invocación en 2 Pasos 4 fases

$\text{Inferencia 1}$

**Detección y Extracción de Argumentos:** Llama 3 recibe el mensaje del usuario junto con las definiciones JSON Schema y emite tokens especiales de llamada a herramienta (`<|start_header_id|>ipython<|end_header_id|>`). 

$\text{Ejecución Backend}$

**Ejecución Segura Fuera del Modelo:** FastAPI valida los argumentos con Pydantic, consulta la base de datos SQL o API externa y obtiene el payload de respuesta en JSON. 

$\text{Inferencia 2}$

**Síntesis Conversacional Empática:** Llama 3 recibe el resultado de la función bajo el rol `tool` y redacta una respuesta en lenguaje natural adaptada a WhatsApp. 

$\mathcal{M}(\text{Historial} \oplus \text{Resultado})$

**Preservación de Coherencia:** Ambos turnos se registran en el historial de sesión para que en mensajes posteriores el modelo recuerde qué función ejecutó y qué datos arrojó. 

¿No entendiste? Te lo explico fácil: El cajero que consulta el saldo

Cuando vas al banco y pides tu saldo, el cajero de la ventanilla no tiene tu dinero guardado en su bolsillo ni sabe de memoria cuántos pesos tienes. El cajero teclea tu número de cuenta en su computadora (Paso 1), la pantalla le muestra _"$15,400 pesos"_ (Paso 2), y el cajero te dice amablemente: _"Señor Olvera, su saldo disponible es de quince mil cuatrocientos pesos"_ (Paso 3). 

Advertencia Crítica: Cuidado con las Dobles Inferencias Innecesarias

Si la respuesta de la base de datos es un valor simple o una plantilla fija (ej. confirmación de cancelación con código numérico), puedes formatear la respuesta directamente en Python y enviarla a WhatsApp, **saltándote la Inferencia 2**. Esto ahorra 1.2 segundos de latencia y la mitad del costo de GPU. 

#### Código de Producción Pipeline Asíncrono de Invocación de Herramientas

Implementación completa del ciclo en dos turnos con cliente OpenAI-compatible (vLLM / Llama Stack):

Python 3.11 · tool_calling_pipeline.py
    
    
    import json
    from openai import AsyncOpenAI
    
    client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="local")
    
    async def ejecutar_ciclo_tool_calling(mensajes: list, tools: list) -> str:
        # 1. INFERENCIA 1: El modelo decide si llama a una función
        resp_1 = await client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=mensajes,
            tools=tools,
            tool_choice="auto"
        )
        
        msg_1 = resp_1.choices[0].message
        if not msg_1.tool_calls:
            return msg_1.content  # Respuesta directa sin herramientas
    
        # 2. EJECUCIÓN EN BACKEND: Procesamos cada tool_call
        mensajes.append(msg_1)
        for tc in msg_1.tool_calls:
            args = json.loads(tc.function.arguments)
            # Ejecutamos la función de base de datos correspondiente
            if tc.function.name == "consultar_disponibilidad_restaurante":
                resultado = await consultar_db_mesas(args["personas"], args["fecha"], args["hora"])
            
            # Inyectamos el resultado en el historial con role='tool'
            mensajes.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(resultado)
            })
    
        # 3. INFERENCIA 2: Llama 3 sintetiza la respuesta conversacional
        resp_2 = await client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct",
            messages=mensajes
        )
        return resp_2.choices[0].message.content

Autoevaluación 2.3.2

¿Quién ejecuta materialmente la consulta a la base de datos cuando se utiliza Function Calling con Llama 3?

Tema 2.3.3 · Especificación Técnica

### Diseño de Esquemas de Herramientas (Tool Schema) con JSON Schema

#### 1\. Declaración Rigurosa Cómo Sabe el Modelo Qué Herramientas Existen

Al iniciar la inferencia, tu backend adjunta en el payload de la API un arreglo llamado `tools`. Cada herramienta define su **nombre** , una **descripción inequívoca** y un esquema tipado de **parámetros** : 

$$P(\text{ValidSchema} \mid \text{Prompt}, \mathcal{T}_{\text{schema}}) = \prod_{k \in \text{required}} \mathbb{I}(k \in \text{args}) \times \prod_{i} \mathbb{I}\Big(\text{type}(\text{args}[k_i]) = \tau_i\Big)$$ 

Desglose Matemático de Conformidad de Esquema 3 variables

$\mathbb{I}(k \in \text{args})$

**Función Indicadora de Presencia:** Vale 1 si el campo obligatorio (`required`) está presente en el JSON emitido por el modelo, y 0 si fue omitido. 

$\text{type}(\text{args}[k_i]) = \tau_i$

**Verificación de Tipos Estrictos:** Confirma que cada variable corresponda con su tipo formal (ej. `personas` es de tipo `integer` y no `string`). 

$P(\text{ValidSchema}) = 1.0$

**Criterio de Validación Productiva:** Solo si todos los campos requeridos y tipos son válidos, el backend procede a ejecutar la consulta SQL. 

JSON Schema · tool_definitions.json
    
    
    {
      "type": "function",
      "function": {
        "name": "consultar_disponibilidad_restaurante",
        "description": "Verifica si hay mesas disponibles en el restaurante para una fecha, hora y número de comensales.",
        "parameters": {
          "type": "object",
          "properties": {
            "personas": { "type": "integer", "description": "Número de comensales (1 a 12)" },
            "fecha": { "type": "string", "description": "Fecha en formato YYYY-MM-DD" },
            "hora": { "type": "string", "description": "Hora en formato HH:MM militar (ej. 20:00)" }
          },
          "required": ["personas", "fecha", "hora"]
        }
      }
    }

¿No entendiste? Te lo explico fácil: El formulario de aduanas con casillas fijas

Un esquema JSON Schema es como un **formulario oficial de aduanas**. En lugar de dejar una hoja en blanco donde cualquiera escribe garabatos confusos, el formulario tiene casillas cuadradas estrictas: _"Edad (solo números)"_ , _"Fecha de nacimiento (AAAA-MM-DD)"_ y _"¿Lleva alimentos? (Sí/No)"_. Si Llama 3 intenta escribir letras donde va un número, el sistema de validación lo detecta al instante antes de enviar el paquete. 

Consejo Pro: Limita el Catálogo a Menos de 8 Herramientas Simultáneas

Enviar más de 10 herramientas en un solo prompt satura la atención de Llama 3 8B y aumenta el riesgo de selección errónea de función (_Tool Hallucination_). Aplica **enrutamiento contextual de herramientas** : si el usuario está en la etapa de pago, envía solo herramientas financieras; si está consultando menús, envía solo herramientas de catálogo. 

Autoevaluación 2.3.3

¿Por qué la propiedad 'description' dentro del esquema de una función es tan crítica para el éxito del Tool Calling?

Tema 2.3.4 · Rendimiento y Latencia

### Latencia del Ciclo Completo: El Desafío de los 3 Segundos

#### 1\. El Presupuesto de Tiempo La Suma de Cada Milisegundo

En WhatsApp, un retraso superior a 5 segundos provoca que el usuario envíe signos de interrogación o abandone la conversación. La latencia total del ciclo de Function Calling es una cascada: 

Etapa de la Cascada | Tiempo Promedio | Estrategia de Optimización  
---|---|---  
**1\. Webhook de Meta a tu Backend** | 150 - 300 ms | Servidor cercano a los centros de datos de Meta (US-East / US-West).  
**2\. Inferencia 1 (Detección de Tool Call)** | 600 - 1,200 ms | Uso de modelos optimizados (Llama 3.1 8B Instruct cuantizado con vLLM o TensorRT-LLM).  
**3\. Ejecución de Base de Datos / API Externa** | 100 - 400 ms | Índices en bases de datos relacionales y pools de conexiones asíncronas con `asyncpg`.  
**4\. Inferencia 2 (Síntesis de Respuesta)** | 800 - 1,400 ms | Prompts concisos con límite de tokens de salida (`max_tokens: 150`).  
**5\. Despacho a WhatsApp Graph API** | 150 - 300 ms | Uso de clientes HTTP asíncronos persistentes (`httpx.AsyncClient`).  
**TOTAL CICLO COMPLETO** | **1.8s - 3.6s** | **Experiencia fluida y profesional.**  
  
$$T_{\text{total}} = T_{\text{webhook\\_in}} + T_{\text{inferencia\\_1}} + T_{\text{database}} + T_{\text{inferencia\\_2}} + T_{\text{graph\\_api\\_out}} \le 3.5\,\text{s}$$ 

Desglose de los Componentes de Latencia E2E 5 métricas

$T_{\text{webhook\\_in}}$

**Tránsito de Red Entrante:** Latencia HTTPS desde el router de Meta Cloud API hasta tu servidor FastAPI ($\approx 200\text{ms}$). 

$T_{\text{inferencia\\_1}}$

**Evaluación de Herramienta:** Tiempo de generación del JSON de parámetros en GPU ($\approx 900\text{ms}$). 

$T_{\text{database}}$

**Consulta de Negocio:** Lectura/escritura en PostgreSQL o CRM con pool asíncrono ($\approx 150\text{ms}$). 

$T_{\text{inferencia\\_2}}$

**Síntesis Conversacional:** Redacción del texto final para WhatsApp ($\approx 1000\text{ms}$). 

$T_{\text{graph\\_api\\_out}}$

**Despacho Saliente:** Petición POST a `graph.facebook.com/v20.0/.../messages` ($\approx 250\text{ms}$). 

¿No entendiste? Te lo explico fácil: La carrera de relevos 4x100

Imagina una **carrera de relevos 4x100 metros** en las olimpiadas. Si el primer corredor (el webhook) corre en 10 segundos, pero el segundo corredor (la GPU) se tropieza y tarda 30 segundos, el equipo pierde la carrera aunque el corredor final sea velocísimo. Cada milisegundo en la cadena cuenta para no hacer esperar al usuario en WhatsApp. 

Consejo Pro: Mensaje de Estado Intermedio (Typing Indicator)

Si tu API externa tarda más de 2 segundos, despacha un mensaje rápido de estado (_"Consultando disponibilidad en tiempo real, un momento por favor..."_) o envía una señal de `typing_on` a WhatsApp para que el usuario sepa que su petición está en curso. 

Autoevaluación 2.3.4

Si una consulta a un sistema legacy externo tarda 6 segundos en responder, ¿qué técnica de experiencia de usuario en WhatsApp debe aplicarse?

Tema 2.3.5 · Resiliencia e Idempotencia

### Idempotencia con `wamid` y Tolerancia a Fallos

#### 1\. El Problema de los Reintentos de Meta Evitar Cobros o Reservas Dobles

Si tu backend tarda más de 3 segundos en responder `HTTP 200` a Meta, los servidores de WhatsApp asumirán que el paquete se perdió y **reenviarán exactamente el mismo webhook 2 o 3 veces**. 

La **idempotencia** garantiza que una misma operación ejecutada múltiples veces produzca exactamente el mismo resultado sin efectos secundarios duplicados. Cada mensaje de WhatsApp posee un identificador global único: `wamid`. 

$$\text{Filtro en Redis: } \text{SETNX}(\text{"processed:"} + \text{wamid}, \text{"OK"}, \text{EX}=86400)$$ $$\text{Si Llave Ya Existe} \implies \text{Retornar HTTP 200 OK Inmediato (Sin re-ejecutar la función)}$$ 

Desglose del Filtro Atómico de Idempotencia 3 variables

$\text{SETNX}$

**Set If Not Exists (Operación Atómica):** Comando en Redis que almacena la llave únicamente si no existía previamente, devolviendo 1 (éxito) o 0 (ya procesado) en $\approx 1\text{ms}$. 

$\text{wamid}$

**WhatsApp Message ID:** Cadena alfanumérica única universal asignada por Meta a cada mensaje entrante (ej. `wamid.HBgLM...`). 

$\text{EX}=86400$

**Expiración TTL de 24 Horas:** Tiempo de vida suficiente para proteger contra reintentos de Meta liberando memoria RAM automáticamente. 

¿No entendiste? Te lo explico fácil: El sello de 'PAGADO' en una factura

Imagina que vas al banco a pagar la luz. La cajera recibe tu factura, le pone un **sello físico de 'PAGADO'** con tinta indeleble y registra el folio en el libro mayor. Si por error el cartero te vuelve a entregar una copia de la misma factura al día siguiente y vas al banco, la cajera revisa el folio, ve el sello y te dice: _"Esta factura ya fue liquidada ayer, no tiene que volver a pagar"_. El `wamid` es el folio de la factura y Redis es el sello de tinta. 

#### Código de Producción Middleware de Idempotencia con Redis & FastAPI

Patrón de producción para interceptar webhooks repetidos de Meta:

Python 3.11 · idempotency_guard.py
    
    
    import redis.asyncio as aioredis
    from fastapi import FastAPI, Request, Response, status
    
    app = FastAPI()
    redis_client = aioredis.from_url("redis://localhost:6379/0")
    
    async def es_mensaje_duplicado(wamid: str) -> bool:
        # SETNX atómico con TTL de 24 horas
        fue_insertado = await redis_client.set(
            f"processed_wamid:{wamid}", 
            "1", 
            nx=True, 
            ex=86400
        )
        return not fue_insertado  # True si ya existía (es duplicado)
    
    @app.post("/webhook")
    async def recibir_webhook_whatsapp(request: Request):
        body = await request.json()
        wamid = body["entry"][0]["changes"][0]["value"]["messages"][0]["id"]
        
        if await es_mensaje_duplicado(wamid):
            # Respondemos HTTP 200 inmediatamente sin re-ejecutar la lógica
            return Response(content="DUPLICATE_IGNORED", status_code=status.HTTP_200_OK)
            
        # Procesar mensaje con Llama 3 en BackgroundTasks...
        return Response(content="EVENT_RECEIVED", status_code=status.HTTP_200_OK)

Autoevaluación 2.3.5

¿Por qué es obligatorio almacenar el 'wamid' en una caché rápida como Redis al recibir un webhook en WhatsApp?

Tema 2.3.6 · Orquestación Avanzada

### Orquestación Multi-Herramienta y Grafos de Ejecución (DAG)

#### 1\. El Problema de las Herramientas Encadenadas De Consultas Simples a Pipelines Complejos

En casos de uso reales de WhatsApp, un usuario rara vez solicita una acción aislada. Por ejemplo: _"Verifica si tengo saldo, calcula el costo del envío a Guadalajara y si me alcanza, crea la orden de compra"_. Este flujo requiere resolver una cadena de dependencias donde el resultado de la **Herramienta A** condiciona la invocación de la **Herramienta B** y los argumentos de la **Herramienta C**. 

Para evitar que Llama 3 sufra de deriva de objetivos (_Goal Drift_) o ejecute pasos fuera de orden, modelamos la interacción como un **Grafo Acíclico Dirigido (DAG)** de ejecución de tareas: 

Fase del Grafo (DAG) | Herramienta Invocada | Validación Previa Requerida | Manejo en Caso de Error  
---|---|---|---  
**Paso 1: Consulta** | `consultar_saldo_usuario(wa_id)` | Autenticación de sesión en Redis. | Notificar al usuario y detener ejecución.  
**Paso 2: Cotización** | `calcular_tarifa_envio(destino, peso)` | Validación de código postal con Pydantic. | Solicitar aclaración de dirección.  
**Paso 3: Transacción** | `crear_orden_compra(items, total)` | Saldo $\ge$ Total + Envío (Condición Atómica). | Rollback en PostgreSQL con Redlock.  
  
$$P( ext{Éxito Pipeline}) = \prod_{k=1}^{M} P( ext{Tool}_k \mid ext{Context}_k) imes P( ext{Schema}_k = 1) \ge 0.95$$ 

Desglose Matemático de Fiabilidad Multi-Paso 3 variables

$\prod_{k=1}^{M} P( ext{Tool}_k)$

**Probabilidad Conjunta Acumulativa:** Cada herramienta adicional en la cadena multiplica el riesgo de fallo. Si cada paso tiene 95% de éxito, una cadena de 4 herramientas tiene solo $0.95^4 pprox 81.4\%$ de éxito si no se implementan guardrails deterministas. 

$P( ext{Schema}_k = 1)$

**Conformidad Sintáctica Estricta:** Garantía de que los argumentos de salida del paso $k-1$ se mapean sin error de tipos al paso $k$ mediante Pydantic. 

$ ext{Manejo DAG}$

**Puntos de Control (Checkpoints):** Cada paso confirmado persiste su estado en Redis para permitir reanudación si el usuario tarda minutos en contestar. 

Python 3.11 · dag_tool_orchestrator.py
    
    
    import asyncio
    from pydantic import BaseModel
    
    class OrdenPipeline(BaseModel):
        wa_id: str
        items: list[str]
        direccion_envio: str
    
    async def ejecutar_pipeline_orden(datos: OrdenPipeline) -> dict:
        # 1. Ejecución paralela de validación de saldo y cotización
        saldo_task = consultar_saldo_usuario(datos.wa_id)
        envio_task = calcular_tarifa_envio(datos.direccion_envio)
        saldo, tarifa_envio = await asyncio.gather(saldo_task, envio_task)
        
        costo_total = sum([250.0 for _ in datos.items]) + tarifa_envio
        
        # 2. Guardia determinista de saldo antes de transaccionar
        if saldo < costo_total:
            return {"status": "INSUFFICIENT_FUNDS", "faltante": costo_total - saldo}
            
        # 3. Transacción atómica
        orden_id = await crear_orden_compra(datos.wa_id, datos.items, costo_total)
        return {"status": "SUCCESS", "orden_id": orden_id, "total": costo_total}

¿No entendiste? Te lo explico fácil: La línea de ensamblaje en una fábrica de autos

Un grafo DAG es como una **línea de ensamblaje de automóviles**. No puedes pintar la carrocería si el chasis no ha sido soldado, ni puedes colocar los neumáticos antes de los ejes. El orquestador se asegura de que cada robot (herramienta) haga su trabajo en el orden exacto. Si falta una pieza, la línea se pausa ordenadamente en lugar de fabricar un auto roto. 

Consejo Pro: Ejecución Paralela con asyncio.gather para Herramientas Read-Only

Si dos herramientas son independientes y de solo lectura (ej. consultar clima y consultar disponibilidad de vuelos), dispáralas concurrentemente con `asyncio.gather()`. Reducirás la latencia acumulada del paso a la mitad ($pprox 300 ext{ms}$ en lugar de $600 ext{ms}$). 

Autoevaluación 2.3.6

¿Por qué no se debe delegar la decisión de cobrar una tarjeta de crédito exclusivamente al texto generado por un LLM?

Laboratorios Prácticos en Vivo

## Bancos Interactivos del Tema 2.3

Prueba la ejecución de Function Calling en 4 etapas, diseña esquemas JSON Schema, calcula latencias E2E y simula filtros de idempotencia.

Banco 2.3.1 · Simulador Visual de Function Calling en Vivo (Llama 3 Tool Dispatcher)

1\. Webhook Entrante

2\. Inferencia 1 (Tool JSON)

3\. Backend DB Execution

4\. Inferencia 2 & Despacho

Respuesta Entregada en WhatsApp:

Haz clic en "Ejecutar Ciclo de Function Calling" para ver la respuesta en vivo...

10:00 AM · Meta AI

Entregado vía Graph API

Banco 2.3.2 · Diseñador & Validador de Esquemas JSON Schema para Herramientas

Nombre de la Función: Descripción para Llama 3: Consulta el estatus de un envío de e-commerce mediante su número de guía.

Parámetro Clave:

Tipo de Dato: string (Texto) integer (Número) boolean (Verdadero/Falso)

Marcar como parámetro obligatorio (required) 

JSON Schema Generado
    
    
    {}

Banco 2.3.3 · Calculadora de Cascada de Latencia E2E & Presupuesto SLA

SLA Cumplido (< 3.0s)

Inferencia 1 (Detección de Tool): **900 ms** Ejecución DB / API Externa: **250 ms** Inferencia 2 (Síntesis de Respuesta): **1050 ms** Latencia de Red Meta Webhooks (Entrada + Salida): **400 ms**

Latencia Total del Ciclo E2E: **2.60 segundos**

Límite SLA WhatsApp: **3.5s Máximo**

Excelente: Tu ciclo responde en menos de 3 segundos, garantizando una interacción fluida en WhatsApp. 

Banco 2.3.4 · Simulador de Filtro de Idempotencia contra Reintentos de Webhook

Simula el envío de un webhook de cobro/reserva y comprueba cómo Redis intercepta los reintentos duplicados de Meta sin volver a ejecutar la función en la base de datos. 

Banco 2.3.5 · Laboratorio de Resiliencia: Fallos en Herramientas & Auto-Healing

Selecciona un escenario de fallo para inspeccionar la estrategia de auto-recuperación sin colapsar el webhook. 

Autoevaluación Práctica & Análisis de Sistemas

## Ejercicios Prácticos Oficiales del Tema 2.3

Resuelve los 3 desafíos de Function Calling y arquitectura de herramientas del temario oficial. Despliega cada solución para revisar el código y los criterios de ingeniería.

Ejercicio 1

#### Diseño de un Tool Schema para Reservas de Restaurante

Enunciado de la Especificación de Herramienta 

Diseña el esquema JSON Schema completo para una herramienta llamada `crear_reserva_restaurante` que requiera: nombre del cliente, número de personas (entero positivo), fecha y hora en formato ISO 8601, teléfono de contacto y zona del restaurante (opcional: 'interior' o 'terraza'). Justifica los tipos de datos elegidos. 

Ver Solución de Ingeniería Paso a Paso & Tool Schema Oficial

1

##### Definición Formal en JSON Schema

JSON Schema
    
    
    {
      "type": "function",
      "function": {
        "name": "crear_reserva_restaurante",
        "description": "Registra una reserva formal en la base de datos tras confirmar disponibilidad de mesa.",
        "parameters": {
          "type": "object",
          "properties": {
            "nombre_cliente": {
              "type": "string",
              "description": "Nombre y apellido del titular de la reserva."
            },
            "personas": {
              "type": "integer",
              "minimum": 1,
              "maximum": 12,
              "description": "Cantidad de comensales."
            },
            "fecha_hora": {
              "type": "string",
              "format": "date-time",
              "description": "Fecha y hora en formato ISO 8601 (ej. '2026-08-28T20:00:00-06:00')."
            },
            "telefono": {
              "type": "string",
              "pattern": "^\\+[1-9]\\d{1,14}$",
              "description": "Número telefónico en formato E.164 con código de país."
            },
            "zona": {
              "type": "string",
              "enum": ["interior", "terraza"],
              "default": "interior",
              "description": "Zona preferida del restaurante."
            }
          },
          "required": ["nombre_cliente", "personas", "fecha_hora", "telefono"]
        }
      }
    }

2

##### Justificación Arquitectónica de Tipos

• **Tipado Estricto de`personas: integer`:** Si Llama 3 emitiera `"cuatro"` (texto), el ORM de SQLAlchemy fallaría con `DataError: invalid input syntax for type integer`, abortando la transacción.  
• **Formato ISO 8601 en`fecha_hora`:** Previene ambigüedades de zona horaria (UTC vs Local) y asegura compatibilidad con índices temporales B-Tree en PostgreSQL.  
• **Regex en`telefono`:** Garantiza que el número sea invocable mediante WhatsApp Graph API sin errores de enrutamiento telefónico. 

Ejercicio 2

#### Cálculo y Optimización de la Latencia del Ciclo Completo

Enunciado de Presupuesto de Rendimiento SLA 

Si la primera inferencia de Llama toma 1,200 ms, la consulta a base de datos tarda 400 ms y la segunda inferencia toma 1,300 ms (más 400 ms de red en webhooks), ¿cuál es la latencia total? ¿Qué estrategia implementarías si la base de datos sube su tiempo a 3,000 ms? 

Ver Solución de Ingeniería Paso a Paso & Desglose Waterfall

1

##### Desglose Matemático de la Latencia E2E

$$T_{\text{total}} = T_{\text{in}} + T_{\text{inferencia\\_1}} + T_{\text{DB}} + T_{\text{inferencia\\_2}} + T_{\text{out}}$$ $$T_{\text{total}} = 200\text{ms} + 1200\text{ms} + 400\text{ms} + 1300\text{ms} + 200\text{ms} = 3,300\text{ms} \quad (3.30\text{ segundos})$$ 

Una latencia de 3.3s se encuentra en el límite de la tolerancia conversacional en mensajería móvil. 

2

##### Estrategia Arquitectónica ante Latencia de DB Elevada ($T_{\text{DB}} = 3.0\text{s}$)

Sin optimización, la latencia total escalaría a **5.90 segundos** , provocando que el usuario piense que el bot se trabó o vuelva a escribir. 

**1\. Mensaje de Estado Intermedio (Sub-300ms):**

El webhook despacha de inmediato un mensaje efímero: _"Consultando disponibilidad en cocina, un momento por favor..."_.

**2\. Desacoplamiento Asíncrono con BackgroundTasks / Celery:**

El backend libera el hilo del servidor HTTP y procesa la DB + Inferencia 2 en segundo plano, enviando la confirmación final por Graph API cuando el cómputo finaliza.

Ejercicio 3

#### Diagnóstico de Fallo y Auto-Corrección en Ejecución de Herramientas

Enunciado de Resiliencia y Auto-Healing 

Llama genera un JSON con `personas: "cuatro"` (string) cuando la función exige `personas: 4` (integer). Explica qué capa debe atrapar este error de validación y cómo debe reaccionar el agente sin colapsar el servicio ni romper el webhook de WhatsApp. 

Ver Solución de Ingeniería Paso a Paso & Patrón de Reintento de Dos Pasos

1

##### Capa de Validación con Pydantic v2

El error debe ser capturado por la capa de validación del backend (FastAPI/Pydantic) **antes** de invocar a la base de datos: 

Python 3.11
    
    
    from pydantic import BaseModel, ValidationError
    
    class ReservaArgs(BaseModel):
        nombre_cliente: str
        personas: int
        fecha_hora: str
    
    try:
        args = ReservaArgs.model_validate_json(raw_tool_args_str)
    except ValidationError as err:
        # Captura controlada del error sin lanzar HTTP 500
        error_msg = f"Error en argumento 'personas': Se recibió un string, pero se requiere un entero numérico (ej. 4)."

2

##### Ciclo de Auto-Corrección (Self-Healing Loop)

En lugar de romper el servicio, el backend inyecta un mensaje de rol `tool` con el error estructurado: `{"tool_call_id": "call_123", "content": "ValidationError: 'personas' debe ser entero"}`. En la siguiente pasada, Llama 3 detecta su propia discrepancia y corrige automáticamente el payload a `{"personas": 4}` o formula una pregunta clarificatoria amigable al cliente. 

Ejercicio 4

#### Enrutador Dinámico de Herramientas con Backoff Exponencial y Fallback

Enunciado de Resiliencia Transaccional 

Implementa un middleware en Python con `tenacity` que ejecute llamadas a APIs externas activadas por Llama 3 con reintentos exponenciales ($2^k \times 100\text{ms}$) y devuelva una respuesta degradada controlada (Graceful Degradation) si el servicio de terceros no responde en menos de 2.5 segundos. 

Ver Solución de Ingeniería Paso a Paso & Pipeline con Tenacity

1

##### Decorador de Reintentos Asíncronos con Jitter

Evitamos el efecto de manada (Thundering Herd) con reintentos aleatorizados: 
    
    
    from tenacity import retry, stop_after_attempt, wait_random_exponential
    
    @retry(wait=wait_random_exponential(multiplier=0.1, max=1.0), stop=stop_after_attempt(3))
    async def ejecutar_herramienta_resiliente(fn_name: str, args: dict):
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.post(f"https://api.negocio.com/{fn_name}", json=args)
            resp.raise_for_status()
            return resp.json()

2

##### Manejo del Fallback Conversacional en Llama 3

Si se agotan los 3 intentos, se inyecta un mensaje `tool` estructurado con `{"status": "TEMPORARY_UNAVAILABLE", "retry_after_sec": 60}` para que Llama 3 redacte una disculpa profesional ofreciendo alternativas (_"En este momento no pude consultar el sistema de reservas, pero he registrado tu solicitud y un asesor te confirmará en breve"_). 

Ejercicio 5 · Nivel Arquitecto 25 min

#### Circuit Breaker & Gateway Multi-Herramienta con Fallback Graceful

Diseña un enrutador de herramientas asíncrono en FastAPI que implemente el patrón **Circuit Breaker** (estados: CLOSED, OPEN, HALF-OPEN). Si el endpoint de la pasarela de pagos supera un umbral de 3 fallos consecutivos en 60 segundos, el sistema debe abrir el circuito automáticamente, registrar la incidencia en Prometheus y responder al usuario de WhatsApp con una opción alternativa sin colapsar el agente. 

Ver Implementación Oficial con Circuit Breaker

Python 3.11 · circuit_breaker_gateway.py
    
    
    import time
    import httpx
    from pydantic import BaseModel
    
    class CircuitBreaker:
        def __init__(self, umbral_fallos=3, tiempo_recuperacion=60):
            self.umbral = umbral_fallos
            self.recuperacion = tiempo_recuperacion
            self.fallos = 0
            self.ultimo_fallo = 0
            self.estado = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
    
        async def ejecutar(self, coro_func, *args):
            ahora = time.time()
            if self.estado == "OPEN":
                if ahora - self.ultimo_fallo > self.recuperacion:
                    self.estado = "HALF_OPEN"
                else:
                    return {"status": "FALLBACK", "msg": "Servicio de pagos en mantenimiento temporal. Te enviaremos un link de pago por SMS."}
    
            try:
                resultado = await coro_func(*args)
                if self.estado == "HALF_OPEN":
                    self.estado = "CLOSED"
                    self.fallos = 0
                return resultado
            except Exception:
                self.fallos += 1
                self.ultimo_fallo = ahora
                if self.fallos >= self.umbral:
                    self.estado = "OPEN"
                return {"status": "FALLBACK", "msg": "No pudimos conectar con el banco. Por favor intenta en un momento."}

Diccionario de Conceptos

## Glosario Técnico Oficial · Tema 2.3

Términos clave sobre Function Calling, esquemas tipados y latencia en WhatsApp.

Function Calling (Tool Use)

Mecanismo estructurado donde el modelo analiza una consulta en lenguaje natural y emite un objeto JSON con el nombre de una función y argumentos tipados para su ejecución en backend. 

Ciclo en Dos Pasos (Two-Pass)

Protocolo de invocación desacoplado: Inferencia 1 genera la llamada a herramienta; el servidor ejecuta la lógica de negocio; e Inferencia 2 sintetiza los datos devueltos en lenguaje natural. 

JSON Schema Draft 2020-12

Estándar de definición de tipos y restricciones para objetos JSON que delimita el rango permitido de parámetros, enums y campos obligatorios de cada herramienta. 

Idempotencia con wamid

Propiedad que garantiza que reintentos de red de Meta con el mismo identificador de mensaje (`wamid`) no ejecuten transacciones duplicadas en la base de datos. 

Cascada de Latencia (Latency Waterfall)

Desglose acumulativo del tiempo de respuesta E2E: red entrante + inferencia 1 + base de datos + inferencia 2 + red saliente de Graph API. 

Tool Hallucination

Fallo en el que el modelo inventa nombres de funciones inexistentes o inventa argumentos no declarados en el JSON Schema por falta de especificidad en las descripciones. 

Pydantic V2 Coercion

Conversión segura y tipada de cadenas de texto a enteros, fechas o modelos anidados con motor en Rust, bloqueando valores malformados antes de consultar SQL. 

Circuit Breaker (Cortocircuito)

Patrón de resiliencia que interrumpe temporalmente llamadas a APIs externas colapsadas para evitar el agotamiento de sockets HTTP en el servidor de WhatsApp. 

Prepared Statements SQL Guard

Ejecución parametrizada de sentencias en bases de datos que neutraliza cualquier intento de inyección SQL indirecta presente en los argumentos generados por el LLM. 

Documentación Oficial & Referencias de Ingeniería

## Fuentes de Referencia Oficiales · Tema 2.3

Estándares JSON Schema, especificaciones de invocación de herramientas en Meta Llama 3 y validación de tipos con Pydantic V2.

JSON Schema Org · 2024 Estándar IETF

#### JSON Schema Specification (Draft 2020-12)

Especificación técnica del estándar de validación de objetos, tipos primitivos, campos obligatorios y enums para Function Calling.

[ Consultar json-schema.org ](https://json-schema.org/)

Meta AI · 2024 Guía de Inferencia

#### Meta Llama 3: Tool Use & Function Calling Format

Guía oficial de sintaxis para inyección de esquemas de funciones y procesamiento de tokens especiales del entorno de ejecución ipython.

[ Consultar Llama 3 Prompt Formats ](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/)

Pydantic · 2024 Validación Tipada

#### Pydantic V2: Tool Schema Generation & Type Coercion

Generación automática de esquemas JSON compatibles con OpenAPI a partir de clases BaseModel y validación defensiva en microsegundos.

[ Consultar Docs Pydantic Schema ](https://docs.pydantic.dev/latest/concepts/json_schema/)

PostgreSQL Global Group Base de Datos

#### asyncpg: High-Performance Asynchronous PostgreSQL Pool

Driver de conexión asíncrona de alto rendimiento para ejecución de consultas parametrizadas en milisegundos bajo alta concurrencia.

[ Consultar Documentación asyncpg ](https://magicstack.github.io/asyncpg/current/)

Tenacity Library · Python Resiliencia

#### Retrying Code Execution with Exponential Backoff & Jitter

Librería estándar para manejo de reintentos en llamadas a herramientas externas para evitar colapsos en cascada.

[ Consultar Tenacity Docs ](https://tenacity.readthedocs.io/)

OpenAI API Compatibility Estándar de Inferencia

#### OpenAI Tool Calling API Specification in vLLM

Implementación del endpoint /v1/chat/completions con soporte de herramientas estructuradas para modelos Llama 3.

[ Consultar vLLM Server API ](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

HTTPX Async Client Cliente HTTP

#### HTTPX: Next-Generation Async HTTP Client for Python

Cliente HTTP asíncrono con soporte HTTP/2 y connection pooling para comunicación con Meta Graph API en submilisegundos.

[ Consultar HTTPX Docs ](https://www.python-httpx.org/)

Meta Graph API v20.0 Mensajería Interactiva

#### Interactive Messages: Buttons, Lists & Flow Triggers

Especificación para formatear respuestas generadas por herramientas en botones de selección rápida y listas en WhatsApp.

[ Consultar Interactive Messages ](https://developers.facebook.com/docs/whatsapp/guides/interactive-messages)

Redlock Distributed Lock Algoritmo Distribuido

#### Distributed Locks with Redis (Redlock Algorithm)

Implementación de bloqueos distribuidos para garantizar que llamadas a funciones transaccionales (pagos, reservas) sean estrictamente atómicas.

[ Consultar Redlock Algorithm ](https://redis.io/docs/latest/develop/use/patterns/distributed-locks/)

Outlines / Guidance Gramáticas Estructuradas

#### Constrained Generation & Guided JSON Sampling

Técnicas de muestreo guiado mediante autómatas de estado finito que garantizan 100% de conformidad sintáctica en salidas JSON.

[ Consultar Outlines GitHub ](https://github.com/dottxt-ai/outlines)

OpenAPI 3.1.0 Specification Estándar API

#### OpenAPI Specification & Schema Dialects

Alineación del catálogo de herramientas de Llama 3 con las especificaciones de OpenAPI para autogeneración de SDKs cliente.

[ Consultar OpenAPI Spec ](https://spec.openapis.org/oas/latest.html)

Python Asyncio Pool Concurrencia

#### Asyncio Semaphore & Worker Pools for Tool Execution

Control de concurrencia mediante semáforos asíncronos para proteger APIs externas de saturación por ráfagas de usuarios en WhatsApp.

[ Consultar Asyncio Sync Docs ](https://docs.python.org/3/library/asyncio-sync.html)

PostgreSQL Transactions ACID Transactions

#### PostgreSQL Serializable Isolation for Tool Transactions

Aislamiento de transacciones de base de datos para prevenir anomalías de lectura fantasma en reservas concurrentes de asientos o inventarios.

[ Consultar Postgres Isolation ](https://www.postgresql.org/docs/current/transaction-iso.html)

Meta Llama 3.1 Function Call Paper Técnico

#### Llama 3.1: Tool Use, Python Interpreter & Web Search

Evaluación del rendimiento en benchmarks de Tool Use (BFCL) alcanzando más del 85% de precisión en llamadas complejas multi-función.

[ Consultar Llama 3.1 Research ](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)

Fault Tolerance Systems Ingeniería del Caos

#### Graceful Degradation Patterns for AI Gateways

Diseño de fallbacks inteligentes que ofrecen opciones pre-calculadas al usuario cuando el servicio externo supera el timeout límite.

[ Consultar Martin Fowler Circuit Breaker ](https://martinfowler.com/bliki/CircuitBreaker.html)

Latency Engineering Rendimiento

#### Time to First Token (TTFT) & Inter-Token Latency Tuning

Técnicas de inferencia acelerada con kernels FlashAttention-2 y cuantización AWQ para reducir el tiempo total del ciclo a menos de 2 segundos.

[ Consultar vLLM Kernels ](https://github.com/vllm-project/vllm)

---

<div align="center">

[⬅️ Anterior](02-agentes-conversacionales-memoria-redis.md) • [🏠 Inicio](../../README.md) • [📁 Módulo 2](README.md) • [Siguiente ➡️](04-produccion-seguridad-llama-guard.md)

</div>
