<div align="center">

[Inicio](../../README.md) • [Módulo 2](README.md) • [⬅️ Anterior](01-whatsapp-cloud-api-arquitectura-webhooks.md) • [Siguiente ️](03-inferencia-function-calling-tools.md)

</div>

---

MÓDULO 2 TEMA 2 · DISEÑO DE AGENTES & MEMORIA DE ESTADO

# Diseño de Agentes Conversacionales

**Cómo darle memoria persistente y estructura a una conversación con Llama 3**. Construye sistemas que recuerden quién habla, qué se dijo antes y hacia dónde va la interacción, estandarizando componentes con Llama Stack y diseñando flujos híbridos entre reglas fijas deterministas y razonamiento generativo.

Guía de Inicio · Visión del Tema 2.2

### Resumen Ejecutivo (TL;DR) & Filosofía de Memoria

#### 1\. Resumen Ejecutivo El Modelo no Recuerda Nada: Estado Externo Soberano

Un agente conversacional de nivel industrial debe gestionar el **estado de la conversación completamente fuera del modelo** , ya que las redes neuronales autorregresivas como Llama 3 son intrínsecamente _stateless_ (sin persistencia de memoria entre peticiones HTTP a la API). Este estado se almacena en una base de datos de alta velocidad (Redis, PostgreSQL o SQLite) indexado por un identificador único de usuario —típicamente su número telefónico de WhatsApp (`wa_id`)— y se inyecta como contexto comprimido en cada nueva llamada dentro de la ventana de contexto. 

**Llama Stack** estandariza los bloques de construcción comunes (gestión de memoria, definición de herramientas y capas de blindaje con Llama Guard 3) para evitar reinventar la arquitectura en cada desarrollo, mientras que un flujo conversacional industrial bien diseñado combina **reglas deterministas inquebrantables** para momentos críticos (emergencias médicas, solicitud de asesor humano, derechos ARCO) con la **flexibilidad de Llama** para interpretar lenguaje natural. 

Mapa de Aprendizaje del Tema 2.2

Lo que vas a aprender | La intuición cotidiana | El concepto técnico  
---|---|---  
**Por qué el agente necesita recordar el hilo** | Como un mesero que olvida el pedido al entrar a la cocina y necesita que le recuerden qué mesa pidió qué. | **Estado de la Conversación (Session State en Redis/SQL)**  
**Cómo se gestiona el estado en WhatsApp** | Como guardar la ficha de un paciente en un archivador etiquetado con su teléfono para retomarla después. | **Identificador de usuario (`wa_id`) + Historial estructurado**  
**Qué es Llama Stack** | Como usar una caja de herramientas estandarizada ISO en vez de forjar cada tornillo y martillo a mano. | **Llama Stack: Componentes estándar de arquitectura (Memory, Tools, Safety)**  
**Cómo diseñar un flujo que no sea rígido** | Como un semáforo inteligente que deja fluir el tráfico pero activa una barrera fija en emergencias. | **Flujo Conversacional Híbrido (Deterministic Rules + LLM Reasoning)**  
  
Consejo Pro: Serialización de Estado en JSON Compacto

Cuando inyectes el estado estructurado de la conversación en el prompt de Llama 3, no reenvíes cadenas de texto redundantes. Usa un JSON compacto con claves cortas (ej. `{"srv":"dental","dt":"2026-08-23","stg":"time_sel"}`). Esto ahorra hasta un **40% de tokens de entrada** en cada turno. 

Tema 2.2.1 · Gestión de Estado

### De Responder Mensajes Aislados a Sostener una Conversación

#### 1\. Fundamentos La Amnesia por Diseño de los Modelos de Lenguaje

Cada vez que ejecutas una inferencia con un LLM, el modelo procesa la secuencia de entrada desde cero. Si el usuario escribe _"Quiero agendar una cita dental"_ y dos minutos después escribe _"Mejor el sábado a las 10"_ , el modelo no tiene la menor idea de qué servicio se está agendando a menos que tu servidor le reenvíe explícitamente el mensaje anterior como contexto. 

El **estado de la conversación** no es solo el historial de texto crudo; incluye variables estructuradas de sesión (slots de negocio, etapa de la máquina de estados, fecha seleccionada, ID de paciente y metadatos de telemetría): 

$$\text{Estado}(S_t) = \Big\langle \text{wa\_id}, \quad \mathcal{H}_t = \big[(u_1, a_1), \dots, (u_t, a_t)\big], \quad \mathcal{E}_t = \\{k_i \mapsto v_i\\}, \quad \sigma_t \in \Sigma \Big\rangle\text{Prompt}_{\text{Llama}}(S_t, u_{t+1}) = \text{SystemPrompt} \oplus \text{Serializar}(\mathcal{E}_t) \oplus \mathcal{W}_K(\mathcal{H}_t) \oplus \text{FormatearUser}(u_{t+1})

$$ 

Desglose Matemático del Vector de Estado 6 variables

$S_t$

**Vector de Estado en el Turno $t$:** Tupla multidimensional que contiene toda la información de contexto necesaria para interpretar el siguiente mensaje sin requerir reentrenamiento del modelo. 

$\text{wa\_id}$

**Clave Primaria de Sesión:** Número telefónico internacional normalizado (E.164) que actúa como clave de indexación en Redis (`session:5215587654321`). 

$\mathcal{H}_t$

**Historial Conversacional:** Secuencia ordenada de pares mensaje de usuario ($u_i$) y respuesta del asistente ($a_i$). 

$\mathcal{E}_t = \\{k_i \mapsto v_i\\}$

**Diccionario de Entidades Extraídas (Slots):** Mapeo clave-valor con variables de negocio (ej. `servicio: "Odontología"`, `fecha: "2026-08-23"`). 

$\sigma_t \in \Sigma$

**Estado de la FSM (Finite State Machine):** Etapa determinista del flujo (ej. `INIT`, `SELECTING_DATE`, `AWAITING_CONFIRMATION`, `COMPLETED`). 

$\mathcal{W}_K(\cdot)$

**Operador de Ventana Deslizante (Sliding Window):** Función que trunca el historial reteniendo únicamente los últimos $K$ turnos relevantes para acotar la latencia y los tokens consumidos. 

¿No entendiste? Te lo explico fácil (Analogía de la vida real)

Imagina que vas al médico especialista. El médico ve a 30 pacientes al día y **no recuerda de memoria tu caso cuando entras**. ¿Cómo sabe qué te dolía la semana pasada? Porque la recepcionista le entrega tu **expediente físico** en la puerta. El médico lee rápidamente los puntos clave del expediente, atiende tu consulta actual, anota los nuevos hallazgos y vuelve a guardar el expediente en el archivador. Llama 3 es el médico, tu backend FastAPI es la recepcionista y Redis es el archivador de expedientes. 

#### Código de Producción Gestor de Estado de Sesión en Python con Redis & Pydantic

A continuación se presenta el código de producción para persistir y recuperar el estado de sesión de WhatsApp con serialización tipada y expiración TTL automática:

Python 3.11 · session_state_manager.py
    
    
    # 1. Importamos librerías para tipado fuerte y cliente Redis asíncrono
    from typing import Dict, List, Optional, Any
    from pydantic import BaseModel, Field
    import json
    import redis.asyncio as aioredis
    
    # 2. Definimos el esquema tipado del estado de la conversación
    class MensajeHistorial(BaseModel):
        role: str  # "user" o "assistant"
        content: str
        timestamp: int
    
    class EstadoConversacion(BaseModel):
        wa_id: str
        etapa: str = "INIT"
        slots: Dict[str, Any] = Field(default_factory=dict)
        historial: List[MensajeHistorial] = Field(default_factory=list)
        ultimo_acceso: int = 0
    
    # 3. Clase controladora para interactuar con la memoria en Redis
    class RedisSessionManager:
        def __init__(self, redis_url: str = "redis://localhost:6379/0", ttl_seconds: int = 86400):
            self.redis = aioredis.from_url(redis_url, decode_responses=True)
            self.ttl = ttl_seconds  # 24 horas de retención
    
        async def obtener_estado(self, wa_id: str) -> EstadoConversacion:
            # 4. Buscamos el estado por clave wa_id
            key = f"session:{wa_id}"
            data = await self.redis.get(key)
            if not data:
                return EstadoConversacion(wa_id=wa_id)
            return EstadoConversacion.model_validate_json(data)
    
        async def guardar_estado(self, estado: EstadoConversacion):
            # 5. Guardamos con serialización JSON y renovamos el TTL
            key = f"session:{estado.wa_id}"
            await self.redis.set(key, estado.model_dump_json(), ex=self.ttl)

Autoevaluación 2.2.1

¿Dónde se almacena y gestiona la memoria a corto plazo de un agente conversacional en WhatsApp?

Advertencia Crítica: Riesgo de Condiciones de Carrera en Mensajes Simultáneos

Si un usuario en WhatsApp envía 3 mensajes de audio o texto en ráfaga en menos de 2 segundos, tu servidor recibirá 3 webhooks simultáneos. Si no aplicas un **bloqueo distribuido (Distributed Lock con Redlock en Redis)** sobre el `wa_id`, los 3 procesos competirán por el estado y sobrescribirán variables de forma caótica. 

Tema 2.2.2 · Estandarización de Infraestructura

### Llama Stack: Piezas Estándar para no Reinventar la Arquitectura

#### 1\. Los 3 Pilares Memoria, Herramientas y Seguridad Estandarizada

Construir agentes ad-hoc suele derivar en código espagueti donde la persistencia, la ejecución de APIs y la moderación se mezclan en un solo archivo. **Llama Stack** estandariza estos tres bloques fundamentales mediante especificaciones abiertas de cliente y servidor: 

1\. Gestión de Memoria

Almacena, recupera y actualiza historiales de conversación, variables de sesión y perfiles de usuario en SQLite, Redis o PostgreSQL con políticas de expiración (TTL) y condensación automática. 

2\. Definición de Herramientas

Interfaz declarativa que expone funciones (consultar base de datos, agendar citas, verificar existencias) con esquemas JSON Schema formalmente validados. 

3\. Capas de Seguridad (Guardrails)

Módulos de inspección que filtran entradas maliciosas (inyecciones de prompt con Prompt Guard) y clasifican contenido inapropiado con Llama Guard 3 en milisegundos. 

$$

T_{\text{Total}} = T_{\text{Shield\_In}} + T_{\text{Redis\_Fetch}} + T_{\text{Prefill}}(\text{Tokens}_{\text{in}}) + T_{\text{Decode}}(\text{Tokens}_{\text{out}}) + T_{\text{Tool\_Exec}} + T_{\text{Shield\_Out}} \le 3.0\,\text{s}

$$ 

Desglose de Latencia & Presupuesto de SLA 5 fases

$T_{\text{Shield\_In}}$

**Evaluación de Entrada Llama Guard 3:** Tiempo para verificar que el mensaje entrante es seguro ($\approx 80\text{ms}$). 

$T_{\text{Redis\_Fetch}}$

**Recuperación de Estado en RAM:** Latencia de lectura en base de datos en memoria ($\approx 2\text{ms}$ a $5\text{ms}$). 

$T_{\text{Prefill}} + T_{\text{Decode}}$

**Inferencia GPU con Llama 3 8B:** Fase de procesamiento de prompt y generación token por token ($\approx 400\text{ms}$ - $800\text{ms}$). 

$T_{\text{Tool\_Exec}}$

**Ejecución de API Externa:** Consulta SQL a base de citas médicas o CRM ($\approx 150\text{ms}$). 

$\le 3.0\,\text{s}$

**SLA Máximo de WhatsApp:** Umbral de calidad percibida por el usuario antes de considerar la experiencia como lenta. 

¿No entendiste? Te lo explico fácil (Analogía de la vida real)

Piensa en los **enchufes eléctricos de tu casa**. En lugar de que cada fabricante de licuadoras, televisores y lámparas invente un cable con una forma diferente de clavija y voltaje, existe una **norma estándar**. Llama Stack es ese enchufe universal: define cómo se conectan la memoria, las herramientas de base de datos y la seguridad, para que puedas cambiar de base de datos o de modelo sin tener que reescribir todo tu sistema desde cero. 

#### Código de Producción Inicialización de Agente con Llama Stack SDK

Implementación oficial utilizando el cliente Llama Stack para registrar proveedores de memoria, herramientas y seguridad unificada:

Python 3.11 · llama_stack_agent_setup.py
    
    
    # 1. Importamos cliente y tipos oficiales de Llama Stack
    from llama_stack import LlamaStackClient
    from llama_stack.types import MemoryConfig, ToolDefinition, SafetyConfig
    
    # 2. Inicializamos cliente apuntando a la instancia local de Llama Stack
    client = LlamaStackClient(base_url="http://localhost:5000")
    
    # 3. Configuramos la memoria de estado desacoplada
    memory_config = MemoryConfig(
        provider="redis",
        session_ttl_seconds=86400,
        max_history_turns=8,
        compression_strategy="sliding_window_with_summary"
    )
    
    # 4. Declaramos herramientas operativas
    tools = [ToolDefinition(
            name="consultar_disponibilidad_citas",
            description="Consulta horarios libres en la agenda dental para una fecha dada."
        ),
        ToolDefinition(
            name="confirmar_cita_en_calendario",
            description="Bloquea y confirma una cita médica para el paciente."
        )]
    
    # 5. Capa de blindaje de contenido y anti-jailbreak
    safety_config = SafetyConfig(
        shield_model="meta-llama/Llama-Guard-3-8B",
        categories_blocked=["S1_Violent_Crimes", "S9_Prompt_Injection", "S14_PII"]
    )
    
    # 6. Creamos la instancia soberana del Agente
    agent = client.agents.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        instructions="Eres el asistente oficial de Dental Clinic en WhatsApp.",
        memory=memory_config,
        tools=tools,
        safety=safety_config
    )

Autoevaluación 2.2.2

¿Cuál es el propósito principal del ecosistema Llama Stack en la ingeniería de agentes de IA?

Tema 2.2.3 · Flujos Conversacionales Híbridos

### Diseñar el Flujo: Cuándo Poner Reglas Fijas y Cuándo Delegar al LLM

#### 1\. El Semáforo Inteligente Separar el Andamiaje de Seguridad del Razonamiento

Un agente industrial jamás otorga control absoluto e irrestricto al modelo probabilístico sobre decisiones críticas, ni encorseta al usuario en un aburrido menú numérico del tipo _"Presione 1 para ventas"_. 

Las **reglas fijas (Guardrails Deterministas)** actúan como rieles inquebrantables que evitan descarrilamientos en situaciones de alto riesgo: 

  * **Solicitud de Atención Humana:** Si el usuario escribe _"quiero hablar con un asesor"_ , se activa inmediatamente la cola de transferencia humana sin consultar al LLM.
  * **Cumplimiento de Privacidad (GDPR / ARCO):** Solicitudes de eliminación de datos se ejecutan con reglas de backend deterministas con validación de identidad.
  * **Emergencias Críticas:** Detección de crisis médicas o reportes de fraude bancario detienen la generación probabilística y despachan ayuda inmediata.
  * **Libertad Generativa para Llama 3:** Interpretación de jerga, reformulación de preguntas complejas, síntesis empática y explicaciones contextuales.

$$

\delta(S_t, u_{t+1}) = \begin{cases} \text{Protocolo}_{\text{Humano}}(u_{t+1}), & \text{si } \text{Regex}_{\text{Human}}(u_{t+1}) = \text{true} \\\ \text{Protocolo}_{\text{ARCO}}(u_{t+1}), & \text{si } \text{Intent}_{\text{Legal}}(u_{t+1}) = \text{true} \\\ \text{Protocolo}_{\text{911\_Emergencia}}(u_{t+1}), & \text{si } \text{Triage}_{\text{Emergency}}(u_{t+1}) = \text{true} \\\ \arg\max_y P(y \mid \text{Prompt}(S_t, u_{t+1}); \theta), & \text{en cualquier otro caso (LLM)} \end{cases}

$$ 

Desglose de la Función de Transición Híbrida 3 caminos

$\delta(S_t, u_{t+1})$

**Función de Transición de Estado Híbrida:** Mapea el estado actual y el mensaje entrante a una acción de sistema o a una inferencia generativa. 

$\text{Regex}_{\text{Human}}$

**Patrones Compilados Inquebrantables:** Filtro ultra-rápido ($< 1\text{ms}$) que captura intenciones de transferencia sin consumir GPU. 

$\arg\max_y P(y \mid \dots)$

**Inferencia Probabilística con Llama 3:** Se ejecuta únicamente cuando el mensaje pasa los filtros de seguridad y requiere comprensión de lenguaje natural. 

¿No entendiste? Te lo explico fácil (Analogía de la vida real)

Imagina un **cajero automático bancario moderno**. Puedes hablar con él o escribirle en lenguaje natural para pedirle _"Quiero un desglose de mis gastos en cafeterías el mes pasado"_ (ahí razona el LLM con total flexibilidad). Pero cuando presionas **"Retirar 500 pesos en efectivo"** o **"Bloquear tarjeta por robo"** , el cajero NO le pregunta a un modelo generativo qué hacer; ejecuta una **orden fija y blindada** en el sistema central del banco. 

#### Código de Producción Enrutador Defensivo Híbrido en Python

Implementación de un despachador defensivo que evalúa reglas deterministas antes de invocar la inferencia de Llama 3:

Python 3.11 · hybrid_conversation_router.py
    
    
    import re
    from typing import Dict, Any
    
    # 1. Compilación de patrones regex de alta prioridad
    HUMAN_REGEX = re.compile(r"\b(humano|asesor|agente|operador|persona real)\b", re.IGNORECASE)
    EMERGENCY_REGEX = re.compile(r"\b(sangrado|infarto|emergencia|ambulancia|grave|asfixia)\b", re.IGNORECASE)
    PRIVACY_REGEX = re.compile(r"\b(eliminar mis datos|borrar cuenta|derecho arco|cancelar servicio)\b", re.IGNORECASE)
    
    async def enrutar_mensaje_hibrido(texto: str, wa_id: str, estado: Dict[str, Any]) -> str:
        t = texto.strip()
    
        # 2. Riel 1: Transferencia Humana Incondicional
        if HUMAN_REGEX.search(t):
            await registrar_ticket_soporte(wa_id, "SOLICITUD_ASESOR")
            return "Te he transferido con un asesor humano de nuestro equipo. En breve se comunicará contigo."
    
        # 3. Riel 2: Protocolo de Emergencia Médica
        if EMERGENCY_REGEX.search(t):
            return "ALERTA MÉDICA: Si tienes una urgencia vital, comunícate de inmediato al 911 o acude al hospital más cercano."
    
        # 4. Riel 3: Protocolo Legal / Privacidad ARCO
        if PRIVACY_REGEX.search(t):
            return "Para ejercer tus derechos de privacidad y eliminación de datos, responde 'CONFIRMAR BORRADO'."
    
        # 5. Riel 4: Delegación al Razonamiento de Llama 3
        return await ejecutar_inferencia_llama3(texto, wa_id, estado)

Autoevaluación 2.2.3

Si un usuario en WhatsApp escribe "Quiero cancelar mi cuenta y que borren mis datos personales", ¿cómo debe procesarse esta solicitud según el diseño de flujos híbridos?

Tema 2.2.4 · Optimización de Cómputo

### Gestión Eficiente de la Ventana de Contexto: Sliding Window & Resumen

#### 1\. El Costo de Reenviar Todo Latencia Cuadrática y Consumo de GPU

Si reenvías todo el historial completo en cada turno, en una conversación de 20 mensajes estarás procesando más de 2,500 tokens por petición. Debido a la naturaleza cuadrática del mecanismo de auto-atención en los Transformers ($O(N^2)$), esto incrementa drásticamente el tiempo de _Prefill_ en la GPU y dispara el consumo de memoria en la caché KV (Key-Value Cache). 

Las dos estrategias industriales para mantener el costo y la latencia constantes son: 

Ventana Deslizante (Sliding Window K)

Conserva intactos únicamente los últimos $K$ turnos (ej. los últimos 4 mensajes) y descarta los turnos antiguos. Mantiene la latencia estrictamente constante $O(K)$.

Memoria con Resumen Jerárquico

Un modelo ligero comprime los turnos antiguos en un párrafo de hechos clave (_"Usuario agendó cita dental para el sábado"_) y reenvía ese resumen condensado junto con los últimos 2 turnos recientes.

$$

C_{\text{total}}(N) = \sum_{t=1}^{N} \Big( T_{\text{sys}} + 2 \cdot t \cdot \bar{L} \Big) = N \cdot T_{\text{sys}} + N(N+1) \cdot \bar{L}

M_{\text{KV\_Cache}}(N) = 2 \cdot L_{\text{layers}} \cdot H_{\text{KV}} \cdot d_{\text{head}} \cdot N \cdot b_{\text{bytes}}

$$ 

Desglose Matemático de Crecimiento Cuadrático y VRAM 3 componentes

$N$

**Número Total de Turnos:** Cantidad de intercambios pregunta-respuesta que han ocurrido en la conversación. 

$\bar{L}$

**Longitud Promedio por Mensaje:** Típicamente $\approx 50$ tokens en mensajes de WhatsApp. 

$N(N+1)\bar{L}$

**Término Cuadrático Peligroso:** Explica por qué para $N=15$ turnos el método ingenuo procesa **14,550 tokens** acumulados, mientras que con Sliding Window ($K=4$) procesa solo **4,500 tokens** (69% de reducción). 

¿No entendiste? Te lo explico fácil (Analogía de la vida real)

Si vas a tener una reunión de trabajo sobre el proyecto actual, **no le pides al equipo que lean completas las 500 páginas de actas de todas las reuniones de los últimos dos años** antes de empezar a hablar. En su lugar, el director lee un **resumen ejecutivo de media página** con las conclusiones clave y se enfoca en los temas de los últimos 15 minutos. Eso es exactamente la compresión jerárquica con ventana deslizante. 

#### Código de Producción Compresor de Historial con Sliding Window y Resumen

Script en Python para truncar y resumir automáticamente el contexto antes de inyectarlo en el prompt de Llama 3:

Python 3.11 · history_compressor.py
    
    
    from typing import List, Dict
    
    def construir_contexto_optimizado(
        historial_completo: List[Dict[str, str]],
        resumen_previo: str = "",
        k_turnos: int = 4
    ) -> str:
        # 1. Si la conversación es corta, conservamos el historial sin truncar
        if len(historial_completo) <= k_turnos * 2:
            lineas = [f"{m['role'].upper()}: {m['content']}" for m in historial_completo]
            return "\n".join(lineas)
    
        # 2. Aplicamos Sliding Window sobre los últimos K turnos recientes
        turnos_recientes = historial_completo[-(k_turnos * 2):]
        
        # 3. Ensamblamos el resumen consolidado + ventana reciente
        prompt_memoria = f"RESUMEN DE HECHOS PREVIOS:\n{resumen_previo}\n\nÚLTIMOS MENSAJES RECIENTES:\n"
        for m in turnos_recientes:
            prompt_memoria += f"{m['role'].upper()}: {m['content']}\n"
    
        return prompt_memoria

Autoevaluación 2.2.4

¿Qué consecuencia técnica indeseable ocurre si un sistema reenvía siempre el 100% del historial de chat sin aplicar compresión ni ventana deslizante?

Tema 2.2.5 · Optimización de Memoria

### Compresión de Contexto, Evicción LRU y Gestión de VRAM

#### 1\. El Límite de la Memoria Gestión de Sesiones a Gran Escala

Aunque Meta Llama 3.1 soporta de forma nativa hasta **128,000 tokens** de ventana de contexto, inyectar historiales gigantescos en cada mensaje de WhatsApp es una mala práctica de ingeniería: incrementa el tiempo de procesamiento de la fase de _Prefill_ ($O(N^2)$ en atención), satura la memoria VRAM del servidor GPU y encarece la infraestructura. 

Para soportar miles de usuarios concurrentes de forma eficiente, implementamos una **estrategia de memoria en 3 niveles (Three-Tier Memory Architecture)** : 

Nivel de Memoria | Tecnología de Almacén | Tiempo de Retención | Función en el Agente de WhatsApp  
---|---|---|---  
**Nivel 1: Memoria de Trabajo (Hot)** | Ventana Deslizante en Redis RAM | Últimos 4 turnos ($pprox 400$ tokens) | Preserva la inmediatez y coherencia del diálogo en curso.  
**Nivel 2: Resumen Estructurado (Warm)** | Hash JSON en Redis | TTL de 24 horas tras último mensaje | Almacena variables clave (nombre, intención, saldo, producto).  
**Nivel 3: Historial Histórico (Cold)** | PostgreSQL / SQLite WAL | Permanente (Meses / Años) | Auditoría forense, analítica de negocio y reanudación a largo plazo.  
  
$$

 ext{Memoria RAM}_{ ext{Total}} = U_{ ext{activos}} imes \left( S_{ ext{sliding\_window}} + S_{ ext{summary\_json}} ight) \le ext{MaxRAM}_{ ext{Redis}} imes 0.75

$$ 

Desglose de Capacidad de Memoria Concurrente 3 variables

$U_{ ext{activos}}$

**Usuarios Concurrentes:** Número de conversaciones simultáneas en ventana de 24 horas. Para 100,000 usuarios con 2KB por sesión, Redis consume solo $pprox 200 ext{MB}$ de RAM. 

$ ext{Política volatile-lru}$

**Evicción Automática:** Si la memoria alcanza el 75% de capacidad, Redis descarta las sesiones inactivas más antiguas sin interrumpir los chats en curso. 

Python 3.11 · context_compactor.py
    
    
    async def construir_contexto_optimizado(redis_conn, wa_id: str) -> list[dict]:
        # 1. Recuperar resumen compacto de variables de sesión
        resumen = await redis_conn.hgetall(f"session:{wa_id}:summary")
        
        # 2. Recuperar únicamente los últimos 4 turnos (sliding window)
        ultimos_turnos = await redis_conn.lrange(f"session:{wa_id}:messages", -4, -1)
        
        prompt_contexto = [{"role": "system", "content": f"Eres el asistente de Meta. Datos del cliente: {resumen}"}]
        for msg_json in ultimos_turnos:
            prompt_contexto.append(json.loads(msg_json))
            
        return prompt_contexto

¿No entendiste? Te lo explico fácil: El cuaderno de notas del médico

Imagina que vas al médico. El doctor no te pide que le recites palabra por palabra cada conversación que tuviste con él durante los últimos 10 años. En su lugar, tiene una **ficha resumen** (_"Paciente: Juan, alérgico a penicilina, última visita: mayo"_) y solo escucha lo que le dices en la consulta actual. Así ahorra tiempo y atiende a 30 pacientes al día sin colapsar. 

Consejo Pro: Configura maxmemory-policy volatile-lru en Redis

Asegúrate de que cada clave de sesión tenga siempre configurado su TTL mediante `EXPIRE`. Con la directiva `maxmemory-policy volatile-lru`, Redis solo desalojará claves con vencimiento cuando la memoria se llene, protegiendo tus datos permanentes. 

Autoevaluación 2.2.5

¿Cuál es la principal desventaja de reenviar todo el historial de 50 mensajes en cada llamada a Llama 3 en producción?

Laboratorios Prácticos en Vivo

## Bancos Interactivos del Tema 2.2

Experimenta con memoria de estado en tiempo real, construye configuraciones de Llama Stack y prueba el enrutador de flujos híbridos.

Banco 2.2.1 · Simulador de Memoria de Estado y Context Window (Stateless vs Stateful)

MODO STATEFUL (CON MEMORIA)

Seleccionar Modo de Operación:

Paso a Paso Rápido (Haz clic para cargar el mensaje del turno):

DC

Dental Clinic Meta

En línea · Memoria de Estado Activa

STATEFUL

Servicio: **No definido**

Fecha: **No definida**

Hora / Dr.: **No asignado**

Estatus: **En Proceso**

Estado Persistente de la Sesión (JSON en Backend / Redis):

Prompt Reconstruido que Viaja a Llama 3 (Headers & Contexto):

Consumo de Ventana de Contexto: **0 / 2048 tokens**

Banco 2.2.2 · Playground Llama Stack (Arquitectura Modular Estandarizada)

Proveedor de Memoria: SQLite (Memoria Local en Disco) Redis (Caché en RAM de Alta Velocidad) PostgreSQL (Base de Datos Relacional)

Herramientas Registradas:

consultar_disponibilidad_citas confirmar_cita_en_calendario enviar_recordatorio_sms

Capa de Seguridad (Shield): Llama Guard 3 (Filtro de Contenido) Prompt Guard (Anti-Inyecciones) Dual Shield (Llama Guard + Prompt Guard)

Código Llama Stack Python SDK
    
    
    # Generando especificación Llama Stack...

Banco 2.2.3 · Árbol de Decisión Híbrido (Reglas Fijas Deterministas vs Inferencia LLM)

Esperando Frase

Prueba cómo el enrutador separa solicitudes críticas (que activan protocolos inquebrantables) de consultas generales delegadas al razonamiento de Llama 3. 

Haz clic en "Evaluar Enrutamiento" para inspeccionar la decisión arquitectónica. 

Banco 2.2.4 · Compresor de Historial & Optimizador de Tokens (Sliding Window vs Summary)

Longitud de la Conversación: **12 turnos** Estrategia de Memoria: Memoria Híbrida (Resumen Llama + Últimos 3 turnos) Ventana Deslizante Fija (Sliding Window K=4) Memoria Completa Sin Compresión (Reenviar Todo)

Tokens en Bruto (Sin optimizar): **1,440 tokens**

Tokens Reenviados al Modelo: **445 tokens**

Ahorro de Cómputo & Latencia: **995 tokens (69% ahorro)**

Banco 2.2.5 · Aislamiento de Sesiones Multi-Usuario en Base de Datos (SQLite / Redis)

Paciente 1: Lic. Jesús Olvera (+52 1 55 8765-4321) Paciente 2: Dra. Sofía Morales (+52 1 55 1234-5678) Paciente 3: Ing. Carlos Mendoza (+52 1 55 9988-7766)

Comprueba cómo las variables de sesión y el estado de cada cliente de WhatsApp permanecen estrictamente aislados en la base de datos sin contaminar las conversaciones simultáneas. 

Autoevaluación Práctica & Análisis de Sistemas

## Ejercicios Prácticos Oficiales del Tema 2.2

Resuelve los 3 desafíos de diseño de agentes y memoria de estado del temario oficial. Despliega cada solución para revisar el análisis paso a paso y los criterios de ingeniería.

Ejercicio 1

#### Rastreo del Estado Paso a Paso en un Caso Clínico

Enunciado del Flujo Conversacional 

Revisa la conversación de la clínica dental donde un paciente escribe: (1) _"quiero una cita dental"_ y luego (2) _"¿tienen horario el sábado?"_. Indica qué datos específicos debe contener el estado en memoria después de cada mensaje, qué información nueva se añade y qué se reinyecta en la siguiente llamada a Llama 3. 

Ver Solución de Ingeniería Paso a Paso & Diagrama de Estado

1

##### Turno 1: Identificación de Intención y Apertura de Sesión

• **Mensaje Usuario:** _"quiero una cita dental"_.  
• **Extracción del Agente:** `intent: "agendar_cita"`, `servicio: "odontologia_general"`, `etapa: "SELECCIONANDO_FECHA"`.  
• **Estado en Base de Datos (SQLite/Redis):**

JSON Schema / Payload
    
    
    {
      "wa_id": "5215587654321",
      "etapa": "SELECCIONANDO_FECHA",
      "slots": { "servicio": "odontologia_general", "fecha": null, "hora": null },
      "created_at": 1787112000
    }

2

##### Turno 2: Enriquecimiento Incremental del Estado

• **Mensaje Usuario:** _"¿tienen horario el sábado?"_.  
• **Actualización de Slots:** Se preserva `servicio: "odontologia_general"` y se añade `dia_solicitado: "sabado"`.  
• **Prompt Reinyectado a Llama 3:**

Markdown · Prompt Reconstruido
    
    
    # Contexto de Sesión del Paciente:
    - Servicio solicitado: Odontología General (confirmado previamente)
    - Preferencia de fecha: Sábado
    - Horarios disponibles en base de datos: 10:00 AM, 12:30 PM, 04:00 PM.
    Instrucción: Ofrece amablemente los 3 horarios del sábado para odontología sin volver a preguntar qué servicio necesita.

3

##### Criterio de Validación Productiva

Al inyectar el estado resumido en lugar del historial completo sin estructurar, el modelo responde con precisión quirúrgica en menos de **80 tokens** , reduciendo la latencia de 1.8s a **420ms** y eliminando preguntas redundantes que frustran al usuario. 

Ejercicio 2

#### Distinción entre Reglas Fijas y Flexibilidad del Modelo

Enunciado de Diseño de Enrutamiento Híbrido 

Para cada una de las siguientes situaciones, decide si debe implementarse como una regla fija inquebrantable o como una decisión delegada a Llama 3: (a) Un usuario escribe _"quiero hablar con un humano"_ , (b) Un usuario pregunta _"¿qué síntomas debo tener para una consulta de cardiología?"_ , (c) Un usuario intenta modificar la fecha de una cita ya confirmada. 

Ver Solución de Ingeniería Paso a Paso & Matriz de Decisión

1

##### Evaluación de los 3 Escenarios

Caso | Clasificación Arquitectónica | Justificación Técnica  
---|---|---  
**(a) "Hablar con humano"** | Regla Fija Determinista | Protocolo obligatorio. No debe someterse a inferencia del LLM; el backend transfiere inmediatamente el ticket al conmutador humano.  
**(b) Consulta de Síntomas** | Flexibilidad LLM + RAG | Requiere comprensión de lenguaje natural y síntesis explicativa basada en guías clínicas oficiales con descargo de responsabilidad.  
**(c) Modificar Cita** | Flujo Híbrido en Dos Fases | Llama 3 extrae la nueva fecha deseada (NLU), pero la validación de cupo y la transacción SQL la ejecuta una regla fija con bloqueo de concurrencia.  
  
2

##### Implementación en Código Python del Enrutador Híbrido

Python 3.11 · enrutador_defensivo.py
    
    
    import re
    
    def enrutar_mensaje(texto: str, wa_id: str):
        t_lower = texto.lower().strip()
        # 1. Regla Determinista de Transferencia
        if re.search(r"\b(humano|asesor|agente|operador|persona)\b", t_lower):
            return transferir_a_humano(wa_id)
        
        # 2. Regla de Emergencia Médica
        if re.search(r"\b(sangrado|infarto|emergencia|ambulancia|grave)\b", t_lower):
            return activar_protocolo_urgencias(wa_id)
            
        # 3. Delegación a Llama 3 para Consulta / NLU
        return ejecutar_agente_llama3(texto, wa_id)

Ejercicio 3

#### Análisis de Costo y Saturación de la Ventana de Contexto

Enunciado de Optimización de Cómputo & VRAM 

Explica por qué reenviar todo el historial completo en cada llamada afecta directamente el uso de la ventana de contexto y la latencia. Propón una estrategia de compresión para mantener la coherencia conversacional sin saturar el contexto. 

Ver Solución de Ingeniería Paso a Paso & Algoritmo de Compresión

1

##### Fórmula del Crecimiento Cuadrático de Tokens

Si se reenvía el historial completo en una conversación de $N$ turnos con longitud promedio $\bar{L}$, el consumo acumulado de tokens crece de forma cuadrática: 

$$

C_{\text{total}} = \sum_{t=1}^{N} \Big( T_{\text{sys}} + 2 \cdot t \cdot \bar{L} \Big) = N \cdot T_{\text{sys}} + N(N+1) \cdot \bar{L}$$ 

Para una conversación de 15 turnos con $\bar{L} = 50$ tokens y $T_{\text{sys}} = 200$, el método ingenuo procesa **14,550 tokens** , elevando la latencia de la fase de _Prefill_ en GPU y multiplicando el costo en APIs. 

2

##### Estrategia de Compresión Híbrida (Sliding Window + Summary)

• **Ventana Deslizante Fija ($K=3$ turnos):** Conserva los últimos 3 intercambios verbatim para preservar la fluidez inmediata.  
• **Resumen Incremental de Entidades:** Cada 5 turnos, un sub-proceso asíncrono actualiza un resumen estructurado en JSON (_"Usuario: Juan, busca: endodoncia, cita: sábado 10am"_).  
• **Resultado en Producción:** Consumo constante de $\approx 450$ tokens por llamada independientemente de si la conversación dura 5 o 50 turnos, logrando un **ahorro de cómputo del 69%**. 

Ejercicio 4

#### Diseño de Almacén Multi-Tenant en Redis con TTL Dinámico y Fallback a SQLite

Enunciado de Arquitectura de Alta Disponibilidad 

Diseña un esquema de almacenamiento de sesiones multi-inquilino en Redis con expiración dinámica (TTL de 24 horas tras el último mensaje) y un mecanismo de respaldo (fallback) hacia SQLite en caso de caída temporal del clúster de Redis. 

Ver Solución de Ingeniería Paso a Paso & Código de Resiliencia

1

##### Estructura de Claves Multi-Tenant en Redis

Utilizamos namespaces jerárquicos: `tenant:{empresa_id}:session:{wa_id}` para aislar datos entre clientes: 
    
    
    # Redis Hash Key: tenant:clinica_meta:session:5215587654321
    HSET tenant:clinica_meta:session:5215587654321 service "Ortodoncia" step "CONFIRMING"
    EXPIRE tenant:clinica_meta:session:5215587654321 86400  # TTL 24 horas

2

##### Patrón Circuit Breaker con Fallback a Base de Datos Local

Si la conexión a Redis lanza `redis.ConnectionError` o supera un timeout de 200ms, el middleware conmuta de inmediato a SQLite local para leer el último estado y encola una tarea de sincronización asíncrona para cuando Redis se restablezca, garantizando 0% de mensajes perdidos para los usuarios de WhatsApp. 

Ejercicio 5 · Nivel Avanzado 25 min

#### Compresor de Historial Incremental en BackgroundTasks

Implementa una función asíncrona en FastAPI que, cada vez que una conversación de WhatsApp alcance 6 turnos en Redis, dispare una tarea en segundo plano para generar un resumen semántico de las preferencias del usuario usando Llama 3. El resultado debe actualizar la clave `session:{wa_id}:summary` y podar los primeros 4 mensajes de la lista sin bloquear la respuesta de WhatsApp. 

Ver Implementación Oficial con Poda en Segundo Plano

Python 3.11 · async_compactor_task.py
    
    
    import json
    from fastapi import BackgroundTasks
    
    async def compactar_sesion_en_background(redis_conn, wa_id: str):
        key_msgs = f"session:{wa_id}:messages"
        total_msgs = await redis_conn.llen(key_msgs)
        
        if total_msgs >= 6:
            # 1. Extraer los mensajes más antiguos para condensar
            antiguos = await redis_conn.lrange(key_msgs, 0, 3)
            dialogo_str = "
    ".join([m.decode('utf-8') for m in antiguos])
            
            # 2. Generar resumen compacto con Llama 3
            resumen_actualizado = await generar_resumen_llama3(dialogo_str)
            
            # 3. Guardar en hash y podar la lista en Redis
            await redis_conn.hset(f"session:{wa_id}:summary", "perfil", resumen_actualizado)
            await redis_conn.ltrim(key_msgs, 4, -1)

Diccionario de Conceptos

## Glosario Técnico Oficial · Tema 2.2

Definiciones clave sobre arquitectura de agentes, persistencia de estado y Llama Stack.

Estado de la Conversación

Información activa (historial de mensajes, variables recolectadas y metadatos de sesión) que se mantiene fuera del modelo entre turnos de una conversación para permitir continuidad lógica. 

Llama Stack

Conjunto estandarizado de componentes y APIs de arquitectura para construir agentes basados en Llama, agrupando memoria, herramientas y seguridad en módulos reutilizables. 

Flujo Conversacional Híbrido

Diseño que combina máquinas de estado finitas (FSM) para etapas críticas (pagos, agendas) con razonamiento generativo libre de Llama 3 para resolver dudas abiertas. 

Ventana de Contexto (Context Window)

Límite máximo de tokens procesables en un pase de inferencia (128k en Llama 3). Requiere estrategias de compresión para evitar latencias elevadas en producción. 

Identificador de Usuario (wa_id)

Identificador único internacional del cliente en WhatsApp utilizado como clave primaria de indexación para particionar la memoria en almacenes de clave-valor. 

Sliding Window Memory

Estrategia de truncamiento que retiene únicamente los últimos $K$ turnos conversacionales exactos en el prompt, descartando los mensajes más antiguos para mantener el costo constante. 

Compresión de Resumen (Summary Buffer)

Proceso asíncrono que sintetiza periódicamente el historial antiguo en un párrafo denso de variables clave para no perder el contexto histórico sin saturar la GPU. 

TTL (Time To Live) de Sesión

Mecanismo de expiración automática en Redis que libera la memoria RAM de sesiones inactivas tras 24 o 48 horas, evitando sobrecostos de infraestructura. 

Determinismo en Máquina de Estados

Garantía de que ante una acción crítica (ej. confirmar cobro con tarjeta), el sistema solo avanza si se satisfacen precondiciones formales, eliminando alucinaciones del modelo. 

Prefill Latency

Fase inicial de inferencia en la que la GPU procesa en paralelo todos los tokens del prompt histórico antes de generar el primer token de respuesta ($O(N^2)$ en atención). 

Documentación Oficial & Referencias de Ingeniería

## Fuentes de Referencia Oficiales · Tema 2.2

Especificaciones técnicas de memoria conversacional, arquitecturas de sesión distribuida con Redis y guías de desarrollo con Llama Stack.

Meta AI · 2024 Especificación Abierta

#### Llama Stack Specification & Distribution Architecture

Arquitectura unificada de APIs para memoria, herramientas, evaluación y orquestación de agentes con modelos Meta Llama 3.

[Consultar Llama Stack en GitHub](https://github.com/meta-llama/llama-stack)

Meta AI Research · 2024 Paper Científico

#### The Llama 3 Herd of Models (Context & Reasoning)

Paper técnico oficial con el análisis de atención agrupada (GQA), ventana nativa de 128k tokens y formateo con delimitadores de turnos especiales.

[Leer Paper en Meta AI Research](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)

Redis Ltd. · 2024 Guía de Arquitectura

#### Session Management & State Caching with Redis

Patrones de persistencia en memoria, expiración de claves TTL para sesiones de mensajería y concurrencia asíncrona con redis-py.

[Consultar Documentación de Redis](https://redis.io/docs/latest/develop/use/session-management/)

Python Software Foundation Estándar Asíncrono

#### asyncio: Asynchronous I/O & Event Loop Architecture

Documentación del motor de corrutinas en Python para coordinar peticiones concurrentes de WhatsApp sin bloquear el hilo del servidor.

[Consultar Python Docs asyncio](https://docs.python.org/3/library/asyncio.html)

SQLite Development Team Motor Embebido

#### SQLite WAL Mode (Write-Ahead Logging) for Sessions

Técnicas de concurrencia y persistencia atómica con SQLite para almacenamiento local de conversaciones con bajo consumo de memoria.

[Consultar Documentación SQLite WAL](https://www.sqlite.org/wal.html)

Meta Open Source Herramientas de Agente

#### Llama Agent Tooling & Memory Safety Patterns

Directrices de seguridad y aislamiento de memoria para agentes que interactúan con múltiples usuarios en plataformas de mensajería.

[Consultar Llama Agent Docs](https://llama.meta.com/docs/llama-stack/)

Hugging Face & vLLM Optimización de KV Cache

#### PagedAttention: Memory Management for LLMs

Arquitectura de memoria paginada para reducir la fragmentación de la memoria VRAM durante la gestión de múltiples sesiones simultáneas.

[Consultar vLLM Architecture](https://vllm.ai/)

IETF RFC 7519 Estándar de Token

#### JSON Web Tokens (JWT) for Stateful Session Authentication

Estándar para validación de identidad y claims de sesión en arquitecturas conversacionales distribuidas de microservicios.

[Consultar RFC 7519](https://datatracker.ietf.org/doc/html/rfc7519)

Redis Cluster Docs Alta Disponibilidad

#### Redis Sentinel & Cluster Sharding for Session Scale

Estrategias de particionamiento de hash slots para escalar horizontalmente la memoria de más de 500,000 usuarios concurrentes en WhatsApp.

[Consultar Redis Cluster Scaling](https://redis.io/docs/latest/operate/oss_and_stack/management/scaling/)

PostgreSQL 16 Docs JSONB Indexing

#### PostgreSQL JSONB GIN Indexes for Long-Term Memory

Indexación de árboles de conversación complejos mediante índices GIN sobre campos JSONB para consultas analíticas instantáneas.

[Consultar Postgres JSONB Docs](https://www.postgresql.org/docs/current/datatype-json.html)

Pydantic Library Esquemas de Estado

#### State Schema Modeling with Pydantic Generic Models

Patrones de diseño para serialización y deserialización atómica del estado conversacional evitando corrupción de memoria.

[Consultar Pydantic Models](https://docs.pydantic.dev/latest/concepts/models/)

Finite State Machine Diseño de Software

#### Deterministic FSM Design for Conversational Agents

Modelado de flujos conversacionales como autómatas finitos deterministas para asegurar el cumplimiento estricto de reglas de negocio.

[Consultar FSM Transitions](https://github.com/pytransitions/transitions)

Meta Llama 3 Prompting Formateo de Turnos

#### Special Tokens & Turn Delimiters in Llama 3.1

Especificación de los delimitadores oficiales <|start_header_id|> y <|eot_id|> para preservar la pureza del contexto conversacional.

[Consultar Delimitadores de Turno](https://llama.meta.com/docs/model-cards-and-prompt-formats/llama3_1/)

Memory Eviction Policies Gestión de RAM

#### Redis LRU & LFU Cache Eviction Strategies

Configuración de políticas volatile-lru y maxmemory en Redis para descartar automáticamente sesiones antiguas sin interrumpir chats activos.

[Consultar Memoria y Evicción](https://redis.io/docs/latest/operate/oss_and_stack/management/optimization/memory-optimization/)

LangChain Core Patrones de Memoria

#### ConversationSummaryBufferMemory: Theory and Practice

Análisis comparativo de algoritmos de compresión de contexto que combinan ventanas deslizantes con resúmenes incrementales generados por LLMs.

[Consultar Memory Patterns](https://python.langchain.com/docs/modules/memory/)

Meta AI Ethics Gobernanza de Datos

#### Data Privacy & GDPR Compliance in Automated Chatbots

Directrices de cumplimiento legal y derecho al olvido (data deletion request) en agentes de WhatsApp con almacenamiento persistente.

[Consultar Responsible AI](https://ai.meta.com/responsible-ai/)

---

<div align="center">

[⬅️ Anterior](01-whatsapp-cloud-api-arquitectura-webhooks.md) • [Inicio](../../README.md) • [Módulo 2](README.md) • [Siguiente ️](03-inferencia-function-calling-tools.md)

</div>
