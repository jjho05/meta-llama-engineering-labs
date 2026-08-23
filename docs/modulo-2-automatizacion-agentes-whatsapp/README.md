# 🚀 Módulo 2: Automatización con Llama & WhatsApp Cloud API

<div align="center">

**Webhooks Asíncronos, Memoria de Sesión con Redis, Function Calling y Blindaje con Llama Guard**

[🏠 Inicio](../../README.md) • [📚 Módulo 1](../modulo-1-fundamentos-ia/README.md) • [💻 Scripts de Producción](../../scripts/)

</div>

---

## 📋 Descripción del Módulo

En este segundo módulo conectarás la inteligencia generativa de Meta Llama 3 con el canal de mensajería más utilizado del mundo: la **WhatsApp Cloud API oficial de Meta**. Construirás agentes autónomos multi-turno con retención de contexto en Redis, capacidad de invocar funciones del mundo real (*Function Calling*) y blindaje de nivel empresarial con **Llama Guard 3** y **Prompt Guard**.

---

## 🎯 Competencias Específicas

1. **Infraestructura de Mensajería:** Configurar la WhatsApp Cloud API, handshake criptográfico GET y recepción de webhooks POST.
2. **Memoria de Sesión Multi-Turno:** Persistir el hilo conversacional con identificadores de teléfono y ventanas de contexto deslizantes.
3. **Ejecución de Herramientas (Function Calling):** Orquestar inferencia en dos pasos con esquemas JSON Schema y validación en Pydantic.
4. **SRE & Ciberseguridad de IA:** Desplegar con Docker Compose, certificados SSL NGINX, filtrado contra inyecciones y monitoreo P95.

---

## 📚 Temario y Documentación

| Tema | Título del Contenido | Enfoque de Ingeniería | Script Asociado |
|---|---|---|---|
| **2.1** | [**WhatsApp Cloud API & Webhooks**](01-whatsapp-cloud-api-arquitectura-webhooks.md) | Handshake criptográfico GET, parseo de eventos JSON, túneles ngrok y envío de mensajes. | [Ver Tema](01-whatsapp-cloud-api-arquitectura-webhooks.md) |
| **2.2** | [**Agentes Conversacionales & Memoria Redis**](02-agentes-conversacionales-memoria-redis.md) | Gestión de estado, ventanas deslizantes de contexto, persistencia y Llama Stack. | [Ver Tema](02-agentes-conversacionales-memoria-redis.md) |
| **2.3** | [**Inferencia, Function Calling & Tools**](03-inferencia-function-calling-tools.md) | Inferencia en dos pasos, esquemas JSON Schema y ejecución segura de bases de datos con Pydantic. | [Ver Tema](03-inferencia-function-calling-tools.md) |
| **2.4** | [**Producción SRE & Seguridad Llama Guard**](04-produccion-seguridad-llama-guard.md) | Blindaje con Llama Guard 3, Prompt Guard, Docker Compose, NGINX SSL y monitoreo de latencia. | [Ver Tema](04-produccion-seguridad-llama-guard.md) |

---

## 💻 Scripts de Terminal

* `scripts/ejecutar_challenge1.py`: Benchmark de inferencia en hardware LPU.
* `scripts/ejecutar_challenge2.py`: Pipeline RAG semántico en terminal.
