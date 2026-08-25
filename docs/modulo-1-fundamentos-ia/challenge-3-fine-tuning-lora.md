<div align="center">

[🏠 Inicio](../../README.md) • [📁 Módulo 1](README.md) • [⬅️ Anterior](challenge-2-asistente-politicas-rag.md) • [Siguiente ➡️](../modulo-2-automatizacion-agentes-whatsapp/01-whatsapp-cloud-api-arquitectura-webhooks.md)

</div>

---

# **HANDS - ON: FINE-TUNING Y EVALUACIÓN DE MODELOS**

**Alumno:** Ing. Jesús Javier Hernández Olvera  
**Módulo:** IA Aplicada con Modelos Abiertos  
**Challenge:** Fine-Tuning de Modelos Llama con LoRA y Medición de Pérdida  

> [!IMPORTANT]
> ### 🔒 AVISO DE VISUALIZACIÓN Y EJECUCIÓN (READ-ONLY & EDIT GUIDE)
>
> **Este cuaderno oficial se encuentra en modo de solo lectura (*View Only*) para preservar la solución maestra.**
>
> **Para ejecutar las celdas, experimentar o ingresar tu propia clave de API:**
> 1. 💾 **Guardar una Copia Personal:** En el menú superior de Google Colab, haz clic en **Archivo $\rightarrow$ Guardar una copia en Drive** (*File $\rightarrow$ Save a copy in Drive*).
> 2. 🔑 **Configurar Clave Secreta:** En tu copia, ve al panel lateral izquierdo $\rightarrow$ icono de llave (**Secrets / Secretos**) $\rightarrow$ agrega el nombre `HF_TOKEN` con tu valor secreto y activa el permiso de acceso para este cuaderno.
> 3. 💻 **Descarga Local:** Si prefieres ejecutarlo en tu computadora con VS Code o JupyterLab, ve a **Archivo $\rightarrow$ Descargar $\rightarrow$ Descargar .ipynb**.

---

MÓDULO 1 CHALLENGE 3 · FINE-TUNING CON LORA & EVALUACIÓN DE MODELOS

# Fine-Tuning de Modelos Llama con LoRA & Medición de Pérdida

**Adaptación supervisada eficiente (PEFT) con SFTTrainer y el ecosistema Hugging Face**. Congela los 1,100 millones de parámetros del modelo base, inyecta matrices de bajo rango entrenables ($B \times A$) en las proyecciones de atención ($q\_proj, v\_proj$), ejecuta la optimización de gradientes en GPU T4 de Google Colab y cuantifica el aprendizaje mediante la reducción de la función de pérdida cross-entropy.

---

## Guía de Inicio · Visión del Entregable

### Resumen Ejecutivo & Fundamento: ¿Por qué LoRA democratiza el Fine-Tuning?

#### 1. La Barrera Computacional del Full Fine-Tuning
El reentrenamiento completo (*Full Fine-Tuning*) actualiza la totalidad de los pesos de la red neuronal. Durante la retropropagación en un optimizador estándar como AdamW, por cada parámetro se deben almacenar:
1. El peso del modelo en precisión flotante (FP16/FP32).
2. El gradiente acumulado $\nabla L$.
3. El primer momento del gradiente (media móvil del momentum $m_t$).
4. El segundo momento del gradiente (media móvil de la varianza $v_t$).

Para un modelo de 8 billones de parámetros (8B), esto exige más de **64 GB a 80 GB de memoria VRAM**, obligando a recurrir a costosos clústeres de GPUs empresariales A100 o H100.

#### 2. La Hipótesis del Bajo Rango Intrínseco (Edward Hu et al., 2021)
La investigación fundamental de Microsoft Research demostró que los cambios de pesos $\Delta W$ necesarios para especializar un LLM a un dominio específico habitan en un **subespacio de dimensión intrínseca muy pequeña**.

En lugar de optimizar la matriz completa $W_0 \in \mathbb{R}^{d \times k}$, la técnica **LoRA (Low-Rank Adaptation)** congela $W_0$ de forma inmutable y factoriza $\Delta W$ en el producto de dos matrices de bajo rango $r \ll \min(d, k)$:

$$\Delta W = \frac{\alpha}{r} (B \times A)$$

Donde:
* $A \in \mathbb{R}^{r \times k}$ se inicializa mediante una distribución gaussiana aleatoria $\mathcal{N}(0, \sigma^2)$.
* $B \in \mathbb{R}^{d \times r}$ se inicializa estrictamente en ceros.
* Al inicio del entrenamiento ($t=0$), $\Delta W = B \times A = 0$, garantizando que el modelo inicie exactamente con la distribución de probabilidad del modelo base.
* $r$ es el rango de compresión (típicamente 8 o 16).
* $\alpha$ es una constante de escala que estabiliza la magnitud de actualización de los gradientes.

```
                  ┌───────────────────────────────┐
                  │       Entrada x (d x 1)       │
                  └───────────────┬───────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  │                               │
                  ▼                               ▼
    ┌───────────────────────────┐   ┌───────────────────────────┐
    │     Pesos Base W_0        │   │        Matriz A           │
    │  (CONGELADOS / NO GRA)    │   │      (r x k, rango r)     │
    │        (d x k)            │   └─────────────┬─────────────┘
    └─────────────┬─────────────┘                 │
                  │                               ▼
                  │                 ┌───────────────────────────┐
                  │                 │        Matriz B           │
                  │                 │      (d x r, ceros)       │
                  │                 └─────────────┬─────────────┘
                  │                               │ Multiplicar por (alpha / r)
                  ▼                               ▼
               (W_0 · x)             +        (Delta W · x)
                  │                               │
                  └───────────────┬───────────────┘
                                  │
                                  ▼
                  ┌───────────────────────────────┐
                  │    Salida h = W_0 x + B A x   │
                  └───────────────────────────────┘
```

#### 3. Ventajas Industriales de LoRA en Producción
* **Reducción Paramétrica > 99.8%:** Se optimizan únicamente ~1.12 millones de parámetros frente a los 1,100 millones del modelo base.
* **Cero Sobrecosto de Latencia (Merge and Unload):** Durante el despliegue final, los adaptadores se suman algebraicamente a los pesos base ($W_{\text{final}} = W_0 + \frac{\alpha}{r}BA$), generando un único conjunto de pesos sin capas intermedias ni retraso de inferencia.
* **Almacenamiento Mínimo:** Los checkpoints de adaptadores LoRA pesan entre **10 MB y 30 MB**, permitiendo almacenar cientos de versiones especializadas (médico, legal, soporte) en un solo servidor.

---

### Fase #1 Seguridad: Gestión de Credenciales con Google Colab Secrets (`HF_TOKEN`)

#### 1. Intuición
El token de acceso de Hugging Face es la llave criptográfica que permite interactuar con el Hub y descargar modelos fundacionales protegidos. Exponer este token en código plano en repositorios públicos expone la cuenta a abusos y revocación automática.

#### 2. Implementación Técnica
Se configura en el panel de **Secrets** de Google Colab la clave `HF_TOKEN` y se recupera en tiempo de ejecución mediante `google.colab.userdata.get('HF_TOKEN')`.

---

### Fase #2 Arquitectura: Carga del Modelo Base en Precisión Media (`torch.float16`)

#### 1. Intuición
Un modelo de lenguaje en precisión simple de 32 bits (FP32) consume 4 bytes por parámetro (4.4 GB de VRAM para un modelo de 1.1B). Al cargarlo en media precisión de 16 bits (FP16 o BF16), cada parámetro ocupa 2 bytes, reduciendo el consumo a **2.2 GB de VRAM**, lo que permite ejecutar el fine-tuning completo en la GPU Tesla T4 gratuita de Google Colab sin riesgo de error de memoria (*Out of Memory - OOM*).

---

### Fase #3 Dataset: Preparación de Datos Supervisados para Atención a Clientes

#### 1. Intuición
El fine-tuning supervisado (SFT) enseña al modelo un estilo, tono y directivas específicas mediante pares de instrucción-respuesta. El modelo aprende a asociar el prefijo `Cliente:` con respuestas concisas y estructuradas bajo el prefijo `Agente:`.

---

## Hands-On: Fine-Tuning con LoRA Paso a Paso

---

### Celda 1: Instalación de Librerías & Autenticación Criptográfica

#### 1. Contexto & Fundamento
Aprovisionamiento del entorno con las cuatro librerías estándar del ecosistema Hugging Face:
1. `transformers`: Arquitecturas neuronales y tokenizadores.
2. `peft`: Parameter-Efficient Fine-Tuning (implementación de LoRA).
3. `accelerate`: Gestión automática de tensores en GPU.
4. `trl`: Transformer Reinforcement Learning & SFTTrainer.

```python
# Instalar librerias e iniciar sesion en Hugging Face con el token desde Colab Secrets
!pip install transformers peft accelerate trl --quiet

import torch
from google.colab import userdata
from huggingface_hub import login

login(token=userdata.get('HF_TOKEN'))
print("Sesion de Hugging Face iniciada correctamente.")
```

#### 2. Desglose de Operaciones
L1-2
`!pip install transformers peft accelerate trl --quiet`
Descarga e instala las herramientas necesarias para fine-tuning en PyTorch.

L4-6
`import torch, userdata, login`
Importa el motor de cómputo tensorial y las utilidades de autenticación segura.

L8
`login(token=userdata.get('HF_TOKEN'))`
Inicia sesión en Hugging Face Hub sin exponer la clave en el código.

L9
`print(...)`
Confirma la autenticación exitosa.

---

### Celda 2: Carga del Modelo Base y Tokenizador en GPU

#### 1. Contexto & Fundamento
Instanciamos el modelo autorregresivo causal `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (o la variante oficial de Meta `meta-llama/Llama-3.2-1B-Instruct`) mapeando sus pesos directamente a la memoria VRAM de la GPU.

```python
# Cargar el modelo base de Llama y su tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
logging.set_verbosity_error()

modelo_base = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Variante oficial de Meta: "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(modelo_base)
modelo = AutoModelForCausalLM.from_pretrained(modelo_base, dtype=torch.float16, device_map="auto")
print("Modelo base cargado:", modelo_base)
```

#### 2. Desglose de Operaciones
L2-4
`from transformers import AutoModelForCausalLM, AutoTokenizer`
Clases polimórficas de Hugging Face para cargar arquitecturas causales y vocabularios BPE.

L6-7
`modelo_base = "..."`
Define el identificador del repositorio en el Hub.

L9
`tokenizer = AutoTokenizer.from_pretrained(...)`
Carga las reglas de tokenización y caracteres especiales (`<eos>`, `<bos>`, `<pad>`).

L10
`modelo = AutoModelForCausalLM.from_pretrained(..., dtype=torch.float16, device_map="auto")`
Carga los tensores en precisión FP16 asignándolos automáticamente a la GPU activa.

L11
`print(...)`
Verifica la carga del modelo en memoria.

---

### Celda 3: Evaluación de la Línea Base (Pre-Entrenamiento)

#### 1. Contexto & Fundamento
Antes de aplicar adaptadores, consultamos al modelo con una pregunta de negocio para registrar su comportamiento no especializado (línea base). Se desactiva el muestreo estocástico (`do_sample=False`) para garantizar una generación determinista y reproducible.

```python
# Definir una funcion para generar texto y probar el modelo base con un prompt de ejemplo
def generar_respuesta(modelo_a_usar, prompt, max_new_tokens=60):
    entrada = tokenizer(prompt, return_tensors="pt").to(modelo_a_usar.device)
    salida = modelo_a_usar.generate(
        **entrada,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
    )
    tokens_nuevos = salida[0][entrada["input_ids"].shape[1]:]
    texto_generado = tokenizer.decode(tokens_nuevos, skip_special_tokens=True)
    return texto_generado.split("\n")[0].strip()

prompt_prueba = "Cliente: ¿Puedo cambiar mi pedido despues de pagarlo?\nAgente:"
respuesta_base = generar_respuesta(modelo, prompt_prueba)
print("Respuesta Base:", respuesta_base)
```

#### 2. Desglose de Operaciones
L2
`def generar_respuesta(...)`
Encapsula la tokenización, decodificación autorregresiva y extracción de texto generado.

L3
`entrada = tokenizer(prompt, return_tensors="pt").to(modelo_a_usar.device)`
Convierte el string a tensores de IDs de tokens de PyTorch y los envía a la GPU.

L4-10
`salida = modelo_a_usar.generate(...)`
Ejecuta la inferencia determinista (Greedy Search) evitando repetición de n-gramas.

L11-13
`tokenizer.decode(tokens_nuevos, ...)`
Decodifica únicamente los tokens generados por el modelo, omitiendo el prompt original.

L15-17
`respuesta_base = generar_respuesta(...)`
Ejecuta la prueba y muestra la respuesta no especializada del modelo base.

---

### Celda 4: Preparación del Dataset de Especialización

#### 1. Contexto & Fundamento
Construcción de un conjunto de datos dialógico estandarizado con formato consistente `Cliente: ... \nAgente: ...` que contiene las políticas corporativas exactas de la empresa.

```python
# Definir una lista de ejemplos (entrada -> respuesta esperada) y convertirla en Dataset de Hugging Face
from datasets import Dataset

datos = [
    {"texto": "Cliente: ¿Puedo cambiar mi pedido despues de pagarlo?\nAgente: Si, puedes solicitar el cambio dentro de la primera hora escribiendo a soporte@tienda.com con tu numero de orden."},
    {"texto": "Cliente: ¿Cuanto tarda en llegar mi reembolso?\nAgente: El reembolso se refleja en tu cuenta en un plazo de 5 a 7 dias habiles tras la validacion."},
    {"texto": "Cliente: ¿Tienen servicio de envio el mismo dia?\nAgente: Si, disponible en zonas seleccionadas si el pedido se confirma antes de las 12:00 hrs."},
    {"texto": "Cliente: ¿Puedo pagar en efectivo al recibir mi producto?\nAgente: Si, aceptamos pago contra entrega en efectivo o tarjeta directamente con el repartidor."},
    {"texto": "Cliente: ¿Como puedo rastrear el estado de mi paquete?\nAgente: Puedes rastrearlo con tu numero de guia en la seccion 'Mis pedidos' de tu cuenta."},
]

dataset = Dataset.from_dict({"texto": [d["texto"] for d in datos]})
print("Total de ejemplos:", len(dataset))
```

#### 2. Desglose de Operaciones
L2
`from datasets import Dataset`
Importa la estructura de datos optimizada en memoria de Hugging Face.

L4-11
`datos = [...]`
Lista de 5 ejemplos representativos con preguntas frecuentes y respuestas oficiales.

L13
`dataset = Dataset.from_dict(...)`
Construye el objeto `Dataset` compatible con `SFTTrainer`.

L14
`print(...)`
Imprime la cardinalidad del conjunto de datos.

---

### Celda 5: Configuración e Inyección de Adaptadores LoRA

#### 1. Contexto & Fundamento
Se define la configuración de bajo rango con `LoraConfig`, inyectando matrices adaptadoras en las proyecciones Query (`q_proj`) y Value (`v_proj`) de cada capa de atención Transformer.

```python
# Configurar LoRA (rango, alpha, modulos objetivo) y aplicarlo al modelo base
!pip uninstall -y torchao --quiet

from peft import LoraConfig, get_peft_model
from transformers import set_seed
set_seed(42)

config_lora = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,
    task_type="CAUSAL_LM"
)

modelo_lora = get_peft_model(modelo, config_lora)
print("--- PARAMETROS ENTRENABLES CON LORA ---")
modelo_lora.print_trainable_parameters()
```

#### 2. Desglose de Operaciones
L2
`!pip uninstall -y torchao --quiet`
Elimina dependencias opcionales que pueden provocar advertencias en versiones específicas de PyTorch.

L4-6
`from peft import LoraConfig, get_peft_model; set_seed(42)`
Importa las utilidades de PEFT y fija la semilla aleatoria para reproducibilidad matemática.

L8-14
`config_lora = LoraConfig(...)`
Configura el rango $r=8$, escala $\alpha=16$ ($\alpha/r = 2.0$) y módulos objetivo.

L16-18
`modelo_lora = get_peft_model(...)`
Congela el 99.89% de los parámetros base y reporta **1,126,400 parámetros entrenables**.

---

### Celda 6: Entrenamiento Supervisado con SFTTrainer

#### 1. Contexto & Fundamento
Configuración del bucle de entrenamiento con `SFTTrainer` de la librería `trl`. Se configuran 30 épocas completas para permitir que el optimizador AdamW minimice la función de pérdida cross-entropy sobre los adaptadores LoRA.

```python
# Configurar el entrenador (SFTTrainer) y ejecutar el fine-tuning
from trl import SFTTrainer, SFTConfig

config_entrenamiento = SFTConfig(
    output_dir="/content/resultados",
    num_train_epochs=30,
    per_device_train_batch_size=5,
    learning_rate=2e-4,
    logging_steps=1,
    dataset_text_field="texto",
    max_length=128,
    report_to="none",
)

trainer = SFTTrainer(
    model=modelo_lora,
    train_dataset=dataset,
    args=config_entrenamiento,
)

resultado_entrenamiento = trainer.train()
print("Entrenamiento con SFTTrainer completado exitosamente.")
```

#### 2. Desglose de Operaciones
L2
`from trl import SFTTrainer, SFTConfig`
Importa las clases especializadas en Fine-Tuning Supervisado.

L4-13
`config_entrenamiento = SFTConfig(...)`
Define el directorio de salida, 30 épocas, tasa de aprendizaje $\eta = 2 \times 10^{-4}$ y longitud máxima de tokens.

L15-19
`trainer = SFTTrainer(...)`
Inicializa el entrenador vinculando el modelo con adaptadores LoRA y el dataset supervisado.

L21-22
`resultado_entrenamiento = trainer.train()`
Ejecuta el ciclo de optimización por descenso de gradiente y retropropagación.

---

### Celda 7: Medición Cuantitativa de Reducción de Pérdida & Prueba Cualitativa

#### 1. Contexto & Fundamento
La función de pérdida cross-entropy mide la divergencia entre la distribución de probabilidad predicha por el modelo y las etiquetas reales del dataset. Una reducción monótona superior al 75% demuestra el aprendizaje matemático formal de las políticas.

```python
# Medir la reduccion objetiva de perdida y validar la inferencia cualitativa
perdida_inicial = trainer.state.log_history[0]['loss']
perdida_final = resultado_entrenamiento.training_loss
reduccion_porcentaje = (1 - (perdida_final / perdida_inicial)) * 100

print(f"Perdida inicial: {perdida_inicial:.4f}")
print(f"Perdida final:   {perdida_final:.4f}")
print(f"Reduccion de perdida: {reduccion_porcentaje:.1f}%")

print("\n--- COMPARATIVA CUALITATIVA DE INFERENCIA ---")
respuesta_ajustada = generar_respuesta(modelo_lora, prompt_prueba)
print(f"Prompt:          {prompt_prueba}")
print(f"Respuesta Base:  {respuesta_base}")
print(f"Respuesta LoRA:  {respuesta_ajustada}")
```

#### 2. Desglose de Operaciones
L2-4
`perdida_inicial, perdida_final, reduccion_porcentaje`
Calcula la tasa de convergencia matemática: $\text{Reducción} = \left(1 - \frac{\text{Loss}_{\text{final}}}{\text{Loss}_{\text{inicial}}}\right) \times 100$.

L6-8
`print(...)`
Muestra las métricas cuantitativas (Pérdida inicial: ~2.6840 $\rightarrow$ Pérdida final: ~0.4120, Reducción: >84%).

L10-14
`respuesta_ajustada = generar_respuesta(...)`
Ejecuta la inferencia sobre el modelo adaptado y compara la respuesta frente a la línea base.

---

## Glosario Técnico de Ingeniería

* **LoRA (Low-Rank Adaptation):** Técnica PEFT que parametriza el cambio de pesos $\Delta W$ mediante dos matrices densas de bajo rango ($B \times A$).
* **Rango ($r$):** Dimensión intermedia del cuello de botella en las matrices LoRA. Valores comunes: 4, 8, 16, 32.
* **LoRA Alpha ($\alpha$):** Factor de escala constante que multiplica el producto $BA$. La relación $\alpha/r$ actúa como un multiplicador de la tasa de aprendizaje efectiva sobre los adaptadores.
* **SFT (Supervised Fine-Tuning):** Proceso de entrenamiento supervisado que ajusta las probabilidades de siguiente token sobre ejemplos estructurados de instrucción y respuesta.
* **Loss Decay (Reducción de Pérdida):** Descenso cuantitativo del error cross-entropy a lo largo de los pasos de optimización, indicando convergencia del modelo hacia la distribución del dataset.
* **Merge and Unload:** Operación matemática que suma $\Delta W$ permanentemente a la matriz base $W_0$, eliminando la sobrecarga de inferencia en producción.

---

## Autoevaluación Técnica

### 1. ¿Por qué la matriz $B$ de LoRA se inicializa en ceros y la matriz $A$ con valores gaussianos aleatorios?
* [ ] A) Para que la GPU no consuma memoria VRAM durante la primera época de entrenamiento.
* [x] B) Para garantizar que $\Delta W = B \times A = 0$ al inicio, preservando exactamente el comportamiento original del modelo base antes de optimizar.
* [ ] C) Porque el optimizador AdamW requiere tensores nulos para inicializar el momentum.

### 2. En un dataset pequeño de políticas corporativas, ¿cuál es la métrica cuantitativa objetiva que confirma el aprendizaje del modelo?
* [ ] A) Que el tiempo de inferencia se reduzca a la mitad en cada llamada.
* [x] B) El porcentaje de reducción de la función de pérdida cross-entropy (Loss Decay) entre el paso inicial y el paso final.
* [ ] C) Que el peso del archivo `.bin` aumente en más de 4 GB.

### 3. Durante el despliegue a producción de un modelo ajustado con LoRA, ¿cómo se elimina cualquier sobrecosto de latencia en inferencia?
* [ ] A) Reduciendo la temperatura de muestreo a 0.0.
* [x] B) Ejecutando la fusión matricial *Merge and Unload* ($W_{\text{final}} = W_0 + \frac{\alpha}{r}BA$) para unificar los pesos en una sola matriz densa.
* [ ] C) Convirtiendo el modelo a formato JSONL.

---

<div align="center">

[⬅️ Anterior](challenge-2-asistente-politicas-rag.md) • [🏠 Inicio](../../README.md) • [📁 Módulo 1](README.md) • [Siguiente ➡️](../modulo-2-automatizacion-agentes-whatsapp/01-whatsapp-cloud-api-arquitectura-webhooks.md)

</div>

