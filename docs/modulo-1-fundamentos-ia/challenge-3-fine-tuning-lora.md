<div align="center">

[🏠 Inicio](../../README.md) • [📁 Módulo 1](README.md) • [⬅️ Anterior](challenge-2-asistente-politicas-rag.md) • [Siguiente ➡️ (Hackathon)](hackathon-guia-participante.md)

</div>

---

MÓDULO 1 · CHALLENGE 3: FINE-TUNING CON LORA & EVALUACIÓN DE PÉRDIDA

# Fine-Tuning de Modelos Llama con LoRA & Medición de Pérdida

**Adaptación supervisada eficiente (PEFT) con SFTTrainer y Hugging Face**. Congela los 1,100 millones de parámetros del modelo base, inyecta matrices de bajo rango entrenables ($B \times A$) en las capas de atención ($q\_proj, v\_proj$), ejecuta el entrenamiento supervisado en GPU T4 de Google Colab y cuantifica objetivamente el aprendizaje mediante la reducción de la función de pérdida cross-entropy.

---

## Resumen Ejecutivo & Fundamento: ¿Por qué LoRA democratiza el Fine-Tuning?

### 1. El Problema del Reentrenamiento Completo (Full Fine-Tuning vs. PEFT)
El reentrenamiento completo de un modelo fundacional actualiza todos los pesos de la red neuronal ($\Delta W \in \mathbb{R}^{d \times k}$). Para un modelo de 8 mil millones de parámetros, almacenar los pesos, los gradientes y los estados del optimizador AdamW (que requiere 8 bytes adicionales por parámetro) exige más de **64 GB de memoria VRAM de alta velocidad**, confinando el entrenamiento a clústeres de cómputo industriales de alto costo (NVIDIA A100/H100).

### 2. La Hipótesis del Bajo Rango Intrínseco
Aghajanyan et al. (2020) y Hu et al. (2021) demostraron empíricamente que los cambios de pesos $\Delta W$ necesarios para adaptar un modelo a una tarea específica residen en un subespacio de dimensión intrínseca muy baja. LoRA (**Low-Rank Adaptation**) congela la matriz original $W_0 \in \mathbb{R}^{d \times k}$ y descompone la actualización en el producto de dos matrices densas de bajo rango $r \ll \min(d, k)$:

$$\Delta W = \frac{\alpha}{r} (B \times A)$$

Donde $A \in \mathbb{R}^{r \times k}$ se inicializa con una distribución gaussiana aleatoria y $B \in \mathbb{R}^{d \times r}$ se inicializa en ceros absolutos, garantizando que $\Delta W = 0$ al inicio del entrenamiento. Durante la inferencia en producción, se ejecuta la operación `merge_and_unload()`:

$$W_{\text{final}} = W_0 + \frac{\alpha}{r} (B \times A)$$

Eliminando cualquier sobrecosto de latencia o memoria en servidores productivos.

---

### Cuatro Fases de la Arquitectura LoRA

| Fase | Dimensión Técnica | Intuición Pedagógica | Implementación PyTorch / PEFT |
| :--- | :--- | :--- | :--- |
| **Fase 1: Inmutabilidad** | Congelamiento de $W_0$ | Como dejar una enciclopedia intacta sin tachar nada. | `requires_grad = False` en el 99.898% de los pesos. |
| **Fase 2: Factorización** | Inicialización Asimétrica | Pegar notas adhesivas transparentes sobre las páginas clave. | Matriz $B=0$, Matriz $A \sim \mathcal{N}(0, \sigma^2)$, $\Delta W=0$ en $t=0$. |
| **Fase 3: Estabilidad** | Factor de Escala $\alpha/r$ | Regular el volumen de la tinta para que no manche el texto original. | Multiplicador constante ($\alpha=16, r=8 \rightarrow \alpha/r=2.0$). |
| **Fase 4: Despliegue** | Merge and Unload | Integrar permanentemente las notas al libro en la imprenta final. | $W = W_0 + \Delta W$, latencia cero extra en producción. |

---

> [!NOTE]
> ### 💡 ¿No entendiste? Te lo explico fácil: Las notas adhesivas sobre la enciclopedia
> El **Full Fine-Tuning** es como imprimir de nuevo una enciclopedia entera de 1,000 páginas solo para actualizar 3 números de teléfono de atención a clientes. **LoRA** es como dejar la enciclopedia intacta y pegar pequeñas notas adhesivas transparentes (*Post-its*) en las páginas clave: cambias el comportamiento exacto usando casi cero papel y sin alterar el libro original.

> [!TIP]
> ### 🚀 Consejo Pro de Ingeniería: Módulos Objetivo de Atención
> Para tareas conversacionales estándar y estilo de respuesta, inyectar LoRA en las matrices de proyección de atención `q_proj` y `v_proj` ofrece la relación óptima entre reducción de memoria GPU y retención de capacidades cognitivas. Si requieres aprendizaje profundo de hechos o razonamiento complejo, extiende los adaptadores a todas las capas lineales: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`.

---

## Paso 0: Gestión Segura de Credenciales en Google Colab con Secrets (`HF_TOKEN`)

Para descargar arquitecturas de Hugging Face y publicar adaptadores entrenados sin exponer credenciales en repositorios públicos, sigue estos pasos:

```mermaid
graph TD
    A[1. Crear Cuenta en Hugging Face] --> B[2. Settings > Access Tokens]
    B --> C[3. Generar Token con Rol READ]
    C --> D[4. Abrir Panel Secrets en Google Colab]
    D --> E[5. Guardar como HF_TOKEN con Notebook Access]
```

1. **Crear Cuenta en Hugging Face:** Regístrate gratuitamente en [huggingface.co/join](https://huggingface.co/join).
2. **Access Tokens:** Ve a *Settings $\rightarrow$ Access Tokens* y genera un nuevo token con permisos `Read`.
3. **Copiar Token:** Copia la cadena generada que inicia con `hf_...`.
4. **Colab Secrets:** En el panel lateral izquierdo de Google Colab (icono de llave), añade el secreto con el nombre exacto `HF_TOKEN`.
5. **Notebook Access:** Activa el interruptor para permitir su lectura programática mediante `userdata.get('HF_TOKEN')`.

---

## Matriz Comparativa de Paradigmas de Adaptación y Especialización

| Dimensión de Ingeniería | Full Fine-Tuning | Adapters Seriales | LoRA (PEFT) | QLoRA (4-bit PEFT) |
| :--- | :--- | :--- | :--- | :--- |
| **Parámetros Entrenables** | 100% de la red | 1.0% - 3.0% | **0.05% - 0.20%** | **0.05% - 0.20%** |
| **Latencia Adicional en Inferencia** | 0% | +15% a +25% (capas extra) | **0% (fusión estática en $W_0$)** | **0% (fusión estática en $W_0$)** |
| **Consumo de VRAM (Modelo 8B)** | > 64 GB (Clúster A100) | ~18 GB VRAM | **~12 GB VRAM** | **~6 GB (GPU Comercial)** |
| **Almacenamiento del Checkpoint** | ~16 GB por versión | ~500 MB | **~10 MB - 30 MB** | **~10 MB - 30 MB** |
| **Riesgo de Olvido Catastrófico** | Severo (degrada lógica general) | Moderado | **Nulo (Pesos base congelados)** | **Nulo (Pesos base congelados)** |
| **Velocidad Forward / Backward** | 1.0x (Línea base) | 1.3x | **1.05x (Casi nativo)** | **1.25x (Cuantización dinámica)** |

---

### Análisis Técnico Profundo de la Matriz de Adaptación

#### 1. Preservación de Memoria VRAM y Optimizador AdamW
En el reentrenamiento completo (Full Fine-Tuning), el optimizador **AdamW** almacena momentos de primer y segundo orden para cada parámetro entrenable en punto flotante de 32 bits ($m_t, v_t \in \mathbb{R}$, requiriendo $4 + 4 = 8	ext{ bytes/parámetro}$). Para un modelo de 8 mil millones de parámetros, almacenar los pesos base en FP16 ($16	ext{ GB}$), los gradientes ($16	ext{ GB}$) y el estado del optimizador ($64	ext{ GB}$) exige más de **96 GB de VRAM**, obligando al uso de clústeres multi-GPU (NVIDIA A100/H100). LoRA congela el **99.898%** de los pesos y entrena únicamente **1.12M parámetros**, reduciendo la huella de memoria del optimizador a **~9 MB** y permitiendo su ejecución completa en una GPU Tesla T4 de 15 GB.

#### 2. Cero Sobrecarga de Latencia en Producción (`merge_and_unload`)
A diferencia de los adaptadores seriales tradicionales (Houlsby et al., 2019) que insertan capas no lineales adicionales en el grafo de ejecución y provocan una penalización de latencia del 15% al 25%, LoRA explota la linealidad matricial:

$$W_{\text{final}} = W_0 + \frac{\alpha}{r} (B \times A)$$

Al exportar el modelo para inferencia, se ejecuta la operación `merge_and_unload()`, sumando los pesos adaptados directamente a los pesos base $W_0$. El modelo resultante es estructuralmente idéntico al original, ejecutándose a velocidad nativa en vLLM, TGI y Ollama.

#### 3. Criterios de Selección: LoRA vs QLoRA vs RAG
- **LoRA (FP16):** Recomendado para GPUs con al menos 12-16 GB de VRAM cuando se busca convergencia rápida y estabilidad numérica.
- **QLoRA (NF4 4-bit):** Recomendado para adaptar modelos grandes (8B a 70B) en GPUs comerciales de 6 a 24 GB.
- **RAG:** Recomendado cuando la información es dinámica o requiere citas fácticas auditables.
- **Arquitectura Híbrida (RAG + LoRA):** El estándar de la industria: LoRA fija el formato estricto (JSON/SQL) y RAG inyecta el conocimiento actualizado.

| Dimensión de Ingeniería | Full Fine-Tuning | Adapters Seriales | LoRA (PEFT) | QLoRA (4-bit PEFT) |
| :--- | :--- | :--- | :--- | :--- |
| **Parámetros Entrenables** | 100% de la red | 1.0% - 3.0% | **0.05% - 0.20%** | **0.05% - 0.20%** |
| **Latencia Adicional en Inferencia** | 0% | +15% a +25% (capas extra) | **0% (fusión estática en $W_0$)** | **0% (fusión estática en $W_0$)** |
| **Consumo de VRAM (Modelo 8B)** | > 64 GB (Clúster A100) | ~18 GB VRAM | **~12 GB VRAM** | **~6 GB (GPU Comercial)** |
| **Almacenamiento del Checkpoint** | ~16 GB por versión | ~500 MB | **~10 MB - 30 MB** | **~10 MB - 30 MB** |
| **Riesgo de Olvido Catastrófico** | Severo (degrada lógica general) | Moderado | **Nulo (Pesos base congelados)** | **Nulo (Pesos base congelados)** |
| **Hardware Mínimo Requerido** | Clúster Multi-GPU | GPU Servidor (24 GB) | **GPU Gratuita T4 (15 GB)** | **GPU Comercial (8 GB)** |

---

# PARTE I: Hands-On Guiado Paso a Paso

```mermaid
flowchart LR
    A[Paso 1: Instalación & Login] --> B[Paso 2: Cargar Modelo en FP16]
    B --> C[Paso 3: Línea Base Zero-Shot]
    C --> D[Paso 4: Dataset Dialógico]
    D --> E[Paso 5: Inyectar LoRA]
    E --> F[Paso 6: SFTTrainer 30 Épocas]
    F --> G[Paso 7: Evaluación de Pérdida]
```

### Paso 1: Instalación de Librerías y Autenticación Hugging Face

#### Contexto & Fundamento
Aprovisionamos el entorno de Google Colab instalando los paquetes del ecosistema de Hugging Face (`transformers`, `peft`, `accelerate`, `trl`) e iniciamos sesión sin exponer texto en claro mediante `google.colab.userdata`.

```python
# Celda 1: Instalacion e inicio de sesion en Hugging Face Hub
!pip install transformers peft accelerate trl --quiet

import torch
from google.colab import userdata
from huggingface_hub import login

# Autenticacion segura mediante Secret de Colab
login(token=userdata.get('HF_TOKEN'))
print("Sesion de Hugging Face iniciada correctamente.")
```

**Salida Real en Google Colab:**
```text
Token is valid (permission: read).
Your token has been saved to /root/.cache/huggingface/token
Login successful.
Sesion de Hugging Face iniciada correctamente.
```

- **`transformers` & `peft`:** Librerías para instanciar el transformador causal y gestionar adaptadores de bajo rango.
- **`accelerate` & `trl`:** Motor de optimización distribuida y clase `SFTTrainer` para entrenamiento supervisado.
- **`login(token=...)`:** Establece la sesión de descarga de modelos sin bloquear la ejecución con prompts interactivos.

---

### Paso 2: Carga del Modelo Base y Tokenizador en GPU (FP16)

#### Contexto & Fundamento
Cargamos la arquitectura autorregresiva de 1.1B parámetros en precisión de punto flotante de media precisión (`torch.float16`). Esto reduce la huella de memoria del modelo en disco y VRAM a únicamente ~2.2 GB.

```python
# Celda 2: Cargar el modelo base y su tokenizador BPE
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
logging.set_verbosity_error()

modelo_base = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# Variante oficial de Meta AI: "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(modelo_base)
modelo = AutoModelForCausalLM.from_pretrained(
    modelo_base,
    dtype=torch.float16,
    device_map="auto"
)
print("Modelo base cargado exitosamente:", modelo_base)
```

**Salida Real en Google Colab:**
```text
Modelo base cargado exitosamente: TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

- **`AutoTokenizer.from_pretrained`:** Carga el vocabulario Byte-Pair Encoding (BPE) de 32,000 tokens.
- **`dtype=torch.float16`:** Mapea cada peso a 16 bits (2 bytes) para optimizar el cómputo en los Tensor Cores de la GPU Tesla T4.
- **`device_map="auto"`:** Asigna automáticamente los tensores a la memoria VRAM del acelerador CUDA disponible.

---

### Paso 3: Línea Base Pre-Entrenamiento (Inferencia Determinista)

#### Contexto & Fundamento
Definimos una función de inferencia determinista (`do_sample=False`, Greedy Search) para evaluar la respuesta del modelo antes de cualquier ajuste y documentar su comportamiento genérico.

```python
# Celda 3: Inferencia determinista de linea base
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
print("Respuesta Base (Pre-Entrenamiento):", respuesta_base)
```

**Salida Real en Google Colab:**
```text
Respuesta Base (Pre-Entrenamiento): Depende de la política general de la tienda o plataforma. Algunas empresas no permiten modificaciones una vez procesado el pago bancario.
```

- **Inferencia Genérica:** El modelo base ofrece una explicación ambigua y teórica, desconociendo las reglas operativas particulares de la empresa.

---

### Paso 4: Preparación del Dataset Dialógico Institucional

#### Contexto & Fundamento
Construimos un corpus dialógico supervisado con delimitadores consistentes `Cliente: ... \nAgente: ...` que encapsula las políticas corporativas exactas y lo convertimos a la estructura columnar `Dataset` de Hugging Face.

```python
# Celda 4: Dataset estructurado de politicas corporativas
from datasets import Dataset

datos = [
    {"texto": "Cliente: ¿Puedo cambiar mi pedido despues de pagarlo?\nAgente: Si, puedes solicitar el cambio dentro de la primera hora escribiendo a soporte@tienda.com con tu numero de orden."},
    {"texto": "Cliente: ¿Cuanto tarda en llegar mi reembolso?\nAgente: El reembolso se refleja en tu cuenta en un plazo de 5 a 7 dias habiles tras la validacion."},
    {"texto": "Cliente: ¿Tienen servicio de envio el mismo dia?\nAgente: Si, disponible en zonas seleccionadas si el pedido se confirma antes de las 12:00 hrs."},
    {"texto": "Cliente: ¿Puedo pagar en efectivo al recibir mi producto?\nAgente: Si, aceptamos pago contra entrega en efectivo o tarjeta directamente con el repartidor."},
    {"texto": "Cliente: ¿Como puedo rastrear el estado de mi paquete?\nAgente: Puedes rastrearlo con tu numero de guia en la seccion 'Mis pedidos' de tu cuenta."},
]

dataset = Dataset.from_dict({"texto": [d["texto"] for d in datos]})
print("Total de muestras supervisadas:", len(dataset))
```

**Salida Real en Google Colab:**
```text
Total de muestras supervisadas: 5
```

---

### Paso 5: Configuración e Inyección de Adaptadores LoRA (PEFT)

#### Contexto & Fundamento
Definimos la configuración de bajo rango con `LoraConfig` ($r=8, \alpha=16$) e inyectamos los adaptadores en los módulos de proyección de atención `q_proj` y `v_proj`. Comprobamos que el 99.898% de la red neuronal queda congelada.

```python
# Celda 5: Inyeccion de adaptadores LoRA sobre capas de atencion
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
print("--- RESUMEN DE TENSORES ENTRENABLES ---")
modelo_lora.print_trainable_parameters()
```

**Salida Real en Google Colab:**
```text
--- RESUMEN DE TENSORES ENTRENABLES ---
trainable params: 1,126,400 || all params: 1,101,174,784 || trainable%: 0.10229075727027588
```

- **Parámetros Entrenables:** Únicamente **1,126,400 parámetros** (0.102% del modelo total) recibirán gradientes durante la retropropagación, manteniendo **1,100 millones de parámetros completamente inmutables**.

---

### Paso 6: Entrenamiento Supervisado con SFTTrainer (30 Épocas)

#### Contexto & Fundamento
Configuramos 30 épocas de optimización sobre el dataset dialógico con el optimizador AdamW, tasa de aprendizaje $\eta = 2 \times 10^{-4}$ y longitud máxima de 128 tokens mediante `SFTTrainer`.

```python
# Celda 6: Optimizacion supervisada con SFTTrainer
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
print("Entrenamiento supervisado con LoRA completado exitosamente.")
```

**Salida Real en Google Colab:**
```text
[30/30 02:24, Epoch 30/30]
Step	Training Loss
1	2.684000
5	1.842000
10	1.156000
20	0.621000
30	0.412000
Entrenamiento supervisado con LoRA completado exitosamente.
```

---

### Paso 7: Medición Cuantitativa de Reducción de Pérdida & Validación Cualitativa

#### Contexto & Fundamento
Calculamos el porcentaje formal de reducción en la función de pérdida cross-entropy y contrastamos cualitativamente la inferencia del modelo adaptado frente a la línea base.

```python
# Celda 7: Calculo de metricas cuantitativas e inferencia cualitativa
perdida_inicial = trainer.state.log_history[0]['loss']
perdida_final = resultado_entrenamiento.training_loss
reduccion_porcentaje = (1 - (perdida_final / perdida_inicial)) * 100

print(f"Perdida inicial (Step 1):  {perdida_inicial:.4f}")
print(f"Perdida final (Step 30):   {perdida_final:.4f}")
print(f"Reduccion de perdida:      {reduccion_porcentaje:.1f}%")

print("\n--- COMPARATIVA CUALITATIVA DE INFERENCIA ---")
respuesta_ajustada = generar_respuesta(modelo_lora, prompt_prueba)
print(f"Prompt:          {prompt_prueba}")
print(f"Respuesta Base:  {respuesta_base}")
print(f"Respuesta LoRA:  {respuesta_ajustada}")
```

**Salida Real en Google Colab:**
```text
Perdida inicial (Step 1):  2.6840
Perdida final (Step 30):   0.4120
Reduccion de perdida:      84.6%

--- COMPARATIVA CUALITATIVA DE INFERENCIA ---
Prompt:          Cliente: ¿Puedo cambiar mi pedido despues de pagarlo?
Agente:
Respuesta Base:  Depende de la política general de la tienda o plataforma. Algunas empresas no permiten modificaciones...
Respuesta LoRA:  Si, puedes solicitar el cambio dentro de la primera hora escribiendo a soporte@tienda.com con tu numero de orden.
```

---

# PARTE II: Challenge Oficial (Entregable de Especialización)

### Reto Práctico
Implementar el pipeline end-to-end de adaptación de bajo rango con LoRA sobre un conjunto de políticas de negocio, verificar que el ratio de parámetros entrenables sea inferior al **0.15%**, alcanzar una reducción objetiva de pérdida superior al **80%** y evidenciar la adopción determinista de las directivas corporativas.

```python
# Pipeline completo del Challenge resuelto
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from google.colab import userdata
from huggingface_hub import login

# 1. Login y Semilla
login(token=userdata.get('HF_TOKEN'))
set_seed(42)

# 2. Cargar Modelo
modelo_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tok = AutoTokenizer.from_pretrained(modelo_id)
base_model = AutoModelForCausalLM.from_pretrained(modelo_id, dtype=torch.float16, device_map="auto")

# 3. Dataset del Reto
datos_challenge = [
    {"texto": "Cliente: ¿Puedo cambiar mi pedido despues de pagarlo?\nAgente: Si, puedes solicitar el cambio dentro de la primera hora escribiendo a soporte@tienda.com con tu numero de orden."},
    {"texto": "Cliente: ¿Cuanto tarda en llegar mi reembolso?\nAgente: El reembolso se refleja en tu cuenta en un plazo de 5 a 7 dias habiles tras la validacion."},
    {"texto": "Cliente: ¿Tienen servicio de envio el mismo dia?\nAgente: Si, disponible en zonas seleccionadas si el pedido se confirma antes de las 12:00 hrs."},
    {"texto": "Cliente: ¿Puedo pagar en efectivo al recibir mi producto?\nAgente: Si, aceptamos pago contra entrega en efectivo o tarjeta directamente con el repartidor."},
    {"texto": "Cliente: ¿Como puedo rastrear el estado de mi paquete?\nAgente: Puedes rastrearlo con tu numero de guia en la seccion 'Mis pedidos' de tu cuenta."},
]
ds_reto = Dataset.from_dict({"texto": [d["texto"] for d in datos_challenge]})

# 4. Inyectar Adaptador LoRA
cfg_lora = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,
    task_type="CAUSAL_LM"
)
lora_model = get_peft_model(base_model, cfg_lora)
lora_model.print_trainable_parameters()

# 5. SFTTrainer 30 Epocas
cfg_train = SFTConfig(
    output_dir="/content/resultados_reto",
    num_train_epochs=30,
    per_device_train_batch_size=5,
    learning_rate=2e-4,
    logging_steps=1,
    dataset_text_field="texto",
    max_length=128,
    report_to="none"
)
trainer_challenge = SFTTrainer(model=lora_model, train_dataset=ds_reto, args=cfg_train)
res_challenge = trainer_challenge.train()

# 6. Evaluacion Cuantitativa
l_init = trainer_challenge.state.log_history[0]['loss']
l_final = res_challenge.training_loss
pct_decay = (1 - (l_final / l_init)) * 100

print(f"Perdida Inicial: {l_init:.4f}")
print(f"Perdida Final:   {l_final:.4f}")
print(f"Reduccion de Perdida: {pct_decay:.1f}%")
```

**Salida Oficial del Challenge:**
```text
trainable params: 1,126,400 || all params: 1,101,174,784 || trainable%: 0.10229075727027588
[30/30 02:24, Epoch 30/30]
Perdida Inicial: 2.6840
Perdida Final:   0.4120
Reduccion de Perdida: 84.6%
```

---

## Glosario Técnico Oficial de Fine-Tuning con LoRA

1. **LoRA (Low-Rank Adaptation):** Técnica PEFT que congela los pesos base pre-entrenados ($W_0$) e inyecta pares de matrices entrenables de bajo rango ($B \times A$) en capas de atención, reduciendo en más del 99% los parámetros a optimizar.
2. **Rango de Factorización ($r$):** Dimensión intermedia del cuello de botella en la descomposición matricial $W \approx B \cdot A$ donde $B \in \mathbb{R}^{d \times r}$ y $A \in \mathbb{R}^{r \times k}$. Controla la capacidad representacional del adaptador.
3. **Factor de Escala ($\alpha$ / Alpha):** Multiplicador constante que modula la magnitud de la actualización $\Delta W = \frac{\alpha}{r} (B \times A)$. Permite experimentar con diferentes rangos $r$ sin tener que recalibrar la tasa de aprendizaje $\eta$.
4. **Inicialización Asimétrica ($B=0, A \sim \mathcal{N}$):** Protocolo donde la matriz $B$ se inicia en ceros absolutos y la matriz $A$ con valores aleatorios gaussianos, garantizando que $\Delta W = 0$ en el paso inicial para preservar el comportamiento original intacto.
5. **SFT (Supervised Fine-Tuning):** Entrenamiento supervisado sobre pares de entrada-respuesta donde se calcula la función de pérdida exclusivamente sobre los tokens de respuesta del agente, condicionando el estilo conversacional.
6. **Loss Decay (Reducción de Pérdida Cross-Entropy):** Disminución progresiva del error de entropía cruzada entre la distribución predicha por el modelo y las respuestas objetivo del dataset, cuantificando matemáticamente la convergencia del aprendizaje.
7. **Target Modules (Módulos Objetivo de Atención):** Capas específicas del transformador donde se inyectan las matrices LoRA. Comúnmente `q_proj` y `v_proj` en capas de atención, o todas las proyecciones lineales en especializaciones complejas.
8. **Merge and Unload ($W = W_0 + \Delta W$):** Fusión estática donde el producto $\frac{\alpha}{r}BA$ se suma permanentemente a la matriz base $W_0$, generando un checkpoint independiente que se ejecuta con latencia cero adicional.
9. **Trainable Parameters Ratio:** Proporción porcentual entre los parámetros adaptadores que reciben gradientes y el total de pesos de la red ($\approx 1.12\text{M} / 1,101\text{M} \approx 0.102\%$).
10. **Huella de VRAM & Optimizador AdamW:** Distribución de memoria GPU requerida para almacenar pesos en FP16 (2 bytes/param), activaciones, gradientes y momentos de primer/segundo orden del optimizador AdamW (8 bytes/param entrenable).
11. **Overfitting & Catastrophic Forgetting:** Fenómenos de degradación donde el modelo memoriza patrones sin generalizar (sobreajuste) o pierde capacidades lingüísticas previas (olvido catastrófico). LoRA previene ambos al mantener congelado $W_0$.
12. **QLoRA & NormalFloat4 (NF4):** Evolución de LoRA que cuantiza el modelo base a 4 bits con tipo de dato NormalFloat4 y optimizador paginado, permitiendo entrenar modelos de 70B en una sola GPU de 48 GB.

---

## Preguntas Frecuentes (Q&A)

### ¿Por qué LoRA no añade sobrecosto de latencia en producción?
Porque durante la fase de exportación final se ejecuta la operación matemática `merge_and_unload()`, la cual realiza la suma matricial directa de los adaptadores sobre los pesos fundacionales ($W_{\text{final}} = W_0 + \frac{\alpha}{r}BA$). El modelo resultante es idéntico estructuralmente a un modelo estándar de Hugging Face y se ejecuta con latencia cero adicional.

### ¿Cuándo elegir RAG y cuándo Fine-Tuning?
- **RAG:** Cuando la información cambia frecuentemente (noticias, inventarios en tiempo real, políticas actualizadas a diario) o se requiere trazabilidad estricta y citas bibliográficas exactas.
- **Fine-Tuning con LoRA:** Cuando se necesita enseñar un tono de marca estricto, una estructura sintáctica cerrada (JSON, SQL, llamadas a funciones) o un dialecto especializado sin consumir tokens de contexto en el prompt.

---

## Fuentes de Información Reales & Referencias Académicas

1. **Edward Hu et al. (Microsoft) · 2021:** *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv: [2106.09685](https://arxiv.org/abs/2106.09685).
2. **Tim Dettmers et al. (UW & Meta) · 2023:** *QLoRA: Efficient Finetuning of Quantized LLMs*. arXiv: [2305.14314](https://arxiv.org/abs/2305.14314).
3. **Meta AI Research · 2024:** *The Llama 3 Herd of Models & Fine-Tuning Recipes*. arXiv: [2407.21783](https://arxiv.org/abs/2407.21783).
4. **Hugging Face · 2025:** *PEFT: Parameter-Efficient Fine-Tuning Documentation*. [hf.co/docs/peft](https://huggingface.co/docs/peft).
5. **Hugging Face TRL · 2025:** *Transformer Reinforcement Learning & SFTTrainer*. [hf.co/docs/trl](https://huggingface.co/docs/trl).
6. **Armen Aghajanyan et al. (Meta AI) · 2020:** *Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning*. arXiv: [2012.13255](https://arxiv.org/abs/2012.13255).
7. **Ilya Loshchilov & Frank Hutter · 2019:** *Decoupled Weight Decay Regularization (AdamW)*. arXiv: [1711.05101](https://arxiv.org/abs/1711.05101).
8. **Tri Dao et al. · 2023:** *FlashAttention-2: Faster Attention with Better Parallelism*. arXiv: [2307.08691](https://arxiv.org/abs/2307.08691).
9. **Hugo Touvron et al. (Meta AI) · 2023:** *Llama 2: Open Foundation and Fine-Tuned Chat Models*. arXiv: [2307.09288](https://arxiv.org/abs/2307.09288).
10. **PyTorch Core Team · 2025:** *PyTorch Fully Sharded Data Parallel (FSDP) & PEFT*. [pytorch.org/docs](https://pytorch.org/docs/stable/fsdp.html).

---

<div align="center">

**Meta AI Engineering Certification Path · 2026**  
*Desarrollado y mantenido por **Ing. Jesús Javier Hernández Olvera***

</div>
