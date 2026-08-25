<div align="center">

[🏠 Inicio](../../README.md) • [📁 Módulo 1](README.md) • [⬅️ Anterior](challenge-2-asistente-politicas-rag.md) • [Siguiente ➡️](../modulo-2-automatizacion-agentes-whatsapp/01-whatsapp-cloud-api-arquitectura-webhooks.md)

</div>

---

MÓDULO 1 CHALLENGE 3 · FINE-TUNING CON LORA & EVALUACIÓN DE PÉRDIDA

# Fine-Tuning de Modelos Llama con LoRA & Medición de Pérdida

**Adaptación eficiente de modelos de lenguaje con Low-Rank Adaptation (LoRA) y SFTTrainer**. Congela los pesos pre-entrenados del modelo base, inyecta matrices de bajo rango entrenables ($B \times A$) en las capas de atención, ejecuta el entrenamiento supervisado en GPU T4 de Google Colab mediante el ecosistema Hugging Face (`transformers`, `peft`, `trl`) y cuantifica objetivamente el aprendizaje mediante la reducción de la función de pérdida cross-entropy.

---

## Guía de Inicio · Visión del Entregable

### Resumen Ejecutivo & Fundamento: ¿Por qué LoRA es el estándar de la industria?

#### 1. La Barrera Computacional del Full Fine-Tuning
Ajustar todos los parámetros de un modelo de lenguaje masivo (Full Fine-Tuning) requiere almacenar en memoria de GPU los pesos del modelo, los gradientes, los estados del optimizador Adam (primer y segundo momento) y las activaciones intermedias. Para un modelo de 8 billones de parámetros (8B), esto exige más de **64 GB a 80 GB de VRAM**, restringiendo el entrenamiento a costosos clústeres de servidores A100/H100 en la nube.

La técnica **LoRA (Low-Rank Adaptation)** parte de una hipótesis fundamental demostrada por Edward Hu et al. (Microsoft Research, 2021): *el cambio en los pesos de una red neuronal durante la adaptación a una tarea específica posee un "rango intrínseco" sumamente bajo*. En lugar de actualizar la matriz completa $W_0 \in \mathbb{R}^{d \times k}$, LoRA congela $W_0$ y descompone la actualización $\Delta W$ en el producto de dos matrices de bajo rango:

$$\Delta W = \frac{\alpha}{r} (B \times A)$$

Donde $A \in \mathbb{R}^{r \times k}$ se inicializa mediante una distribución gaussiana aleatoria, $B \in \mathbb{R}^{d \times r}$ se inicializa en ceros (haciendo que $\Delta W = 0$ al inicio del entrenamiento), $r$ es el rango de adaptación ($r \ll \min(d, k)$, típicamente 8 o 16), y $\alpha$ es una constante de escala que estabiliza los hiperparámetros.

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

---

## Comparativa Técnica: Full Fine-Tuning vs. LoRA (PEFT)

| Dimensión de Evaluación | Full Fine-Tuning Tradicional | Fine-Tuning con LoRA (PEFT) | Ventaja Operativa |
| :--- | :--- | :--- | :--- |
| **Parámetros Entrenables** | 100% de la red neuronal | < 1% de los pesos totales (0.1% - 0.5%) | **Reducción de 99% en memoria** |
| **VRAM de GPU Requerida** | 40 GB - 80 GB (A100 / H100) | 6 GB - 15 GB (GPU T4 Gratuita) | **Ejecución local / Colab sin costo** |
| **Tamaño de Artefactos** | ~16 GB por cada modelo ajustado | 10 MB - 50 MB (Solo adaptadores) | **Distribución ultra-ligera en producción** |
| **Riesgo de Olvido Catastrófico** | Elevado (destruye conocimiento base) | Nulo (pesos base intactos) | **Preservación total de capacidades lógicas** |
| **Conmutación en Producción** | Requiere recargar el modelo completo | Carga dinámica de adaptadores LoRA | **Un solo modelo base atiende N clientes** |

---

## Paso 0 · Configuración de Seguridad

### Gestión del Token de Hugging Face en Google Colab con Secrets (🔑)

Para descargar modelos de lenguaje e iniciar sesión en el ecosistema Hugging Face de forma segura sin exponer credenciales en repositorios públicos:

1. **Crea tu Token de Acceso:**  
   Ingresa a [Hugging Face Settings -> Tokens](https://huggingface.co/settings/tokens), crea un token de rol `read` y cópialo.
2. **Abre el Panel de Secrets:**  
   En Google Colab, pulsa el ícono de llave **Secrets (🔑)** en la barra lateral izquierda.
3. **Registra la Variable:**  
   * Nombre: `HF_TOKEN`
   * Valor: `hf_tu_token_secreto_aqui`
   * Activa el interruptor **Notebook access**.

---

## Implementación Técnica Celda por Celda

### Celda 1: `01_instalacion_autenticacion.py`
Instalamos las librerías del ecosistema Hugging Face (`transformers`, `peft`, `accelerate`, `trl`) y autenticamos la sesión de manera cifrada con Colab Secrets.

```python
# Instalar librerías e iniciar sesión en Hugging Face con el token desde Colab Secrets
!pip install transformers peft accelerate trl --quiet

import torch
from google.colab import userdata
from huggingface_hub import login

# Autenticación segura mediante Secrets
login(token=userdata.get('HF_TOKEN'))
print("Sesión de Hugging Face iniciada correctamente.")
```

#### Desglose Línea por Línea
* `L1` **`!pip install transformers peft accelerate trl --quiet`:** Instala el núcleo de modelos de Hugging Face (`transformers`), el framework de ajuste eficiente de parámetros (`peft`), el optimizador de hardware (`accelerate`) y la librería de entrenamiento supervisado de LLMs (`trl`).
* `L3-5` **`import torch, userdata, login`:** Carga el motor de tensores de PyTorch, el lector de secretos de Colab y la utilidad de autenticación de Hugging Face Hub.
* `L8` **`login(token=userdata.get('HF_TOKEN'))`:** Valida la sesión criptográfica en el Hub de Hugging Face permitiendo descargar modelos base y subir adaptadores entrenados.

---

### Celda 2: `02_cargar_modelo_base.py`
Cargamos el modelo de pesos abiertos en precisión `torch.float16` y su respectivo tokenizador con asignación automática de hardware.

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

#### Desglose Línea por Línea
* `L2-4` **`AutoModelForCausalLM, AutoTokenizer`:** Clases automáticas que infieren la arquitectura causal de lenguaje y las reglas de tokenización del repositorio especificado.
* `L6` **`modelo_base = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"`:** Identificador del modelo pre-entrenado de 1.1 billones de parámetros con arquitectura idéntica a Llama.
* `L9-10` **`dtype=torch.float16, device_map="auto"`:** Carga los tensores en precisión media de 16 bits para reducir el consumo de VRAM a solo ~2.2 GB y asigna automáticamente la memoria hacia la GPU T4 (`cuda:0`).

---

### Celda 3: `03_evaluacion_linea_base.py`
Definimos la función de inferencia determinista y evaluamos la respuesta del modelo base sin entrenar ante una consulta de atención a clientes.

```python
# Definir una función para generar texto y probar el modelo base con un prompt de ejemplo
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

prompt_prueba = "Cliente: ¿Puedo cambiar mi pedido después de pagarlo?\nAgente:"
respuesta_base = generar_respuesta(modelo, prompt_prueba)
print("--- RESPUESTA BASE (SIN ENTRENAR) ---")
print(respuesta_base)
```

#### Desglose Línea por Línea
* `L2-3` **`entrada = tokenizer(prompt, return_tensors="pt").to(...)`:** Convierte la cadena de texto en IDs numéricos de tokens y transfiere el tensor a la memoria de la GPU.
* `L4-10` **`modelo_a_usar.generate(...)`:** Ejecuta el pase hacia adelante autoregresivo con decodificación greedy (`do_sample=False`) y penalización de repetición (`no_repeat_ngram_size=3`).
* `L11-13` **`tokens_nuevos = salida[0][entrada["input_ids"].shape[1]:]`:** Corta el vector de salida para aislar exclusivamente los tokens recién generados por el modelo, decodificándolos a texto legible.

---

### Celda 4: `04_preparacion_dataset.py`
Construimos un dataset estructurado en formato supervisado con ejemplos de preguntas de clientes y respuestas estandarizadas de la empresa.

```python
# Definir una lista de ejemplos (entrada -> respuesta esperada) y convertirla en dataset
from datasets import Dataset

ejemplos = [
    {"texto": "Cliente: ¿Puedo cambiar mi pedido después de pagarlo?\nAgente: Sí, puedes solicitar el cambio dentro de la primera hora escribiendo a soporte@tienda.com."},
    {"texto": "Cliente: ¿Cuánto tarda el reembolso?\nAgente: El reembolso se refleja en un plazo de 5 a 7 días hábiles."},
    {"texto": "Cliente: ¿Tienen envío el mismo día?\nAgente: Sí, disponible en zonas seleccionadas si el pedido se confirma antes de las 12:00."},
    {"texto": "Cliente: ¿Puedo pagar en el momento de la entrega?\nAgente: Sí, aceptamos pago contra entrega en efectivo o tarjeta."},
    {"texto": "Cliente: ¿Cómo rastreo mi paquete?\nAgente: Puedes rastrearlo con el número de guía en la sección 'Mis pedidos' de tu cuenta."},
]

dataset = Dataset.from_list(ejemplos)
print(dataset)
```

#### Desglose Línea por Línea
* `L2` **`from datasets import Dataset`:** Importa la estructura de datos orientada a memoria y streaming de Hugging Face.
* `L4-10` **`ejemplos = [...]`:** Pares de diálogo etiquetados con sintaxis consistente `Cliente: ... \nAgente: ...`.
* `L12` **`Dataset.from_list(ejemplos)`:** Transforma la lista de diccionarios de Python en un objeto Dataset columnar listo para el procesamiento por lotes del entrenador.

---

### Celda 5: `05_configurar_aplicar_lora.py`
Configuramos los hiperparámetros de LoRA e inyectamos los adaptadores de bajo rango sobre las matrices de proyección de atención `q_proj` y `v_proj`.

```python
# Configurar LoRA (rango, alpha, módulos objetivo) y aplicarlo al modelo base
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
print("--- PARÁMETROS ENTRENABLES CON LORA ---")
modelo_lora.print_trainable_parameters()
```

#### Desglose Línea por Línea
* `L6` **`set_seed(42)`:** Fija la semilla aleatoria para garantizar la reproducibilidad matemática de la inicialización gaussiana de la matriz $A$.
* `L8-14` **`LoraConfig(r=8, lora_alpha=16, ...)`:** Establece un rango $r=8$, un factor de escalado $\alpha=16$ (multiplicador de 2.0) y focaliza la adaptación en las proyecciones Query (`q_proj`) y Value (`v_proj`) del mecanismo de Multi-Head Attention.
* `L16-18` **`get_peft_model(modelo, config_lora)`:** Congela los 1,100 millones de parámetros del modelo base y activa los gradientes únicamente en los tensores de LoRA, reportando menos del **0.15% de parámetros entrenables**.

---

### Celda 6: `06_entrenar_con_sfttrainer.py`
Configuramos el ciclo de entrenamiento supervisado con `SFTTrainer` (Supervised Fine-Tuning Trainer) de la librería `trl` y ejecutamos la optimización por descenso de gradiente.

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
print("Pérdida final del entrenamiento:", resultado_entrenamiento.training_loss)
```

#### Desglose Línea por Línea
* `L2` **`from trl import SFTTrainer, SFTConfig`:** Carga las herramientas de SFT diseñadas para empaquetar secuencias de texto y calcular automáticamente la función de pérdida cross-entropy.
* `L4-13` **`SFTConfig(...)`:** Configura 30 épocas completas sobre el lote de 5 ejemplos, una tasa de aprendizaje $\eta = 2 \times 10^{-4}$ con optimizador AdamW y límite de contexto de 128 tokens.
* `L15-22` **`trainer.train()`:** Ejecuta el ciclo de retropropagación (backpropagation) actualizando exclusivamente las matrices $B$ y $A$ de LoRA en VRAM.

---

### Celda 7: `07_medicion_mejora_perdida.py`
Calculamos el porcentaje de reducción de pérdida y generamos la respuesta del modelo adaptado con LoRA para validar la adquisición del estilo corporativo.

```python
# Comparar las pérdidas y verificar la mejora objetiva
perdida_inicial = trainer.state.log_history[0]['loss']
perdida_final = resultado_entrenamiento.training_loss
reduccion_pct = (1 - (perdida_final / perdida_inicial)) * 100

print("================ EVALUACIÓN CUANTITATIVA ================")
print(f"Pérdida al inicio del entrenamiento: {perdida_inicial:.4f}")
print(f"Pérdida final del entrenamiento:      {perdida_final:.4f}")
print(f"Porcentaje de Reducción de Pérdida:   {reduccion_pct:.2f}%")
print("=========================================================")

# Inferencia cualitativa
respuesta_ajustada = generar_respuesta(modelo_lora, prompt_prueba)
print("\nRespuesta del modelo ajustado (referencia):")
print(respuesta_ajustada)
```

#### Desglose Línea por Línea
* `L2` **`perdida_inicial = trainer.state.log_history[0]['loss']`:** Extrae la métrica de pérdida registrada en el primer paso del entrenamiento supervisado.
* `L3` **`perdida_final = resultado_entrenamiento.training_loss`:** Obtiene el valor medio de convergencia tras completar las 30 épocas.
* `L4-8` **`reduccion_pct = (1 - (perdida_final / perdida_inicial)) * 100`:** Cuantifica la ganancia de aprendizaje (reducción típica de entre **60% y 85% en la función de pérdida**), demostrando de forma matemática que el modelo incorporó el conocimiento objetivo.

---

## Autoevaluación & Preguntas de Verificación

1. **¿Por qué LoRA inicializa la matriz $B$ en ceros y la matriz $A$ con valores gaussianos aleatorios?**  
   *Para garantizar que al inicio del entrenamiento $\Delta W = B \times A = 0$, asegurando que la salida del modelo adaptado sea exactamente idéntica a la del modelo base original antes de realizar cualquier actualización de gradiente.*

2. **¿Cuál es la función del parámetro $\alpha$ (lora_alpha) en relación con el rango $r$?**  
   *Actúa como una constante de escala $\frac{\alpha}{r}$ sobre la magnitud de las actualizaciones de gradiente. Al mantener la proporción $\frac{\alpha}{r} = 2.0$, se puede modificar el rango $r$ sin necesidad de recalibrar drásticamente la tasa de aprendizaje ($\text{learning\_rate}$).*

3. **¿Por qué la métrica de reducción de pérdida (Loss) es el criterio objetivo principal frente a la inspección manual del texto en datasets pequeños?**  
   *En datasets reducidos de demostración (5 a 10 ejemplos), la generación de texto libre puede presentar variaciones estocásticas de muestreo. La pérdida cross-entropy mide directamente la probabilidad acumulada que el modelo asigna a los tokens correctos del dataset, siendo la métrica matemática determinista de convergencia.*
