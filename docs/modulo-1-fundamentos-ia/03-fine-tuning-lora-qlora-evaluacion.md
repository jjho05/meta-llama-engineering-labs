<div align="center">

[🏠 Inicio](../../README.md) • [📁 Módulo 1](README.md) • [⬅️ Anterior](02-prompt-engineering-avanzado-rag.md) • [Siguiente ➡️](04-del-prototipo-al-pipeline-productivo.md)

</div>

---

MÓDULO 1 TEMA 3 · FINE-TUNING Y EVALUACIÓN DE MODELOS

# Fine-tuning y Evaluación de Modelos

**Ajustar Llama sin reentrenarlo desde cero**. Domina la matemática de matrices de bajo rango (LoRA), cuantización 4-bit (QLoRA) en GPUs comerciales, preparación de datasets SFT en JSONL, métricas cuantitativas (PPL, BLEU, ROUGE) y auditoría con Llama Guard 3.

Guía de Inicio · Visión del Tema 1.3

### Resumen Ejecutivo & Visión: Especialización Eficiente de Llama

#### 1\. Resumen Ejecutivo De Modelo Generalista a Motor de Dominio

El reentrenamiento completo (*Full Fine-Tuning*) de un modelo como **Meta Llama 3 8B** requiere actualizar más de 8 mil millones de pesos neuronales y almacenar gradientes y optimizadores que demandan más de **120 GB de VRAM** en clusters multi-GPU inasequibles para la mayoría de los desarrolladores. 

Mediante **Adaptación de Bajo Rango (LoRA)** y **Cuantización NormalFloat a 4 bits (QLoRA)** , es posible congelar el 99.9% de los parámetros base y entrenar únicamente matrices adaptadoras delgadas, reduciendo la huella de memoria a menos de **6 GB de VRAM**. Esto permite especializar Llama 3 en una sola GPU comercial (como RTX 3060/4060 o Apple Silicon) preservando el 99% de su capacidad cognitiva general. 

¿Qué vas a dominar en este tema?

Aprenderás a estructurar datasets SFT en JSONL, aplicar Chat Templates con _loss masking_ , planificar hiperparámetros de entrenamiento estables (Learning Rate con Cosine Decay), medir el desempeño con Perplexity, BLEU-4 y ROUGE-L, y auditar las respuestas con evaluadores automáticos (LLM-as-a-Judge) y filtros de seguridad Llama Guard 3. 

Tema 1.3

## Fine-tuning y Evaluación de Modelos

Ajustar Llama sin reentrenarlo desde cero: adaptación eficiente con LoRA/QLoRA, métricas de evaluación cuantitativas y auditoría de seguridad.

¿No entendiste? Te lo explico fácil: Las notas adhesivas en una enciclopedia

Hacer un **Full Fine-Tuning** es como reescribir e imprimir de nuevo los 30 tomos de una enciclopedia cada vez que quieres corregir un dato (cuesta millones de pesos y tarda semanas). **LoRA** es como pegar **pequeñas notas adhesivas transparentes (Post-its)** sobre las páginas clave con tus correcciones. La enciclopedia original queda intacta, el libro pesa exactamente lo mismo y solo gastaste unos cuantos centavos en las notas adhesivas. 

Consejo Pro: Configuración Óptima de Rango LoRA ($r=16, \alpha=32$)

En la gran mayoría de casos de uso empresariales (atención a clientes, extracción de JSON, clasificación médica), un rango $r=16$ con factor de escala $\alpha=32$ y dropout de $0.05$ adaptando los módulos `q_proj, k_proj, v_proj, o_proj` logra el **99.2% del rendimiento de un Full Fine-Tuning** con menos de 12 GB de VRAM. 

Tema 1.3.1 · Paradigmas de Adaptación

### Matriz de Decisión: ¿Cuándo Fine-Tuning vs RAG vs Prompting?

#### 1\. Concepto Formal El Dilema Arquitectónico de la IA Aplicada

Uno de los errores más costosos en la industria es intentar usar **Fine-Tuning** para actualizar conocimientos fácticos dinámicos (como precios de catálogo o leyes recientes). El Fine-Tuning no es una base de datos: es un mecanismo para **enseñar formato, tono, razonamiento, jerga y estructura**. Para datos cambiantes y citas verificables, **RAG** es la arquitectura correcta. 

$$\text{Memoria}_{\text{Full FT}} \approx 2P_{\text{pesos}} + 2P_{\text{grad}} + 8P_{\text{AdamW}} + 4P_{\text{act}} \approx 16 P \text{ bytes}\text{Memoria}_{\text{QLoRA}} \approx 0.5P_{\text{base}} + 2P_{\text{adapter}} + 0.8P_{\text{adapter}} + 1.5\text{ GB}_{\text{act}} \approx 0.6 P + 2\text{ GB}

$$ 

Desglose de Memoria: Full Fine-Tuning vs QLoRA 6 variables

$P$ (Parámetros)

**Número de Pesos Neuronales:** 8 mil millones en Llama 3 8B ($P = 8.03 \times 10^9$) y 70 mil millones en Llama 3 70B. 

$2P$ (Pesos FP16)

**Almacenamiento de Pesos Base:** Cada parámetro en precisión flotante de 16 bits ocupa 2 bytes (16 GB para 8B). 

$8P$ (Optimizador AdamW)

**Momentum y Varianza:** AdamW almacena dos estados en FP32 (4 bytes cada uno) por cada parámetro entrenable, quintuplicando la memoria requerida. 

$P_{\text{adapter}} \ll P$

**Fracción de Parámetros LoRA:** En PEFT solo se calculan gradientes para el 0.1% de los pesos ($P_{\text{adapter}} \approx 8\text{M}$ parámetros), reduciendo el estado de AdamW a megabytes. 

$0.5P$ (Cuantización INT4)

**Base Congelada en QLoRA:** Al comprimir los pesos a 4 bits NormalFloat, los 8B parámetros ocupan apenas 4.4 GB de VRAM. 

$\Delta\text{VRAM}$ (Ahorro Total)

**Factor de Reducción:** Full Fine-Tuning de 8B requiere 128 GB de VRAM (A100); QLoRA permite entrenarlo en una RTX 3060 de 8 GB/12 GB. 

Banco de Pruebas 1.3.1: Matriz Interactiva de Decisión Arquitectónica 

Simulador Multicriterio RAG vs FT

#### ¿Cómo elegir la arquitectura óptima entre Prompting, RAG, LoRA y Full FT?

La selección arquitectónica depende del balance entre 4 factores fundamentales: el dinamismo con el que cambia la información, la rigidez del formato requerido, el presupuesto de GPU y la tolerancia a alucinaciones. Este simulador calcula la técnica recomendada con su respectiva justificación técnica.

**Regla de Decisión:** RAG es insustituible para conocimiento fáctico dinámico; Fine-Tuning (LoRA) es insustituible para comportamiento, tono, formato estricto (JSON/SQL) y reducción de latencia en inferencia.

Cargar Escenario de Negocio Preconfigurado: 

Caso A Caso B Caso C Caso D Caso E Custom

Frecuencia de Cambio de Datos: Muy Alta (Diaria)

Estática (Años) Dinámica (Horas)

Rigidez de Formato / Jerga: Estándar

Flexible (Lenguaje Natural) Sintaxis Estricta / Código

Presupuesto / Cómputo GPU: Bajo (0 USD / API)

Serverless / Sin GPUs Cluster A100 / H100

Tolerancia a Alucinaciones: Cero Tolerancia

Cero (Auditoría Estricta) Alta (Modo Creativo)

####  Enfoque Recomendado: RAG Semántico (Retrieval-Augmented Generation) 

100% RAG Puro

Tus datos cambian frecuentemente y necesitas cero alucinaciones con citas exactas. Fine-tuning **no** es la herramienta para inyectar hechos dinámicos; RAG resuelve esto a costo casi nulo y actualización inmediata. 

**RAG Semántico:** 88%

**Fine-Tuning LoRA:** 22%

**Prompt Engineering:** 45%

**Arquitectura Híbrida:** 55%

Autoevaluación 1.3.1

Una clínica médica desea que Llama 3 consulte diariamente los expedientes clínicos de pacientes para resumir análisis de sangre sin equivocarse en los valores de laboratorio. ¿Cuál es la arquitectura correcta?

Advertencia Crítica: Olvido Catastrófico por Sobreentrenamiento

Si entrenas a Llama 3 con más de 5 épocas sobre un dataset muy pequeño o con un Learning Rate superior a $5\times 10^{-4}$, el modelo sufrirá **olvido catastrófico** : aprenderá de memoria las respuestas de tu dataset pero perderá su capacidad de razonar, hablar en español correcto o seguir instrucciones generales. Mantén siempre entre 2 y 3 épocas con $3\%$ de Warmup. 

Tema 1.3.2 · Matemática de PEFT

### Matemática de LoRA (Low-Rank Adaptation) & Descomposición Intrínseca

#### 1\. Concepto Formal La Hipótesis del Rango Intrínseco

Aghajanyan et al. (Meta AI, 2020) y Hu et al. (Microsoft, 2021) demostraron que los cambios de peso $\Delta W$ necesarios para adaptar un modelo masivo a una tarea específica tienen una **dimensión intrínseca muy baja**. 

En lugar de actualizar la matriz completa de pesos $W_0 \in \mathbb{R}^{d \times k}$, LoRA la descompone en el producto de dos matrices de bajo rango $B \in \mathbb{R}^{d \times r}$ y $A \in \mathbb{R}^{r \times k}$, donde el rango $r \ll \min(d, k)$ (típicamente $r \in [4, 64]$). 

$$

h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B \cdot A) x

\text{Inicialización: } A \sim \mathcal{N}\left(0,\, \frac{1}{r}\right), \quad B = 0 \implies \Delta W = 0 \quad (\text{al inicio})

$$ 

Desglose de la Ecuación LoRA 6 elementos

$W_0 \in \mathbb{R}^{d \times k}$

**Matriz de Pesos Preentrenada (Congelada):** Los pesos originales de Llama 3 se mantienen inmutables ($\text{requires\_grad} = \text{False}$), evitando el olvido catastrófico. 

$r$ (Rango LoRA)

**Dimensión del Cuello de Botella:** Número de columnas en $B$ y filas en $A$. Controla la capacidad expresiva del adaptador ($r=8$ o $r=16$ es óptimo para la mayoría de tareas). 

$\alpha$ (Factor de Escala)

**Constante de Escalado de Hiperparámetros:** Escala la magnitud del gradiente. Mantener $\alpha = 2r$ estabiliza el aprendizaje al experimentar con diferentes rangos. 

$B \cdot A$ (Producto de Bajo Rango)

**Matriz de Actualización $\Delta W$:** Matriz de dimensión completa $[d \times k]$ generada multiplicando las dos matrices delgadas entrenables. 

$B = 0$ (Inicialización en Ceros)

**Preservación de Estado Inicial:** Como $B$ inicia en cero, $\Delta W = 0 \cdot A = 0$, garantizando que el modelo comience con exactamente el mismo comportamiento que el modelo preentrenado. 

$\Delta W \to 0$ (Zero Latency)

**Despliegue sin Sobrecosto:** En producción, se calcula $W_{\text{final}} = W_0 + \frac{\alpha}{r}(BA)$ una sola vez, eliminando cualquier latencia extra en inferencia. 

Banco de Pruebas 1.3.2: Calculadora de Matrices LoRA & Visualizador Gráfico 

Álgebra Lineal & Canvas 2D

#### ¿Cómo funciona la descomposición de bajo rango (Low-Rank Adaptation)?

En lugar de actualizar toda la matriz densa $W_0 \in \mathbb{R}^{d \times d}$ (que requiere gigabytes de gradientes), LoRA congela $W_0$ y entrena únicamente dos matrices delgadas $B \in \mathbb{R}^{d \times r}$ y $A \in \mathbb{R}^{r \times d}$ donde $r \ll d$. Modifica el rango $r$ y el factor $\alpha$ para ver cómo cambia la compresión en tiempo real.

**Regla de Inicialización:** Inicializar $B=0$ asegura que $\Delta W = \frac{\alpha}{r}(B \cdot A) = 0$ al inicio, garantizando cero regresión sobre las capacidades base de Llama 3.

1\. Modelo Base Meta Llama 3:

8B 70B 405B

2\. Capas Objetivo (Target Modules):

qv qkov all-linear

Rango LoRA ($r$ - Dimensión de Cuello de Botella): r = 16

r = 2 (Ultraligero) r = 16 (Óptimo) r = 128 (Máxima Expresividad)

Factor de Escala Alfa ($\alpha$): α = 32

α = 4 α = 2r = 32 (Recomendado) α = 256

Ecuación Matricial LoRA: $h = W_0 x + \frac{\alpha}{r}(B \cdot A)x$

Escalado: ΔW = 2.00 · (B·A)

Congelada

W₀ 4096 × 4096

Pesos Base (Grad=False)

+

Escala α/r = 2.00

Adaptadores LoRA (Bajo Rango)

Entrenable

B 4096 × 16

Init: B = 0

·

Entrenable

A 16 × 4096

Init: $\mathcal{N}$(0, 1/r)

=

Salida Activada

h = W₀x + 2.00 · (B·A)x 

**Sin regresión:** Al inicio $B=0 \implies \Delta W=0$.

**99.9% ahorro:** Solo 14.7M params activos.

8.0B

Parámetros Congelados

14.7M params

Parámetros Entrenables LoRA

0.183%

Porcentaje Entrenable

29.4 MB

Peso del Adaptador (FP16)

Autoevaluación 1.3.2

¿Por qué en el algoritmo LoRA la matriz adaptadora $B$ se inicializa con ceros ($B=0$) mientras que $A$ se inicializa con valores gaussianos normales?

Tema 1.3.3 · Cuantización en Entrenamiento

### QLoRA: Cuantización 4-bit NormalFloat (NF4) & Double Quantization

#### 1\. Concepto Formal Entrenamiento de Modelos Gigantes en GPUs Comerciales

Propuesto por Tim Dettmers et al. (Universidad de Washington, 2023), **QLoRA** resuelve la barrera de memoria VRAM mediante tres innovaciones fundamentales: 

1\. Tipo de Dato 4-bit NormalFloat (NF4)

Un tipo de dato teóricamente óptimo para cuantizar variables con distribución normal $\mathcal{N}(0, \sigma^2)$, reteniendo más información que INT4 estándar.

2\. Doble Cuantización (Double Quantization)

Cuantiza las propias constantes de escala de cuantización (de 32 bits a 8 bits), ahorrando $0.37$ bits por parámetro (3 GB de VRAM en Llama 70B).

3\. Optimizadores Paginados (Paged Optimizers)

Mueve automáticamente los estados del optimizador de la VRAM a la memoria RAM del sistema durante picos de gradiente, eliminando los temidos errores OOM.

$$

\tilde{W} = \text{dequantize}\left( c_1^{\text{FP32}}, c_2^{\text{FP8}}, q^{\text{NF4}} \right) + \frac{\alpha}{r} (B \cdot A)

q_i^{\text{NF4}} = \arg\min_{q \in Q_{\text{NF4}}} |w_i - q|

$$ 

Desglose de la Ecuación QLoRA 5 variables

$q^{\text{NF4}}$ (Cuantil 4 bits)

**Índice de Cuantización NormalFloat:** Almacena cada peso congelado en 4 bits de precisión compacta en VRAM (0.5 bytes por parámetro). 

$c_1, c_2$ (Constantes de Escala)

**Escalas de Doble Cuantización:** Factores de normalización de bloques que permiten reconstruir el valor continuo en precisión FP16/BF16 en el forward pass. 

$\text{dequantize}(c_1, c_2, q)$

**Descuantización On-The-Fly:** Durante el cálculo matricial, el tensor cuantizado se expande momentáneamente a FP16 en la caché de la GPU y se descarta de inmediato. 

$\frac{\alpha}{r} (BA)$ (LoRA FP16)

**Adaptadores en Alta Precisión:** Las matrices $A$ y $B$ permanecen en precisión flotante completa de 16 bits para garantizar la convergencia matemática del gradiente. 

$\tilde{W}$ (Peso Efectivo)

**Representación Funcional Final:** La suma del peso base descomprimido más la perturbación de bajo rango ajustada por el entrenamiento. 

Banco de Pruebas 1.3.3: Simulador de Memoria VRAM de Entrenamiento 

Estimador Hardware & Diagnóstico OOM

#### ¿Por qué la memoria VRAM no solo almacena los pesos del modelo?

Durante el entrenamiento, la VRAM de la GPU se divide en 4 componentes críticos: **1\. Pesos del Modelo** (FP16/FP8/NF4), **2\. Gradientes** (activados solo para parámetros entrenables), **3\. Estados de AdamW** (8 bytes por parámetro para momentos $m_t$ y $v_t$), y **4\. Activaciones / KV Cache** (que crecen linealmente con la longitud de contexto $S$).

**Por qué QLoRA es revolucionario:** Al cuantizar los pesos base a 4-bit (NF4) y congelar el 99.9% de los parámetros, elimina el 90% de los estados de optimizador, permitiendo afinar Llama 3 8B en una sola GPU comercial de 16–24 GB.

1\. Método de Entrenamiento:

full-fp16 lora-fp16 qlora-int4

2\. Modelo Base Llama 3:

8b 70b

Longitud de Contexto ($S$ - Secuencia): 2048 tokens

512 tokens 2,048 tokens (Estándar) 8,192 tokens

Tarjeta Gráfica / Hardware:

Google Colab Free (Tesla T4 15 GB) NVIDIA RTX 4060 Laptop/Desktop (8 GB) NVIDIA RTX 4090 Desktop (24 GB) Apple Silicon M3 Max (36 GB UMA) NVIDIA A100 SXM (80 GB Cloud)

Consumo Total Estimado de VRAM: 6.5 GB

Pesos Base: **4.4 GB**

Gradientes: **0.08 GB**

Optimizador: **0.15 GB**

Activaciones: **1.8 GB**

Capacidad: 24 GB (NVIDIA RTX 4090) Viable y Estable (Holgura Segura)

Excelente configuración. El entrenamiento de Llama 3 8B consumirá aproximadamente 6.5 GB de los 24 GB disponibles. 

Autoevaluación 1.3.3

¿Cuál es la función principal de los Optimizadores Paginados (Paged Optimizers) introducidos en QLoRA?

Tema 1.3.4 · Curaduría de Datasets SFT

### Preparación de Datasets (JSONL) & Enmascaramiento de Pérdida (Loss Masking)

#### 1\. Concepto Formal Entrenar solo en las Respuestas del Asistente

En el **Ajuste Fino Supervisado (SFT)** , los datos se preparan en formato de diálogo estructurado en JSONL. Un error crítico de principiante es calcular la función de pérdida de entropía cruzada sobre _todo el texto_ (incluidas las instrucciones del sistema y las preguntas del usuario). 

El **Enmascaramiento de Pérdida (Loss Masking)** asigna el valor especial `label = -100` (ignorado por PyTorch) a todos los tokens del prompt, obligando al modelo a actualizar sus gradientes exclusivamente sobre los tokens emitidos por el asistente. 

$$

\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^{|Y|} \log P_\theta\\!\left(y_t \mid X,\, y_{1:t-1}\right) \quad \text{donde } \mathrm{label}(x_i) = {-100}\; \forall\; x_i \in X

$$ 

Desglose de Pérdida SFT Enmascarada 5 variables

$\mathcal{L}_{\text{SFT}}(\theta)$ (Loss SFT)

**Pérdida de Entropía Cruzada Condicional:** Métrica de error que los optimizadores minimizan ajustando los parámetros adaptadores $\theta$. 

$X$ (Tokens de Entrada)

**Contexto y Prompt del Usuario:** Incluye el system prompt y la pregunta. Sus etiquetas se fijan en $-100$ para que el modelo no aprenda a predecir la pregunta. 

$Y$ (Tokens de Salida)

**Respuesta Objetivo del Asistente:** La secuencia exacta de tokens de respuesta sobre los cuales sí se calcula el gradiente $\nabla_\theta \mathcal{L}$. 

$\text{label} = -100$ (Ignored Index)

**Índice de Ignorado en PyTorch:** Convención estándar en `torch.nn.CrossEntropyLoss(ignore_index=-100)` para omitir tokens en el cálculo del gradiente. 

$\log P_\theta(y_t \mid \dots)$ (Log-Prob)

**Log-Verosimilitud Autoregresiva:** Probabilidad asignada por Llama 3 al token objetivo $y_t$ condicionado en todos los tokens previos. 

Banco de Pruebas 1.3.4: Validador JSONL & Inspector Interactivo de Loss Masking 

Chat Template Llama 3 & PyTorch Tensors

#### ¿Por qué es indispensable el Loss Masking en SFT?

En Fine-Tuning Supervisado, cada registro contiene el contexto (`system`), la consulta (`user`) y la respuesta experta (`assistant`). Si no enmascaras el prompt, PyTorch calculará gradientes sobre las preguntas del usuario, haciendo que el modelo aprenda a repetir o adivinar preguntas. Con **Loss Masking** , se asigna `label = -100` al prompt para que **el 100% del gradiente optimice exclusivamente la respuesta del asistente**.

Paso 1 · Selecciona un Caso de Uso Industrial para Fine-Tuning: 

support code medical

Paso 2 · Registro JSONL del Dataset SFT (Editor en Color con Resaltado VS Code): 

dataset_sft_sample.jsonl

JSON Válido

Auditoría SFT: Compatible con SFTTrainer / TRL / Unsloth

Secuencia válida en vocabulario Llama 3 (128,256 tokens). Enmascaramiento activo en system y user (label = -100). 

Auditoría Aprobada

Paso 3 · Inspección Interna: ¿Cómo procesa PyTorch la secuencia? 

Estructura de delimitadores oficiales de Meta Llama 3:

Gradiente Activo (Asistente) label = -100 (Prompt)

Pasa el cursor sobre cada token para inspeccionar su **input_id** en el vocabulario (128,256 tokens) y su valor en el tensor de etiquetas **labels** de PyTorch: 

**label = -100:** Token del prompt. CrossEntropyLoss lo ignora. Gradiente = 0. **label = ID Real:** Token del asistente. Gradiente $\nabla_\theta \mathcal{L}$ actualiza pesos LoRA.

Sin Loss Masking (Error Típico)

  * El modelo calcula pérdida sobre las preguntas del usuario y el system prompt.
  * **Consecuencia 1:** Desperdicia capacidad aprendiendo a autocompletar preguntas en lugar de resolverlas.
  * **Consecuencia 2:** En inferencia tiende a repetir el prompt o alucinar turnos de usuario inexistentes.
  * **Consecuencia 3:** Sobreajuste (overfitting) a la sintaxis específica del conjunto de entrenamiento.

Con Loss Masking (Estándar SFT Oficial)

  * PyTorch recibe `label = -100` para los turnos `system` y `user`.
  * **Beneficio 1:** El 100% de los pasos de optimización afinan la calidad técnica de la respuesta.
  * **Beneficio 2:** Generalización óptima ante cualquier formulación o variación de la pregunta.
  * **Beneficio 3:** Cero pérdida de cómputo en tokens que el usuario proporcionará en tiempo real.

0

Tokens Totales en Muestra

0 (0%)

Enmascarados (label = -100)

0 (0%)

Entrenables (Gradiente Activo)

100% Gradiente

Eficiencia en Respuestas

Autoevaluación 1.3.4

¿Qué consecuencia negativa ocurre si NO aplicas enmascaramiento de pérdida (Loss Masking) en los tokens de las preguntas del usuario durante el fine-tuning?

Tema 1.3.5 · Régimen de Optimización

### Hiperparámetros Críticos: Learning Rate, Warmup & Batch Size Efectivo

#### 1\. Concepto Formal Estabilidad y Convergencia en Fine-Tuning

Ajustar adaptadores LoRA requiere una tasa de aprendizaje (*learning rate*) sustancialmente más alta que el preentrenamiento (típicamente $\eta \in [1\times 10^{-4}, 2\times 10^{-4}]$ frente a $\eta \approx 1\times 10^{-6}$ en full fine-tuning). 

Para garantizar una convergencia suave sin desestabilizar las representaciones latentes, se utiliza un planificador con **calentamiento lineal (*Warmup*)** seguido de un decaimiento por **Coseno (*Cosine Annealing*)** , complementado con **Acumulación de Gradientes (*Gradient Accumulation*)** para simular lotes masivos sin agotar la memoria de la tarjeta gráfica. 

$$

\text{Batch Size}_{\text{Efectivo}} = \text{Micro Batch Size} \times \text{Gradient Accumulation Steps} \times N_{\text{GPUs}}

\text{Steps Totales} = \left\lceil \frac{N_{\text{ejemplos}}}{\text{Batch Size}_{\text{Efectivo}}} \right\rceil \times \text{Epocas}

$$ 

Desglose de Hiperparámetros de Entrenamiento 5 variables

$B$ (Micro Batch)

**Lote Físico por GPU:** Cantidad de secuencias procesadas en cada forward pass simultáneo (ej. 2 o 4 ejemplos, limitado por la VRAM). 

$G$ (Grad Accum)

**Pasos de Acumulación:** Acumula gradientes a lo largo de múltiples micro-batches antes de ejecutar una sola actualización de pesos en el optimizador. 

$B_{\text{eff}}$ (Batch Efectivo)

**Tamaño de Lote Teórico:** Un batch size efectivo de 16 a 64 ejemplos estabiliza la estimación del gradiente estocástico de AdamW. 

$\text{Warmup}$ (Ratio)

**Porcentaje de Calentamiento:** Típicamente el 3% al 5% de los steps totales, donde el learning rate asciende linealmente desde 0 hasta el valor máximo. 

$\eta(t)$ (Cosine Decay)

**Reducción Progresiva:** Disminuye suavemente la tasa de aprendizaje siguiendo una curva cosenoidal hasta alcanzar $\eta_{\text{min}} \approx 0.1 \times \eta_{\text{max}}$. 

Banco de Pruebas 1.3.5: Planificador de Hiperparámetros & Curva de Learning Rate 

Cosine Annealing & TRL Optimizer

#### ¿Cómo configurar un régimen de entrenamiento estable y predecible?

En el fine-tuning de LLMs, un mal régimen de hiperparámetros destruye las capacidades base del modelo (olvido catastrófico) o genera inestabilidad numérica en los gradientes. Este simulador calcula en tiempo real el **Batch Size Efectivo ($B_{\text{eff}} = \text{Micro Batch} \times \text{Grad Accum}$)** , el total de pasos de optimización y la curva de decaimiento cosenoidal con **calentamiento lineal (Warmup)**.

**Regla de Oro de la Industria:** Mantén un $B_{\text{eff}} \in [16, 64]$ ejemplos. Con LoRA ($r=16$), un $\text{learning rate} = 2\times 10^{-4}$ con 3% a 5% de warmup logra la convergencia óptima en 3 épocas.

Paso 1 · Configura la Escala del Dataset y Batching por GPU: 

Tamaño del Dataset ($N$): 2,000 ejemplos

200 (Micro-SFT) 2,000 (Estándar) 20,000 (Masivo)

Cantidad de pares de diálogo en el dataset JSONL. Entre 1,000 y 3,000 ejemplos de alta calidad bastan para la mayoría de dominios.

Micro Batch Size ($B$): 2

1 (Mínimo VRAM) 2 (Recomendado 24GB) 8 (A100 80GB)

Ejemplos cargados simultáneamente en la VRAM de la GPU. Con 24 GB de VRAM y secuencia 2048, $B=2$ evita errores Out-Of-Memory.

Gradient Accumulation ($G$): 8

1 step (Sin acumulación) 8 steps (Estándar) 32 steps

Pasos de forward/backward acumulados antes de ejecutar una sola actualización de pesos en AdamW. Simula un batch mayor.

Épocas de Entrenamiento ($E$): 3 épocas

1 época 3 (Recomendado SFT) 5 épocas

Veces que el optimizador recorre el dataset completo. En LLMs, más de 3–4 épocas suele provocar sobreajuste (memorización de sintaxis).

Paso 2 · Selecciona la Tasa de Aprendizaje y Porcentaje de Calentamiento: 

Learning Rate Máximo ($\eta_{\text{max}}$):

LoRA adapta matrices inicializadas en cero, requiriendo un LR ~100x mayor que Full FT ($2\times 10^{-4}$ vs $1\times 10^{-6}$).

5e-5 1e-4 2e-4 5e-4

Warmup Ratio (% Calentamiento): 5% (19 steps)

1% (Rápido) 5% (Óptimo) 15% (Largo)

Porcentaje inicial donde el LR asciende linealmente desde 0 hasta $\eta_{\text{max}}$. Evita que gradientes iniciales caóticos desestabilicen el modelo.

Planificador de Tasa de Aprendizaje: Warmup Lineal + Decaimiento Coseno (Cosine Annealing) Curva de Convergencia

16 ejemplos

Batch Size Efectivo ($B \times G$)

375 steps

Pasos Totales de Optimización

19 steps

Pasos de Calentamiento (Warmup)

~1h 8m

Tiempo Estimado (1x RTX 4090)

Diagnóstico de Régimen de Entrenamiento:

Régimen Óptimo SFT

El Batch Size Efectivo de **16 ejemplos** con **375 pasos totales** y **19 steps de warmup (5%)** garantiza un balance ideal entre estabilidad del estimador AdamW y velocidad de convergencia en 1x GPU comercial. 

Paso 3 · Script de Configuración Generado en Vivo para Hugging Face / TRL: 

training_args_config.py (Hugging Face Transformers / TRL)
    
    
    from transformers import TrainingArguments
    
    training_args = TrainingArguments(
        output_dir="./llama3-sft-checkpoint",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # Batch efectivo = 16
        learning_rate=2e-4,
        num_train_epochs=3,
        warmup_ratio=0.05,              # 19 steps de calentamiento
        lr_scheduler_type="cosine",
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
    )

Autoevaluación 1.3.5

Si tu GPU solo tiene memoria VRAM para un Micro Batch Size de 2 secuencias, pero necesitas un Batch Size Efectivo de 32 para estabilizar el optimizador, ¿qué valor de Gradient Accumulation Steps debes configurar?

Tema 1.3.6 · Métricas de Evaluación Cuantitativa

### Métricas Cuantitativas: Perplexity (PPL), BLEU-4 y ROUGE-L

#### 1\. Concepto Formal Medición Matemática del Rendimiento NLP

Para validar que el fine-tuning mejoró las respuestas del modelo sin degradar su fluidez, se emplean métricas matemáticas estandarizadas: 

Perplexity (PPL)

Exponencial de la entropía cruzada. Mide qué tan "sorprendido" está el modelo ante un texto de prueba. Valores más bajos indican mayor certeza y fluidez.

BLEU (Bilingual Evaluation Understudy)

Precisión geométrica ponderada de 1-gramas hasta 4-gramas con penalización por brevedad ($BP$). Estándar para traducción y generación de código exacto.

ROUGE (Recall-Oriented Understudy)

Mide la cobertura de información. ROUGE-1 (palabras individuales), ROUGE-2 (bigramas) y ROUGE-L (Longest Common Subsequence). Ideal para resúmenes.

$$

\text{PPL}(W) = \exp\left( -\frac{1}{N} \sum_{i=1}^{N} \log P(w_i \mid w_{1:i-1}) \right)

\text{BLEU-4} = \text{BP} \cdot \exp\left( \sum_{n=1}^{4} \frac{1}{4} \log p_n \right), \quad \text{ROUGE-L} = \frac{(1 + \beta^2) R_{\text{LCS}} P_{\text{LCS}}}{R_{\text{LCS}} + \beta^2 P_{\text{LCS}}}

$$ 

Desglose de Métricas de Evaluación NLP 6 elementos

$\text{PPL}(W)$ (Perplexity)

**Perplexity Intrínseca:** Si $\text{PPL} = 5.2$, significa que en promedio el modelo dudaba entre aproximadamente 5 palabras equiprobables en cada token. 

$p_n$ (Precisión n-gramas)

**Fracción de Coincidencia:** Proporción de secuencias contiguas de $n$ palabras generadas por el modelo que aparecen en la referencia humana. 

$\text{BP}$ (Brevity Penalty)

**Penalización por Respuestas Cortas:** $\min(1, e^{1 - r/c})$. Evita que una respuesta de una sola palabra correcta obtenga un puntaje de precisión engañoso del 100%. 

$\text{LCS}$ (Subsecuencia)

**Longest Common Subsequence:** La secuencia más larga de palabras compartidas en el mismo orden cronológico relativo, sin requerir contigüidad estricta. 

$R_{\text{LCS}}, P_{\text{LCS}}$ (Recall/Prec)

**Recall y Precisión de Estructura:** $R_{\text{LCS}} = \text{LCS} / |Ref|$ y $P_{\text{LCS}} = \text{LCS} / |Hyp|$. 

$\text{F1}_{\text{ROUGE}}$ (Media Armónica)

**Media Armónica:** Balance equilibrado entre precisión léxica y cobertura exhaustiva de conceptos. 

Banco de Pruebas 1.3.6: Calculadora en Vivo de PPL, BLEU-4 y ROUGE-L 

Métricas de Evaluación NLP

#### ¿Cómo evaluar cuantitativamente la fidelidad de un LLM adaptado?

La evaluación de texto generado requiere combinar múltiples métricas complementarias: **Perplexity (PPL)** evalúa la sorpresa probabilística (valores bajos $\approx 2–6$ indican alta fluidez); **BLEU-4** evalúa coincidencia exacta de 4-gramas contiguos (estricto en código/SQL); y **ROUGE-L** evalúa la subsecuencia común más larga (ideal para síntesis y respuestas explicativas).

**Criterio de Validación Productiva:** Se requiere $\text{ROUGE-L} \ge 0.70$ y $\text{BLEU-4} \ge 0.45$ para certificar adaptadores LoRA en banca, salud o legal-tech.

Cargar Caso de Prueba Comparativo: 

exact synonym hallucinated incomplete

Texto de Referencia Humana (Ground Truth): Objetivo Real

Predicción Generada por Meta Llama 3: Salida Modelo

1.85

Perplexity (PPL)

75.0%

BLEU-1 (Unigramas)

35.2%

BLEU-4 (4-gramas)

80.0%

ROUGE-1 (F1)

60.0%

ROUGE-2 (F1)

75.0%

ROUGE-L (LCS F1)

**Paráfrasis Semántica:** El modelo capturó el significado general (ROUGE-L alto), pero utilizó sinónimos o un orden de palabras distinto, reduciendo BLEU-4 exacto. 

Autoevaluación 1.3.6

¿Por qué un modelo con puntuación BLEU moderada (ej. 45%) puede seguir siendo una respuesta médica excelente en lenguaje natural?

Tema 1.3.7 · Evaluación Avanzada & Seguridad

### LLM-as-a-Judge, Benchmarks Estandarizados & Blindaje con Llama Guard 3

#### 1\. Concepto Formal Evaluación Automatizada y Gobernanza Ética

Las métricas léxicas tradicionales (BLEU/ROUGE) no pueden evaluar razonamiento complejo. El paradigma **LLM-as-a-Judge** (Zheng et al., LMSYS 2023) utiliza un modelo de alta capacidad como evaluador automático guiado por una rúbrica multidimensional estricta. 

Para evitar que el modelo genere contenido peligroso, tóxico o vulnerable a inyecciones de prompt, Meta desarrolló **Llama Guard 3** : un clasificador especializado de 8B parámetros que audita entradas y salidas según 14 categorías de riesgo estandarizadas (violencia, automedicación no autorizada, robo de credenciales, jailbreaks, etc.). 

$$

\bar{S} = \frac{1}{M} \sum_{j=1}^{M} S_j, \quad S_j = f_{\text{Judge}}\left(X, Y_{\text{ref}}, \hat{Y}, \mathcal{R}\right)

\text{LlamaGuard}(X, \hat{Y}) \in \\{\text{safe}\\} \cup \\{\text{unsafe}, S_1, S_2, \dots, S_{14}\\}

$$ 

Desglose de LLM-as-a-Judge & Llama Guard 5 variables

$\bar{S}$ (Puntuación Consolidada)

**Promedio Multidimensional:** Calificación ponderada en escala de 1 a 5 estrellas que evalúa Factualidad, Relevancia, Coherencia y Adherencia al Formato. 

$\mathcal{R}$ (Rúbrica Juez)

**Criterios de Evaluación Explícitos:** Prompt del juez con instrucciones precisas para mitigar sesgos de verbosidad (preferencia por respuestas largas) y sesgo de posición. 

$S_1, \dots, S_{14}$ (Riesgos)

**Taxonomía de Riesgo Llama Guard 3:** Códigos de violación de seguridad oficiales (ej. $S_{11}$: Asesoramiento Médico No Calificado, $S_{14}$: Inyección de Prompt / Jailbreak). 

$\text{safe} / \text{unsafe}$ (Veredicto)

**Veredicto Binario en Pipeline:** Si Llama Guard emite `unsafe`, el microservicio HTTP interrumpe el streaming y emite un mensaje de seguridad institucional. 

$\Delta t_{\text{Guard}}$ (Latencia Guard)

**Auditoría Ultrarrápida:** Llama Guard 3 8B cuantizado en 4 bits toma menos de 40 ms en clasificar la petición del usuario antes de pasarla al agente principal. 

Banco de Pruebas 1.3.7: Simulador LLM-as-a-Judge & Auditor Llama Guard 3 

Juez Sintético & Blindaje

#### Arquitectura de Evaluación en Dos Capas: Llama Guard 3 & LLM-as-a-Judge

La evaluación en producción no puede depender de revisiones humanas manuales. Se despliega una defensa de dos capas: **1\. Llama Guard 3** (microservicio ultra-rápido <40ms que clasifica 14 categorías de riesgo $S_1$–$S_{14}$ y frena inyecciones de prompt); y **2\. LLM-as-a-Judge** (modelo evaluador sintético que califica Factualidad, Relevancia, Coherencia y Adherencia Regulatoria en escala 1 a 5).

**Umbral de Pase:** Puntuación media $\bar{S} \ge 4.5/5.0$ y veredicto de Llama Guard `safe` para autorizar el despliegue automático del modelo.

Seleccionar Caso de Auditoría en Pipeline: 

safe-legal medical-unauthorized prompt-injection hallucinated-fin

Prompt del Usuario (Entrada): Input Stream

Resume la cláusula de indemnización del contrato adjunto.

Respuesta Emitida por Llama 3: Model Output

Según la Cláusula 14.2, la indemnización máxima es del 100% de los honorarios.

##### Rúbrica LLM-as-a-Judge

5.00 / 5.0

Factualidad: 5 / 5

Relevancia: 5 / 5

Coherencia: 5 / 5

Adherencia a Formato: 5 / 5

**Evaluación del Juez:** Respuesta impecable. Cita con precisión la cláusula específica, no inventa excepciones y mantiene estricta objetividad jurídica. 

##### Auditoría Llama Guard 3

SAFE (Seguro)

Categoría de Violación:

Ninguna (Sin Violación de Políticas)

Acción del Microservicio:

Permitir Emisión (HTTP 200)

Evaluado con el clasificador oficial Llama-Guard-3-8B con 14 políticas de gobernanza de Meta. 

Autoevaluación 1.3.7

¿Cuál es el beneficio de integrar Llama Guard 3 como filtro independiente en un pipeline de producción en lugar de confiar únicamente en el system prompt del modelo principal?

Tema 1.3.8 · Exportación y Despliegue Industrial

### Del Adaptador Entrenado a Producción: vLLM, Ollama y GGUF

#### 1\. Concepto Formal Pipeline de Exportación Post-Fine-Tuning

Una vez completado el entrenamiento con QLoRA, el modelo no está listo para producción en su forma de adaptadores separados. El flujo de exportación sigue tres etapas obligatorias: **Merge** (fusión de pesos LoRA en el modelo base), **Cuantización de Inferencia** (GGUF con llama.cpp o AWQ con AutoAWQ) y **Servicio** (Ollama, vLLM o TensorRT-LLM). Omitir el merge e intentar servir directamente adaptadores PEFT añade latencia adicional por bifurcaciones de cálculo en cada forward pass. 

**vLLM** maximiza el throughput en GPU con _PagedAttention_ y _Continuous Batching_ , ideal para APIs de alto tráfico. **Ollama** prioriza la facilidad de despliegue local con cuantización GGUF integrada, opción estándar para equipos pequeños y edge. **TensorRT-LLM** ofrece latencia mínima absoluta en hardware NVIDIA de producción a costa de una compilación previa específica por modelo y GPU. 

#### 2\. Pipeline de Despliegue Merge, Cuantización GGUF y Servicio

1

#####  Merge: Fusión de Adaptadores LoRA en Pesos Base 

El método `model.merge_and_unload()` suma matemáticamente los pesos LoRA al modelo base ($W_{\text{final}} = W_0 + \frac{\alpha}{r}BA$) y produce un checkpoint consolidado en `.safetensors` con cero sobrecarga computacional en inferencia. 

merge_lora.py (Fusión Definitiva)
    
    
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    
    # 1. Cargamos el modelo base en precisión bfloat16
    base = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", torch_dtype="bfloat16")
    
    # 2. Enlazamos los pesos del adaptador entrenado
    lora_model = PeftModel.from_pretrained(base, "./llama3-sft-checkpoint")
    
    # 3. Fusionamos matrices: W_final = W_0 + (alpha/r)*B*A
    merged = lora_model.merge_and_unload()
    merged.save_pretrained("./llama3-merged")

2

#####  Cuantización GGUF para Despliegue Local con Ollama / llama.cpp 

Convertimos el modelo fusionado a formato GGUF con cuantización `Q4_K_M` (4.5 bits por peso con k-quants), reduciendo la huella de memoria de Llama 3 8B de 16 GB (FP16) a ~4.8 GB con nula degradación de calidad. 

convert_gguf.sh (Conversión y Cuantización a 4 bits)
    
    
    # Paso 1: Convertir checkpoint Hugging Face a binario GGUF FP16
    python llama.cpp/convert_hf_to_gguf.py ./llama3-merged --outfile ./llama3-f16.gguf
    
    # Paso 2: Cuantizar a 4 bits Q4_K_M (Balance óptimo velocidad / precisión)
    ./llama.cpp/llama-quantize ./llama3-f16.gguf ./llama3-q4km.gguf q4_k_m

3

#####  Servicio en Producción: Modelfile de Ollama y Servidor vLLM 

Para uso local o edge usamos un `Modelfile` declarativo en Ollama. Para APIs de alta concurrencia usamos el servidor de **vLLM** con _PagedAttention_ y _Continuous Batching_. 

Modelfile (Despliegue Local con Ollama)
    
    
    FROM ./llama3-q4km.gguf
    SYSTEM "Eres un asistente legal experto en derecho corporativo mexicano. Cita artículos del Código de Comercio."
    PARAMETER temperature 0.2
    PARAMETER top_p 0.9

vLLM Server (API Compatible con OpenAI)
    
    
    vllm serve ./llama3-merged --host 0.0.0.0 --port 8000 --dtype bfloat16 --max-model-len 4096

#### 3\. Tabla Comparativa Motores de Servicio

Criterio | vLLM | Ollama | TensorRT-LLM  
---|---|---|---  
Throughput (tokens/s) | 3,000–8,000 | 400–800 | 5,000–12,000  
Hardware Requerido | GPU NVIDIA (CUDA) | CPU, GPU, Apple Silicon | NVIDIA H100 / A100  
API Compatible | OpenAI v1 | Ollama REST + OpenAI | Triton Inference Server  
Formato de Modelo | HuggingFace / AWQ | GGUF (llama.cpp) | TRT Engine (.trt)  
Caso de Uso Ideal | API producción multi-usuario | Dev local / Edge | Latencia crítica empresarial  
  
Regla Práctica de Selección

Para desarrollo y demos usa **Ollama + GGUF Q4_K_M** : cero configuración, cualquier hardware, Modelfile declarativo. Para APIs de producción con más de 50 usuarios concurrentes, despliega **vLLM** con PagedAttention en GPU NVIDIA — elimina la fragmentación del KV-Cache y multiplica el throughput por 3–5x frente a HuggingFace Inference nativo.

Práctica & Aplicación de Ingeniería

## Ejercicios Prácticos del Tema 1.3

Pon a prueba tu comprensión técnica resolviendo casos reales de ingeniería de LLMs. Antes de ver la solución, trabaja con la guía de conceptos clave que precede cada enunciado.

Ejercicio 1

#### Cálculo de Parámetros y VRAM para LoRA en Llama 3 8B

Conceptos necesarios para este ejercicio 

  * **Arquitectura LoRA:** Añade dos matrices de baja dimensión $B \in \mathbb{R}^{d \times r}$ y $A \in \mathbb{R}^{r \times d}$ por proyección adaptada, donde el rango $r \ll d_{\text{model}}$. Solo se entrenan estas matrices; los pesos originales $W_0$ se congelan.
  * **Rango $r$ y escala $\alpha$:** $r=16$ define el cuello de botella dimensional. La actualización efectiva es $\Delta W = \frac{\alpha}{r} B \cdot A$. Con $\alpha=32$ y $r=16$, el factor de escala es 2.
  * **VRAM en FP16:** Cada parámetro en FP16 (half-precision) ocupa 2 bytes. Total bytes = número de parámetros × 2.
  * **Fusión (Merge):** Antes del despliegue se calcula $W_{\text{final}} = W_0 + \frac{\alpha}{r}(B \cdot A)$ una sola vez. El resultado tiene las mismas dimensiones que $W_0$, sin overhead de inferencia.

Para un modelo Meta Llama 3 8B ($d_{\text{model}} = 4096$, 32 capas transformer), calcula la cantidad exacta de parámetros entrenables si se aplica LoRA con $r=16$ y $\alpha=32$ adaptando únicamente $W_q$ y $W_v$. Compara la VRAM requerida por los pesos entrenables en FP16 frente a un Full Fine-Tuning de 8B parámetros, y explica por qué la inferencia no sufre latencia adicional tras la fusión. 

Ver Criterio de Solución & Derivación Matemática

1

**Cálculo Exacto de Parámetros Entrenables:**

• Para cada módulo adaptado ($W_q$ y $W_v$), las matrices $A \in \mathbb{R}^{16 \times 4096}$ y $B \in \mathbb{R}^{4096 \times 16}$ suman: $(4096 \times 16) + (16 \times 4096) = 65{,}536 + 65{,}536 = 131{,}072\text{ parámetros}$.  
• Adaptando $W_q$ y $W_v$ por capa: $2 \times 131{,}072 = 262{,}144\text{ parámetros por capa}$.  
• Para las 32 capas: $32 \times 262{,}144 = \mathbf{8{,}388{,}608\text{ parámetros}}$ (~8.39M params, que representa solo el **0.104% del modelo base**). 

2

**Comparativa de VRAM y Fusión Zero-Overhead:**

• **VRAM de Pesos LoRA (FP16):** $8{,}388{,}608 \times 2\text{ bytes} = \mathbf{16.78\text{ MB}}$. En Full FT se requerirían 16 GB de pesos + 16 GB de gradientes + 64 GB de optimizador AdamW = **96 GB de VRAM**.  
• **Cero Latencia en Inferencia:** Al fusionar $W_{\text{final}} = W_0 + \frac{\alpha}{r}(B \cdot A)$, las dimensiones resultantes son idénticas a $W_0$, eliminando cualquier bifurcación de cálculo durante el runtime. 

calcular_lora_params.py
    
    
    def calcular_vram_lora(d_model=4096, r=16, capas=32, modulos=2):
        params_capa = modulos * (d_model * r + r * d_model)
        total_params = params_capa * capas
        vram_mb = (total_params * 2) / (1024 * 1024)
        print(f"Parámetros LoRA: {total_params:,} ({vram_mb:.2f} MB en FP16)")
        return total_params, vram_mb
    
    calcular_vram_lora()  # Output: 8,388,608 (16.00 MB en FP16)

Ejercicio 2

#### Construcción y Validación de Dataset SFT en JSONL para PostgreSQL DBA

Conceptos necesarios para este ejercicio 

  * **Esquema Multi-Turno SFT:** Cada línea del archivo JSONL debe contener un objeto `{"messages": [...]}` con roles `system`, `user` y `assistant`.
  * **Loss Masking (Enmascaramiento):** Asignar `label = -100` al prompt del usuario para concentrar los gradientes únicamente en la respuesta del asistente.

Diseña un registro JSONL estructurado para entrenar a un asistente experto en optimización de bases de datos PostgreSQL. El prompt debe consultar sobre una consulta lenta con 10M de filas y la respuesta debe proporcionar un diagnóstico conciso con índices B-Tree y plan de ejecución. Justifica formalmente el uso de Loss Masking. 

Ver Criterio de Solución & JSONL Oficial

1

**Registro Estructurado en Formato JSONL Oficial:**

dataset_sft_postgresql.jsonl
    
    
    {
      "messages": [
        {
          "role": "system",
          "content": "Eres un DBA senior experto en PostgreSQL. Proporciona diagnósticos con EXPLAIN ANALYZE e índices concisos."
        },
        {
          "role": "user",
          "content": "SELECT * FROM ordenes WHERE cliente_id=450 AND estado='PAGADO' tarda 4.2s con 10M filas. ¿Cómo la optimizo?"
        },
        {
          "role": "assistant",
          "content": "1. Crea índice compuesto B-Tree:\nCREATE INDEX idx_ord ON ordenes (cliente_id, estado);\n2. Ejecuta: ANALYZE ordenes;\n3. El Seq Scan pasará a Index Scan en <5ms."
        }
      ]
    }

2

**Justificación Matemática del Loss Masking:** $$

\mathcal{L}_{\text{SFT}} = -\sum_{t \in \mathcal{A}} \log P_\theta\\!\left(y_t \;\middle|\; x_{\text{sys}},\, x_{\text{usr}},\, y_{1:t-1}\right)

$$ 

Donde $\mathcal{A}$ es el conjunto exclusivo de índices de tokens del asistente. Asignar `label = -100` a los tokens de $x_{\text{sys}}$ y $x_{\text{usr}}$ garantiza que `torch.nn.CrossEntropyLoss(ignore_index=-100)` no calcule gradientes sobre el prompt del usuario. 

Ejercicio 3

#### Diseño de Pipeline con Llama Guard 3 y LLM-as-a-Judge

Conceptos necesarios para este ejercicio 

  * **ROUGE-L:** Métrica automática que mide el solapamiento de la subsecuencia común más larga (LCS) entre la respuesta generada y la referencia. Rango 0–1; valores $\geq 0.70$ indican alta fidelidad léxica en dominio específico.
  * **Llama Guard 3:** Modelo de clasificación de seguridad Meta que evalúa prompts y respuestas contra 14 categorías de riesgo ($S_1$–$S_{14}$). Se despliega como microservicio y retorna `safe` o la categoría de riesgo detectada.
  * **LLM-as-a-Judge:** Un LLM evaluador (distinto al productivo) califica respuestas en múltiples dimensiones con un prompt de evaluación estandarizado. Detecta alucinaciones semánticas que métricas automáticas como ROUGE no capturan.
  * **Alucinaciones numéricas:** En contextos financieros, los LLMs pueden fabricar tasas de interés, montos o fechas. El juez sintético los detecta cruzando la respuesta con fórmulas de referencia o datos conocidos.

Una institución financiera implementa un asistente de créditos con Llama 3 8B adaptado con QLoRA. Diseña la arquitectura del pipeline de inferencia y evaluación combinando ROUGE-L (contra 300 casos bancarios auditados) con Llama Guard 3 como filtro de seguridad. Explica qué categoría de riesgo ($S_1$–$S_{14}$) debe interceptar intentos de fuga de información y qué función cumple el juez sintético en la prevención de alucinaciones numéricas. 

Ver Criterio de Solución & Arquitectura de 3 Capas

1

**Validación Offline Pre-Despliegue:** El adaptador LoRA se evalúa contra 300 pares de referencia bancarios. Se requiere ROUGE-L $\geq 0.70$ y BLEU-4 $\geq 0.45$ para aprobar el despliegue. Un umbral inferior cancela el release automáticamente. 

2

**Filtro de Entrada/Salida — Llama Guard 3:** Cada prompt y respuesta pasa por Llama Guard 3 8B cuantizado. Si detecta **$S_{14}$** (Inyección de Prompt / Jailbreak) o **$S_2$** (Exfiltración de Datos Financieros Confidenciales), el microservicio bloquea el stream y retorna HTTP 403. 

3

**Auditoría Continua — LLM-as-a-Judge:** Un modelo evaluador califica aleatoriamente el 5% del tráfico en 4 dimensiones: Factualidad, Relevancia, Coherencia, Adherencia Regulatoria (escala 1–5). Si la media cae bajo 4.5/5.0, se genera alerta a MLOps. Los montos y tasas de las tablas de amortización se validan contra las fórmulas financieras de referencia. 

**Arquitectura complementaria:** Llama Guard 3 opera en <40ms protegiendo contra ataques externos, mientras el LLM Juez detecta alucinaciones numéricas que ninguna métrica automática puede capturar — una defensa en profundidad de dos capas.

Ejercicio 4

#### Implementación de DPO para Alineación por Preferencias

Conceptos necesarios para este ejercicio 

  * **RLHF con PPO:** Proceso de 4 etapas — entrenar reward model, inicializar política, optimizar con PPO (Proximal Policy Optimization), monitorear divergencia KL. Requiere ajustar 15+ hiperparámetros y puede colapsar la política si la KL no se controla.
  * **DPO (Direct Preference Optimization):** Elimina el reward model y el bucle RL. Optimiza directamente la política a partir de pares preferidos/rechazados $(y_w, y_l)$ usando una función de pérdida cerrada derivada de la solución analítica del RLHF.
  * **Dataset de Preferencias:** Formato de tripletas $(x, y_w, y_l)$. TRL espera columnas `prompt`, `chosen`, `rejected`. El 70-80% del dataset debe cubrir casos ambiguos donde la diferencia de calidad es sutil.
  * **Coeficiente $\beta$:** Penalización KL entre $\pi_\theta$ (política entrenada) y $\pi_{\text{ref}}$ (checkpoint SFT congelado). $\beta=0.1$ es conservador; valores altos hacen que el modelo no se separe de la referencia.

Tras un SFT exitoso, el equipo de ML de una legal-tech aplica DPO para alinear Llama 3 8B con preferencias de abogados senior usando 1,200 pares $(x, y_w, y_l)$. (a) Escribe la función de pérdida DPO formal con $\beta=0.1$ identificando cada variable; (b) Explica en tres puntos por qué DPO es más estable que RLHF-PPO; (c) Escribe el script completo con TRL `DPOTrainer` con los hiperparámetros críticos justificados. 

Ver Criterio de Solución & Script DPO Completo

**a) Función de Pérdida DPO:**

$$

\mathcal{L}_{\text{DPO}}(\pi_\theta;\, \pi_{\text{ref}}) = -\mathbb{E}_{(x,\, y_w,\, y_l) \sim \mathcal{D}} \left[ \log \sigma \\!\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$ 

  * **$\pi_\theta$** — política en entrenamiento (Llama 3 + LoRA activo).
  * **$\pi_{\text{ref}}$** — política de referencia congelada: el checkpoint SFT sin gradientes.
  * **$\beta = 0.1$** — penalización KL; valores pequeños permiten mayor divergencia de la referencia.
  * **$\sigma$** — sigmoide: transforma la diferencia de log-ratios en probabilidad $[0, 1]$.

**b) DPO vs. RLHF-PPO:**

  * **Sin reward model:** Elimina la etapa de entrenamiento del RM separado — recorta el cómputo a la mitad y elimina el riesgo de reward hacking.
  * **Sin RL inestable:** PPO requiere ajustar clip ratio, KL coeff, GAE $\lambda$, value loss, entropy bonus. DPO es una pérdida de clasificación binaria con gradientes estables en cada paso.
  * **Eficiencia VRAM:** Con `ref_model=None` TRL desactiva los adaptadores LoRA internamente para computar los log-probs de referencia, sin necesidad de una segunda GPU.

**c) Script TRL DPOTrainer:**

dpo_legal_llama3.py
    
    
    from trl import DPOTrainer, DPOConfig
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # 1. Modelo SFT como base con QLoRA (congelado)
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        load_in_4bit=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    
    # 2. Adaptador LoRA — proyecciones de atención full
    lora_cfg = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05, bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    
    # 3. Hiperparámetros DPO con justificación técnica
    dpo_cfg = DPOConfig(
        beta=0.1,                        # KL conservador para dominio legal
        learning_rate=5e-7,              # LR bajo: previene colapso de política
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,   # Batch efectivo = 16
        max_length=2048,
        max_prompt_length=512,
        output_dir="./llama3-legal-dpo",
        logging_steps=10,
    )
    
    # 4. Entrenar — ref_model=None: TRL desactiva LoRA internamente
    trainer = DPOTrainer(
        model=model, ref_model=None,
        args=dpo_cfg,
        train_dataset=dpo_dataset,       # Columnas: prompt, chosen, rejected
        tokenizer=tokenizer,
    )
    trainer.train()
    # -> Exportar: merge_and_unload() -> GGUF -> Ollama/vLLM

Ejercicio 5

#### Análisis de Viabilidad: Fine-Tuning QLoRA en Despacho Contable

Enunciado Oficial 

Un despacho contable utilizó 200 ejemplos de dictámenes previos y una GPU gratuita en Google Colab con QLoRA. Basándote en la diferencia entre 93% y 61% de precisión terminológica, argumenta por qué QLoRA fue decisivo para que un equipo pequeño pudiera obtener esa mejora sin infraestructura empresarial. 

Ver Solución de Ingeniería Paso a Paso & Análisis de Viabilidad

1

##### Barrera de Entrada del Fine-Tuning Completo

Ajustar los 8,000 millones de parámetros de Llama 3 en 16 bits requiere almacenar los pesos base (16 GB), gradientes (16 GB) y estados del optimizador AdamW (64 GB), exigiendo al menos **96 GB de VRAM** (clúster de 2x A100 de $10,000+ USD), inaccesible para una firma mediana. 

2

##### Democratización Técnica con QLoRA (NF4 + LoRA)

• **Pesos Base Cuantizados en 4 bits:** El modelo base ocupa solo **5.5 GB** en VRAM.  
• **Matrices Adaptadoras Ligeras:** Solo se entrenan ~16 MB de parámetros ($r=16, \alpha=32$).  
• **Consumo Total:** ~7.5 GB de VRAM durante el entrenamiento, ejecutándose perfectamente en la GPU T4 gratuita de Google Colab (16 GB VRAM). 

3

##### Impacto Medible en Negocio (+32% Precisión)

El salto del **61% al 93%** demuestra que para alinear el estilo, la jerga contable y los formatos de dictamen, no se necesita reescribir todo el cerebro del modelo: basta con modular las rutas de atención con adaptadores de bajo rango. QLoRA convirtió un proyecto de decenas de miles de dólares en un entregable exitoso con costo cero de infraestructura. 

Terminología Oficial del Curso

## Glosario Técnico del Tema 1.3

Definiciones formales y rigurosas de los conceptos clave de Fine-Tuning PEFT, cuantización y métricas de evaluación. 

Oficial #01 Temario

LoRA (Low-Rank Adaptation)

Técnica de ajuste eficiente de parámetros (PEFT) que congela los pesos base $W_0$ e inyecta matrices de descomposición de bajo rango $B \cdot A$ con rango $r \ll d$, reduciendo los parámetros entrenables hasta en un 99.8%.

Ecuación Central: $h = W_0 x + \frac{\alpha}{r} (B \cdot A) x$.

Oficial #02 Temario

QLoRA (Quantized LoRA)

Variante de LoRA que cuantiza el modelo base congelado a 4 bits NormalFloat (NF4), introduce Doble Cuantización para constantes de escala y Paged Optimizers para evitar errores de memoria Out-Of-Memory (OOM).

Impacto: Permite entrenar Llama 3 8B en 6 GB de VRAM y Llama 3 70B en 48 GB.

Oficial #03 Temario

Loss Masking (Enmascaramiento de Pérdida)

Técnica de entrenamiento supervisado que asigna la etiqueta `label = -100` a todos los tokens del prompt del usuario y del sistema, garantizando que los gradientes solo se calculen sobre las respuestas del asistente.

Efecto: Concentra la capacidad del modelo en generar respuestas expertas sin sobreajustar en preguntas.

Oficial #04 Temario

Perplexity (PPL)

Métrica intrínseca calculada como el exponencial de la pérdida de entropía cruzada media ($\text{PPL} = \exp(\mathcal{L})$). Mide el número promedio de opciones equiprobables entre las que el modelo duda al predecir el siguiente token.

Interpretación: Valores más bajos indican mayor certeza y mejor modelado del lenguaje.

Oficial #05 Temario

BLEU-4 & ROUGE-L

Métricas automáticas basadas en n-gramas. BLEU evalúa la precisión léxica con penalización por brevedad ($BP$), mientras que ROUGE-L mide el recall estructural a través de la subsecuencia común más larga (LCS).

Uso: BLEU es ideal para traducción y código; ROUGE-L para resúmenes y extracción de conceptos.

Oficial #06 Temario

Llama Guard 3 & LLM-as-a-Judge

Framework de auditoría y moderación. Llama Guard 3 clasifica riesgos en 14 categorías en milisegundos, mientras LLM-as-a-Judge utiliza modelos de alta capacidad con rúbricas cuantitativas para evaluar calidad semántica.

Gobernanza: Blindaje en tiempo real contra ataques de jailbreak e inyección de prompt.

PEFT #07

Rango LoRA ($r$)

Dimensión interna del cuello de botella en las matrices adaptadoras $B \in \mathbb{R}^{d \times r}$ y $A \in \mathbb{R}^{r \times d}$. Determina la capacidad expresiva de adaptación y el número de parámetros entrenables.

Valores Típicos: $r \in [8, 64]$; valores mayores a 64 suelen incrementar riesgo de sobreajuste.

PEFT #08

Factor de Escala LoRA ($\alpha$)

Hiperparámetro constante que modula la magnitud de la actualización $\Delta W = \frac{\alpha}{r}(B \cdot A)$. Permite ajustar la tasa de aprendizaje efectiva sin alterar el learning rate base del optimizador.

Regla Práctica: Se suele fijar $\alpha = 2r$ o $\alpha = r$.

PEFT #09

4-bit NormalFloat (NF4)

Tipo de dato de cuantización no lineal teóricamente óptimo para pesos de redes neuronales preentrenadas que siguen una distribución normal $\mathcal{N}(0, \sigma^2)$, reteniendo mayor información que enteros INT4.

Cuantización Cuantílica: Divide el área bajo la curva normal en 16 intervalos de igual probabilidad.

PEFT #10

Doble Cuantización (Double Quant)

Proceso que cuantiza las propias constantes de escala de cuantización de 32 bits a 8 bits FP8, ahorrando aproximadamente 0.37 bits por parámetro (cerca de 3 GB en un modelo 65B/70B).

Optimización: Reduce la sobrecarga de memoria de los bloques de cuantización.

PEFT #11

Paged Optimizers (Paginación de Memoria)

Mecanismo que utiliza la memoria unificada de CUDA para paginar automáticamente los estados de AdamW entre la VRAM de la GPU y la RAM del sistema ante picos repentinos de memoria durante secuencias largas.

Ventaja: Elimina caídas catastróficas por errores OOM en ráfagas de cálculo.

EVAL #12

Cosine Annealing Schedule

Estrategia de decaimiento del learning rate que reduce suavemente $\eta_t$ siguiendo una curva cosenoidal tras un periodo inicial de calentamiento (*warmup*), favoreciendo la convergencia en mínimos planos.

Ecuación: $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\frac{t}{T}\pi))$.

PEFT #13

Supervised Fine-Tuning (SFT)

Entrenamiento supervisado sobre pares curados de instrucción-demostración que enseña al modelo preentrenado a adoptar un rol conversacional estructurado y seguir contratos de entrada-salida.

Alineación: Es el primer paso antes de técnicas de alineación por preferencias como DPO o RLHF.

EVAL #14

Direct Preference Optimization (DPO)

Método de alineación que optimiza directamente la política del modelo a partir de pares de respuestas preferidas ($y_w$) y rechazadas ($y_l$), prescindiendo del entrenamiento inestable de modelos de recompensa RLHF.

Estabilidad: Convierte el aprendizaje por refuerzo en una optimización de clasificación binaria cerrada.

EVAL #15

Olvido Catastrófico (Catastrophic Forgetting)

Degradación de las capacidades cognitivas generales del modelo base (razonamiento matemático, sentido común) al sobreajustar agresivamente todos los pesos en una tarea de dominio hiperespecífico.

Mitigación: LoRA y QLoRA previenen este fenómeno al mantener congelados los pesos $W_0$.

PEFT #16

Weight Merging (Fusión de Pesos)

Operación aritmética lineal que suma los adaptadores entrenados $\Delta W$ a los pesos base $W_0$ para generar un nuevo checkpoint consolidado (.safetensors / GGUF), eliminando bifurcaciones de cálculo en inferencia.

Producción: Permite exportar el modelo afinado a vLLM, Ollama o TensorRT-LLM sin dependencias PEFT.

PEFT #17

Gradient Checkpointing

Técnica de gestión de memoria que evita almacenar todas las activaciones intermedias del forward pass, recalculándolas bajo demanda en el backward pass a cambio de un 20% más de tiempo de cómputo.

Ahorro: Reduce la memoria de activaciones hasta en un 75%, permitiendo mayores longitudes de contexto.

EVAL #18

Sesgo de Posición (Position Bias en Juez IA)

Tendencia sistemática de los modelos evaluadores a favorecer la primera respuesta presentada en comparaciones por pares (A vs B), mitigable mediante evaluación bidireccional permutada y promedios.

Mitigación: Permutación obligatoria de pares (evaluar A-B y luego B-A).

Evidencia Científica & Recursos Oficiales

## Fuentes de Información Reales & Referencias Académicas

Todo el contenido técnico, formulaciones matemáticas y métodos de optimización presentados en este tema están fundamentados en investigaciones científicas publicadas y librerías oficiales de código abierto. 

Microsoft Research · 2021 Paper Fundacional PEFT

#### LoRA: Low-Rank Adaptation of Large Language Models

El artículo seminal de Edward Hu et al. que introdujo la descomposición $W_0 + \frac{\alpha}{r}(BA)$, reduciendo los parámetros entrenables en un 99.8% sin introducir latencia en inferencia. 

[ Consultar en arXiv: 2106.09685 ](https://arxiv.org/abs/2106.09685)

Univ. of Washington · 2023 Cuantización 4-bit NF4

#### QLoRA: Efficient Finetuning of Quantized LLMs

Tim Dettmers et al. introducen el tipo de dato 4-bit NormalFloat, Double Quantization y Paged Optimizers para entrenar modelos de 70B en GPUs de 48GB y de 8B en GPUs de 8GB. 

[ Consultar en arXiv: 2305.14314 ](https://arxiv.org/abs/2305.14314)

Meta AI Research · 2020 Dimensión Intrínseca

#### Intrinsic Dimensionality Explains Fine-Tuning

Aghajanyan et al. demuestran que los modelos grandes de lenguaje operan en subespacios de baja dimensión, fundamentando la base teórica de la adaptación de bajo rango. 

[ Consultar en arXiv: 2012.13255 ](https://arxiv.org/abs/2012.13255)

Stanford University · 2023 Alineación DPO

#### Direct Preference Optimization (DPO)

Rafailov et al. formulan matemáticamente cómo alinear modelos a partir de pares de preferencias humanas mediante optimización directa sin entrenar un modelo de recompensa separado. 

[ Consultar en arXiv: 2305.18290 ](https://arxiv.org/abs/2305.18290)

Meta AI · 2023 Seguridad & Moderación

#### Llama Guard: LLM-based Safeguard for Dialog

Documento técnico oficial de Llama Guard, detallando la taxonomía de 14 riesgos de seguridad, el entrenamiento del clasificador y su integración en pipelines corporativos. 

[ Consultar en arXiv: 2312.06674 ](https://arxiv.org/abs/2312.06674)

LMSYS & UC Berkeley · 2023 Evaluación con Juez IA

#### Judging LLM-as-a-Judge with MT-Bench

Zheng et al. evalúan la consistencia de modelos como evaluadores automáticos frente a humanos, identificando técnicas para mitigar sesgos de posición y verbosidad. 

[ Consultar en arXiv: 2306.05685 ](https://arxiv.org/abs/2306.05685)

IBM Research · 2002 Métrica Clásica BLEU

#### BLEU: A Method for Automatic Evaluation

Papineni et al. presentan la formulación de precisión n-grama con Brevity Penalty, el estándar histórico para evaluación de traducción y generación estructurada. 

[ Consultar en ACL: P02-1040 ](https://aclanthology.org/P02-1040/)

Univ. of Southern California · 2004 Métrica de Resumen ROUGE

#### ROUGE: A Package for Evaluation of Summaries

Chin-Yew Lin introduce las métricas de solapamiento de n-gramas orientadas a recall y el algoritmo de Longest Common Subsequence (ROUGE-L). 

[ Consultar en ACL: W04-1013 ](https://aclanthology.org/W04-1013/)

UC Berkeley · 2020 Benchmark Multitarea MMLU

#### Measuring Massive Multitask Understanding

Hendrycks et al. construyen el benchmark estandarizado MMLU con 57 materias académicas y profesionales para medir conocimiento general en modelos de lenguaje. 

[ Consultar en arXiv: 2009.03300 ](https://arxiv.org/abs/2009.03300)

Unsloth AI · 2024 Motor de Inferencia y Fine-Tuning

#### Unsloth: Fast & Memory-Efficient LLM Engine

Kernel personalizado en Triton y CUDA que acelera el entrenamiento de LoRA/QLoRA en Llama 3 hasta 2x y reduce el consumo de memoria en un 70%. 

[ Consultar en GitHub: unsloth ](https://github.com/unslothai/unsloth)

Hugging Face · 2022 Librería PEFT Oficial

#### PEFT: Parameter-Efficient Fine-Tuning

Documentación técnica y suite de herramientas de Hugging Face para inyección y fusión de adaptadores LoRA, prefix tuning y quantizadores bitsandbytes. 

[ Consultar en Hugging Face: peft ](https://github.com/huggingface/peft)

OpenAI · 2022 SFT e Instruct Tuning

#### Training Language Models to Follow Instructions

Ouyang et al. (InstructGPT) detallan el proceso de Supervised Fine-Tuning (SFT) y recolección de datasets de demostración para alinear modelos autorregresivos. 

[ Consultar en arXiv: 2203.02155 ](https://arxiv.org/abs/2203.02155)

Qdrant Systems · 2024 Base Vectorial

#### Qdrant: Vector Similarity Engine with Payload Filtering

Motor de búsqueda vectorial en Rust con soporte para filtrado geoespacial y atributos estructurados integrado con Llama 3 RAG. 

[ Consultar Qdrant Docs ](https://qdrant.tech/documentation/)

Cohere AI · 2024 Re-Ranking Neuronal

#### Cohere Rerank: Boosting Dense Retrieval Precision

Modelo cross-encoder de alta precisión que reordena los 20 mejores resultados de búsqueda densa antes de inyectarlos en el contexto de Llama 3. 

[ Consultar Cohere Rerank ](https://docs.cohere.com/docs/reranking)

LlamaIndex Core Framework RAG

#### LlamaIndex: Data Framework for LLM Applications

Herramientas avanzadas de ingesta de datos, chunking semántico y enrutamiento jerárquico de consultas sobre documentos no estructurados. 

[ Consultar LlamaIndex ](https://docs.llamaindex.ai/)

BAAI Research · 2024 Modelo de Embeddings

#### BAAI General Embedding (BGE-M3): Multi-Lingual & Multi-Functionality

Modelo líder en el benchmark MTEB para representaciones vectoriales densas y dispersas con soporte para más de 100 idiomas incluyendo español. 

[ Consultar BGE GitHub ](https://github.com/FlagOpen/FlagEmbedding)

---

<div align="center">

[⬅️ Anterior](02-prompt-engineering-avanzado-rag.md) • [🏠 Inicio](../../README.md) • [📁 Módulo 1](README.md) • [Siguiente ➡️](04-del-prototipo-al-pipeline-productivo.md)

</div>
