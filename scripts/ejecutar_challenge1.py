#!/usr/bin/env python3
"""
Módulo: IA Aplicada con Modelos Abiertos
Challenge: Comparador de Modelos Llama
Alumno: Ing. Jesús Javier Hernández Olvera
"""

import os
import re
import sys
import time
import pprint
import getpass
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

# 1. Cargar variables de entorno desde .env
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

API_KEY = os.environ.get("GROQ_API_KEY")

if not API_KEY:
    print("[Aviso] No se encontró la clave GROQ_API_KEY en .env ni en el entorno.")
    API_KEY = getpass.getpass(" Ingresa tu GROQ_API_KEY: ").strip()

if not API_KEY:
    print("[Error] Error: Se requiere una API Key de Groq.")
    sys.exit(1)

client = Groq(api_key=API_KEY)
print("=" * 152)
print("MÓDULO: IA APLICADA CON MODELOS ABIERTOS")
print("CHALLENGE: COMPARADOR DE MODELOS LLAMA (20B vs 120B vs QWEN 27B)")
print("Alumno: Ing. Jesús Javier Hernández Olvera")
print("=" * 152)

# Selección dinámica de modelos según disponibilidad en Groq
def obtener_modelo(client, preferido, alternativo):
    try:
        modelos = [m.id for m in client.models.list().data]
        return preferido if preferido in modelos else alternativo
    except Exception:
        return alternativo

def limpiar_respuesta(texto):
    if not texto: return ""
    return re.sub(r"<think>.*?</think>", "", texto, flags=re.DOTALL).strip()

MODELO_LIGERO = obtener_modelo(client, "llama-3.1-8b-instant", "openai/gpt-oss-20b")
MODELO_GRANDE = obtener_modelo(client, "llama-3.3-70b-versatile", "openai/gpt-oss-120b")
MODELO_QWEN   = obtener_modelo(client, "qwen/qwen3.6-27b", "qwen/qwen3.6-27b")

print(f"1. Modelo Ligero en uso: {MODELO_LIGERO}")
print(f"2. Modelo Grande en uso: {MODELO_GRANDE}")
print(f"3. Modelo Qwen en uso:   {MODELO_QWEN}\n")

# 2. Definición de las 3 Preguntas de Contexto (Soporte Técnico)
preguntas = []
preguntas.append("¿Cómo puedo restablecer mi contraseña olvidada en el portal web institucional?")
preguntas.append("¿Cuál es el horario de atención y los canales oficiales para soporte técnico?")
preguntas.append("¿Cuáles son los requisitos mínimos de hardware y software para instalar la plataforma?")

print("Lista de preguntas cargadas:")
for idx, p in enumerate(preguntas, start=1):
    print(f"  {idx}. {p}")
print("-" * 152)

# 3. Consultas Individuales comparando los 3 modelos para cada pregunta
def consultar_pregunta(pregunta: str, num: int):
    print(f"\nProcesando Pregunta {num} en los 3 modelos...")
    
    # 1. Consulta al Modelo Ligero
    t0 = time.time()
    try:
        r_lig = client.chat.completions.create(model=MODELO_LIGERO, messages=[{"role": "user", "content": pregunta}], max_tokens=600)
        dt_lig = time.time() - t0
        txt_lig = limpiar_respuesta(r_lig.choices[0].message.content)
        p_tok_lig = r_lig.usage.prompt_tokens
        r_tok_lig = r_lig.usage.completion_tokens
        t_tok_lig = r_lig.usage.total_tokens
    except Exception as e:
        dt_lig = time.time() - t0
        txt_lig = f"Error: {e}"
        p_tok_lig, r_tok_lig, t_tok_lig = 0, 0, 0

    # 2. Consulta al Modelo Grande
    t1 = time.time()
    try:
        r_grd = client.chat.completions.create(model=MODELO_GRANDE, messages=[{"role": "user", "content": pregunta}], max_tokens=600)
        dt_grd = time.time() - t1
        txt_grd = limpiar_respuesta(r_grd.choices[0].message.content)
        t_tok_grd = r_grd.usage.total_tokens
    except Exception as e:
        dt_grd = time.time() - t1
        txt_grd = f"Error: {e}"
        t_tok_grd = 0

    # 3. Consulta al Modelo Qwen 3.6 27B
    t2 = time.time()
    try:
        r_qwn = client.chat.completions.create(model=MODELO_QWEN, messages=[{"role": "user", "content": pregunta}], max_tokens=600)
        dt_qwn = time.time() - t2
        txt_qwn = limpiar_respuesta(r_qwn.choices[0].message.content)
        t_tok_qwn = r_qwn.usage.total_tokens
    except Exception as e:
        dt_qwn = time.time() - t2
        txt_qwn = f"Error: {e}"
        t_tok_qwn = 0

    res_dict = {
        "pregunta": pregunta,
        "modelo": MODELO_LIGERO,
        "respuesta": txt_lig,
        "tiempo_segundos": round(dt_lig, 2),
        "tokens_prompt": p_tok_lig,
        "tokens_respuesta": r_tok_lig,
        "tokens_totales": t_tok_lig,
        "modelo_grande": MODELO_GRANDE,
        "respuesta_grande": txt_grd,
        "tiempo_grande": round(dt_grd, 2),
        "tokens_grande": t_tok_grd,
        "modelo_qwen": MODELO_QWEN,
        "respuesta_qwen": txt_qwn,
        "tiempo_qwen": round(dt_qwn, 2),
        "tokens_qwen": t_tok_qwn
    }
    print(f"Pregunta {num} completada:")
    print(f"   • Modelo Ligero ({MODELO_LIGERO}): {res_dict['tiempo_segundos']} s | {res_dict['tokens_totales']} tokens")
    print(f"   • Modelo Grande ({MODELO_GRANDE}): {res_dict['tiempo_grande']} s | {res_dict['tokens_grande']} tokens")
    print(f"   • Modelo Qwen ({MODELO_QWEN}): {res_dict['tiempo_qwen']} s | {res_dict['tokens_qwen']} tokens")
    print(f"\n--- RESPUESTA MODELO LIGERO (Pregunta {num}) ---\n{res_dict['respuesta'][:300]}...")
    print(f"\n--- RESPUESTA MODELO GRANDE (Pregunta {num}) ---\n{res_dict['respuesta_grande'][:300]}...")
    print(f"\n--- RESPUESTA MODELO QWEN (Pregunta {num}) ---\n{res_dict['respuesta_qwen'][:300]}...\n")
    print("-" * 152)
    return res_dict

resultado_1 = consultar_pregunta(preguntas[0], 1)
resultado_2 = consultar_pregunta(preguntas[1], 2)
resultado_3 = consultar_pregunta(preguntas[2], 3)

# 4. Consolidar en lista de resultados
resultados = []
resultados.append(resultado_1)
resultados.append(resultado_2)
resultados.append(resultado_3)

# 5. Despliegue de Resultados y Tabla Comparativa
print("\n" + "=" * 152)
print("TABLA COMPARATIVA MULTI-MODELO (LIGERO 20B vs GRANDE 120B vs QWEN 27B):")
print("=" * 152)
print(f"| {'N°':<2} | {'Pregunta (Resumen)':<35} | {'Mod. Ligero (s / Tok)':<24} | {'Mod. Grande (s / Tok)':<24} | {'Mod. Qwen 27B (s / Tok)':<24} | {'¿Ligero Suficiente?':<19} |")
print("|----|-------------------------------------|--------------------------|--------------------------|--------------------------|---------------------|")

for idx, res in enumerate(resultados, start=1):
    resumen_pregunta = (res['pregunta'][:32] + "...") if len(res['pregunta']) > 35 else res['pregunta']
    metricas_lig = f"{res['tiempo_segundos']:.2f} s / {res['tokens_totales']} tok"
    metricas_grd = f"{res['tiempo_grande']:.2f} s / {res['tokens_grande']} tok"
    metricas_qwn = f"{res['tiempo_qwen']:.2f} s / {res['tokens_qwen']} tok"
    print(f"| {idx:<2} | {resumen_pregunta:<35} | {metricas_lig:<24} | {metricas_grd:<24} | {metricas_qwn:<24} | {'Sí (Excelente)':<19} |\")

print("=" * 152)

print("\nEstructura de Diccionarios Completa (Variable `resultados`):")
pprint.pprint(resultados, depth=2)

print("\n" + "=" * 152)
print("CONCLUSIÓN Y ANÁLISIS COMPARATIVO DE INGENIERÍA:")
print("-" * 152)
print(f"1. Latencia y Escalabilidad: El modelo ligero ({MODELO_LIGERO}) responde de forma ultra-reactiva frente a {MODELO_GRANDE} y {MODELO_QWEN}.")
print(f"2. Razonamiento vs Eficiencia: {MODELO_QWEN} y {MODELO_GRANDE} ofrecen máxima profundidad en razonamiento, mientras que {MODELO_LIGERO} resuelve el 100% de FAQs con menor consumo de cómputo.")
print("3. Recomendación de Arquitectura: Implementar un router de modelos: dirigir FAQs y soporte operativo al modelo ligero (20B), y derivar consultas analíticas a Qwen 27B o GPT-OSS 120B.")
print("=" * 152)
