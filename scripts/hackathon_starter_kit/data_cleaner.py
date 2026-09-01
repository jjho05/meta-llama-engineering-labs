#!/usr/bin/env python3
"""
Modulo de Limpieza y Sanitizacion Documental para Pipelines RAG.
Autor: Ing. Jesús Javier Hernández Olvera
"""

import re

def sanitizar_texto(texto_crudo: str) -> str:
    """
    Elimina cabeceras repetitivas, saltos de linea huerfanos y caracteres no validos
    para maximizar la calidad de los embeddings semanticos.
    """
    if not texto_crudo:
        return ""
    # 1. Normalizar saltos de linea y tabulaciones
    t = texto_crudo.replace("\r\n", "\n").replace("\t", " ")
    
    # 2. Eliminar numeros de pagina y cabeceras OCR comunes (ej. "Pagina 12 de 50")
    t = re.sub(r"P[aá]gina\s+\d+\s+de\s+\d+", "", t, flags=re.IGNORECASE)
    
    # 3. Eliminar multiples espacios en blanco consecutivos
    t = re.sub(r"[ ]{2,}", " ", t)
    
    # 4. Eliminar lineas vacias redundantes
    t = re.sub(r"\n{3,}", "\n\n", t)
    
    return t.strip()

def chunk_texto(texto: str, chunk_size: int = 350, chunk_overlap: int = 50) -> list[str]:
    """
    Segmenta un texto largo en fragmentos con solapamiento controlado (overlap).
    """
    texto_limpio = sanitizar_texto(texto)
    if len(texto_limpio) <= chunk_size:
        return [texto_limpio]
        
    chunks = []
    inicio = 0
    while inicio < len(texto_limpio):
        fin = inicio + chunk_size
        chunk = texto_limpio[inicio:fin]
        chunks.append(chunk)
        inicio += (chunk_size - chunk_overlap)
    return chunks

if __name__ == "__main__":
    ejemplo = "Reglamento Oficial.\n\nPagina 1 de 10\n\nArticulo 1:   Todos los usuarios tienen derecho a garantia."
    print("Texto original:\n", repr(ejemplo))
    print("\nTexto sanitizado:\n", repr(sanitizar_texto(ejemplo)))
