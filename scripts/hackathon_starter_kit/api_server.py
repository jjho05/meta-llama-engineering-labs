"""
Servidor FastAPI para el Proyecto Integrador del Hackathon.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time
from router import ModelRouter
from rag_engine import VectorRAGEngine

app = FastAPI(
    title="Motor de Asistencia IA - Hackathon Integrador",
    description="Pipeline de IA con Model Router, RAG Semantico y Adaptadores LoRA",
    version="1.0.0"
)

router = ModelRouter()
rag = VectorRAGEngine()

# Indexar documentos base de prueba
docs_demo = [
    "La garantia cubre defectos de fabrica durante los primeros 12 meses tras la compra.",
    "El proceso de devolucion requiere presentar el numero de pedido y fotografia del producto.",
    "Nuestros canales de atencion operan las 24 horas a traves de WhatsApp y correo electronico."
]
rag.index_documents(docs_demo)

class ChatRequest(BaseModel):
    mensaje: str
    usuario_id: str = "usr_demo"

class ChatResponse(BaseModel):
    respuesta: str
    ruta_usada: str
    latencia_ms: float
    fuentes: list[str] = []

@app.get("/")
def read_root():
    return {"status": "online", "service": "Hackathon IA Engine", "version": "1.0.0"}

@app.post("/v1/chat", response_model=ChatResponse)
async def procesar_chat(req: ChatRequest):
    t0 = time.perf_counter()
    if not req.mensaje.strip():
        raise HTTPException(status_code=400, detail="El mensaje no puede estar vacio.")

    decision = router.route(req.mensaje)
    fuentes_encontradas = []
    
    if decision == "RAG_PIPELINE":
        resultados = rag.search(req.mensaje, top_k=2, threshold=0.35)
        if resultados:
            contexto = " ".join([r["text"] for r in resultados])
            respuesta = f"Segun nuestras politicas: {contexto}"
            fuentes_encontradas = [r["text"][:50] + "..." for r in resultados]
        else:
            respuesta = "No se encontro informacion especifica en los manuales sobre esta consulta."
    elif decision == "LORA_ADAPTER":
        respuesta = '{"diagnostico": "Consulta especializada", "accion_recomendada": "Revisar parametros tecnicos"}'
    else:
        respuesta = "Hola, soy el asistente inteligente de IA. ¿En que puedo ayudarte hoy?"

    latencia = (time.perf_counter() - t0) * 1000
    return ChatResponse(
        respuesta=respuesta,
        ruta_usada=decision,
        latencia_ms=round(latencia, 2),
        fuentes=fuentes_encontradas
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
