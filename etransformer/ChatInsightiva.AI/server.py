from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from utils_tools.config_loader import load_config
from core.query_router import MultiIndexFAQRouter

app = FastAPI()
config = load_config()

faq_router = MultiIndexFAQRouter(
    salutations=config['salutations'],
    welcome_messages=config['welcome_messages'],
    notification_warning=config['notification_warning_bot']
)

@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(content={"erro": "Corpo da requisição inválido ou vazio"}, status_code=400)

    pergunta = data.get("pergunta", "")

    if not pergunta:
        return JSONResponse(content={"erro": "Pergunta não enviada"}, status_code=400)

    print(f"📝 Recebida pergunta: {pergunta}")
    print(f"🔎 faq_router: {faq_router}")

    try:
        resposta = faq_router.responder(pergunta=pergunta)
        print(f"✅ Resposta: {resposta}")
    except Exception as e:
        print(f"❌ Erro na faq_router.consultar: {e}")
        return JSONResponse(content={"erro": f"Falha no processamento interno: {e}"}, status_code=500)

    # 💡 Serializa a resposta mesmo se for RespostaRAG ou outro objeto não-serializável
    if hasattr(resposta, "resposta"):
        resposta_texto = resposta.resposta or resposta.comentario
    elif hasattr(resposta, "final_output"):
        resposta_texto = resposta.final_output
    else:
        resposta_texto = str(resposta)

    return JSONResponse(content={"resposta": resposta_texto})

