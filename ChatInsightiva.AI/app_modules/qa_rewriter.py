"""
Função de reescrita de perguntas com exibição em tempo real e chamada ao bot.
"""

import random
import streamlit as st
from core.query_router import MultiIndexFAQRouter
from utils_tools.langfuse_logger import registrar_trace_reescrita

def reescrever_perguntas(
    dataset,
    modelo="gpt-4",
    provedor="OpenAI",
    temperatura=0.7,
    top_p=1.0,
    presence_penalty=0.0,
    frequency_penalty=0.0,
    estilos=None,
    perguntas_personalizadas=None,
    max_qas=20,
    modo_qas="Amostra Aleatória",
    bloco_texto=None
):
    router = MultiIndexFAQRouter()
    total = min(len(dataset), max_qas if max_qas else len(dataset))

    if modo_qas == "Amostra Aleatória":
        dataset = random.sample(dataset, total)
    elif modo_qas == "Sequencial":
        dataset = dataset[:total]

    if bloco_texto is None:
        bloco_texto = st.empty()

    progress_bar = st.progress(0)
    respostas = []

    for i, item in enumerate(dataset):
        pergunta = item["question"]

        # Substitui por perguntas personalizadas se fornecidas
        if perguntas_personalizadas and i < len(perguntas_personalizadas):
            pergunta = perguntas_personalizadas[i]

        # Adiciona estilo se houver
        if estilos:
            estilo = ", ".join(estilos)
            prompt_final = f"Reescreva de forma {estilo}: {pergunta}"
        else:
            prompt_final = pergunta

        bloco_texto.markdown(f"""🔢 **Pergunta {i+1}/{total}**  
🧾 **Original:** {pergunta}  
🎭 **Prompt:** {prompt_final}  
⌛ Aguardando resposta do chatbot...""")

        resposta = router.responder(prompt_final).resposta
        registrar_trace_reescrita(prompt_final, resposta)

        bloco_texto.markdown(f"""✅ **Pergunta {i+1}/{total} respondida**  
🧾 **Original:** {pergunta}  
🎭 **Prompt:** {prompt_final}  
🤖 **Resposta:** {resposta}""")

        item["question"] = prompt_final
        item["answer"] = resposta
        respostas.append(item)
        progress_bar.progress((i+1)/total)

    return respostas
