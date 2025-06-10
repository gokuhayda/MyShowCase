
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

from datasets import Dataset

# Simulação de dataset para teste
dados = {
    "question": [
        "Como funciona a demonstração do processo de diagnóstico oferecida na pascoa pela Insightiva?"
    ],
    "answer": [
        "Durante a Páscoa, a Insightiva oferece uma demonstração que inclui a segmentação de equipes e definição de líderes."
    ],
    "contexts": [[
        "A demonstração do diagnóstico inclui onboarding, valores e estrutura inicial de pesquisa para análise cultural.",
        "Não há menção a Páscoa nos conteúdos institucionais da Insightiva."
    ]],
    "ground_truth": [
        "A demonstração do diagnóstico cultural da Insightiva envolve onboarding, definição de líderes e preenchimento de valores."
    ]
}

dataset = Dataset.from_dict(dados)

# Avaliação
resultado = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)

print(resultado)
