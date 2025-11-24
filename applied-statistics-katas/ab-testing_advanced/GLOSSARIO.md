# 📘 Glossário de Tradução Técnica (Tech ↔ Business)

Este glossário ajuda a **traduzir conceitos complexos de estatística e engenharia de software** para stakeholders, sem perder rigor técnico. Útil para **entrevistas, apresentações ou reuniões do dia a dia**.

---

## 1. Arquivo: `ab_testing_advanced_cohens_h.py` (A Abordagem Clássica)

| 🧩 Termo Técnico | 💬 Explicação para Stakeholder | 🎯 Valor de Negócio |
|-----------------|-------------------------------|------------------|
| **Strategy Pattern (Design Pattern)** | 🕹️ *"O Sistema de Cartuchos"*: Imagine um console de videogame. O console é nosso sistema de testes. O "jogo" (estratégia) pode ser trocado (Frequentista ou Bayesiano) sem precisar comprar um console novo. | 🔄 Permite mudar a matemática do teste no futuro sem quebrar o sistema atual. Flexibilidade e segurança. |
| **Cohen's h (Estatística)** | 📏 *"A Régua Universal"*: Aumentar a conversão de 1% para 2% é mais difícil que de 50% para 51%. Cohen's h ajusta essa dificuldade para medir o impacto real de forma justa. | ⚖️ Evita superestimar ou subestimar impacto de mudanças. Garantia de investimento baseado em dados corretos. |
| **Frequentist Approach** | ⚖️ *"O Tribunal"*: Assume que a nova versão NÃO funciona (inocente até provar o contrário). Só declaramos vitória se tivermos provas esmagadoras (95% de confiança). | ✅ Padrão da indústria para decisões de alto risco. Evita falsos positivos. |
| **Feasibility Analyzer (Viabilidade)** | 💡 *"O Choque de Realidade"*: Calculadora que olha para o tráfego diário e diz se o teste vai levar 2 semanas ou 2 anos. | ⏱️ Evita começar testes impossíveis. Economiza tempo de engenharia e produto. |

---

## 2. Arquivo: `ab_testing_advanced_Bayesian.py` (A Abordagem Moderna)

| 🧩 Termo Técnico | 💬 Explicação para Stakeholder | 🎯 Valor de Negócio |
|-----------------|-------------------------------|------------------|
| **Bayesian Simulation (Simulação Bayesiana)** | 🎲 *"A Aposta Inteligente"*: Perguntamos: "Dadas as vendas de hoje, qual a probabilidade da Versão B ser melhor que A?". Atualizamos chances à medida que o jogo acontece. | 🚀 Permite decisões rápidas em cenários de incerteza. Responde à pergunta que o negócio realmente faz. |
| **Monte Carlo Simulation** | 🌌 *"O Multiverso"*: Computador joga o "dado" milhares de vezes para ver todos os futuros possíveis e calcular risco. | 📊 Fornece visão de risco robusta, melhor que fórmulas simplistas. |
| **Priors (Priores)** | 🏛️ *"O Histórico"*: Usamos conhecimento prévio (ex: taxas de conversão nunca passam de 5%) para calibrar o teste, em vez de começar do zero. | ⏩ Aproveita conhecimento acumulado da empresa para acelerar testes. |

---

## 3. Arquivo: `test_ab_powervisualizer.py` (Visualização & Qualidade)

| 🧩 Termo Técnico | 💬 Explicação para Stakeholder | 🎯 Valor de Negócio |
|-----------------|-------------------------------|------------------|
| **Power Curve (Curva de Poder)** | 📈 *"Gráfico de Custo-Benefício"*: Mostra visualmente que detectar uma melhoria minúscula (formiga) exige muito mais usuários que uma grande (elefante). | 🤝 Ajuda a negociar requisitos. Stakeholders entendem custo vs. impacto. |
| **MDE (Minimum Detectable Effect)** | 🛠️ *"Sensibilidade do Radar"*: Qual o tamanho mínimo da melhoria que queremos capturar? | 🎯 Define meta de sucesso antes do experimento. |
| **Parametrized Tests (@pytest)** | ⚡ *"Teste de Estresse Automatizado"*: Testa dezenas de cenários automaticamente para garantir que a calculadora nunca minta. | 🔒 Garante que a ferramenta de decisão é confiável, mesmo em uso crítico. |

--- 
