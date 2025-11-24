# 📊 A/B Testing Kata — Sample Size Calculator

> “Conduct experiments and develop hypotheses using advanced statistics.”

Este exercício demonstra raciocínio estatístico rigoroso, engenharia de software limpa e a capacidade de traduzir conceitos matemáticos complexos para decisões práticas — competências essenciais para Cientistas de Dados Sêniores e Engenheiros de ML em empresas de alto nível.

---

## 🎬 O Problema

A maior parte dos cientistas de dados iniciantes comete o mesmo erro crítico:

- Roda um teste A/B sem planejamento.  
- Vê um aumento de 5% na métrica alvo.  
- Conclui prematuramente que a "variante B ganhou".  
- Sobe para produção.

O que foi ignorado?

- ❌ Tamanho mínimo de amostra.  
- ❌ Poder estatístico (Power).  
- ❌ Significância estatística (Alpha).  
- ❌ Viabilidade temporal.  

Isso frequentemente resulta em decisões baseadas em **falsos positivos** — ruído aleatório interpretado como sinal.

---

## 🎯 Objetivo do Exercício

Este kata implementa uma classe em Python para calcular quantos usuários são necessários **por grupo** antes de iniciar um experimento A/B.

Esse cálculo é essencial para evitar:

- Testes inconclusivos  
- Peeking (olhar antes da hora)  
- P-hacking  ((ou data dredging, fishing) é um erro estatístico grave que acontece quando alguém manipula o processo de análise até “encontrar”
um resultado estatisticamente significativo — mesmo quando esse resultado não é real.É basicamente “forçar” os dados a contar uma história que eles não contam.)
- Decisões enviesadas baseadas em intuição  

### A Linguagem dos Stakeholders

Stakeholders não querem saber de p-values.  
Eles querem respostas como:

> “Em quantos dias teremos um resultado confiável?”

Este kata demonstra como transformar **estatística em engenharia**, e **engenharia em decisões de negócio**.

---

## 🧱 Tradução Técnica: Do Matemático para o Negócio

A estatística aqui não é tratada como números abstratos, mas como ferramentas de **gestão de risco**.

### **Alpha (α) — “A Trava de Segurança”**

- **Técnico:** Probabilidade de erro tipo I.  
- **Executivo:** Evita que recursos sejam investidos em uma feature que “parece boa” mas não é.

### **Power (1−β) — “O Detector de Oportunidades”**

- **Técnico:** Probabilidade de rejeitar H0 quando H1 é verdadeira.  
- **Executivo:** Garante que boas ideias não sejam descartadas como “não conclusivas”.

### **MDE — Minimum Detectable Effect (“A Régua de Relevância”)**

- **Técnico:** Menor diferença detectável pelo teste.  
- **Executivo:** Evita gastar meses testando para descobrir melhorias irrelevantes.

### **Tamanho da Amostra — “O Custo do Experimento”**

- **Técnico:** Número calculado via Z-Test Power Analysis.  
- **Executivo:** Antes de começar, respondemos:  
  > “Vale a pena travar esse tráfego por 2 semanas para testar essa hipótese?”

---

## 📁 Estrutura do Projeto
applied-statistics-katas/
└── ab-testing/
├── README.md # Você está aqui
└── ab_testing.py # Implementação da classe SampleSizeCalculator


---

## 🔍 Sobre a Implementação

A classe utilitária encapsula o cálculo estatístico para determinar o tamanho da amostra necessária para testes A/B de **proporções** (ex: taxa de conversão).

### Parâmetros padrão da indústria

| Parâmetro | Valor | Significado |
|----------|-------|-------------|
| Alpha (α) | 5% | Aceitamos 5% de chance de falso positivo |
| Power | 80% | Chance de detectar efeito real |

Esses valores refletem o padrão adotado em empresas de engenharia e produto com rigor científico.

---

## 🎓 Pergunta Frequente de Entrevista (Behavioral/Technical)

### O Cenário

O Product Manager diz:

> “Vamos rodar o teste só por dois dias. Se a Variante B estiver ganhando, a gente para e sobe pra produção!”

Seu cálculo apontava que eram necessários **14 dias de dados**.

### A Pergunta

**Que erro estatístico isso representa e por que é perigoso?**

### Resposta Sênior

Isso se chama **Peeking** (Early Stopping sem correção).

- **O Erro:** Nos primeiros dias, a variância é alta, o comportamento dos dados segue um random walk.  
- **O Risco:** Aumenta drasticamente a chance de falso positivo.  
- **A Consequência:** Você pode estar implantando algo que **não funciona** — ou pior, prejudica métricas reais.

---

## 🛠️ Como Executar
from ab_testing import SampleSizeCalculator

calculator = SampleSizeCalculator()

tamanho_amostra = calculator.calculate_sample_size(
baseline_rate=0.10,
minimum_detectable_effect=0.02
)

print(f"Necessários {tamanho_amostra} usuários por variante.")

---
