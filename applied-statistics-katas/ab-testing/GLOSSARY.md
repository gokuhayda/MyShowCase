# 📘 Glossário Técnico — Fundamentos por trás de `zt_ind_solve_power`

Este glossário explica todos os conceitos estatísticos utilizados pela função  
`zt_ind_solve_power` da biblioteca *statsmodels*, aplicada em testes A/B com duas proporções independentes.

---

## 📌 Teste Z para Duas Proporções

Um **Teste Z para duas proporções** avalia se a taxa de conversão do grupo A é estatisticamente diferente da taxa do grupo B.

É adequado quando:

- cada usuário pertence a apenas um grupo (independência)  
- as métricas são **proporções** (ex: conversão, clique, churn)  
- o tamanho de amostra é suficientemente grande para aproximar a normal

---

## 📌 Poder Estatístico (Power)

O **poder** é a probabilidade de detectar um efeito real quando ele realmente existe.

Formalmente:  
> Power = P(Rejeitar H0 | H1 é verdadeira)

Em experimentos sérios, usa-se **80%** como padrão mínimo.

Quanto maior o poder:

- maior o tamanho da amostra  
- menor o risco de um **falso negativo**

---

## 📌 Significância Estatística (Alpha, α)

A significância **alpha** é o limite aceito para a probabilidade de um **falso positivo**.

Padrão da indústria:  
**α = 0.05 (5%)**

Interpretação:

> Estamos dispostos a aceitar 5% de chance de dizer que B é melhor que A por puro acaso.

---

## 📌 Efeito (Effect Size)

No contexto de proporções, o efeito representa **o tamanho da diferença entre os grupos** que estamos tentando detectar.

Exemplo:  
Taxa A = 10%  
Taxa B = 12%  
MDE = 2 pontos percentuais (0.02)

---

## 📌 Cohen’s h (Efeito Padronizado)

Para testes de proporções, a métrica estatisticamente correta para representar o efeito é o **Cohen's h**, definida como:

h = 2 * arcsin(√p1) – 2 * arcsin(√p2)


Ela padroniza proporções para a escala da distribuição normal.

A função `zt_ind_solve_power` pode receber esse valor diretamente se fornecido.

---

## 📌 Razão Entre Tamanhos dos Grupos (ratio)

Em testes A/B comuns:
ratio = 1.0

ou seja:

- metade do tráfego para A  
- metade para B  

Se A recebe o dobro de tráfego de B:

ratio = 2.0


---

## 📌 Lado do Teste (Alternative = 'two-sided')

Indica se o teste é:

- **two-sided** → queremos saber se A ≠ B  
- **one-sided** → queremos saber se B > A  

Na maioria dos testes A/B de produto:

➡ **two-sided** é o padrão recomendado

---

## 📌 Distribuição Normal Padrão (Z-score)

O teste utiliza a **distribuição normal padrão** para aproximar a distribuição das proporções.

Os valores críticos típicos:

- z(α/2) para alpha  
- z(β) para poder

Esses valores determinam o tamanho da amostra necessária.

---

## 📌 Tamanho da Amostra (Sample Size, n)

É o valor final retornado por `zt_ind_solve_power`.

Interpretação:

> Quantos usuários **por grupo** são necessários para detectar o MDE com  
> α = 5% e power = 80%.

---

## 📌 O Que `zt_ind_solve_power` Faz Exatamente

### Entrada:
- efeito (effect size)  
- alpha  
- power  
- razão entre tamanhos dos grupos  
- tipo de teste (one/two-sided)  

### Saída:
➡ **o número mínimo de observações necessárias por grupo**

---

## 📌 Por Que Isso Importa?

Porque sem esse cálculo:

- testes podem durar tempo demais (inviabilidade operacional)  
- ou durar pouco demais (falsos positivos/falsos negativos)  
- ou consumir tráfego desnecessário  
- ou levar a conclusões erradas de negócio

O cálculo de tamanho de amostra é **a base da experimentação científica aplicada a produto**.

---

## 🧠 Resumo do Glossário em Uma Frase

`zt_ind_solve_power` transforma rigor estatístico (α, β, Cohen’s h, normal padrão)  
em uma decisão prática:

> “Quantos usuários precisamos para ter um resultado confiável?”

