# 📊 Window Functions: A Superpotência do SQL

## 🎯 Objetivo

Dominar Window Functions para resolver problemas analíticos complexos que seriam impossíveis (ou muito difíceis) com `GROUP BY` tradicional.

---

## 🧠 A Analogia Definitiva: Sala de Aula

Imagine uma sala com 30 alunos de 3 turmas (A, B, C).

**Pergunta 1:** "Qual a média de nota da minha turma?"
- **Solução:** `GROUP BY turma` → 3 linhas (uma por turma)

**Pergunta 2:** "Qual a média de nota da minha turma, mas quero ver **cada aluno** com sua nota individual?"
- **Problema:** `GROUP BY` colapsa as linhas (perde os alunos)
- **Solução:** **Window Function!**

---

## 🔑 Conceito Chave: Manter Todas as Linhas

| Técnica | O que faz | Quando usar |
|---------|-----------|-------------|
| `GROUP BY` | **Agrupa** e **reduz** linhas | "Quero 1 linha por grupo" |
| `Window Function` | **Agrega mantendo** todas as linhas | "Quero todas as linhas + contexto do grupo" |

---

## 📝 Sintaxe Fundamental
```sql
<função> OVER (
    [PARTITION BY coluna1, coluna2]  -- Dividir em grupos
    [ORDER BY coluna3]               -- Ordenar dentro de cada grupo
    [ROWS/RANGE frame]               -- Definir "janela" de linhas
)
```

**Componentes:**
1. **Função:** `ROW_NUMBER()`, `RANK()`, `SUM()`, `AVG()`, `LAG()`, etc.
2. **PARTITION BY:** "Crie universos paralelos para cada valor"
3. **ORDER BY:** "Como ordenar dentro de cada universo"
4. **Frame:** "Quantas linhas considerar" (opcional)

---

## 🎭 As 5 Funções Essenciais

### 1. ROW_NUMBER() - "O Imparcial"

**Personalidade:** Numera sequencialmente, ignorando empates.

**Quando usar:**
- ✅ Eliminar duplicatas
- ✅ Paginação
- ✅ Top N por grupo

**Exemplo:**
```sql
-- Top 3 produtos mais vendidos por categoria
WITH ranked AS (
    SELECT 
        product_name,
        category,
        sales,
        ROW_NUMBER() OVER (
            PARTITION BY category 
            ORDER BY sales DESC
        ) AS rank
    FROM products
)
SELECT * FROM ranked WHERE rank <= 3;
```

---

### 2. RANK() - "O Olímpico"

**Personalidade:** Empates ganham mesmo lugar, mas pula próximos números.

**Ranking:**
```
1º lugar: Alice (100 pontos)
1º lugar: Bob (100 pontos)   ← Empate!
3º lugar: Charlie (95 pontos) ← Pulou o 2º
```

**Quando usar:**
- ✅ Competições reais
- ✅ Rankings com empates
- ✅ Medalhas olímpicas

---

### 3. DENSE_RANK() - "O Justo"

**Personalidade:** Empates ganham mesmo lugar, mas **não** pula números.

**Ranking:**
```
1º lugar: Alice (100 pontos)
1º lugar: Bob (100 pontos)
2º lugar: Charlie (95 pontos) ← Não pulou!
```

**Quando usar:**
- ✅ Níveis de jogo/RPG
- ✅ Classificações sem pulos

---

### 4. LAG() / LEAD() - "A Máquina do Tempo"

**LAG():** Olha para trás
**LEAD():** Olha para frente

**Exemplo: Crescimento mês a mês**
```sql
SELECT 
    month,
    revenue,
    LAG(revenue) OVER (ORDER BY month) AS prev_month,
    revenue - LAG(revenue) OVER (ORDER BY month) AS growth
FROM monthly_sales;
```

**Quando usar:**
- ✅ Comparar com período anterior
- ✅ Detectar mudanças
- ✅ Calcular deltas

---

### 5. Agregações (SUM, AVG, etc.)

**Usar agregação como window function:**
```sql
SELECT 
    name,
    department,
    salary,
    -- Média do departamento (SEM agrupar!)
    AVG(salary) OVER (PARTITION BY department) AS dept_avg,
    salary - AVG(salary) OVER (PARTITION BY department) AS diff_from_avg
FROM employees;
```

**Resultado:** Cada funcionário vê a média do **seu** departamento.

---

## 🎯 Padrões Comuns

### Padrão 1: Top N por Grupo
```sql
WITH ranked AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (PARTITION BY grupo ORDER BY metrica DESC) AS rn
    FROM tabela
)
SELECT * FROM ranked WHERE rn <= N;
```

---

### Padrão 2: Comparação com Período Anterior
```sql
SELECT 
    date,
    value,
    LAG(value) OVER (ORDER BY date) AS prev_value,
    value - LAG(value) OVER (ORDER BY date) AS change
FROM time_series;
```

---

### Padrão 3: Média Móvel
```sql
SELECT 
    date,
    value,
    AVG(value) OVER (
        ORDER BY date
        ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ) AS moving_avg_7d
FROM daily_metrics;
```

---

## 🚨 Armadilhas Comuns

### Armadilha 1: Window Function no WHERE
```sql
-- ❌ ERRO: Window functions não funcionam em WHERE
SELECT * FROM employees
WHERE RANK() OVER (ORDER BY salary DESC) <= 10;

-- ✅ CORRETO: Usar CTE ou subquery
WITH ranked AS (
    SELECT *, RANK() OVER (ORDER BY salary DESC) AS r
    FROM employees
)
SELECT * FROM ranked WHERE r <= 10;
```

---

### Armadilha 2: LAST_VALUE sem Frame
```sql
-- ❌ ERRADO: Retorna linha atual, não a última
SELECT LAST_VALUE(salary) OVER (ORDER BY hire_date)
FROM employees;

-- ✅ CORRETO: Especificar frame completo
SELECT 
    LAST_VALUE(salary) OVER (
        ORDER BY hire_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    )
FROM employees;
```

---

## 🎓 Próximos Passos

1. Execute os exemplos em `examples.sql`
2. Resolva os exercícios em `exercises.sql`
3. Compare suas soluções com `solutions.sql`
4. Avance para CTEs quando dominar window functions

---

**💡 Dica:** Na entrevista, sempre verbalize: "Vou usar window function porque preciso manter todas as linhas enquanto calculo [métrica] por [grupo]."
