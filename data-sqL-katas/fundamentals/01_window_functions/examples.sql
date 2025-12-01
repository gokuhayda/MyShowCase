-- ============================================================
-- WINDOW FUNCTIONS: EXEMPLOS FUNDAMENTAIS
-- ============================================================
-- Execute este arquivo linha por linha para entender cada conceito.
-- Recomendação: Rode cada exemplo e analise o resultado antes
-- de avançar para o próximo.
-- ============================================================

-- ============================================================
-- SETUP: Criar tabela de exemplo (vendas de produtos)
-- ============================================================

DROP TABLE IF EXISTS sample_sales CASCADE;

CREATE TABLE sample_sales (
    sale_id SERIAL PRIMARY KEY,
    product_name VARCHAR(100),
    category VARCHAR(50),
    sales_amount DECIMAL(10,2),
    sale_date DATE,
    region VARCHAR(50)
);

-- Inserir dados de exemplo
INSERT INTO sample_sales (product_name, category, sales_amount, sale_date, region) VALUES
    ('iPhone 15', 'Electronics', 5999.00, '2024-01-15', 'Southeast'),
    ('MacBook Pro', 'Electronics', 12999.00, '2024-01-16', 'Southeast'),
    ('AirPods Pro', 'Electronics', 1999.00, '2024-01-17', 'Southeast'),
    ('iPad Air', 'Electronics', 4999.00, '2024-01-18', 'Southeast'),
    ('Nike Air Max', 'Footwear', 699.00, '2024-01-15', 'South'),
    ('Adidas Ultraboost', 'Footwear', 799.00, '2024-01-16', 'South'),
    ('Vans Old Skool', 'Footwear', 399.00, '2024-01-17', 'South'),
    ('Levi''s 501', 'Clothing', 299.00, '2024-01-15', 'North'),
    ('Nike Dri-FIT', 'Clothing', 149.00, '2024-01-16', 'North'),
    ('H&M Hoodie', 'Clothing', 179.00, '2024-01-17', 'North');

-- ============================================================
-- EXEMPLO 1: Diferença entre GROUP BY e Window Function
-- ============================================================

-- 1A. Com GROUP BY (perde as linhas individuais)
-- ❌ Problema: Não consigo ver os produtos, só a agregação
SELECT 
    category,
    AVG(sales_amount) AS avg_sales
FROM sample_sales
GROUP BY category;

/*
Resultado:
category    | avg_sales
------------|----------
Electronics | 6499.00
Footwear    | 632.33
Clothing    | 209.00

Problema: Cadê os produtos individuais?
*/

-- 1B. Com Window Function (mantém todas as linhas)
-- ✅ Solução: Cada produto vê a média da sua categoria
SELECT 
    product_name,
    category,
    sales_amount,
    ROUND(AVG(sales_amount) OVER (PARTITION BY category), 2) AS category_avg,
    ROUND(sales_amount - AVG(sales_amount) OVER (PARTITION BY category), 2) AS diff_from_avg
FROM sample_sales
ORDER BY category, sales_amount DESC;

/*
Resultado:
product_name     | category    | sales_amount | category_avg | diff_from_avg
-----------------|-------------|--------------|--------------|---------------
MacBook Pro      | Electronics | 12999.00     | 6499.00      | +6500.00
iPhone 15        | Electronics | 5999.00      | 6499.00      | -500.00
iPad Air         | Electronics | 4999.00      | 6499.00      | -1500.00
AirPods Pro      | Electronics | 1999.00      | 6499.00      | -4500.00

✅ Agora temos produtos individuais + contexto da categoria!
*/

-- ============================================================
-- EXEMPLO 2: ROW_NUMBER() - Numeração Sequencial
-- ============================================================

-- 2A. Numerar produtos por vendas (global)
SELECT 
    product_name,
    category,
    sales_amount,
    ROW_NUMBER() OVER (ORDER BY sales_amount DESC) AS overall_rank
FROM sample_sales
ORDER BY overall_rank;

/*
Resultado:
product_name     | category    | sales_amount | overall_rank
-----------------|-------------|--------------|-------------
MacBook Pro      | Electronics | 12999.00     | 1
iPhone 15        | Electronics | 5999.00      | 2
iPad Air         | Electronics | 4999.00      | 3
AirPods Pro      | Electronics | 1999.00      | 4
...
*/

-- 2B. Numerar produtos DENTRO de cada categoria
-- 🎯 Caso de uso: Top 3 produtos por categoria
SELECT 
    product_name,
    category,
    sales_amount,
    ROW_NUMBER() OVER (
        PARTITION BY category      -- Separar por categoria
        ORDER BY sales_amount DESC -- Ordenar por vendas dentro da categoria
    ) AS rank_in_category
FROM sample_sales
ORDER BY category, rank_in_category;

/*
Resultado:
product_name     | category    | sales_amount | rank_in_category
-----------------|-------------|--------------|------------------
MacBook Pro      | Electronics | 12999.00     | 1  ← Top 1 em Electronics
iPhone 15        | Electronics | 5999.00      | 2  ← Top 2 em Electronics
iPad Air         | Electronics | 4999.00      | 3  ← Top 3 em Electronics
Adidas Ultraboost| Footwear    | 799.00       | 1  ← Top 1 em Footwear
Nike Air Max     | Footwear    | 699.00       | 2  ← Top 2 em Footwear
...

Cada categoria tem seu próprio ranking!
*/

-- 2C. Filtrar apenas Top 2 por categoria usando CTE
WITH ranked_products AS (
    SELECT 
        product_name,
        category,
        sales_amount,
        ROW_NUMBER() OVER (
            PARTITION BY category 
            ORDER BY sales_amount DESC
        ) AS rank
    FROM sample_sales
)
SELECT 
    category,
    product_name,
    sales_amount,
    rank
FROM ranked_products
WHERE rank <= 2
ORDER BY category, rank;

/*
Resultado: Apenas os top 2 de cada categoria
category    | product_name     | sales_amount | rank
------------|------------------|--------------|-----
Clothing    | Levi's 501       | 299.00       | 1
Clothing    | H&M Hoodie       | 179.00       | 2
Electronics | MacBook Pro      | 12999.00     | 1
Electronics | iPhone 15        | 5999.00      | 2
Footwear    | Adidas Ultraboost| 799.00       | 1
Footwear    | Nike Air Max     | 699.00       | 2
*/

-- ============================================================
-- EXEMPLO 3: RANK() vs DENSE_RANK() vs ROW_NUMBER()
-- ============================================================

-- 3A. Criar tabela com empates
DROP TABLE IF EXISTS exam_scores;

CREATE TABLE exam_scores (
    student_name VARCHAR(50),
    score INT
);

INSERT INTO exam_scores VALUES
    ('Alice', 95),
    ('Bob', 95),      -- Empate com Alice!
    ('Charlie', 90),
    ('Diana', 85),
    ('Eve', 85),      -- Empate com Diana!
    ('Frank', 80);

-- 3B. Comparar as três funções
SELECT 
    student_name,
    score,
    ROW_NUMBER() OVER (ORDER BY score DESC) AS row_num,
    RANK() OVER (ORDER BY score DESC) AS rank,
    DENSE_RANK() OVER (ORDER BY score DESC) AS dense_rank
FROM exam_scores
ORDER BY score DESC;

/*
Resultado:
student_name | score | row_num | rank | dense_rank
-------------|-------|---------|------|------------
Alice        | 95    | 1       | 1    | 1
Bob          | 95    | 2       | 1    | 1  ← Empate!
Charlie      | 90    | 3       | 3    | 2  ← RANK pulou o 2
Diana        | 85    | 4       | 4    | 3
Eve          | 85    | 5       | 4    | 3  ← Outro empate
Frank        | 80    | 6       | 6    | 4  ← RANK pulou o 5

Diferenças:
- ROW_NUMBER: 1,2,3,4,5,6 (ignora empates)
- RANK: 1,1,3,4,4,6 (pula números após empate)
- DENSE_RANK: 1,1,2,3,3,4 (não pula números)
*/

-- 3C. Quando usar cada uma?
/*
ROW_NUMBER():
  ✅ Eliminar duplicatas (pegar só a primeira ocorrência)
  ✅ Paginação (OFFSET/LIMIT)
  ✅ Quando precisa de números únicos

RANK():
  ✅ Competições esportivas (medalhas olímpicas)
  ✅ Rankings onde empates devem "contar"
  ✅ Quando "2º lugar vago" faz sentido

DENSE_RANK():
  ✅ Níveis de jogo/RPG (Bronze, Prata, Ouro)
  ✅ Quando não faz sentido pular números
  ✅ Classificações contínuas
*/

-- ============================================================
-- EXEMPLO 4: LAG() e LEAD() - Acessar Linhas Anteriores/Posteriores
-- ============================================================

-- 4A. Criar tabela de vendas mensais
DROP TABLE IF EXISTS monthly_revenue;

CREATE TABLE monthly_revenue (
    month DATE,
    revenue DECIMAL(10,2)
);

INSERT INTO monthly_revenue VALUES
    ('2024-01-01', 100000.00),
    ('2024-02-01', 110000.00),
    ('2024-03-01', 105000.00),
    ('2024-04-01', 120000.00),
    ('2024-05-01', 115000.00),
    ('2024-06-01', 130000.00);

-- 4B. Calcular crescimento mês a mês
SELECT 
    month,
    revenue,
    LAG(revenue) OVER (ORDER BY month) AS prev_month_revenue,
    revenue - LAG(revenue) OVER (ORDER BY month) AS absolute_growth,
    ROUND(
        100.0 * (revenue - LAG(revenue) OVER (ORDER BY month)) / 
        LAG(revenue) OVER (ORDER BY month),
        2
    ) AS growth_percentage
FROM monthly_revenue
ORDER BY month;

/*
Resultado:
month      | revenue   | prev_month | absolute_growth | growth_percentage
-----------|-----------|------------|-----------------|-------------------
2024-01-01 | 100000.00 | NULL       | NULL            | NULL
2024-02-01 | 110000.00 | 100000.00  | 10000.00        | 10.00
2024-03-01 | 105000.00 | 110000.00  | -5000.00        | -4.55  ← Queda!
2024-04-01 | 120000.00 | 105000.00  | 15000.00        | 14.29
2024-05-01 | 115000.00 | 120000.00  | -5000.00        | -4.17
2024-06-01 | 130000.00 | 115000.00  | 15000.00        | 13.04

LAG() trouxe o valor do mês anterior!
*/

-- 4C. LEAD() para ver o próximo mês
SELECT 
    month,
    revenue,
    LEAD(revenue) OVER (ORDER BY month) AS next_month_revenue,
    LEAD(revenue) OVER (ORDER BY month) - revenue AS expected_growth
FROM monthly_revenue
ORDER BY month;

/*
Resultado:
month      | revenue   | next_month | expected_growth
-----------|-----------|------------|----------------
2024-01-01 | 100000.00 | 110000.00  | 10000.00
2024-02-01 | 110000.00 | 105000.00  | -5000.00
2024-03-01 | 105000.00 | 120000.00  | 15000.00
2024-04-01 | 120000.00 | 115000.00  | -5000.00
2024-05-01 | 115000.00 | 130000.00  | 15000.00
2024-06-01 | 130000.00 | NULL       | NULL        ← Não há próximo mês

LEAD() trouxe o valor do próximo mês!
*/

-- 4D. LAG com offset e default
SELECT 
    month,
    revenue,
    -- Olhar 2 meses atrás
    LAG(revenue, 2) OVER (ORDER BY month) AS two_months_ago,
    -- Olhar 1 mês atrás, mas se não existir, retornar 0
    LAG(revenue, 1, 0) OVER (ORDER BY month) AS prev_or_zero
FROM monthly_revenue
ORDER BY month;

/*
Resultado:
month      | revenue   | two_months_ago | prev_or_zero
-----------|-----------|----------------|-------------
2024-01-01 | 100000.00 | NULL           | 0          ← Default aplicado
2024-02-01 | 110000.00 | NULL           | 100000.00
2024-03-01 | 105000.00 | 100000.00      | 110000.00
2024-04-01 | 120000.00 | 110000.00      | 105000.00

Sintaxe: LAG(coluna, offset, default_value)
*/

-- ============================================================
-- EXEMPLO 5: FIRST_VALUE() e LAST_VALUE()
-- ============================================================

-- 5A. Comparar cada venda com a primeira e última do grupo
SELECT 
    product_name,
    category,
    sales_amount,
    -- Primeira venda da categoria (menor valor)
    FIRST_VALUE(sales_amount) OVER (
        PARTITION BY category 
        ORDER BY sales_amount ASC
    ) AS min_in_category,
    -- Última venda da categoria (maior valor)
    LAST_VALUE(sales_amount) OVER (
        PARTITION BY category 
        ORDER BY sales_amount ASC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS max_in_category,
    -- Percentual do máximo
    ROUND(
        100.0 * sales_amount / LAST_VALUE(sales_amount) OVER (
            PARTITION BY category 
            ORDER BY sales_amount ASC
            ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
        ),
        2
    ) AS pct_of_max
FROM sample_sales
ORDER BY category, sales_amount DESC;

/*
Resultado:
product_name | category    | sales_amount | min_in_cat | max_in_cat | pct_of_max
-------------|-------------|--------------|------------|------------|------------
MacBook Pro  | Electronics | 12999.00     | 1999.00    | 12999.00   | 100.00
iPhone 15    | Electronics | 5999.00      | 1999.00    | 12999.00   | 46.15
iPad Air     | Electronics | 4999.00      | 1999.00    | 12999.00   | 38.46
AirPods Pro  | Electronics | 1999.00      | 1999.00    | 12999.00   | 15.38

⚠️ IMPORTANTE: LAST_VALUE precisa do frame completo!
   Sem "ROWS BETWEEN UNBOUNDED...", retorna a linha atual!
*/

-- ============================================================
-- EXEMPLO 6: Agregações como Window Functions
-- ============================================================

-- 6A. Soma acumulada (Running Total)
SELECT 
    sale_date,
    sales_amount,
    SUM(sales_amount) OVER (ORDER BY sale_date) AS running_total
FROM sample_sales
ORDER BY sale_date;

/*
Resultado:
sale_date  | sales_amount | running_total
-----------|--------------|---------------
2024-01-15 | 5999.00      | 5999.00
2024-01-15 | 699.00       | 6698.00
2024-01-15 | 299.00       | 6997.00
2024-01-16 | 12999.00     | 19996.00
2024-01-16 | 799.00       | 20795.00
...

Cada linha mostra o acumulado até aquele ponto!
*/

-- 6B. Média móvel de 3 linhas
SELECT 
    sale_date,
    sales_amount,
    ROUND(
        AVG(sales_amount) OVER (
            ORDER BY sale_date
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ),
        2
    ) AS moving_avg_3
FROM sample_sales
ORDER BY sale_date;

/*
Resultado:
sale_date  | sales_amount | moving_avg_3
-----------|--------------|-------------
2024-01-15 | 5999.00      | 5999.00      ← Só 1 linha
2024-01-15 | 699.00       | 3349.00      ← Média de 2 linhas
2024-01-15 | 299.00       | 2332.33      ← Média de 3 linhas
2024-01-16 | 12999.00     | 4665.67      ← Janela deslizou!
2024-01-16 | 799.00       | 4699.00

ROWS BETWEEN 2 PRECEDING AND CURRENT ROW = janela de 3 linhas
*/

-- ============================================================
-- EXEMPLO 7: Múltiplas Window Functions na Mesma Query
-- ============================================================

SELECT 
    product_name,
    category,
    sales_amount,
    region,
    -- Ranking dentro da categoria
    ROW_NUMBER() OVER (
        PARTITION BY category 
        ORDER BY sales_amount DESC
    ) AS rank_in_category,
    -- Média da categoria
    ROUND(AVG(sales_amount) OVER (PARTITION BY category), 2) AS category_avg,
    -- Diferença da média da categoria
    ROUND(sales_amount - AVG(sales_amount) OVER (PARTITION BY category), 2) AS diff_cat_avg,
    -- Média da região
    ROUND(AVG(sales_amount) OVER (PARTITION BY region), 2) AS region_avg,
    -- Percentual do total geral
    ROUND(
        100.0 * sales_amount / SUM(sales_amount) OVER (),
        2
    ) AS pct_of_total
FROM sample_sales
ORDER BY category, rank_in_category;

/*
Resultado: Cada produto com múltiplos contextos!
product_name | category | sales | rank | cat_avg | diff_cat | region_avg | pct_total
-------------|----------|-------|------|---------|----------|------------|----------
MacBook Pro  | Elect    | 12999 | 1    | 6499.00 | +6500.00 | 6499.00    | 38.46
iPhone 15    | Elect    | 5999  | 2    | 6499.00 | -500.00  | 6499.00    | 17.75
...

Uma query, múltiplas análises!
*/

-- ============================================================
-- EXEMPLO 8: Window Functions com WHERE e HAVING
-- ============================================================

-- 8A. ❌ ERRO: Window function no WHERE não funciona
-- SELECT * FROM sample_sales
-- WHERE ROW_NUMBER() OVER (ORDER BY sales_amount DESC) <= 5;
-- ERRO: window functions are not allowed in WHERE

-- 8B. ✅ CORRETO: Usar CTE ou subquery
WITH ranked AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (ORDER BY sales_amount DESC) AS rank
    FROM sample_sales
)
SELECT * FROM ranked WHERE rank <= 5;

-- 8C. Filtrar antes de aplicar window function
-- Exemplo: Top 3 vendas de Electronics apenas
WITH electronics_only AS (
    SELECT * 
    FROM sample_sales 
    WHERE category = 'Electronics'
)
SELECT 
    product_name,
    sales_amount,
    ROW_NUMBER() OVER (ORDER BY sales_amount DESC) AS rank
FROM electronics_only;

/*
⚠️ ORDEM DE EXECUÇÃO DO SQL:
1. FROM / JOIN
2. WHERE          ← Window functions ainda não existem aqui!
3. GROUP BY
4. HAVING
5. SELECT         ← Window functions são calculadas aqui!
6. ORDER BY

Por isso: WHERE não pode usar window functions!
Solução: CTE ou subquery
*/

-- ============================================================
-- EXEMPLO 9: Frames de Window (ROWS vs RANGE)
-- ============================================================

-- 9A. ROWS: Baseado em número de linhas
SELECT 
    sale_date,
    sales_amount,
    SUM(sales_amount) OVER (
        ORDER BY sale_date
        ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
    ) AS sum_3_rows
FROM sample_sales
ORDER BY sale_date, sales_amount;

/*
ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
= linha anterior + linha atual + próxima linha
= janela de 3 linhas

Para cada linha:
  Soma = linha[i-1] + linha[i] + linha[i+1]
*/

-- 9B. RANGE: Baseado em valores
SELECT 
    sale_date,
    sales_amount,
    SUM(sales_amount) OVER (
        ORDER BY sale_date
        RANGE BETWEEN INTERVAL '1 day' PRECEDING AND CURRENT ROW
    ) AS sum_last_2_days
FROM sample_sales
ORDER BY sale_date, sales_amount;

/*
RANGE: Considera valores da coluna de ORDER BY

RANGE BETWEEN INTERVAL '1 day' PRECEDING AND CURRENT ROW
= todas as vendas de ontem + vendas de hoje

Útil para: janelas temporais, valores próximos
*/

-- ============================================================
-- RESUMO DE PADRÕES COMUNS
-- ============================================================

/*
✅ PADRÃO 1: Top N por grupo
WITH ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY grupo ORDER BY metrica DESC) AS rn
    FROM tabela
)
SELECT * FROM ranked WHERE rn <= N;

✅ PADRÃO 2: Comparar com período anterior
SELECT 
    data,
    valor,
    LAG(valor) OVER (ORDER BY data) AS valor_anterior,
    valor - LAG(valor) OVER (ORDER BY data) AS diferenca
FROM serie_temporal;

✅ PADRÃO 3: Média móvel
SELECT 
    data,
    valor,
    AVG(valor) OVER (
        ORDER BY data
        ROWS BETWEEN N PRECEDING AND CURRENT ROW
    ) AS media_movel
FROM metrica_diaria;

✅ PADRÃO 4: Percentual do total
SELECT 
    categoria,
    valor,
    ROUND(100.0 * valor / SUM(valor) OVER (), 2) AS pct_total
FROM vendas;

✅ PADRÃO 5: Running total (soma acumulada)
SELECT 
    data,
    valor,
    SUM(valor) OVER (ORDER BY data) AS acumulado
FROM transacoes;
*/

-- ============================================================
-- FIM DOS EXEMPLOS
-- ============================================================

-- 🎯 Próximo passo: Resolver os exercícios em exercises.sql!
