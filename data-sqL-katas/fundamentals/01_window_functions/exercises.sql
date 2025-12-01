-- ============================================================
-- WINDOW FUNCTIONS: EXERCÍCIOS PRÁTICOS
-- ============================================================
-- Resolva cada exercício antes de ver as soluções!
-- Dificuldade: 🔥 (Fácil) → 🔥🔥🔥🔥🔥 (Muito Difícil)
-- ============================================================

-- ============================================================
-- SETUP: Banco de dados disponível
-- ============================================================
-- Certifique-se de ter rodado setup/init_database.sql antes!
-- Tabelas disponíveis:
--   - products (product_id, product_name, category, price, stock)
--   - customers (customer_id, name, email, city, signup_date)
--   - orders (order_id, customer_id, order_date, total_amount, status)
--   - order_items (order_item_id, order_id, product_id, quantity, unit_price)
--   - employees (employee_id, name, department, salary, manager_id, hire_date)
--   - daily_sales (date, revenue, orders_count, avg_order_value)
--   - stock_prices (ticker, date, open_price, close_price, volume)

-- ============================================================
-- EXERCÍCIO 1: Top 5 Produtos Mais Caros por Categoria
-- Dificuldade: 🔥
-- ============================================================
/*
Objetivo: Encontrar os 5 produtos mais caros de cada categoria.

Resultado esperado:
category    | product_name      | price  | rank
------------|-------------------|--------|-----
Electronics | MacBook Pro       | 12999  | 1
Electronics | iPhone 15         | 5999   | 2
...

Dicas:
- Use ROW_NUMBER() ou RANK()
- PARTITION BY categoria
- Filtrar com CTE onde rank <= 5
*/

-- Escreva sua solução aqui:





-- ============================================================
-- EXERCÍCIO 2: Clientes Acima da Média de Gastos da Cidade
-- Dificuldade: 🔥🔥
-- ============================================================
/*
Objetivo: Encontrar clientes que gastaram acima da média de sua cidade.

Tabelas: customers, orders

Resultado esperado:
customer_name | city        | total_spent | city_avg | diff
--------------|-------------|-------------|----------|------
João Silva    | São Paulo   | 5000.00     | 3500.00  | +1500
...

Dicas:
- Junte customers com orders
- Use AVG() OVER (PARTITION BY city)
- Filtre onde total_spent > city_avg
*/

-- Escreva sua solução aqui:





-- ============================================================
-- EXERCÍCIO 3: Crescimento Diário de Receita (MoM)
-- Dificuldade: 🔥🔥
-- ============================================================
/*
Objetivo: Calcular crescimento % da receita em relação ao dia anterior.

Tabela: daily_sales

Resultado esperado:
date       | revenue  | prev_day | growth_pct
-----------|----------|----------|------------
2024-01-01 | 10000    | NULL     | NULL
2024-01-02 | 11000    | 10000    | 10.00
2024-01-03 | 10500    | 11000    | -4.55
...

Dicas:
- Use LAG(revenue) OVER (ORDER BY date)
- Fórmula: 100.0 * (atual - anterior) / anterior
- ROUND para 2 casas decimais
*/

-- Escreva sua solução aqui:





-- ============================================================
-- EXERCÍCIO 4: Média Móvel de 7 Dias
-- Dificuldade: 🔥🔥🔥
-- ============================================================
/*
Objetivo: Calcular média móvel de 7 dias da receita.

Tabela: daily_sales

Resultado esperado:
date       | revenue  | ma_7d
-----------|----------|--------
2024-01-07 | 12000    | 11500.00  (média dos últimos 7 dias)
2024-01-08 | 13000    | 11800.00
...

Dicas:
- Use AVG() OVER (...)
- Frame: ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
- Primeiros 6 dias terão média parcial
*/

-- Escreva sua solução aqui:





-- ============================================================
-- EXERCÍCIO 5: Funcionários com Salário Maior que Gerente
-- Dificuldade: 🔥🔥
-- ============================================================
/*
Objetivo: Encontrar funcionários que ganham mais que seus gerentes diretos.

Tabela: employees (com self-reference manager_id)

Resultado esperado:
employee_name | employee_salary | manager_name | manager_salary | diff
--------------|-----------------|--------------|----------------|------
Diana VP      | 120000          | Bob CTO      | 110000         | 10000
...

Dicas:
- Self-join: employees e LEFT JOIN employees m ON e.manager_id = m.employee_id
- Filtrar onde e.salary > m.salary
- Calcular diferença
*/

-- Escreva sua solução aqui:





-- ============================================================
-- EXERCÍCIO 6: Top 3 Dias de Maior Receita por Mês
-- Dificuldade: 🔥🔥🔥
-- ============================================================
/*
Objetivo: Para cada mês de 2024, encontrar os 3 dias de maior receita.

Tabela: daily_sales

Resultado esperado:
month      | date       | revenue  | rank_in_month
-----------|------------|----------|---------------
2024-01    | 2024-01-25 | 15000    | 1
2024-01    | 2024-01-18 | 14500    | 2
2024-01    | 2024-01-12 | 14000    | 3
2024-02    | 2024-02-14 | 16000    | 1
...

Dicas:
- Extrair mês: DATE_TRUNC('month', date) ou TO_CHAR(date, 'YYYY-MM')
- PARTITION BY mês
- ORDER BY revenue DESC
- Filtrar rank <= 3
*/

-- Escreva sua solução aqui:





-- ============================================================
-- EXERCÍCIO 7: Detectar Quedas Consecutivas de Receita
-- Dificuldade: 🔥🔥🔥🔥
-- ============================================================
/*
Objetivo: Encontrar datas onde a receita caiu por 3 dias consecutivos.

Tabela: daily_sales

Resultado esperado:
date       | revenue  | day1_ago | day2_ago | day3_ago | is_3day_drop
-----------|----------|----------|----------|----------|-------------
2024-03-15 | 9000     | 9500     | 10000    | 10500    | true
...

Dicas:
- Use LAG() para pegar 3 dias anteriores
- Criar CASE WHEN para verificar: atual < dia1 < dia2 < dia3
- Filtrar apenas onde is_3day_drop = true
*/

-- Escreva sua solução aqui:





-- ============================================================
-- EXERCÍCIO 8: Cohort Analysis - Retenção Mensal
-- Dificuldade: 🔥🔥🔥🔥🔥
-- ============================================================
/*
Objetivo: Análise de cohort - quantos clientes de cada mês de signup 
          ainda fazem pedidos N meses depois?

Tabelas: customers, orders

Resultado esperado:
signup_month | months_after | active_customers | cohort_size | retention_pct
-------------|--------------|------------------|-------------|---------------
2024-01      | 0            | 100              | 100         | 100.00
2024-01      | 1            | 75               | 100         | 75.00
2024-01      | 2            | 60               | 100         | 60.00
...

Dicas:
- CTE 1: Definir cohort (mês de signup)
- CTE 2: Marcar meses de atividade (pedidos)
- CTE 3: Calcular months_after = diferença entre mês do pedido e signup
- CTE 4: Contar clientes ativos por (cohort, months_after)
- Final: Calcular % de retenção
- AVANÇADO: Envolve DATE_TRUNC, AGE, múltiplas CTEs
*/

-- Escreva sua solução aqui (exercício desafiador!):





-- ============================================================
-- EXERCÍCIO 9: Ranking com Empates (RANK vs DENSE_RANK)
-- Dificuldade: 🔥🔥
-- ============================================================
/*
Objetivo: Classificar produtos por número de vendas, mostrando diferença
          entre RANK e DENSE_RANK quando há empates.

Tabelas: order_items (JOIN com products)

Resultado esperado:
product_name | total_sold | row_num | rank | dense_rank
-------------|------------|---------|------|------------
iPhone 15    | 50         | 1       | 1    | 1
MacBook Pro  | 50         | 2       | 1    | 1  (empate)
iPad         | 45         | 3       | 3    | 2  (RANK pulou)
...

Dicas:
- Agregar vendas: SUM(quantity) GROUP BY product_id
- Aplicar as 3 funções: ROW_NUMBER(), RANK(), DENSE_RANK()
- Observar comportamento em empates
*/

-- Escreva sua solução aqui:





-- ============================================================
-- EXERCÍCIO 10: Máxima Histórica de Ações (High Water Mark)
-- Dificuldade: 🔥🔥🔥
-- ============================================================
/*
Objetivo: Para cada dia, mostrar a máxima histórica do preço da ação
          (high water mark).

Tabela: stock_prices

Resultado esperado:
ticker | date       | close_price | historical_max | pct_of_max
-------|------------|-------------|----------------|------------
AAPL   | 2024-01-01 | 180.00      | 180.00         | 100.00
AAPL   | 2024-01-02 | 178.00      | 180.00         | 98.89
AAPL   | 2024-01-03 | 185.00      | 185.00         | 100.00 (novo máximo!)
...

Dicas:
- Use MAX() OVER (PARTITION BY ticker ORDER BY date)
- Frame: ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
- Calcular % de quão próximo está do máximo histórico
*/

-- Escreva sua solução aqui:





-- ============================================================
-- EXERCÍCIO BÔNUS: Percentis por Departamento
-- Dificuldade: 🔥🔥🔥🔥
-- ============================================================
/*
Objetivo: Para cada funcionário, mostrar em qual quartil de salário
          ele está dentro do departamento.

Tabela: employees

Resultado esperado:
name        | department | salary  | quartile
------------|------------|---------|----------
Alice CEO   | Executive  | 150000  | 4  (top 25%)
Bob CTO     | Technology | 130000  | 4
Diana VP    | Engineering| 120000  | 4
Frank Dev   | Engineering| 100000  | 3
...

Dicas:
- Use NTILE(4) OVER (PARTITION BY department ORDER BY salary DESC)
- Quartile 1 = top 25%, Quartile 4 = bottom 25%
*/

-- Escreva sua solução aqui:





-- ============================================================
-- FIM DOS EXERCÍCIOS
-- ============================================================

-- 🎯 Próximo passo:
-- 1. Tente resolver cada exercício
-- 2. Compare com solutions.sql
-- 3. Entenda o "por quê" de cada solução
-- 4. Refaça os que errou sem olhar a solução

-- 💡 Dica para entrevista:
-- Sempre verbalize seu raciocínio:
-- "Vou usar ROW_NUMBER porque preciso eliminar empates..."
-- "LAG faz sentido aqui porque quero comparar com o período anterior..."
