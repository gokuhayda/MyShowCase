-- ============================================================
-- WINDOW FUNCTIONS: SOLUÇÕES COMENTADAS
-- ============================================================
-- Soluções detalhadas dos 10 exercícios + explicações.
-- Leia os comentários para entender o "por quê" de cada decisão.
-- ============================================================

-- ============================================================
-- SOLUÇÃO 1: Top 5 Produtos Mais Caros por Categoria
-- Dificuldade: 🔥
-- ============================================================

/*
ESTRATÉGIA:
1. Usar ROW_NUMBER() para ranquear produtos dentro de cada categoria
2. PARTITION BY category = criar ranking separado para cada categoria
3. ORDER BY price DESC = mais caro primeiro
4. CTE + WHERE para filtrar apenas top 5
*/

WITH ranked_products AS (
    SELECT 
        category,
        product_name,
        price,
        stock,
        ROW_NUMBER() OVER (
            PARTITION BY category 
            ORDER BY price DESC
        ) AS rank
    FROM products
)
SELECT 
    category,
    product_name,
    price,
    stock,
    rank
FROM ranked_products
WHERE rank <= 5
ORDER BY category, rank;

/*
POR QUE ROW_NUMBER() E NÃO RANK()?
- ROW_NUMBER(): Garante que pegamos exatamente 5 produtos, mesmo com empates
- RANK(): Se 3 produtos empatam em 5º lugar, traria os 3 (mais de 5 produtos)

Escolha depende do requisito:
- "Top 5 produtos" (quantidade exata) → ROW_NUMBER()
- "Top 5 posições" (pode ter empates) → RANK()

RESULTADO ESPERADO:
category    | product_name     | price   | rank
------------|------------------|---------|-----
Clothing    | Levi's 501       | 299.00  | 1
Clothing    | Zara Jeans       | 199.00  | 2
Clothing    | H&M Hoodie       | 179.00  | 3
...
Electronics | MacBook Pro      | 12999   | 1
Electronics | iPhone 15        | 5999    | 2
...
*/

-- ============================================================
-- SOLUÇÃO 2: Clientes Acima da Média de Gastos da Cidade
-- Dificuldade: 🔥🔥
-- ============================================================

/*
ESTRATÉGIA:
1. JOIN customers com orders para saber quanto cada cliente gastou
2. Agregar gastos por cliente (SUM)
3. Window function para calcular média da cidade
4. Filtrar clientes acima da média
*/

WITH customer_spending AS (
    -- Passo 1: Calcular quanto cada cliente gastou no total
    SELECT 
        c.customer_id,
        c.name AS customer_name,
        c.city,
        COALESCE(SUM(o.total_amount), 0) AS total_spent
    FROM customers c
    LEFT JOIN orders o ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'  -- Só pedidos completos
    GROUP BY c.customer_id, c.name, c.city
),
customers_with_avg AS (
    -- Passo 2: Adicionar média da cidade usando window function
    SELECT 
        customer_name,
        city,
        total_spent,
        ROUND(AVG(total_spent) OVER (PARTITION BY city), 2) AS city_avg
    FROM customer_spending
)
-- Passo 3: Filtrar apenas quem está acima da média
SELECT 
    customer_name,
    city,
    total_spent,
    city_avg,
    ROUND(total_spent - city_avg, 2) AS diff_from_avg
FROM customers_with_avg
WHERE total_spent > city_avg
ORDER BY city, total_spent DESC;

/*
PONTOS-CHAVE:

1. LEFT JOIN vs INNER JOIN:
   - LEFT JOIN: Incluir clientes que nunca compraram (total_spent = 0)
   - INNER JOIN: Só clientes que compraram
   - Decisão: Depende do requisito. Aqui usamos LEFT para ser inclusivo.

2. COALESCE(SUM(...), 0):
   - Se cliente não tem pedidos, SUM retorna NULL
   - COALESCE converte NULL em 0
   - Alternativa: usar INNER JOIN (elimina clientes sem pedidos)

3. WHERE status = 'completed':
   - Só considerar pedidos finalizados
   - Pedidos 'cancelled' ou 'pending' não contam

4. Por que AVG() OVER (PARTITION BY city)?
   - AVG() normal com GROUP BY colapsaria as linhas
   - OVER (PARTITION BY) mantém cada cliente como linha
   - Cada cliente vê a média da SUA cidade

RESULTADO ESPERADO:
customer_name | city          | total_spent | city_avg | diff_from_avg
--------------|---------------|-------------|----------|---------------
João Silva    | São Paulo     | 5000.00     | 3500.00  | 1500.00
Maria Santos  | São Paulo     | 4200.00     | 3500.00  | 700.00
Carlos Lima   | Rio de Janeiro| 6000.00     | 4000.00  | 2000.00
...
*/

-- ============================================================
-- SOLUÇÃO 3: Crescimento Diário de Receita (Day-over-Day)
-- Dificuldade: 🔥🔥
-- ============================================================

/*
ESTRATÉGIA:
1. Usar LAG() para pegar receita do dia anterior
2. Calcular diferença absoluta e percentual
3. ORDER BY date para sequência temporal correta
*/

SELECT 
    date,
    revenue,
    -- Receita do dia anterior
    LAG(revenue) OVER (ORDER BY date) AS prev_day_revenue,
    -- Diferença absoluta
    ROUND(revenue - LAG(revenue) OVER (ORDER BY date), 2) AS absolute_change,
    -- Crescimento percentual
    ROUND(
        100.0 * (revenue - LAG(revenue) OVER (ORDER BY date)) / 
        NULLIF(LAG(revenue) OVER (ORDER BY date), 0),
        2
    ) AS growth_pct
FROM daily_sales
ORDER BY date;

/*
PONTOS-CHAVE:

1. LAG(revenue) OVER (ORDER BY date):
   - Pega o valor de 'revenue' da linha ANTERIOR
   - ORDER BY date: Define o que é "anterior" (ordenação temporal)
   - Primeira linha retorna NULL (não há dia anterior)

2. NULLIF(..., 0):
   - Proteção contra divisão por zero
   - Se prev_day_revenue = 0, NULLIF retorna NULL
   - NULL / 0 = NULL (sem erro)
   - Alternativa: CASE WHEN prev_day_revenue > 0 THEN ... ELSE NULL END

3. 100.0 (não 100):
   - Força aritmética de ponto flutuante
   - 100 / 200 = 0 (divisão inteira!)
   - 100.0 / 200 = 0.5 ✓

4. Repetição de LAG():
   - LAG() é chamado 3 vezes na mesma query
   - Isso é ineficiente? Na verdade, PostgreSQL otimiza!
   - Alternativa: usar CTE (mais legível, mesma performance)

RESULTADO ESPERADO:
date       | revenue   | prev_day | absolute_change | growth_pct
-----------|-----------|----------|-----------------|------------
2024-01-01 | 10000.00  | NULL     | NULL            | NULL
2024-01-02 | 11000.00  | 10000.00 | 1000.00         | 10.00
2024-01-03 | 10500.00  | 11000.00 | -500.00         | -4.55
2024-01-04 | 12000.00  | 10500.00 | 1500.00         | 14.29
...

INSIGHTS VISUAIS:
- growth_pct > 0: Crescimento 📈
- growth_pct < 0: Queda 📉
- growth_pct = NULL: Primeiro dia (sem comparação)
*/

-- ALTERNATIVA: Versão com CTE (mais legível)
WITH daily_with_lag AS (
    SELECT 
        date,
        revenue,
        LAG(revenue) OVER (ORDER BY date) AS prev_day_revenue
    FROM daily_sales
)
SELECT 
    date,
    revenue,
    prev_day_revenue,
    ROUND(revenue - prev_day_revenue, 2) AS absolute_change,
    ROUND(
        100.0 * (revenue - prev_day_revenue) / NULLIF(prev_day_revenue, 0),
        2
    ) AS growth_pct
FROM daily_with_lag
ORDER BY date;

/*
QUANDO USAR CADA VERSÃO:

Inline LAG (primeira versão):
✅ Queries curtas e simples
✅ Performance levemente melhor (menos materialização)
❌ Menos legível se LAG() usado muitas vezes

CTE (segunda versão):
✅ Mais legível e manutenível
✅ Facilita debugging (pode testar CTE isoladamente)
✅ Melhor para queries complexas
❌ Performance similar (otimizador resolve)

RECOMENDAÇÃO: Use CTE em entrevistas (demonstra clean code)!
*/

-- ============================================================
-- SOLUÇÃO 4: Média Móvel de 7 Dias
-- Dificuldade: 🔥🔥🔥
-- ============================================================

/*
ESTRATÉGIA:
1. Usar AVG() como window function
2. Frame: ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
3. Isso cria uma janela deslizante de 7 dias
*/

SELECT 
    date,
    revenue,
    orders_count,
    -- Média móvel de 7 dias da receita
    ROUND(
        AVG(revenue) OVER (
            ORDER BY date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ),
        2
    ) AS ma_7d_revenue,
    -- Média móvel de 7 dias do número de pedidos
    ROUND(
        AVG(orders_count) OVER (
            ORDER BY date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ),
        2
    ) AS ma_7d_orders
FROM daily_sales
ORDER BY date;

/*
PONTOS-CHAVE:

1. ROWS BETWEEN 6 PRECEDING AND CURRENT ROW:
   - 6 PRECEDING: 6 linhas anteriores
   - CURRENT ROW: Linha atual
   - Total: 7 linhas (6 + 1)
   
   Visualização:
   Dia 10: [Dia 4, Dia 5, Dia 6, Dia 7, Dia 8, Dia 9, Dia 10]
            ↑─────────── 6 anteriores ────────────↑  ↑ atual
   
2. Primeiros 6 dias:
   - Dia 1: Média de 1 valor (só ele)
   - Dia 2: Média de 2 valores
   - ...
   - Dia 7: Média de 7 valores (primeira janela completa)
   
3. Por que média móvel?
   - Suaviza flutuações diárias
   - Identifica tendências de médio prazo
   - Útil para detectar sazonalidade

4. ROWS vs RANGE:
   - ROWS: Baseado em NÚMERO de linhas (físico)
   - RANGE: Baseado em VALORES da coluna ORDER BY (lógico)
   
   Para média móvel de 7 DIAS (não 7 linhas):
   RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW
   
   Diferença:
   - ROWS: Sempre 7 linhas (mesmo se faltar dia)
   - RANGE: 7 dias consecutivos (pula fins de semana se não houver dados)

RESULTADO ESPERADO:
date       | revenue  | orders | ma_7d_revenue | ma_7d_orders
-----------|----------|--------|---------------|-------------
2024-01-01 | 10000    | 50     | 10000.00      | 50.00  (só 1 dia)
2024-01-02 | 11000    | 55     | 10500.00      | 52.50  (média de 2)
2024-01-03 | 10500    | 52     | 10500.00      | 52.33  (média de 3)
...
2024-01-07 | 12000    | 60     | 11214.29      | 56.14  (7 dias completos)
2024-01-08 | 11500    | 58     | 11285.71      | 56.71  (janela deslizou)
...

INSIGHT:
- Se ma_7d crescente → Tendência de alta
- Se ma_7d decrescente → Tendência de baixa
- Revenue cruza ma_7d de baixo pra cima → Sinal de compra (análise técnica)
*/

-- ALTERNATIVA: Média móvel de 7 DIAS (não 7 linhas)
SELECT 
    date,
    revenue,
    ROUND(
        AVG(revenue) OVER (
            ORDER BY date
            RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW
        ),
        2
    ) AS ma_7d_calendar
FROM daily_sales
ORDER BY date;

/*
DIFERENÇA PRÁTICA:

Suponha que não há dados de fim de semana:
Sex (dia 1): 10000
Seg (dia 4): 11000  ← Pulou sábado e domingo

ROWS BETWEEN 6 PRECEDING:
  → Média de [dia 1, dia 4] = 10500 (só 2 valores)

RANGE BETWEEN INTERVAL '6 days':
  → Média de [dia 1, dia 2, dia 3, dia 4]
  → Como dia 2 e 3 não existem, média de [dia 1, dia 4] = 10500

Para dados diários COMPLETOS: ROWS e RANGE dão igual.
Para dados com gaps: RANGE é semanticamente correto.

RECOMENDAÇÃO: Use ROWS (mais simples) se dados são diários sem gaps.
*/

-- ============================================================
-- SOLUÇÃO 5: Funcionários com Salário Maior que Gerente
-- Dificuldade: 🔥🔥
-- ============================================================

/*
ESTRATÉGIA:
1. Self-join da tabela employees
2. Conectar funcionário ao gerente via manager_id
3. Filtrar onde salário do funcionário > salário do gerente
*/

SELECT 
    e.name AS employee_name,
    e.salary AS employee_salary,
    e.department AS employee_dept,
    m.name AS manager_name,
    m.salary AS manager_salary,
    m.department AS manager_dept,
    e.salary - m.salary AS salary_diff
FROM employees e
-- Self-join: juntar employee com seu manager
INNER JOIN employees m ON e.manager_id = m.employee_id
-- Filtrar: funcionário ganha mais que gerente
WHERE e.salary > m.salary
ORDER BY salary_diff DESC;

/*
PONTOS-CHAVE:

1. Self-Join Pattern:
   FROM employees e
   JOIN employees m ON e.manager_id = m.employee_id
   
   e = employee (funcionário)
   m = manager (gerente do funcionário)
   
   Visualização:
   ┌─── Tabela employees (como e) ───┐
   │ employee_id | name    | manager_id | salary │
   │ 2           | Bob     | 1          | 130000 │ ─┐
   └──────────────────────────────────────────────┘  │
                                                      │ JOIN
   ┌─── Tabela employees (como m) ───┐              │
   │ employee_id | name    | salary    │◄────────────┘
   │ 1           | Alice   | 150000    │
   └───────────────────────────────────┘

2. INNER JOIN vs LEFT JOIN:
   - INNER JOIN: Só funcionários que TÊM gerente
   - LEFT JOIN: Inclui CEO (manager_id = NULL)
   
   Se usar LEFT JOIN:
   WHERE e.salary > COALESCE(m.salary, 0)
   
   Decisão: INNER faz sentido (CEO não tem gerente para comparar)

3. Por que self-join e não window function?
   - Window function serve para comparar com linha anterior/próxima
   - Aqui precisamos comparar linhas "qualquer" (gerente pode estar longe)
   - Self-join é a solução correta

4. Caso de borda: Gerente tem múltiplos subordinados
   - O join retorna uma linha por subordinado
   - Isso está correto (cada subordinado é avaliado individualmente)

RESULTADO ESPERADO:
employee_name | employee_salary | employee_dept | manager_name | manager_salary | salary_diff
--------------|-----------------|---------------|--------------|----------------|-------------
Diana VP Eng  | 120000          | Engineering   | Bob CTO      | 110000         | 10000
Eve VP Prod   | 120000          | Product       | Bob CTO      | 110000         | 10000
...

INSIGHTS:
- Pode indicar problema de remuneração (inversão hierárquica)
- Ou pode ser intencional (especialista ganha mais que gerente generalista)
- Útil para RH detectar inconsistências salariais
*/

-- ALTERNATIVA: Incluir CEO (que não tem gerente)
SELECT 
    e.name AS employee_name,
    e.salary AS employee_salary,
    COALESCE(m.name, 'N/A') AS manager_name,
    m.salary AS manager_salary,
    CASE 
        WHEN m.salary IS NULL THEN 'No manager'
        WHEN e.salary > m.salary THEN 'Earns more'
        ELSE 'Normal'
    END AS status
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.employee_id
ORDER BY e.salary DESC;

/*
Esta versão mostra TODOS os funcionários com status.
Útil para análise exploratória.
*/

-- ============================================================
-- SOLUÇÃO 6: Top 3 Dias de Maior Receita por Mês
-- Dificuldade: 🔥🔥🔥
-- ============================================================

/*
ESTRATÉGIA:
1. Extrair mês da data
2. Ranquear dias dentro de cada mês
3. Filtrar apenas top 3
*/

WITH daily_with_month AS (
    -- Passo 1: Adicionar coluna de mês
    SELECT 
        date,
        revenue,
        orders_count,
        TO_CHAR(date, 'YYYY-MM') AS month  -- Formato: 2024-01
        -- Alternativa: DATE_TRUNC('month', date) AS month
    FROM daily_sales
),
ranked_by_month AS (
    -- Passo 2: Ranquear dias dentro de cada mês
    SELECT 
        month,
        date,
        revenue,
        orders_count,
        ROW_NUMBER() OVER (
            PARTITION BY month 
            ORDER BY revenue DESC
        ) AS rank_in_month
    FROM daily_with_month
)
-- Passo 3: Filtrar top 3
SELECT 
    month,
    date,
    revenue,
    orders_count,
    rank_in_month
FROM ranked_by_month
WHERE rank_in_month <= 3
ORDER BY month, rank_in_month;

/*
PONTOS-CHAVE:

1. Extrair mês: TO_CHAR vs DATE_TRUNC
   
   TO_CHAR(date, 'YYYY-MM'):
   ✅ Retorna texto: '2024-01'
   ✅ Formato customizável
   ✅ Melhor para display
   
   DATE_TRUNC('month', date):
   ✅ Retorna DATE: '2024-01-01'
   ✅ Mantém tipo de data
   ✅ Melhor para cálculos temporais
   
   Para PARTITION BY: Ambos funcionam!

2. Por que CTE em 2 passos?
   - Passo 1: Preparar dados (adicionar mês)
   - Passo 2: Aplicar window function
   - Passo 3: Filtrar
   
   Vantagem: Cada passo é testável isoladamente

3. ROW_NUMBER vs RANK:
   - ROW_NUMBER: Garante exatamente 3 dias por mês
   - RANK: Se 3 dias empatam em 3º lugar, traz os 3 (>3 dias)

4. PARTITION BY month:
   - Cria ranking separado para cada mês
   - Janeiro tem ranking 1,2,3
   - Fevereiro tem ranking 1,2,3 (independente)

RESULTADO ESPERADO:
month   | date       | revenue  | orders | rank_in_month
--------|------------|----------|--------|---------------
2024-01 | 2024-01-25 | 15000.00 | 75     | 1
2024-01 | 2024-01-18 | 14500.00 | 72     | 2
2024-01 | 2024-01-12 | 14000.00 | 70     | 3
2024-02 | 2024-02-14 | 16000.00 | 80     | 1  (Dia dos Namorados?)
2024-02 | 2024-02-20 | 15500.00 | 77     | 2
2024-02 | 2024-02-08 | 15200.00 | 76     | 3
...

INSIGHTS:
- Detectar sazonalidade (Black Friday, Natal, etc)
- Identificar campanhas de marketing bem-sucedidas
- Planejar promoções futuras
*/

-- ALTERNATIVA: Versão compacta (sem CTE intermediária)
WITH ranked AS (
    SELECT 
        TO_CHAR(date, 'YYYY-MM') AS month,
        date,
        revenue,
        ROW_NUMBER() OVER (
            PARTITION BY DATE_TRUNC('month', date)
            ORDER BY revenue DESC
        ) AS rank
    FROM daily_sales
)
SELECT * FROM ranked WHERE rank <= 3;

/*
Versão mais compacta, mas menos legível.
Em entrevista: Preferir versão com CTEs (demonstra pensamento estruturado).
*/

-- ============================================================
-- SOLUÇÃO 7: Detectar Quedas Consecutivas de Receita
-- Dificuldade: 🔥🔥🔥🔥
-- ============================================================

/*
ESTRATÉGIA:
1. Usar LAG() para pegar 3 dias anteriores
2. Verificar se houve queda em todos os 3 dias
3. Filtrar apenas quedas consecutivas
*/

WITH daily_with_previous AS (
    -- Passo 1: Adicionar receita dos 3 dias anteriores
    SELECT 
        date,
        revenue,
        LAG(revenue, 1) OVER (ORDER BY date) AS day1_ago,
        LAG(revenue, 2) OVER (ORDER BY date) AS day2_ago,
        LAG(revenue, 3) OVER (ORDER BY date) AS day3_ago
    FROM daily_sales
),
with_drop_check AS (
    -- Passo 2: Verificar se houve queda em cada dia
    SELECT 
        date,
        revenue,
        day1_ago,
        day2_ago,
        day3_ago,
        -- Verificar: atual < dia1 < dia2 < dia3
        CASE 
            WHEN revenue < day1_ago 
                AND day1_ago < day2_ago 
                AND day2_ago < day3_ago 
            THEN true
            ELSE false
        END AS is_3day_drop,
        -- Calcular queda acumulada
        ROUND(revenue - day3_ago, 2) AS total_drop,
        -- Calcular queda percentual
        ROUND(
            100.0 * (revenue - day3_ago) / NULLIF(day3_ago, 0),
            2
        ) AS drop_pct
    FROM daily_with_previous
)
-- Passo 3: Filtrar apenas quedas de 3 dias
SELECT 
    date,
    revenue,
    day1_ago,
    day2_ago,
    day3_ago,
    total_drop,
    drop_pct
FROM with_drop_check
WHERE is_3day_drop = true
ORDER BY date;

/*
PONTOS-CHAVE:

1. LAG(revenue, N):
   - LAG(revenue, 1): 1 dia atrás
   - LAG(revenue, 2): 2 dias atrás
   - LAG(revenue, 3): 3 dias atrás
   
   Visualização para dia 10:
   Dia 10: 9000  ← atual
   Dia 09: 9500  ← LAG(revenue, 1)
   Dia 08: 10000 ← LAG(revenue, 2)
   Dia 07: 10500 ← LAG(revenue, 3)

2. Condição de queda consecutiva:
   revenue < day1_ago < day2_ago < day3_ago
   
   Significa:
   - Dia 10 < Dia 9: Queda ontem
   - Dia 9 < Dia 8: Queda anteontem
   - Dia 8 < Dia 7: Queda 3 dias atrás
   
   = 3 quedas consecutivas!

3. NULL handling:
   - Primeiros 3 dias terão NULL (não há 3 dias anteriores)
   - CASE WHEN automaticamente retorna false para NULL
   - Alternativa: adicionar AND day3_ago IS NOT NULL

4. Por que CTE em 2 passos?
   - Separar extração de dados (LAG) da lógica (CASE WHEN)
   - Facilita debugging (pode inspecionar daily_with_previous)
   - Mais legível em entrevista

RESULTADO ESPERADO:
date       | revenue | day1_ago | day2_ago | day3_ago | total_drop | drop_pct
-----------|---------|----------|----------|----------|------------|----------
2024-03-15 | 9000    | 9500     | 10000    | 10500    | -1500.00   | -14.29
2024-06-22 | 8500    | 9000     | 9200     | 9500     | -1000.00   | -10.53
...

INSIGHTS:
- Alerta de queda consistente (não é flutuação)
- Pode indicar problema operacional
- Gatilho para investigação (campanha terminou? Bug no site?)

USO PRÁTICO:
- Criar alerta automático (send email if 3-day drop detected)
- Dashboard de "health" do negócio
*/

-- VARIAÇÃO: Detectar SUBIDAS consecutivas
WITH daily_with_previous AS (
    SELECT 
        date,
        revenue,
        LAG(revenue, 1) OVER (ORDER BY date) AS day1_ago,
        LAG(revenue, 2) OVER (ORDER BY date) AS day2_ago,
        LAG(revenue, 3) OVER (ORDER BY date) AS day3_ago
    FROM daily_sales
)
SELECT 
    date,
    revenue,
    day3_ago,
    revenue - day3_ago AS total_growth
FROM daily_with_previous
WHERE revenue > day1_ago 
  AND day1_ago > day2_ago 
  AND day2_ago > day3_ago
ORDER BY date;

/*
Mesma lógica, condição invertida (>).
Útil para detectar momentum positivo!
*/

-- ============================================================
-- SOLUÇÃO 8: Cohort Analysis - Retenção Mensal
-- Dificuldade: 🔥🔥🔥🔥🔥
-- ============================================================

/*
ESTRATÉGIA:
1. Definir cohort de cada usuário (mês de signup)
2. Marcar todos os meses em que cada usuário fez pedido
3. Calcular "months_after" = diferença entre mês do pedido e cohort
4. Contar usuários ativos por (cohort, months_after)
5. Calcular % de retenção
*/

WITH 
-- Passo 1: Definir cohort (mês de signup de cada cliente)
user_cohorts AS (
    SELECT 
        customer_id,
        DATE_TRUNC('month', signup_date) AS cohort_month
    FROM customers
),
-- Passo 2: Marcar meses de atividade (quando fizeram pedidos)
user_activities AS (
    SELECT DISTINCT
        uc.customer_id,
        uc.cohort_month,
        DATE_TRUNC('month', o.order_date) AS activity_month
    FROM user_cohorts uc
    JOIN orders o ON uc.customer_id = o.customer_id
    WHERE o.status = 'completed'
),
-- Passo 3: Calcular "months_after" (quanto tempo após signup)
cohort_activities AS (
    SELECT 
        cohort_month,
        activity_month,
        -- Diferença em meses entre atividade e signup
        EXTRACT(YEAR FROM AGE(activity_month, cohort_month)) * 12 +
        EXTRACT(MONTH FROM AGE(activity_month, cohort_month)) AS months_after,
        customer_id
    FROM user_activities
),
-- Passo 4: Contar usuários ativos por (cohort, months_after)
cohort_counts AS (
    SELECT 
        cohort_month,
        months_after,
        COUNT(DISTINCT customer_id) AS active_customers
    FROM cohort_activities
    GROUP BY cohort_month, months_after
),
-- Passo 5: Tamanho de cada cohort (mês 0)
cohort_sizes AS (
    SELECT 
        cohort_month,
        active_customers AS cohort_size
    FROM cohort_counts
    WHERE months_after = 0
)
-- Passo 6: Calcular % de retenção
SELECT 
    TO_CHAR(cc.cohort_month, 'YYYY-MM') AS cohort,
    cc.months_after,
    cc.active_customers,
    cs.cohort_size,
    ROUND(100.0 * cc.active_customers / cs.cohort_size, 2) AS retention_pct
FROM cohort_counts cc
JOIN cohort_sizes cs ON cc.cohort_month = cs.cohort_month
ORDER BY cc.cohort_month, cc.months_after;

/*
PONTOS-CHAVE:

1. O que é Cohort Analysis?
   - Agrupar usuários por quando começaram (signup_date)
   - Acompanhar comportamento desse grupo ao longo do tempo
   - Pergunta: "Dos 100 usuários que se inscreveram em Jan/2024, quantos ainda estão ativos em Fev? Mar? Abr?"

2. DATE_TRUNC('month', ...):
   - Converte data completa em primeiro dia do mês
   - 2024-01-15 → 2024-01-01
   - Permite agrupar por mês

3. AGE(activity_month, cohort_month):
   - Retorna INTERVAL (ex: '2 months 5 days')
   - EXTRACT(YEAR) e EXTRACT(MONTH) para converter em número de meses
   - Exemplo: AGE('2024-03-01', '2024-01-01') = '2 months'
              → months_after = 2

4. DISTINCT customer_id:
   - Cliente pode fazer múltiplos pedidos no mesmo mês
   - DISTINCT garante que contamos o cliente uma vez por mês
   - Sem DISTINCT: mesma pessoa contaria 5 vezes se fez 5 pedidos

5. Por que tantas CTEs?
   - Cada CTE = um passo lógico
   - Facilita debug (SELECT * FROM user_cohorts para ver resultado)
   - Demonstra pensamento estruturado em entrevista
   - Performance: PostgreSQL otimiza (não há overhead)

6. months_after = 0:
   - Mês de signup
   - cohort_size = número de pessoas que se inscreveram naquele mês
   - retention_pct = 100% (todos estavam ativos no mês de signup)

RESULTADO ESPERADO:
cohort  | months_after | active_customers | cohort_size | retention_pct
--------|--------------|------------------|-------------|---------------
2024-01 | 0            | 100              | 100         | 100.00
2024-01 | 1            | 75               | 100         | 75.00
2024-01 | 2            | 60               | 100         | 60.00
2024-01 | 3            | 50               | 100         | 50.00
2024-02 | 0            | 120              | 120         | 100.00
2024-02 | 1            | 90               | 120         | 75.00
2024-02 | 2            | 75               | 120         | 62.50
...

VISUALIZAÇÃO (Cohort de Jan/2024):
Mês 0 (Jan): ████████████████████ 100% (100 pessoas)
Mês 1 (Fev): ███████████████      75% (75 pessoas)
Mês 2 (Mar): ████████████         60% (60 pessoas)
Mês 3 (Abr): ██████████           50% (50 pessoas)

INSIGHTS:
- Queda de 25% no primeiro mês → Problema de onboarding?
- Retenção estabiliza em 50% após 3 meses → "Core users"
- Comparar cohorts: Jan/2024 vs Fev/2024
  → Se Fev/2024 tem retenção melhor, algo mudou para melhor!

USO PRÁTICO:
- Medir impacto de mudanças de produto
- Calcular LTV (Lifetime Value)
- Detectar churn precoce
*/

-- VARIAÇÃO: Cohort Table (formato pivot)
-- Formato mais visual para apresentações
SELECT 
    cohort_month,
    MAX(CASE WHEN months_after = 0 THEN retention_pct END) AS month_0,
    MAX(CASE WHEN months_after = 1 THEN retention_pct END) AS month_1,
    MAX(CASE WHEN months_after = 2 THEN retention_pct END) AS month_2,
    MAX(CASE WHEN months_after = 3 THEN retention_pct END) AS month_3
FROM (
    -- Reusar a query anterior
    SELECT 
        cc.cohort_month,
        cc.months_after,
        ROUND(100.0 * cc.active_customers / cs.cohort_size, 2) AS retention_pct
    FROM cohort_counts cc
    JOIN cohort_sizes cs ON cc.cohort_month = cs.cohort_month
) sub
GROUP BY cohort_month
ORDER BY cohort_month;

/*
Resultado em formato tabela:
cohort_month | month_0 | month_1 | month_2 | month_3
-------------|---------|---------|---------|--------
2024-01      | 100.00  | 75.00   | 60.00   | 50.00
2024-02      | 100.00  | 75.00   | 62.50   | 55.00
2024-03      | 100.00  | 80.00   | 70.00   | NULL

Mais fácil de visualizar tendências!
*/

-- ============================================================
-- SOLUÇÃO 9: Ranking com Empates (RANK vs DENSE_RANK)
-- Dificuldade: 🔥🔥
-- ============================================================

/*
ESTRATÉGIA:
1. Agregar vendas por produto (SUM quantity)
2. Aplicar as 3 funções de ranking
3. Observar diferença de comportamento em empates
*/

WITH product_sales AS (
    -- Passo 1: Calcular total vendido de cada produto
    SELECT 
        p.product_id,
        p.product_name,
        p.category,
        COALESCE(SUM(oi.quantity), 0) AS total_sold
    FROM products p
    LEFT JOIN order_items oi ON p.product_id = oi.product_id
    LEFT JOIN orders o ON oi.order_id = o.order_id AND o.status = 'completed'
    GROUP BY p.product_id, p.product_name, p.category
)
-- Passo 2: Aplicar as 3 funções de ranking
SELECT 
    product_name,
    category,
    total_sold,
    -- ROW_NUMBER: Números únicos (ignora empates)
    ROW_NUMBER() OVER (ORDER BY total_sold DESC) AS row_num,
    -- RANK: Pula números após empate
    RANK() OVER (ORDER BY total_sold DESC) AS rank,
    -- DENSE_RANK: Não pula números
    DENSE_RANK() OVER (ORDER BY total_sold DESC) AS dense_rank
FROM product_sales
ORDER BY total_sold DESC, product_name;

/*
PONTOS-CHAVE:

1. LEFT JOIN:
   - Incluir produtos que nunca foram vendidos (total_sold = 0)
   - INNER JOIN: Excluiria produtos sem vendas
   - COALESCE(SUM(...), 0): Converter NULL em 0

2. Comportamento das funções em empates:

Exemplo com empates:
┌───────────┬────────────┬─────────┬──────┬─────────────┐
│ produto   │ total_sold │ row_num │ rank │ dense_rank  │
├───────────┼────────────┼─────────┼──────┼─────────────┤
│ iPhone    │ 50         │ 1       │ 1    │ 1           │
│ MacBook   │ 50         │ 2       │ 1    │ 1   ← Empate│
│ iPad      │ 45         │ 3       │ 3    │ 2   ← RANK pulou 2│
│ AirPods   │ 40         │ 4       │ 4    │ 3           │
│ Apple TV  │ 40         │ 5       │ 4    │ 3   ← Empate│
│ Watch     │ 35         │ 6       │ 6    │ 4   ← RANK pulou 5│
└───────────┴────────────┴─────────┴──────┴─────────────┘

ROW_NUMBER: 1,2,3,4,5,6 (ignora empates completamente)
RANK:       1,1,3,4,4,6 (pula números: não há 2, não há 5)
DENSE_RANK: 1,1,2,3,3,4 (não pula: sequência contínua)

3. Quando usar cada uma?

ROW_NUMBER():
✅ Eliminar duplicatas (pegar só primeira ocorrência)
✅ Paginação (LIMIT/OFFSET)
✅ Quando precisa de número único por linha
❌ Não reflete empates (arbitrário)

RANK():
✅ Competições reais (Olimpíadas, vendas)
✅ Quando "2º lugar vago" faz sentido
✅ Fiel à realidade de empates
❌ Deixa gaps na sequência

DENSE_RANK():
✅ Níveis/categorias (Bronze, Prata, Ouro)
✅ Quando não faz sentido pular números
✅ Classificação contínua
❌ Pode ter "muita gente" no topo

4. Por que ORDER BY total_sold DESC, product_name?
   - total_sold DESC: Mais vendido primeiro
   - product_name: Desempate (ordem alfabética)
   - Garante resultados determinísticos

RESULTADO ESPERADO:
product_name     | category    | total_sold | row_num | rank | dense_rank
-----------------|-------------|------------|---------|------|------------
iPhone 15        | Electronics | 50         | 1       | 1    | 1
MacBook Pro      | Electronics | 50         | 2       | 1    | 1
iPad Air         | Electronics | 45         | 3       | 3    | 2
AirPods Pro      | Electronics | 40         | 4       | 4    | 3
Apple Watch      | Electronics | 40         | 5       | 4    | 3
Nike Air Max     | Footwear    | 35         | 6       | 6    | 4
...

INTERPRETAÇÃO:
- iPhone e MacBook empatam em 1º lugar
- ROW_NUMBER arbitrariamente coloca iPhone como 1 e MacBook como 2
- RANK: Ambos são 1º, próximo é 3º (não há 2º)
- DENSE_RANK: Ambos são 1º, próximo é 2º (sequência continua)

RECOMENDAÇÃO PARA ENTREVISTA:
"Vou usar RANK() porque reflete empates reais. Se dois produtos vendem igual, ambos merecem o mesmo ranking."
*/

-- ============================================================
-- SOLUÇÃO 10: Máxima Histórica de Ações (High Water Mark)
-- Dificuldade: 🔥🔥🔥
-- ============================================================

/*
ESTRATÉGIA:
1. Para cada dia, calcular máximo histórico até aquele dia
2. Usar MAX() OVER com frame UNBOUNDED PRECEDING
3. Calcular % de quanto o preço atual está do máximo
*/

SELECT 
    ticker,
    date,
    close_price,
    -- Máximo histórico até hoje
    MAX(close_price) OVER (
        PARTITION BY ticker
        ORDER BY date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS historical_max,
    -- Percentual do máximo histórico
    ROUND(
        100.0 * close_price / MAX(close_price) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ),
        2
    ) AS pct_of_max,
    -- Drawdown (queda desde o pico)
    ROUND(
        close_price - MAX(close_price) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ),
        2
    ) AS drawdown
FROM stock_prices
WHERE ticker = 'AAPL'  -- Filtrar apenas Apple para exemplo
ORDER BY date;

/*
PONTOS-CHAVE:

1. ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW:
   - UNBOUNDED PRECEDING: Desde o início da partição
   - CURRENT ROW: Até a linha atual
   - = "Todos os dias desde o primeiro até hoje"
   
   Visualização para dia 10:
   Máx([Dia 1, Dia 2, ..., Dia 9, Dia 10])

2. High Water Mark (Máxima Histórica):
   - O maior valor que a ação já atingiu até hoje
   - Se hoje quebrou o recorde, historical_max = close_price
   - Se não quebrou, historical_max = último pico

3. Drawdown:
   - Queda desde o pico
   - Negativo = ação está abaixo do pico
   - Zero = ação está no pico (novo recorde)
   
   Exemplo:
   - Pico histórico: $200
   - Preço hoje: $180
   - Drawdown: -$20 (-10%)

4. PARTITION BY ticker:
   - Calcular máximo separadamente para cada ação
   - Apple tem seu próprio histórico
   - Google tem seu próprio histórico
   - Não misturam

5. Por que MAX() e não LAST_VALUE()?
   - LAST_VALUE() pegaria o último valor da janela (não o máximo)
   - MAX() encontra o maior valor na janela (o que queremos)

RESULTADO ESPERADO:
ticker | date       | close_price | historical_max | pct_of_max | drawdown
-------|------------|-------------|----------------|------------|----------
AAPL   | 2024-01-01 | 180.00      | 180.00         | 100.00     | 0.00
AAPL   | 2024-01-02 | 178.00      | 180.00         | 98.89      | -2.00
AAPL   | 2024-01-03 | 185.00      | 185.00         | 100.00     | 0.00  ← Novo pico!
AAPL   | 2024-01-04 | 183.00      | 185.00         | 98.92      | -2.00
AAPL   | 2024-01-05 | 187.00      | 187.00         | 100.00     | 0.00  ← Novo pico!
AAPL   | 2024-01-06 | 182.00      | 187.00         | 97.33      | -5.00
...

INSIGHTS:
- pct_of_max = 100%: Ação no pico histórico (comprar? vender?)
- pct_of_max < 90%: Queda significativa (oportunidade de compra?)
- drawdown crescente: Tendência de baixa
- Novo pico após drawdown: Recuperação (sinal de força)

USO PRÁTICO:
- Alertas: "AAPL atingiu novo máximo histórico!"
- Risk management: "Stop loss se drawdown > 20%"
- Análise técnica: Identificar suportes e resistências

COMPARAÇÃO ENTRE AÇÕES:
Qual ação está "mais cara" historicamente?
*/

-- VARIAÇÃO: Comparar múltiplas ações
WITH stock_analysis AS (
    SELECT 
        ticker,
        date,
        close_price,
        MAX(close_price) OVER (
            PARTITION BY ticker
            ORDER BY date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS historical_max,
        ROUND(
            100.0 * close_price / MAX(close_price) OVER (
                PARTITION BY ticker
                ORDER BY date
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ),
            2
        ) AS pct_of_max
    FROM stock_prices
)
-- Mostrar apenas o dia mais recente de cada ação
SELECT 
    ticker,
    MAX(date) AS latest_date,
    MAX(close_price) AS current_price,
    MAX(historical_max) AS all_time_high,
    MAX(pct_of_max) AS pct_of_ath
FROM stock_analysis
GROUP BY ticker
ORDER BY pct_of_ath DESC;

/*
Resultado:
ticker | latest_date | current_price | all_time_high | pct_of_ath
-------|-------------|---------------|---------------|------------
MSFT   | 2024-12-31  | 395.00        | 395.00        | 100.00  ← No pico!
GOOGL  | 2024-12-31  | 145.00        | 150.00        | 96.67
AAPL   | 2024-12-31  | 185.00        | 195.00        | 94.87   ← 5% do pico

INSIGHT: MSFT está mais "cara" (no all-time high)
         AAPL está 5% mais "barata" que seu pico
*/

-- ============================================================
-- EXERCÍCIO BÔNUS: Percentis por Departamento (NTILE)
-- Dificuldade: 🔥🔥🔥🔥
-- ============================================================

/*
ESTRATÉGIA:
1. Usar NTILE(4) para dividir em 4 grupos (quartis)
2. PARTITION BY department para quartis dentro de cada departamento
3. Quartil 1 = top 25%, Quartil 4 = bottom 25%
*/

SELECT 
    name,
    department,
    salary,
    -- Dividir em 4 grupos (quartis)
    NTILE(4) OVER (
        PARTITION BY department 
        ORDER BY salary DESC
    ) AS salary_quartile,
    -- Também calcular percentil exato
    ROUND(
        PERCENT_RANK() OVER (
            PARTITION BY department 
            ORDER BY salary DESC
        ) * 100,
        2
    ) AS percentile,
    -- Salário médio do departamento (para contexto)
    ROUND(AVG(salary) OVER (PARTITION BY department), 2) AS dept_avg
FROM employees
ORDER BY department, salary DESC;

/*
PONTOS-CHAVE:

1. NTILE(N):
   - Divide linhas em N grupos aproximadamente iguais
   - NTILE(4): Quartis (4 grupos de 25% cada)
   - NTILE(10): Decis (10 grupos de 10% cada)
   - NTILE(100): Percentis (100 grupos de 1% cada)

2. Como NTILE funciona:
   - Ordena as linhas (por ORDER BY)
   - Divide em N grupos do mesmo tamanho
   - Se não divide exato, primeiros grupos ficam maiores
   
   Exemplo: 10 funcionários em 4 quartis
   - Quartil 1: 3 pessoas (top 30%)
   - Quartil 2: 3 pessoas
   - Quartil 3: 2 pessoas
   - Quartil 4: 2 pessoas (bottom 20%)

3. PERCENT_RANK():
   - Retorna posição relativa (0 a 1)
   - 0 = menor valor
   - 0.5 = mediana
   - 1 = maior valor
   - Fórmula: (rank - 1) / (total_rows - 1)

4. NTILE vs PERCENT_RANK:
   - NTILE: Grupos discretos (1, 2, 3, 4)
   - PERCENT_RANK: Valor contínuo (0.00, 0.25, 0.50, ...)
   - Use NTILE para classificação simples
   - Use PERCENT_RANK para análise granular

5. ORDER BY salary DESC:
   - Quartil 1 = maiores salários (top performers)
   - Quartil 4 = menores salários (bottom)
   - Se fosse ASC, inverteria (Quartil 1 = menores)

RESULTADO ESPERADO:
name         | department  | salary  | quartile | percentile | dept_avg
-------------|-------------|---------|----------|------------|----------
Alice CEO    | Executive   | 150000  | 1        | 0.00       | 150000.00
Bob CTO      | Technology  | 130000  | 1        | 0.00       | 130000.00
Diana VP     | Engineering | 120000  | 1        | 0.00       | 100000.00
Frank Senior | Engineering | 100000  | 1        | 25.00      | 100000.00
Grace Senior | Engineering | 98000   | 2        | 37.50      | 100000.00
Henry Mid    | Engineering | 85000   | 3        | 62.50      | 100000.00
Ivy Junior   | Engineering | 70000   | 4        | 87.50      | 100000.00
Jack Junior  | Engineering | 72000   | 4        | 100.00     | 100000.00

INTERPRETAÇÃO:
- Diana (Engineering): Quartil 1 = top 25% do departamento
- Ivy (Engineering): Quartil 4 = bottom 25%
- Percentile 0 = maior salário
- Percentile 100 = menor salário

USO PRÁTICO:
- Análise de equidade salarial
- Identificar funcionários sub/super pagos
- Planejar aumentos salariais
- Benchmark interno
*/

-- VARIAÇÃO: Detectar outliers (top 10% e bottom 10%)
WITH salary_analysis AS (
    SELECT 
        name,
        department,
        salary,
        NTILE(10) OVER (PARTITION BY department ORDER BY salary DESC) AS decile
    FROM employees
)
SELECT 
    name,
    department,
    salary,
    CASE 
        WHEN decile = 1 THEN 'Top 10%'
        WHEN decile = 10 THEN 'Bottom 10%'
        ELSE 'Middle 80%'
    END AS salary_tier
FROM salary_analysis
WHERE decile IN (1, 10)  -- Apenas extremos
ORDER BY department, salary DESC;

/*
Identifica funcionários nos extremos salariais.
Útil para:
- Retenção de top performers
- Revisão de salários muito baixos
*/

-- ============================================================
-- FIM DAS SOLUÇÕES
-- ============================================================

/*
🎯 PRÓXIMOS PASSOS:

1. ✅ Refazer exercícios sem olhar soluções
2. ✅ Criar variações próprias
3. ✅ Praticar explicar em voz alta
4. ✅ Avançar para módulo 02_ctes/

💡 DICAS PARA ENTREVISTA:

1. Sempre verbalize seu raciocínio:
   "Vou usar ROW_NUMBER porque preciso exatamente 3 produtos..."
   "LAG faz sentido aqui porque quero comparar com anterior..."

2. Comece simples, depois refine:
   "Primeiro vou confirmar que os dados estão corretos..."
   "Agora vou adicionar a window function..."

3. Use CTEs para clareza:
   "Vou dividir em 3 passos: preparar, ranquear, filtrar"

4. Teste incrementalmente:
   "Deixa eu rodar só a primeira CTE para ver se está certo..."

5. Considere edge cases:
   "E se não houver dia anterior? LAG retorna NULL..."
   "E se houver empate? Vou usar RANK ao invés de ROW_NUMBER..."

🔥 PADRÕES PARA MEMORIZAR:

Top N por grupo:
  ROW_NUMBER() OVER (PARTITION BY grupo ORDER BY metrica DESC)

Comparar com anterior:
  LAG(coluna) OVER (ORDER BY sequencia)

Soma acumulada:
  SUM(valor) OVER (ORDER BY data)

Média móvel:
  AVG(valor) OVER (ORDER BY data ROWS BETWEEN N PRECEDING AND CURRENT ROW)

Máximo histórico:
  MAX(valor) OVER (ORDER BY data ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)

Percentual do total:
  100.0 * valor / SUM(valor) OVER ()
*/
