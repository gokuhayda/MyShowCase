-- ============================================================
-- SQL Katas: Inicialização do Banco de Dados
-- ============================================================
-- Este script cria todas as tabelas e carrega dados de exemplo
-- para os exercícios do repositório.
-- ============================================================

-- Limpar banco (apenas para desenvolvimento)
DROP SCHEMA IF EXISTS public CASCADE;
CREATE SCHEMA public;

-- ============================================================
-- DATASET 1: E-commerce
-- ============================================================

-- Tabela de produtos
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(100) NOT NULL,
    category VARCHAR(50) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    stock INT NOT NULL,
    created_at DATE DEFAULT CURRENT_DATE
);

-- Tabela de clientes
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    city VARCHAR(50),
    signup_date DATE NOT NULL
);

-- Tabela de pedidos
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INT REFERENCES customers(customer_id),
    order_date DATE NOT NULL,
    total_amount DECIMAL(10,2) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending'
);

-- Tabela de itens do pedido
CREATE TABLE order_items (
    order_item_id SERIAL PRIMARY KEY,
    order_id INT REFERENCES orders(order_id),
    product_id INT REFERENCES products(product_id),
    quantity INT NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL
);

-- ============================================================
-- DATASET 2: Empresa / RH
-- ============================================================

-- Tabela de funcionários
CREATE TABLE employees (
    employee_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    department VARCHAR(50) NOT NULL,
    salary DECIMAL(10,2) NOT NULL,
    manager_id INT REFERENCES employees(employee_id),
    hire_date DATE NOT NULL
);

-- ============================================================
-- DATASET 3: SaaS Metrics
-- ============================================================

-- Tabela de assinaturas
CREATE TABLE subscriptions (
    subscription_id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    plan VARCHAR(20) NOT NULL,  -- 'basic', 'pro', 'enterprise'
    start_date DATE NOT NULL,
    end_date DATE,  -- NULL = ativa
    monthly_value DECIMAL(10,2) NOT NULL
);

-- Tabela de eventos de usuário
CREATE TABLE user_events (
    event_id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    event_time TIMESTAMP NOT NULL,
    properties JSONB
);

-- ============================================================
-- DATASET 4: Séries Temporais
-- ============================================================

-- Tabela de vendas diárias
CREATE TABLE daily_sales (
    date DATE PRIMARY KEY,
    revenue DECIMAL(12,2) NOT NULL,
    orders_count INT NOT NULL,
    avg_order_value DECIMAL(10,2) NOT NULL
);

-- Tabela de preços de ações (para exercícios financeiros)
CREATE TABLE stock_prices (
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open_price DECIMAL(10,2) NOT NULL,
    close_price DECIMAL(10,2) NOT NULL,
    volume BIGINT NOT NULL,
    PRIMARY KEY (ticker, date)
);

-- ============================================================
-- INSERIR DADOS DE EXEMPLO
-- ============================================================

-- Produtos (100 produtos em 5 categorias)
INSERT INTO products (product_name, category, price, stock) VALUES
    ('iPhone 15', 'Electronics', 5999.00, 50),
    ('AirPods Pro', 'Electronics', 1999.00, 100),
    ('MacBook Pro', 'Electronics', 12999.00, 30),
    ('iPad Air', 'Electronics', 4999.00, 75),
    ('Apple Watch', 'Electronics', 2999.00, 60),
    ('Nike Air Max', 'Footwear', 699.00, 150),
    ('Adidas Ultraboost', 'Footwear', 799.00, 120),
    ('Vans Old Skool', 'Footwear', 399.00, 200),
    ('Converse Chuck Taylor', 'Footwear', 299.00, 180),
    ('Puma Suede', 'Footwear', 449.00, 90),
    ('Levi\'s 501', 'Clothing', 299.00, 300),
    ('Nike Dri-FIT', 'Clothing', 149.00, 250),
    ('Adidas T-Shirt', 'Clothing', 129.00, 400),
    ('Zara Jeans', 'Clothing', 199.00, 200),
    ('H&M Hoodie', 'Clothing', 179.00, 180);

-- Clientes (50 clientes)
INSERT INTO customers (name, email, city, signup_date)
SELECT 
    'Cliente ' || i,
    'cliente' || i || '@example.com',
    CASE (i % 5)
        WHEN 0 THEN 'São Paulo'
        WHEN 1 THEN 'Rio de Janeiro'
        WHEN 2 THEN 'Belo Horizonte'
        WHEN 3 THEN 'Curitiba'
        ELSE 'Porto Alegre'
    END,
    DATE '2023-01-01' + (i || ' days')::INTERVAL
FROM generate_series(1, 50) AS i;

-- Pedidos (200 pedidos ao longo de 2024)
INSERT INTO orders (customer_id, order_date, total_amount, status)
SELECT 
    (random() * 49 + 1)::INT,  -- customer_id aleatório
    DATE '2024-01-01' + (random() * 300)::INT * INTERVAL '1 day',
    ROUND((random() * 2000 + 100)::NUMERIC, 2),
    CASE 
        WHEN random() < 0.8 THEN 'completed'
        WHEN random() < 0.95 THEN 'shipped'
        ELSE 'cancelled'
    END
FROM generate_series(1, 200);

-- Funcionários (30 funcionários com hierarquia)
INSERT INTO employees (name, department, salary, manager_id, hire_date) VALUES
    ('Alice CEO', 'Executive', 150000, NULL, '2020-01-01'),
    ('Bob CTO', 'Technology', 130000, 1, '2020-02-01'),
    ('Charlie CFO', 'Finance', 130000, 1, '2020-02-01'),
    ('Diana VP Eng', 'Engineering', 120000, 2, '2020-03-01'),
    ('Eve VP Product', 'Product', 120000, 2, '2020-03-01'),
    ('Frank Senior Dev', 'Engineering', 100000, 4, '2020-04-01'),
    ('Grace Senior Dev', 'Engineering', 98000, 4, '2020-05-01'),
    ('Henry Mid Dev', 'Engineering', 85000, 4, '2021-01-01'),
    ('Ivy Junior Dev', 'Engineering', 70000, 6, '2022-01-01'),
    ('Jack Junior Dev', 'Engineering', 72000, 7, '2022-02-01');

-- Assinaturas SaaS (100 usuários)
INSERT INTO subscriptions (user_id, plan, start_date, end_date, monthly_value)
SELECT 
    i,
    CASE 
        WHEN random() < 0.5 THEN 'basic'
        WHEN random() < 0.85 THEN 'pro'
        ELSE 'enterprise'
    END,
    DATE '2023-01-01' + (random() * 365)::INT * INTERVAL '1 day',
    CASE 
        WHEN random() < 0.2 THEN DATE '2024-01-01' + (random() * 180)::INT * INTERVAL '1 day'
        ELSE NULL  -- Assinatura ativa
    END,
    CASE 
        WHEN random() < 0.5 THEN 29.90
        WHEN random() < 0.85 THEN 99.90
        ELSE 299.90
    END
FROM generate_series(1, 100) AS i;

-- Vendas diárias (365 dias de 2024)
INSERT INTO daily_sales (date, revenue, orders_count, avg_order_value)
SELECT 
    date,
    ROUND((5000 + random() * 10000 + 1000 * sin(EXTRACT(DOY FROM date) / 365.0 * 2 * pi()))::NUMERIC, 2),
    (20 + random() * 30)::INT,
    ROUND((200 + random() * 300)::NUMERIC, 2)
FROM generate_series(
    DATE '2024-01-01',
    DATE '2024-12-31',
    INTERVAL '1 day'
) AS date;

-- Preços de ações (AAPL, GOOGL, MSFT - 252 dias úteis de 2024)
WITH trading_days AS (
    SELECT date
    FROM generate_series(DATE '2024-01-01', DATE '2024-12-31', INTERVAL '1 day') AS date
    WHERE EXTRACT(DOW FROM date) NOT IN (0, 6)  -- Excluir fins de semana
    LIMIT 252
)
INSERT INTO stock_prices (ticker, date, open_price, close_price, volume)
SELECT 
    ticker,
    date,
    ROUND((base_price + random() * 10)::NUMERIC, 2),
    ROUND((base_price + random() * 10)::NUMERIC, 2),
    (1000000 + random() * 5000000)::BIGINT
FROM trading_days
CROSS JOIN (
    VALUES 
        ('AAPL', 180.0),
        ('GOOGL', 140.0),
        ('MSFT', 380.0)
) AS stocks(ticker, base_price);

-- ============================================================
-- CRIAR ÍNDICES PARA PERFORMANCE
-- ============================================================

CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_order_items_order ON order_items(order_id);
CREATE INDEX idx_order_items_product ON order_items(product_id);
CREATE INDEX idx_employees_manager ON employees(manager_id);
CREATE INDEX idx_subscriptions_user ON subscriptions(user_id);
CREATE INDEX idx_subscriptions_dates ON subscriptions(start_date, end_date);
CREATE INDEX idx_stock_prices_ticker_date ON stock_prices(ticker, date);

-- ============================================================
-- CRIAR VIEWS ÚTEIS
-- ============================================================

-- View: Vendas por categoria
CREATE OR REPLACE VIEW sales_by_category AS
SELECT 
    p.category,
    COUNT(DISTINCT o.order_id) AS orders_count,
    SUM(oi.quantity * oi.unit_price) AS total_revenue,
    AVG(oi.unit_price) AS avg_price
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
JOIN products p ON oi.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY p.category;

-- View: Hierarquia de funcionários
CREATE OR REPLACE VIEW employee_hierarchy AS
WITH RECURSIVE org_tree AS (
    -- Caso base: CEO
    SELECT 
        employee_id,
        name,
        department,
        manager_id,
        salary,
        1 AS level,
        name AS path
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Caso recursivo
    SELECT 
        e.employee_id,
        e.name,
        e.department,
        e.manager_id,
        e.salary,
        ot.level + 1,
        ot.path || ' > ' || e.name
    FROM employees e
    JOIN org_tree ot ON e.manager_id = ot.employee_id
)
SELECT * FROM org_tree;

-- ============================================================
-- MENSAGEM DE SUCESSO
-- ============================================================

DO $$
BEGIN
    RAISE NOTICE '✅ Banco de dados inicializado com sucesso!';
    RAISE NOTICE '📊 Datasets criados:';
    RAISE NOTICE '  - E-commerce: products, customers, orders, order_items';
    RAISE NOTICE '  - RH: employees';
    RAISE NOTICE '  - SaaS: subscriptions, user_events';
    RAISE NOTICE '  - Financeiro: daily_sales, stock_prices';
    RAISE NOTICE '🚀 Você está pronto para começar os exercícios!';
END $$;
