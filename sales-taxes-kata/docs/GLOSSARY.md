# Glossário Técnico --- Sales Taxes Kata

## 🧱 Arquitetura & Design

### **Functional Core**

Parte do sistema onde toda a lógica é pura, determinística e livre de
efeitos colaterais.

### **Imperative Shell**

Camada externa onde ocorrem efeitos colaterais (I/O, prints, composição
do sistema).

### **Composition Root**

Local onde dependências são instanciadas.\
No projeto: `factory.py`.

### **Domain Model**

Representação do domínio do problema.\
Aqui: entidade `Product`.

------------------------------------------------------------------------

## 🎯 Princípios SOLID

### **SRP --- Single Responsibility Principle**

Cada classe possui apenas uma razão para mudar.\
Ex.: `BasicSalesTax` e `ImportDutyTax` são separados.

### **OCP --- Open/Closed Principle**

Aberto para extensão, fechado para modificação.\
Adicionar novo imposto não altera `TaxCalculator`.

### **LSP --- Liskov Substitution Principle**

Qualquer implementação concreta substitui a abstração sem quebrar o
sistema.

### **ISP --- Interface Segregation Principle**

Interfaces pequenas e focadas.\
Aqui: `TaxStrategy`.

### **DIP --- Dependency Inversion Principle**

Código depende de abstrações, não implementações.\
`TaxCalculator` opera sem conhecer estratégias concretas.

------------------------------------------------------------------------

## 🧠 Padrões de Projeto

### **Strategy Pattern**

Encapsula regras variáveis (impostos) em classes intercambiáveis.

### **Factory Pattern**

Centraliza criação de objetos e define quais estratégias estarão ativas.

------------------------------------------------------------------------

## 🔁 Testes & TDD

### **TDD --- Test-Driven Development**

Escreve-se o teste antes do código de produção.

### **Unit Test**

Testes determinísticos que validam comportamentos isolados.

------------------------------------------------------------------------

## 🧮 Matemática & Finanças

### **Arredondamento para múltiplo de 0.05**

Regra: sempre arredondar para cima no próximo 0.05.

### **Decimal**

Tipo numérico com precisão exata para cálculos financeiros.

------------------------------------------------------------------------

## 🗂️ Engenharia de Código

### **Imutabilidade**

Objetos não mudam após criados.

### **Polimorfismo**

Tratamento uniforme via interface comum (`TaxStrategy`).

### **Coesão**

Módulos com propósito único.

### **Acoplamento Baixo**

Mudanças em uma classe não quebram outras.

------------------------------------------------------------------------

## ⚙️ Pair Programming

### **Verbalização**

Explicar o raciocínio enquanto programa.

### **Trade-off**

Escolha consciente entre alternativas.

### **Refatoração**

Melhorar estrutura interna sem alterar comportamento externo.

------------------------------------------------------------------------

## 🔒 Conceitos Relevantes

### **Pure Function**

Mesmas entradas → mesmo resultado, sem efeitos colaterais.

### **Side Effect**

Alteração de estado externo: I/O, prints, arquivos, instâncias.

### **Determinismo**

Código sempre produz o mesmo resultado para inputs iguais.
