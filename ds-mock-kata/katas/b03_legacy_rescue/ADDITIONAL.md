"""
🔴 CÓDIGO LEGADO - ORIGINAL (ANTES DA REFATORAÇÃO)

Este arquivo representa o "código espaguete" que vamos refatorar.
É um exemplo REAL de como código de produção pode ficar ruim ao longo do tempo.

❌ PROBLEMAS DESTE CÓDIGO:
==========================

1. ACOPLAMENTO A I/O EXTERNO:
   - A função faz uma chamada HTTP diretamente (requests.get)
   - Impossível testar sem rede ou sem mockar internamente
   - Qualquer mudança na API quebra nossos testes

2. DEPENDÊNCIA GLOBAL OCULTA:
   - O modelo ML (MODEL) é uma variável global
   - Não consigo substituir por um mock facilmente
   - Em produção, isso carregaria 5GB de memória sempre!

3. LÓGICA DE NEGÓCIO MISTURADA COM I/O:
   - As regras de cálculo (base_score) estão enterradas no meio do código
   - Não consigo testar a LÓGICA isoladamente
   - Difícil de entender qual é a regra de negócio real

4. VIOLAÇÃO DO SINGLE RESPONSIBILITY PRINCIPLE:
   - Esta função faz 3 coisas: busca dados, calcula score, faz predição
   - Mudanças em qualquer uma das partes afetam tudo

5. FALTA DE INVERSÃO DE CONTROLE:
   - As dependências são criadas DENTRO da função
   - Não consigo injetar versões mockadas para testes

📚 CONCEITOS DE DESIGN QUE ESTÃO FALTANDO:
==========================================
- Dependency Injection (DI)
- Separation of Concerns
- Single Responsibility Principle
- Testability by Design

🎯 O QUE VAMOS FAZER:
=====================
Vamos refatorar este código em 3 componentes:
1. api_client.py    → Responsável APENAS por I/O (boundary)
2. scoring_logic.py → Lógica PURA de negócio (core)
3. orchestrator.py  → Coordenação com Dependency Injection

Veja os arquivos refatorados e compare com este!
"""


# ❌ PROBLEMA #1: DEPENDÊNCIA GLOBAL OCULTA
# Esta variável global torna o código impossível de testar com diferentes modelos
# Em produção real, isso seria um modelo de 5GB carregado na memória
# Comentário: Se você tentasse testar esta função, seria OBRIGADO a usar este modelo
# MODEL = LogisticRegression()  # Importação comentada para exercício sem sklearn


    Gera um score de crédito para um cliente.
    
    ❌ ESTA FUNÇÃO É UM PESADELO PARA TESTAR!
    
    Por quê?
    1. Faz I/O real (HTTP request)
    2. Usa modelo global (não posso mockar facilmente)
    3. Lógica de negócio misturada com infraestrutura
    

    Fluxo:
    ┌──────────────┐
    │ 1. HTTP GET  │  ← I/O (Boundary)
    └──────┬───────┘
           │
    ┌──────▼───────────┐
    │ 2. Calcula Base  │  ← Lógica (Core) - MAS MISTURADA!
    └──────┬───────────┘
           │
    ┌──────▼───────────┐
    │ 3. ML Predict    │  ← I/O (Boundary) - Global!
    └──────┬───────────┘
           │
    ┌──────▼───────────┐
    │ 4. Score Final   │  ← Lógica (Core) - MAS MISTURADA!
    └──────────────────┘
    
    ⚠️ Todos esses passos estão ACOPLADOS na mesma função!
    """
    
    # ❌ PROBLEMA #2: I/O MISTURADO COM LÓGICA
    # Esta linha faz uma chamada HTTP real.
    # Em testes, isso significa:
    # - Preciso de rede funcionando
    # - Preciso que a API esteja no ar
    # - Os testes ficam lentos (latência de rede)
    # - Os testes ficam flaky (podem falhar por motivos de rede)
