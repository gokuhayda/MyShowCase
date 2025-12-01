bash# Certifique-se de estar na pasta do projeto
cd ~/Documentos/thoughtWorks/KATA/MyShowCase/ds-cd4ml-kata

# Criar arquivo de parâmetros com comentários completos
cat > params.yaml << 'EOF'
# ============================================================================
# CONFIGURAÇÃO DE HIPERPARÂMETROS E QUALITY GATES
# ============================================================================
# Este arquivo centraliza TODOS os parâmetros do pipeline de ML
# Benefícios:
#   - Rastreabilidade: MLflow loga esses valores
#   - Reprodutibilidade: Mesmos params = mesmo resultado
#   - Experimentação: Fácil testar diferentes configurações
#   - Versionamento: Git versiona mudanças nos parâmetros
# ============================================================================

# ----------------------------------------------------------------------------
# HIPERPARÂMETROS DO MODELO
# ----------------------------------------------------------------------------
model:
  algorithm: RandomForest                # Algoritmo escolhido
  
  # Número de árvores no ensemble
  # - Mais árvores = mais estável, mas mais lento
  # - Típico: 50-500
  n_estimators: 100
  
  # Profundidade máxima de cada árvore
  # - Controla overfitting (menor = menos overfitting)
  # - None = sem limite (mais overfitting)
  # - Típico: 5-20
  max_depth: 10
  
  # Mínimo de amostras para dividir um nó
  # - Maior valor = menos divisões = menos overfitting
  # - Típico: 2-20
  min_samples_split: 5
  
  # Mínimo de amostras em cada folha
  # - Maior valor = folhas mais robustas
  # - Típico: 1-10
  min_samples_leaf: 2
  
  # Seed para reprodutibilidade
  # - Garante que o treino é determinístico
  # - Use o mesmo valor em todos os experimentos comparáveis
  random_state: 42
  
  # Balanceamento de classes
  # - balanced: Ajusta pesos automaticamente (recomendado para classes desbalanceadas)
  # - None: Sem balanceamento
  # - dict: {0: 1, 1: 2} para pesos customizados
  class_weight: balanced

# ----------------------------------------------------------------------------
# DIVISÃO DE DADOS (TRAIN/TEST SPLIT)
# ----------------------------------------------------------------------------
data:
  # Proporção de dados para teste
  # - 0.2 = 80% treino, 20% teste
  # - Típico: 0.2 - 0.3
  test_size: 0.2
  
  # Seed para reprodutibilidade do split
  # - Garante que sempre pega as mesmas amostras
  random_state: 42
  
  # Estratificar por target
  # - true: Mantém proporção de classes em train e test
  # - false: Split aleatório puro
  # - SEMPRE use true para classificação!
  stratify: true

# ----------------------------------------------------------------------------
# QUALITY GATES - THRESHOLDS MÍNIMOS
# ----------------------------------------------------------------------------
# O modelo SÓ passa se atingir TODOS esses valores
# Ajuste baseado em:
#   - Requisitos de negócio
#   - Baseline (modelo simples)
#   - Benchmarks da literatura
metrics:
  # Acurácia mínima no conjunto de teste
  # - (TP + TN) / Total
  # - 0.75 = 75% de predições corretas
  min_accuracy: 0.75
  
  # Precisão mínima
  # - TP / (TP + FP)
  # - "Das predições positivas, quantas estavam certas?"
  # - Importante quando custo de FP é alto
  min_precision: 0.73
  
  # Recall mínimo (sensibilidade)
  # - TP / (TP + FN)
  # - "Dos casos positivos reais, quantos detectamos?"
  # - Importante quando custo de FN é alto
  min_recall: 0.73
  
  # F1-Score mínimo
  # - Média harmônica de precision e recall
  # - Balanceia precision e recall
  min_f1: 0.73
  
  # Gap máximo entre treino e teste (overfitting check)
  # - train_acc - test_acc
  # - 0.10 = máximo 10% de diferença
  # - Valores maiores indicam overfitting
  max_train_test_gap: 0.10

# ----------------------------------------------------------------------------
# CROSS-VALIDATION
# ----------------------------------------------------------------------------
# Validação cruzada para estimar performance de forma mais robusta
cv:
  # Número de folds (divisões)
  # - 5 = divide em 5 partes, treina 5 vezes
  # - Típico: 3-10 (5 é padrão)
  # - Mais folds = mais confiável, mas mais lento
  n_splits: 5
  
  # Embaralhar dados antes de dividir
  # - true: Aleatoriza ordem (recomendado)
  # - false: Usa ordem original
  shuffle: true
  
  # Seed para reprodutibilidade do CV
  random_state: 42

# ============================================================================
# DICAS DE AJUSTE (TUNING)
# ============================================================================
# 
# 🔧 Se OVERFITTING (train_acc >> test_acc):
#    - Diminuir max_depth (ex: 10 → 5)
#    - Aumentar min_samples_split (ex: 5 → 10)
#    - Aumentar min_samples_leaf (ex: 2 → 5)
#    - Diminuir n_estimators (ex: 100 → 50)
#
# 🔧 Se UNDERFITTING (train_acc e test_acc baixos):
#    - Aumentar max_depth (ex: 10 → 15)
#    - Diminuir min_samples_split (ex: 5 → 2)
#    - Aumentar n_estimators (ex: 100 → 200)
#    - Adicionar features
#
# 🔧 Se classes DESBALANCEADAS:
#    - Manter class_weight: balanced
#    - Ou testar: {0: 1, 1: 3} para dar 3x mais peso à classe 1
#
# 🔧 Para TUNING sistemático:
#    - Use GridSearchCV ou RandomizedSearchCV
#    - Exemplo: n_estimators: [50, 100, 200]
#              max_depth: [5, 10, 15, 20]
#
# ============================================================================