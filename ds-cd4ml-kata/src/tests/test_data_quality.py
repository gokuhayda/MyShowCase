"""
Data Quality Tests
==================
Testes automatizados para validar qualidade e integridade dos dados.

Categorias de testes:
1. Schema Compliance: Estrutura e tipos corretos
2. Data Integrity: Valores faltantes, duplicatas
3. Statistical Properties: Distribuições, ranges
4. Business Rules: Regras de negócio específicas
5. Data Leakage Prevention: Evitar vazamento de informação

Todos os testes são executados automaticamente no CI/CD antes do treino.
"""
import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path para importar schemas
sys.path.insert(0, str(Path(__file__).parent.parent))
from data.schemas import processed_schema

# ============================================================================
# FIXTURES - Configuração de dados para os testes
# ============================================================================

@pytest.fixture
def processed_data():
    """
    Carrega dados processados para testes.
    
    Fixture é reutilizada em múltiplos testes, evitando leitura duplicada.
    
    Returns:
        pd.DataFrame: Dados processados do wine_features.csv
    """
    data_path = "data/processed/wine_features.csv"
    
    if not Path(data_path).exists():
        pytest.skip(f"Processed data not found: {data_path}")
    
    df = pd.read_csv(data_path)
    return df

@pytest.fixture
def raw_data():
    """
    Carrega dados raw para comparações.
    
    Returns:
        pd.DataFrame: Dados raw do wine_quality.csv
    """
    data_path = "data/raw/wine_quality.csv"
    
    if not Path(data_path).exists():
        pytest.skip(f"Raw data not found: {data_path}")
    
    df = pd.read_csv(data_path)
    return df


# ============================================================================
# TESTES DE SCHEMA (Estrutura dos dados)
# ============================================================================

def test_schema_compliance(processed_data):
    """
    Test: Dados atendem schema definido (Pandera).
    
    Valida:
    - Colunas esperadas estão presentes
    - Tipos de dados corretos (float, int)
    - Constraints de range (ex: pH entre 2.5-4.5)
    
    Por que importante:
    - Modelo espera features específicas na ordem correta
    - Tipos incorretos causam erros em produção
    - Ranges inválidos indicam problemas na coleta
    """
    # Validação usando Pandera schema
    # lazy=True: coleta todos os erros ao invés de parar no primeiro
    processed_schema.validate(processed_data, lazy=True)


def test_expected_columns(processed_data):
    """
    Test: Todas colunas esperadas estão presentes.
    
    Colunas obrigatórias:
    - 11 features físico-químicas
    - 1 target (quality_binary)
    """
    expected_columns = [
        'fixed_acidity',
        'volatile_acidity', 
        'citric_acid',
        'residual_sugar',
        'chlorides',
        'free_sulfur_dioxide',
        'total_sulfur_dioxide',
        'density',
        'pH',
        'sulphates',
        'alcohol',
        'quality_binary'
    ]
    
    missing = set(expected_columns) - set(processed_data.columns)
    extra = set(processed_data.columns) - set(expected_columns)
    
    assert not missing, f"Missing columns: {missing}"
    assert not extra, f"Unexpected columns: {extra}"


def test_column_types(processed_data):
    """
    Test: Tipos de dados são corretos.
    
    Features: float64
    Target: int64
    """
    # Features devem ser float
    feature_cols = processed_data.columns.drop('quality_binary')
    for col in feature_cols:
        assert processed_data[col].dtype == np.float64, \
            f"Column {col} should be float64, got {processed_data[col].dtype}"
    
    # Target deve ser int
    assert processed_data['quality_binary'].dtype in [np.int64, np.int32], \
        f"Target should be int, got {processed_data['quality_binary'].dtype}"


# ============================================================================
# TESTES DE INTEGRIDADE (Completude dos dados)
# ============================================================================

def test_no_missing_values(processed_data):
    """
    Test: Nenhum valor faltante (NaN, None, null).
    
    Por que importante:
    - Modelos sklearn não aceitam NaN por padrão
    - Missing values indicam problemas no ETL
    - Em produção, causaria erro imediato
    
    Ação se falhar:
    - Investigar etapa de ETL
    - Implementar imputação se apropriado
    """
    missing = processed_data.isnull().sum()
    total_missing = missing.sum()
    
    if total_missing > 0:
        # Detalhar quais colunas têm missing
        cols_with_missing = missing[missing > 0]
        pytest.fail(
            f"Found {total_missing} missing values:\n"
            f"{cols_with_missing.to_dict()}"
        )


def test_no_duplicates(processed_data):
    """
    Test: Nenhuma linha duplicada.
    
    Por que importante:
    - Duplicatas inflam artificialmente o dataset
    - Podem causar data leakage (mesmo sample em train/test)
    - Enviesam métricas de avaliação
    
    Ação se falhar:
    - Investigar fonte dos dados
    - Remover duplicatas no ETL
    """
    duplicates = processed_data.duplicated().sum()
    
    assert duplicates == 0, \
        f"Found {duplicates} duplicate rows ({duplicates/len(processed_data)*100:.2f}%)"


def test_no_infinite_values(processed_data):
    """
    Test: Nenhum valor infinito (inf, -inf).
    
    Por que importante:
    - Infinitos causam erros numéricos no treino
    - Indicam divisão por zero ou overflow
    """
    feature_cols = processed_data.columns.drop('quality_binary')
    
    for col in feature_cols:
        inf_count = np.isinf(processed_data[col]).sum()
        assert inf_count == 0, \
            f"Column {col} has {inf_count} infinite values"


# ============================================================================
# TESTES ESTATÍSTICOS (Propriedades dos dados)
# ============================================================================

def test_target_distribution(processed_data):
    """
    Test: Classes do target estão balanceadas (mínimo 30% cada).
    
    Por que importante:
    - Classes muito desbalanceadas (<10%) exigem técnicas especiais
    - Modelo pode simplesmente ignorar classe minoritária
    - Accuracy se torna métrica enganosa
    
    Thresholds:
    - ⚠️  Warning: < 40%
    - ❌ Fail: < 30%
    """
    dist = processed_data['quality_binary'].value_counts(normalize=True)
    
    min_proportion = 0.30  # 30%
    
    for class_label, proportion in dist.items():
        assert proportion >= min_proportion, \
            f"Class {class_label} has only {proportion:.1%} of samples " \
            f"(minimum: {min_proportion:.0%}). Consider resampling techniques."


def test_feature_ranges(processed_data):
    """
    Test: Features estão em ranges plausíveis.
    
    Ranges baseados em conhecimento de domínio:
    - pH de vinho: 2.5 - 4.5
    - Álcool: 8% - 15%
    - Densidade: próxima de 1.0 (água)
    
    Por que importante:
    - Valores fora do range indicam erros de medição
    - Podem ser outliers extremos que prejudicam modelo
    """
    # pH deve estar entre 2.5 e 4.5 (acidez do vinho)
    assert processed_data['pH'].between(2.5, 4.5).all(), \
        f"pH out of range [2.5, 4.5]: " \
        f"min={processed_data['pH'].min():.2f}, " \
        f"max={processed_data['pH'].max():.2f}"
    
    # Álcool entre 8% e 15%
    assert processed_data['alcohol'].between(8, 15).all(), \
        f"Alcohol out of range [8, 15]: " \
        f"min={processed_data['alcohol'].min():.2f}, " \
        f"max={processed_data['alcohol'].max():.2f}"
    
    # Densidade próxima de 1.0 (água = 1.0)
    assert processed_data['density'].between(0.99, 1.01).all(), \
        f"Density out of range [0.99, 1.01]: " \
        f"min={processed_data['density'].min():.4f}, " \
        f"max={processed_data['density'].max():.4f}"


def test_no_extreme_outliers(processed_data):
    """
    Test: Sem outliers extremos (> 5 desvios padrão).
    
    Por que importante:
    - Outliers extremos podem ser erros de digitação
    - Podem dominar o treino e prejudicar generalização
    
    Método: Z-score
    - |z| > 5: outlier extremo (muito raro em distribuição normal)
    """
    feature_cols = processed_data.columns.drop('quality_binary')
    
    for col in feature_cols:
        # Calcular z-scores
        z_scores = np.abs(
            (processed_data[col] - processed_data[col].mean()) / 
            processed_data[col].std()
        )
        
        extreme_outliers = (z_scores > 5).sum()
        
        assert extreme_outliers == 0, \
            f"Column {col} has {extreme_outliers} extreme outliers (|z| > 5)"


# ============================================================================
# TESTES DE REGRAS DE NEGÓCIO
# ============================================================================

def test_no_data_leakage(processed_data):
    """
    Test: Nenhuma feature tem correlação perfeita com target.
    
    Por que importante:
    - Correlação > 0.95 indica possível data leakage
    - Leakage = informação do futuro vazando para features
    - Causa overfitting severo (100% treino, 50% teste)
    
    Exemplo de leakage:
    - Feature "foi_aprovado" para prever "aprovacao"
    - Feature calculada DEPOIS do evento que queremos prever
    """
    correlations = processed_data.corr()['quality_binary'].abs()
    
    # Remover correlação do target consigo mesmo
    correlations = correlations.drop('quality_binary')
    
    max_corr = correlations.max()
    max_corr_feature = correlations.idxmax()
    
    threshold = 0.95
    
    assert max_corr < threshold, \
        f"Suspiciously high correlation detected: " \
        f"{max_corr_feature} = {max_corr:.3f} (threshold: {threshold}). " \
        f"Possible data leakage!"


def test_positive_features(processed_data):
    """
    Test: Features que devem ser positivas não têm valores negativos.
    
    Validações de domínio:
    - Concentrações químicas não podem ser negativas
    - Medidas físicas (densidade, álcool) não podem ser negativas
    """
    positive_features = [
        'fixed_acidity',
        'volatile_acidity',
        'citric_acid',
        'residual_sugar',
        'chlorides',
        'free_sulfur_dioxide',
        'total_sulfur_dioxide',
        'sulphates',
        'alcohol'
    ]
    
    for col in positive_features:
        min_value = processed_data[col].min()
        assert min_value >= 0, \
            f"Feature {col} has negative values: min={min_value:.4f}"


# ============================================================================
# TESTES DE VOLUME E COBERTURA
# ============================================================================

def test_sample_size(processed_data):
    """
    Test: Dataset tem tamanho mínimo para treino confiável.
    
    Regra geral (Machine Learning):
    - Classificação binária: mínimo 1000 samples
    - 10x número de features: 11 features → 110 samples (muito pouco!)
    - Ideal: 100x número de features: 11 features → 1100 samples ✅
    
    Por que importante:
    - Poucos dados → overfitting garantido
    - Métricas não são confiáveis estatisticamente
    """
    min_samples = 1000
    n_samples = len(processed_data)
    
    assert n_samples >= min_samples, \
        f"Dataset has only {n_samples} samples (minimum: {min_samples}). " \
        f"Consider collecting more data or using data augmentation."


def test_feature_count(processed_data):
    """
    Test: Número correto de features.
    
    Esperado:
    - 11 features + 1 target = 12 colunas
    """
    expected_features = 12  # 11 features + 1 target
    actual_features = len(processed_data.columns)
    
    assert actual_features == expected_features, \
        f"Expected {expected_features} columns, found {actual_features}. " \
        f"Columns: {list(processed_data.columns)}"


def test_target_binary(processed_data):
    """
    Test: Target é binário (apenas 0 e 1).
    
    Por que importante:
    - Problema é classificação binária
    - Valores diferentes de 0/1 causarão erro no treino
    """
    unique_values = processed_data['quality_binary'].unique()
    
    assert set(unique_values) == {0, 1}, \
        f"Target should be binary (0, 1), found: {sorted(unique_values)}"


# ============================================================================
# TESTES DE CONSISTÊNCIA ENTRE RAW E PROCESSED
# ============================================================================

def test_processed_has_same_count_as_raw(raw_data, processed_data):
    """
    Test: ETL não perdeu nem ganhou amostras.
    
    Por que importante:
    - Perda de amostras indica bug no ETL
    - Ganho de amostras indica duplicação acidental
    """
    assert len(processed_data) == len(raw_data), \
        f"Sample count mismatch: " \
        f"raw={len(raw_data)}, processed={len(processed_data)}"


def test_feature_statistics_reasonable(processed_data):
    """
    Test: Estatísticas descritivas são razoáveis.
    
    Validações:
    - Médias não são zero (indicaria normalização errada)
    - Desvios padrão não são zero (features constantes)
    """
    feature_cols = processed_data.columns.drop('quality_binary')
    
    for col in feature_cols:
        mean = processed_data[col].mean()
        std = processed_data[col].std()
        
        # Média não deve ser exatamente zero (improvável em dados reais)
        assert abs(mean) > 1e-10, \
            f"Feature {col} has mean ~0 ({mean:.10f}). Possible normalization error."
        
        # Desvio padrão não deve ser zero (feature constante = inútil)
        assert std > 0, \
            f"Feature {col} has std=0. Constant feature, should be removed."


# ============================================================================
# SUMÁRIO DOS TESTES
# ============================================================================
"""
RESUMO DOS TESTES IMPLEMENTADOS:
================================

✅ Schema (3 testes):
   - Schema compliance (Pandera)
   - Colunas esperadas
   - Tipos corretos

✅ Integridade (3 testes):
   - Sem missing values
   - Sem duplicatas
   - Sem infinitos

✅ Estatísticos (3 testes):
   - Target balanceado (≥30% cada classe)
   - Features em ranges válidos
   - Sem outliers extremos (|z| > 5)

✅ Regras de Negócio (2 testes):
   - Sem data leakage (correlação < 0.95)
   - Features positivas são positivas

✅ Volume (3 testes):
   - Tamanho mínimo (≥1000 samples)
   - Número de features correto (12)
   - Target binário (0 ou 1)

✅ Consistência (2 testes):
   - Mesmo número de samples que raw
   - Estatísticas razoáveis

TOTAL: 16 TESTES

Para rodar:
-----------
pytest src/tests/test_data_quality.py -v

Para rodar com coverage:
------------------------
pytest src/tests/test_data_quality.py -v --cov=src/data --cov-report=html

Para rodar apenas um teste:
----------------------------
pytest src/tests/test_data_quality.py::test_no_missing_values -v
"""