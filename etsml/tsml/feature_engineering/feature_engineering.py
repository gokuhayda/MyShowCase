import os
import pickle
from itertools import combinations
from collections import defaultdict
import time
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from scipy.special import log1p
from scipy.stats import boxcox
from pmdarima import auto_arima
# import cudf
# import cupy
import logging

# Configurar logging para melhor gerenciamento de mensagens
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Classe para engenharia de recursos com suporte a transformações numéricas e categóricas,
    detecção de outliers, normalização, criação de interações e seleção de recursos.
    """

    def __init__(self, apply_boxcox=False, min_poly_degree=1, max_poly_degree=2, min_comb_size=2, max_comb_size=2,
                 lag_features=None, auto_lag=None, handle_missing='drop', impute_strategy='mean',
                 custom_transformations=None, save_transformers_dir=None, categorical_cols=None, encoding_strategy='auto',
                 verbose=False, use_gpu=False):
        self.apply_boxcox = apply_boxcox
        self.min_poly_degree = min_poly_degree
        self.max_poly_degree = max_poly_degree
        self.min_comb_size = min_comb_size
        self.max_comb_size = max_comb_size
        self.lag_features = lag_features
        self.auto_lag = auto_lag
        self.handle_missing = handle_missing
        self.impute_strategy = impute_strategy
        self.custom_transformations = custom_transformations
        self.save_transformers_dir = save_transformers_dir
        self.categorical_cols = categorical_cols
        self.encoding_strategy = encoding_strategy
        self.verbose = verbose
        self.use_gpu = use_gpu  # Adicionado para suporte a GPU
        self.transformers = defaultdict(dict)
        self.history = []

    def fit(self, X, y=None):
        """
        Ajusta transformadores necessários, como Box-Cox e codificadores para dados categóricos.
        """
        if self.categorical_cols is None:
            self.categorical_cols = self.detect_categorical_columns(X)

        if self.encoding_strategy in ['ordinal', 'target'] and self.categorical_cols:
            self.transformers['categorical'] = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            self.transformers['categorical'].fit(X[self.categorical_cols])

        if self.apply_boxcox:
            for col in self.detect_numerical_columns(X):
                if (X[col] > 0).all():
                    _, lmbda = boxcox(X[col] + 1)  # Calcula o valor de lambda
                    self.transformers['boxcox'][col] = lmbda  # Armazena o valor ajustado de lambda
                else:
                    logger.warning(f"Skipping Box-Cox for column '{col}' due to non-positive values.")

        return self

    def transform(self, X, y=None):
        """
        Aplica todas as transformações no DataFrame.
        """
        df = X.copy()
            
        steps = [
            ("Handle missing values", self.handle_missing_values),  
            ("Categorical transformations", self.apply_categorical_transformations),  
            ("Drop low variance features", self.drop_low_variance_features),  
            ("Handle outliers", self.handle_outliers),  
            ("Box-Cox transformations", self.apply_boxcox_transform),  
            ("Numeric transformations", self.apply_numeric_transformations),  
            ("Create interactions", self.create_interactions),  
            ("Polynomial features", self.apply_polynomial_features),  
            ("Combination features", self.apply_combination_features),  
            ("Lagged features", self.apply_lagged_features),  
            ("Normalize features", self.normalize_numerical_features),  
            ("Custom transformations", self.apply_custom_transformations), 
        ]

        start_time = time.time()

        for step_count, (step_name, step_function) in enumerate(steps, 1):
            self.log_step(step_name, step_count, len(steps), start_time)
            if step_function in [self.apply_categorical_transformations]:
                df = step_function(df, y)
            else:
                df = step_function(df)
            self.history.append(step_name)

        if y is not None:
            self.log_step("Select important features", len(steps) + 1, len(steps) + 1, start_time)
            df = self.select_important_features(df, y)

        if self.save_transformers_dir:
            self.save_transformers(df)

        self.log_final_performance(start_time)
        return df

    def log_step(self, step_name, step_count, total_steps, start_time):
        """
        Exibe o progresso da etapa no console e no log.
        """
        elapsed_time = time.time() - start_time
        logger.info(f"Step {step_count}/{total_steps}: {step_name}... (Elapsed time: {elapsed_time:.2f} seconds)")

    def log_final_performance(self, start_time):
        """
        Loga o desempenho final da transformação.
        """
        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"Feature engineering completed in {int(hours)}h {int(minutes)}m {int(seconds)}s.")

    def handle_missing_values(self, df):
        """
        Trata valores ausentes no DataFrame.
        """
        if self.handle_missing == 'impute':
            imputer = SimpleImputer(strategy=self.impute_strategy)
            df[:] = imputer.fit_transform(df)
        elif self.handle_missing == 'drop':
            df.dropna(inplace=True)
        return df
        
    def boxcox1p(self, x, lmbda):
        """
        Implementação manual da transformação boxcox1p.
    
        Parâmetros:
        - x: Coluna ou array para transformação.
        - lmbda: Parâmetro de transformação ajustado.
    
        Retorna:
        - Transformação Box-Cox aplicada a x.
        """
        return boxcox(x + 1, lmbda) if lmbda != 0 else np.log1p(x)
        
    def detect_categorical_columns(self, df):
        """
        Detecta automaticamente colunas categóricas.

        Parâmetros:
        - df: DataFrame de entrada.

        Retorna:
        - Lista de colunas categóricas.
        """
        return [col for col in df.columns if df[col].dtype == 'object' or df[col].dtype.name == 'category']

    def detect_numerical_columns(self, df):
        """
        Detecta automaticamente colunas numéricas.

        Parâmetros:
        - df: DataFrame de entrada.

        Retorna:
        - Lista de colunas numéricas.
        """
        return [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]

    def apply_categorical_transformations(self, df, y=None):
        """
        Aplica transformações em colunas categóricas com base na estratégia definida.

        Parâmetros:
        - df: DataFrame com as colunas categóricas.
        - y: Série com a variável alvo (necessária para Target Encoding).

        Retorna:
        - DataFrame com transformações categóricas aplicadas.
        """
        if not self.categorical_cols:
            return df  # Sem colunas categóricas, retorna o DataFrame original

        if self.encoding_strategy == 'auto':
            for col in self.categorical_cols:
                if df[col].nunique() <= 10:
                    df = self.apply_one_hot_encoding(df, col)
                else:
                    df = self.apply_frequency_encoding(df, col)
        elif self.encoding_strategy == 'onehot':
            for col in self.categorical_cols:
                df = self.apply_one_hot_encoding(df, col)
        elif self.encoding_strategy == 'ordinal':
            if 'categorical' in self.transformers:
                df[self.categorical_cols] = self.transformers['categorical'].transform(df[self.categorical_cols])
        elif self.encoding_strategy == 'target' and y is not None:
            for col in self.categorical_cols:
                df = self.apply_target_encoding(df, col, y)
        elif self.encoding_strategy == 'frequency':
            for col in self.categorical_cols:
                df = self.apply_frequency_encoding(df, col)

        return df

    def apply_one_hot_encoding(self, df, col):
        """
        Aplica One-Hot Encoding para uma coluna.

        Parâmetros:
        - df: DataFrame de entrada.
        - col: Nome da coluna a ser transformada.

        Retorna:
        - DataFrame com One-Hot Encoding aplicado.
        """
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        encoded = encoder.fit_transform(df[[col]])
        encoded_cols = [f"{col}_{category}" for category in encoder.categories_[0][1:]]
        df = df.drop(col, axis=1).join(pd.DataFrame(encoded, columns=encoded_cols, index=df.index))
        self.transformers['onehot'][col] = encoder
        return df

    def apply_frequency_encoding(self, df, col):
        """
        Aplica Frequency Encoding para uma coluna.

        Parâmetros:
        - df: DataFrame de entrada.
        - col: Nome da coluna a ser transformada.

        Retorna:
        - DataFrame com Frequency Encoding aplicado.
        """
        freq_map = df[col].value_counts(normalize=True).to_dict()
        df[f'{col}_freq'] = df[col].map(freq_map)
        self.transformers['frequency'][col] = freq_map
        df.drop(col, axis=1, inplace=True)
        return df

    def apply_target_encoding(self, df, col, y):
        """
        Aplica Target Encoding para uma coluna.

        Parâmetros:
        - df: DataFrame de entrada.
        - col: Nome da coluna a ser transformada.
        - y: Série com a variável alvo.

        Retorna:
        - DataFrame com Target Encoding aplicado.
        """
        target_mean = df.groupby(col)[y.name].mean()
        df[f'{col}_target'] = df[col].map(target_mean)
        self.transformers['target'][col] = target_mean
        df.drop(col, axis=1, inplace=True)
        return df

    def drop_low_variance_features(self, df, threshold=0.01):
        """
        Remove colunas com baixa variância.

        Parâmetros:
        - df: DataFrame de entrada.
        - threshold: Variância mínima para manter a coluna.

        Retorna:
        - DataFrame sem colunas de baixa variância.
        """
        variances = df.var()
        low_variance_cols = variances[variances < threshold].index.tolist()
        return df.drop(columns=low_variance_cols)

    def handle_outliers(self, df, strategy='clip', multiplier=1.5):
        """
        Lida com outliers em colunas numéricas.

        Parâmetros:
        - df: DataFrame de entrada.
        - strategy: Estratégia para tratar outliers ('clip', 'remove').
        - multiplier: Multiplicador para o intervalo interquartil (IQR).

        Retorna:
        - DataFrame com outliers tratados.
        """
        for col in self.detect_numerical_columns(df):
            q1, q3 = np.percentile(df[col], [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - multiplier * iqr
            upper_bound = q3 + multiplier * iqr

            if strategy == 'clip':
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            elif strategy == 'remove':
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df

    def normalize_numerical_features(self, df, method='minmax'):
        numerical_cols = self.detect_numerical_columns(df)
        if numerical_cols:
            scaler = MinMaxScaler() if method == 'minmax' else StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        return df

    def create_interactions(self, df):
        """
        Cria interações não-lineares entre colunas numéricas.
        """
        interaction_features = {}
        numerical_cols = self.detect_numerical_columns(df)
        for col1, col2 in combinations(numerical_cols, 2):
            interaction_features[f'interaction_{col1}_{col2}'] = df[col1] * df[col2]
        return pd.concat([df, pd.DataFrame(interaction_features, index=df.index)], axis=1)

    def apply_custom_transformations(self, df):
        """
        Aplica transformações personalizadas no DataFrame.

        Parâmetros:
        - df: DataFrame de entrada.

        Retorna:
        - DataFrame com transformações personalizadas aplicadas.
        """
        if self.custom_transformations:
            for col, func in self.custom_transformations.items():
                if col in df.columns:
                    df[f'custom_{col}'] = func(df[col])
        return df

    def apply_boxcox_transform(self, df):
        """
        Aplica a transformação Box-Cox nas colunas ajustadas.
        """
        for col in self.detect_numerical_columns(df):
            if col in self.transformers.get('boxcox', {}):
                lmbda = self.transformers['boxcox'][col]
                df[f'boxcox_{col}'] = self.boxcox1p(df[col], lmbda)
            else:
                logger.warning(f"Skipping Box-Cox for column '{col}' due to non-positive values.")
        return df

    def apply_polynomial_features(self, df):
        """
        Cria recursos polinomiais.
        
        Parâmetros:
        - df: DataFrame de entrada.
        
        Retorna:
        - DataFrame com recursos polinomiais adicionados.
        """
        poly_features = {}  # Dictionary to store new features
        
        for col in self.detect_numerical_columns(df):
            for degree in range(self.min_poly_degree, self.max_poly_degree + 1):
                poly_features[f'{col}_poly_{degree}'] = df[col] ** degree
        
        # Concatenate all new features at once
        df = pd.concat([df, pd.DataFrame(poly_features, index=df.index)], axis=1)
        return df

    def apply_combination_features(self, df):
        """
        Cria recursos baseados em combinações de colunas.
        """
        comb_features = {}
        for comb_size in range(self.min_comb_size, self.max_comb_size + 1):
            for comb in combinations(df.columns, comb_size):
                comb_name = '_'.join(comb)
                comb_features[f'prod_{comb_name}'] = df[list(comb)].prod(axis=1)
                if len(comb) == 2:
                    comb_features[f'div_{comb[0]}_by_{comb[1]}'] = np.where(
                        df[comb[1]] != 0, df[comb[0]] / df[comb[1]], 0)
        return pd.concat([df, pd.DataFrame(comb_features, index=df.index)], axis=1)

    def apply_lagged_features(self, df, y=None):
        """
        Cria recursos defasados com base nas configurações fornecidas.
    
        Parâmetros:
        - df: DataFrame de entrada.
        - y: Série com a variável alvo (opcional).
    
        Retorna:
        - DataFrame com recursos defasados adicionados.
        """
        lagged_features = {}
    
        # Adiciona lags definidos manualmente
        if self.lag_features:
            for col, lags in self.lag_features.items():
                for lag in lags:
                    lagged_features[f'{col}_lag_{lag}'] = df[col].shift(lag)
    
        # Configura o auto_arima para os lags automáticos
        if self.auto_lag:
            for col, arima_config in self.auto_lag.items():
                # Verifica se há valores nulos e aplica o modelo
                try:
                    series = df[col].dropna()  # Remove valores ausentes para o ajuste
                    if len(series) > 10:  # Verifica se há dados suficientes para ajustar o modelo
                        auto_model = auto_arima(
                            series,
                            seasonal=arima_config.get("seasonal", False),
                            m=arima_config.get("m", 1),
                            max_order=arima_config.get("max_order", None),
                            max_p=arima_config.get("max_p", 5),
                            max_q=arima_config.get("max_q", 5),
                            max_d=arima_config.get("max_d", 2),
                            suppress_warnings=True,
                            stepwise=True,
                            error_action='ignore'
                        )
                        # Adiciona os lags baseados na ordem do modelo ajustado
                        for lag in range(1, auto_model.order[0] + 1):  # Ordem AR (Auto-Regressivo)
                            lagged_features[f'{col}_auto_lag_{lag}'] = df[col].shift(lag)
                    else:
                        logger.warning(f"Skipping auto_lag for column '{col}' due to insufficient data.")
                except Exception as e:
                    logger.error(f"Error fitting auto_arima for column '{col}': {e}")
                    continue
    
        # Adiciona os recursos defasados ao DataFrame
        if lagged_features:
            df = pd.concat([df, pd.DataFrame(lagged_features, index=df.index)], axis=1)
    
        return df


    def select_important_features(self, df, y, model=None, top_n=10):
        """
        Seleciona as colunas mais importantes com base em um modelo ou correlação.

        Parâmetros:
        - df: DataFrame com recursos.
        - y: Série com a variável alvo.
        - model: Modelo para calcular a importância dos recursos.
        - top_n: Número de recursos importantes a selecionar.

        Retorna:
        - DataFrame reduzido aos recursos mais importantes.
        """
        if model is None:
            model = RandomForestClassifier(random_state=42)

        model.fit(df, y)
        importances = pd.Series(model.feature_importances_, index=df.columns)
        important_features = importances.nlargest(top_n).index.tolist()
        return df[important_features]

    def save_transformers(self, df):
        """
        Salva os transformadores e o DataFrame transformado.

        Parâmetros:
        - df: DataFrame final transformado.
        """
        os.makedirs(self.save_transformers_dir, exist_ok=True)
        with open(os.path.join(self.save_transformers_dir, 'transformers.pkl'), 'wb') as f:
            pickle.dump(self.transformers, f)
        df.to_csv(os.path.join(self.save_transformers_dir, 'final_feature_engineering.csv'), index=False)
        if self.verbose:
            print(f"Transformers and DataFrame saved to {self.save_transformers_dir}")

    def transform(self, X, y=None):
        """
        Aplica todas as transformações no DataFrame.
    
        Parâmetros:
        - X: DataFrame de entrada.
        - y: Opcional, série com a variável alvo (necessária para Target Encoding e seleção de recursos).
    
        Retorna:
        - DataFrame transformado.
        """
        df = X.copy()
    
            
        steps = [
            ("Handle missing values", self.handle_missing_values),  
            ("Categorical transformations", self.apply_categorical_transformations),  
            ("Drop low variance features", self.drop_low_variance_features),  
            ("Handle outliers", self.handle_outliers),  
            ("Box-Cox transformations", self.apply_boxcox_transform),  
            ("Numeric transformations", self.apply_numeric_transformations),  
            ("Create interactions", self.create_interactions),  
            ("Polynomial features", self.apply_polynomial_features),  
            ("Combination features", self.apply_combination_features),  
            ("Lagged features", self.apply_lagged_features),  
            ("Normalize features", self.normalize_numerical_features),  
            ("Custom transformations", self.apply_custom_transformations), 
        ]
        
        start_time = time.time()
    
        for step_count, (step_name, step_function) in enumerate(steps, 1):
            self.log_step(step_name, step_count, len(steps), start_time)
            if step_function in [self.apply_categorical_transformations]:
                df = step_function(df, y)
            else:
                df = step_function(df)
            self.history.append(step_name)
    
        if y is not None:
            self.print_step("Select important features", len(steps) + 1, len(steps) + 1, start_time)
            df = self.select_important_features(df, y)
    
        if self.save_transformers_dir:
            self.save_transformers(df)
    
        return df

    def apply_numeric_transformations(self, df):
        """
        Aplica transformações matemáticas a colunas numéricas.

        Parâmetros:
        - df: DataFrame de entrada.

        Retorna:
        - DataFrame com transformações matemáticas adicionadas.
        """
        transformed_features = {}
        transformations_applied = {}

        numerical_cols = self.detect_numerical_columns(df)

        for feature_col in numerical_cols:
            # Outras transformações matemáticas
            for transformation in ['sqrt', 'square', 'cbrt', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan']:
                new_feature = f'{transformation}_{feature_col}'
                try:
                    # Verifica se a função existe no módulo numpy
                    if hasattr(np, transformation):
                        func = getattr(np, transformation)
                        # Evita transformações inválidas, como sqrt de valores negativos
                        if transformation in ['sqrt', 'arcsin', 'arccos'] and (df[feature_col] < 0).any():
                            continue
                        # Aplica a transformação
                        transformed_features[new_feature] = func(df[feature_col])
                        transformations_applied[new_feature] = transformation
                except Exception as e:
                    logger.warning(f"Transformation {transformation} failed for {feature_col}: {e}")

        # Adiciona as novas colunas transformadas ao DataFrame
        if transformed_features:
            df = pd.concat([df, pd.DataFrame(transformed_features, index=df.index)], axis=1)

        return df

    def use_gpu_for_large_data(self, df):
        """
        Converte DataFrame para formato compatível com GPU, se configurado.
        """
        if self.use_gpu:
            try:
                import cudf
                logger.info("Using GPU with cuDF for faster processing.")
                return cudf.DataFrame.from_pandas(df)
            except ImportError:
                logger.warning("cuDF not installed. Falling back to CPU.")
        return df
