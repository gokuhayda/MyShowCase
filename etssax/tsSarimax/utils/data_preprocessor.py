import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataPreprocessor:
    """
    Classe para pré-processamento de dados com suporte à identificação e tratamento
    de variáveis categóricas e não categóricas.
    """

    def __init__(self, df, target_col, categorical_cols=None):
        """
        Inicializa o pré-processador com o DataFrame, a coluna alvo e as colunas categóricas.

        Args:
            df (pd.DataFrame): DataFrame contendo os dados.
            target_col (str): Nome da coluna alvo.
            categorical_cols (list, optional): Lista de colunas categóricas. Detectadas automaticamente se None.
        """
        self.df = df.copy()
        self.target_col = target_col
        self.categorical_cols = categorical_cols or self._detect_categorical_columns()
        self.numeric_cols = [col for col in self.df.columns if col not in self.categorical_cols + [self.target_col]]
        self.scaler = None

    def _detect_categorical_columns(self):
        """
        Detecta colunas categóricas automaticamente com base no tipo de dado.

        Returns:
            list: Lista de colunas categóricas.
        """
        return self.df.select_dtypes(include=['object', 'category']).columns.tolist()

    def preprocess(self, scale_numeric=True, drop_first=True):
        """
        Executa o pré-processamento dos dados.

        Args:
            scale_numeric (bool, optional): Se True, escala variáveis numéricas. Default é True.
            drop_first (bool, optional): Se True, remove a primeira coluna ao criar dummies para evitar multicolinearidade. Default é True.

        Returns:
            pd.DataFrame: DataFrame pré-processado.
        """
        # Processar variáveis categóricas
        self._process_categorical_variables(drop_first)

        # Processar variáveis numéricas
        if scale_numeric:
            self._scale_numeric_variables()

        return self.df

    def _process_categorical_variables(self, drop_first):
        """
        Cria variáveis dummies para as colunas categóricas.

        Args:
            drop_first (bool): Se True, remove a primeira categoria para cada variável dummy.
        """
        if self.categorical_cols:
            self.df = pd.get_dummies(self.df, columns=self.categorical_cols, drop_first=drop_first)

    def _scale_numeric_variables(self):
        """
        Escala variáveis numéricas usando StandardScaler.
        """
        if self.numeric_cols:
            self.scaler = StandardScaler()
            self.df[self.numeric_cols] = self.scaler.fit_transform(self.df[self.numeric_cols])

    def inverse_transform_numeric(self, df):
        """
        Reverte a escala das variáveis numéricas.

        Args:
            df (pd.DataFrame): DataFrame com variáveis numéricas escaladas.

        Returns:
            pd.DataFrame: DataFrame com variáveis numéricas na escala original.
        """
        if self.scaler:
            df[self.numeric_cols] = self.scaler.inverse_transform(df[self.numeric_cols])
        return df

    def get_processed_columns(self):
        """
        Obtém as colunas categóricas (dummies) e numéricas processadas.

        Returns:
            tuple: (list, list) Lista de colunas categóricas processadas e numéricas.
        """
        dummies_cols = [col for col in self.df.columns if any(cat in col for cat in self.categorical_cols)]
        return dummies_cols, self.numeric_cols
