# tsml/models/time_series_model_trainer.py

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import RFE
from scipy.stats import t
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
import warnings
warnings.filterwarnings('ignore')

class TSModelTrainer:
    """
    Classe para treinar e avaliar modelos de séries temporais.

    Parâmetros:
    - models_params (dict): Dicionário com modelos e seus parâmetros.
    - df (pd.DataFrame): DataFrame contendo os dados.
    - feature_cols (list): Lista das colunas de features.
    - target_col (str): Nome da coluna alvo.
    - metric (str): Métrica para avaliação ('mse', 'rmse', 'mae', 'combined').
    - n_splits (int): Número de divisões para o TimeSeriesSplit.
    - n_features_to_select (int, opcional): Número de features a serem selecionadas.
    - reserve_percent (float): Percentual de dados a serem reservados para teste.
    - scaler_type (str): Tipo de scaler ('standard' ou 'minmax').
    - remove_outliers_type (str, opcional): Método para remover outliers ('mad', 'iqr' ou None).
    - reserved_data (bool): Se True, reserva dados para teste.
    - sparse_data (bool): Se True, não centraliza os dados no StandardScaler.
    """

    def __init__(self, models_params, df, feature_cols, target_col, metric='mse', n_splits=3,
                 n_features_to_select=None, reserve_percent=0.3, scaler_type='standard',
                 remove_outliers_type=None, reserved_data=True, sparse_data=True):
        # Atributos iniciais
        self.models_params = models_params
        self.df = df.copy()
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.metric = metric
        self.n_splits = n_splits
        self.n_features_to_select = n_features_to_select
        self.reserve_percent = reserve_percent
        self.scaler_type = scaler_type
        self.remove_outliers_type = remove_outliers_type
        self.reserved_data_flag = reserved_data
        self.sparse_data = sparse_data
        
        
        # Atributos adicionais
        self.feature_importances = None
        self.best_model = None
        self.best_params = None
        self.best_score = float('inf')
        self.best_model_name = ""
        self.selected_features = []
        self.conclusion_analyze_residuals = None
        self.models_performance = {}
        self.performance_table = None
        self.best_scores_models = {}
        self.time_series_split = TimeSeriesSplit(n_splits=self.n_splits)

        # Prepara os dados
        self.prepare_data()

    def remove_outliers_mad(self, df, column, threshold=3.5):
        """Remove outliers usando o método MAD."""
        median = df[column].median()
        absolute_deviation = (df[column] - median).abs()
        mad = absolute_deviation.median()
        modified_z_score = 0.6745 * absolute_deviation / mad
        return df[modified_z_score < threshold]

    def remove_outliers_iqr(self, df, column):
        """Remove outliers usando o método IQR."""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    def prepare_data(self):
        """Prepara os dados para treinamento e teste."""
        # Remoção de outliers
        if self.remove_outliers_type == 'mad':
            self.df = self.remove_outliers_mad(self.df, column=self.target_col)
        elif self.remove_outliers_type == 'iqr':
            self.df = self.remove_outliers_iqr(self.df, column=self.target_col)

        # Separação dos dados reservados
        if self.reserved_data_flag:
            reserve_index = int(len(self.df) * (1 - self.reserve_percent))
            self.reserved_data = self.df.iloc[reserve_index:]
            self.df = self.df.iloc[:reserve_index]
            self.X_test = self.reserved_data[self.feature_cols].values
            self.y_test = self.reserved_data[self.target_col].values
        else:
            self.X_test = None
            self.y_test = None

        # Dados de treinamento
        self.X = self.df[self.feature_cols].values
        self.y = self.df[self.target_col].values

    def select_scaler(self):
        """Seleciona o scaler de acordo com o tipo especificado."""
        if self.scaler_type == 'minmax':
            return MinMaxScaler()
        else:
            return StandardScaler(with_mean=self.sparse_data)

    def create_pipeline(self, model):
        """Cria o pipeline de treinamento."""
        scaler = self.select_scaler()
        steps = [('scaler', scaler)]
        if self.n_features_to_select:
            steps.append(('feature_selection', RFE(estimator=model, n_features_to_select=self.n_features_to_select)))
        steps.append(('model', model))
        return Pipeline(steps)

    def train_models(self):
        """Treina os modelos e avalia o desempenho."""
        for model_name, (model, param_grid) in self.models_params.items():
            try:
                print(f"Treinando {model_name}...")
                pipeline = self.create_pipeline(model)
                params = {f'model__{key}': val for key, val in param_grid.items()}
                if self.n_features_to_select:
                    params.update({f'feature_selection__estimator__{key}': val for key, val in param_grid.items()})

                grid_search = GridSearchCV(
                    pipeline,
                    params,
                    cv=self.time_series_split,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                grid_search.fit(self.X, self.y)
                val_score = -grid_search.best_score_

                best_params = grid_search.best_params_
                best_params_cleaned = {key.replace('model__', ''): val for key, val in best_params.items() if 'model__' in key}
                model_class = type(model)
                pipeline = self.create_pipeline(model_class(**best_params_cleaned))
                pipeline.fit(self.X, self.y)

                # Avaliação no conjunto de teste reservado
                if self.X_test is not None and self.y_test is not None:
                    y_pred = pipeline.predict(self.X_test)
                    mse, rmse, mae, r2, combined_score = self.compute_metrics(self.y_test, y_pred)
                    score = self.select_metric(mse, rmse, mae, r2, combined_score, val_score)
                    self.models_performance[model_name] = self.evaluate_model_test(
                        model=pipeline,
                        score=val_score,
                        model_name=model_name
                    )
                else:
                    score = val_score

                # Atualização do melhor modelo
                if score < self.best_score:
                    self.best_score = score
                    self.best_model = pipeline
                    self.best_params = best_params_cleaned
                    self.best_model_name = model_name.upper()
                    self.selected_features = self.get_feature_importances(pipeline)
                    self.best_scores_models[model_name.upper()] = score
                    if self.n_features_to_select:
                        self.get_feature_importances(pipeline)
 
                        
            except Exception as e:
                print(f"Erro ao treinar {model_name}: {e}")
                continue

        # Criação da tabela de desempenho
        if self.models_performance:
            self.performance_table = pd.DataFrame(self.models_performance).T
            if 'Combined Score' in self.performance_table.columns:
                self.performance_table.sort_values('Combined Score', inplace=True)
            else:
                print("Coluna 'Combined Score' não encontrada nos dados de desempenho.")
        else:
            print("Nenhum modelo foi treinado com sucesso.")

    def compute_metrics(self, y_true, y_pred, weights=None):
        """Calcula as métricas de desempenho."""
        if weights is None:
            weights = {'MAE': 0.25, 'RMSE': 0.25, 'MSE': 0.25, 'R2': 0.25}

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        combined_score = sum([
            weights['MAE'] * mae,
            weights['RMSE'] * rmse,
            weights['MSE'] * mse,
            weights['R2'] * (1 - r2)
        ])
        return mse, rmse, mae, r2, combined_score

    def select_metric(self, mse, rmse, mae, r2, combined_score, val_score):
        """Seleciona a métrica para avaliação."""
        if self.metric == 'mse':
            return mse
        elif self.metric == 'rmse':
            return rmse
        elif self.metric == 'mae':
            return mae
        elif self.metric == 'combined':
            return combined_score
        else:
            return val_score

    def evaluate_model_test(self, model, score, model_name):
        """Avalia o modelo no conjunto de teste."""
        y_pred = model.predict(self.X_test)
        mse, rmse, mae, r2, combined_score = self.compute_metrics(self.y_test, y_pred)
        print(f"Avaliação do Modelo {model_name.upper()} || MSE = {round(mse, 5)}, RMSE = {round(rmse, 5)}, MAE = {round(mae, 5)}, R2 = {round(r2, 5)}, Combined Score = {round(combined_score, 5)}, Score = {score}")
        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Combined Score': combined_score, 'Score': score}

    def get_feature_importances(self, pipeline):
        if self.n_features_to_select:
            model = pipeline.named_steps['model']
            selector = pipeline.named_steps['feature_selection']
        
            if hasattr(model, 'coef_'):
                feature_importances = model.coef_
            elif hasattr(model, 'feature_importances_'):
                feature_importances = model.feature_importances_
            else:
                print("Model does not support feature importance.")
                return None
        
            self.selected_features = np.array(self.feature_cols)[selector.support_]
            self.feature_importances = feature_importances
            return feature_importances  
        else:
            self.selected_features = self.feature_cols
            return None  

    def plot_feature_importance(self):
        if self.feature_importances is None:
            print("Feature importances not set.")
            return
    
        plt.figure(figsize=(10, 6))
        indices = np.argsort(self.feature_importances)
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), self.feature_importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [self.selected_features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()


    def plot_predictions_vs_actual(self, X_test_new_horizont=None, confidence_intervals=False, only_reserved_data=False):
        """Plota as previsões em comparação com os valores reais."""
        if self.best_model is None:
            print("Nenhum modelo disponível para plotar previsões.")
            return

        # Preparar os dados para plotagem
        if only_reserved_data:
            y_test_pred = self.best_model.predict(self.X_test)
            y_test_actual = self.y_test
        else:
            y_train_pred = self.best_model.predict(self.X)
            y_test_pred = self.best_model.predict(self.X_test)
            y_train_actual = self.y
            y_test_actual = self.y_test

        # Se X_test_new_horizont for fornecido, prever e preparar para plotagem
        if X_test_new_horizont is not None:
            if isinstance(X_test_new_horizont, pd.Series):
                X_test_new_horizont = X_test_new_horizont.to_frame()
            y_test_new_horizont_pred = self.best_model.predict(X_test_new_horizont)
            new_horizont_index_start = len(self.y) + len(self.y_test)
            new_horizont_index_end = new_horizont_index_start + len(y_test_new_horizont_pred)

        # Configurar o plot
        plt.figure(figsize=(25, 20))

        # Plotar intervalos de confiança se solicitado
        if confidence_intervals:
            if only_reserved_data:
                lower_bounds_test, upper_bounds_test, mean_prediction_test = self.jackknife_confidence_intervals(self.X_test, self.y_test)
                plt.fill_between(range(len(lower_bounds_test)), lower_bounds_test, upper_bounds_test, color='lightgreen', alpha=0.4, label='Intervalo de Confiança (Teste)')
                plt.plot(range(len(lower_bounds_test)), mean_prediction_test, 'g-', label='Previsão Média (Teste)')
            else:
                lower_bounds_train, upper_bounds_train, mean_prediction_train = self.jackknife_confidence_intervals(self.X, self.y)
                lower_bounds_test, upper_bounds_test, mean_prediction_test = self.jackknife_confidence_intervals(self.X_test, self.y_test)
                plt.fill_between(range(len(lower_bounds_train)), lower_bounds_train, upper_bounds_train, color='skyblue', alpha=0.4, label='Intervalo de Confiança (Treino)')
                plt.plot(range(len(lower_bounds_train)), mean_prediction_train, 'b-', label='Previsão Média (Treino)')
                plt.fill_between(range(len(lower_bounds_test)), lower_bounds_test, upper_bounds_test, color='lightgreen', alpha=0.4, label='Intervalo de Confiança (Teste)')
                plt.plot(range(len(lower_bounds_test)), mean_prediction_test, 'g-', label='Previsão Média (Teste)')

        # Plotar valores reais vs. previsões
        elif only_reserved_data:
            plt.plot(np.arange(len(y_test_actual)), y_test_actual, label='Real (Teste)', color='orange')
            plt.plot(np.arange(len(y_test_pred)), y_test_pred, label='Previsão (Teste)', linestyle='--', color='blue')
        else:
            plt.plot(np.arange(len(y_train_actual)), y_train_actual, label='Real (Treino)', color='orange')
            plt.plot(np.arange(len(y_train_pred)), y_train_pred, label='Previsão (Treino)', linestyle='--', color='blue')
            plt.plot(np.arange(len(y_train_actual), len(y_train_actual) + len(y_test_actual)), y_test_actual, label='Real (Teste)', color='yellow')
            plt.plot(np.arange(len(y_train_pred), len(y_train_pred) + len(y_test_pred)), y_test_pred, label='Previsão (Teste)', linestyle='--', color='purple')

            # Plotar novo horizonte se fornecido
            if X_test_new_horizont is not None:
                plt.plot(np.arange(new_horizont_index_start, new_horizont_index_end), y_test_new_horizont_pred, label='Previsão (Novo Horizonte)', linestyle='--', color='red')

        # Finalizar o plot
        plt.xlabel('Índice de Tempo')
        plt.ylabel(self.target_col)
        plt.title(f'Valores Reais vs. Previstos para {self.best_model_name.upper()}')
        plt.legend()
        plt.grid(True)
        plt.show()

    def analyze_residuals(self, only_reserved_data=False):
        """Analisa os resíduos do modelo."""
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import statsmodels.api as sm
        from statsmodels.stats.diagnostic import het_breuschpagan, het_white
        from statsmodels.stats.stattools import durbin_watson
        import scipy.stats as stats
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    
        if self.best_model is None:
            print("Nenhum modelo disponível para análise de resíduos.")
            return
    
        if only_reserved_data:
            X = self.X_test
            y = self.y_test
        else:
            X = self.X
            y = self.y
    
        # Calcular valores ajustados e resíduos
        fitted_values = self.best_model.predict(X)
        residuals = y - fitted_values
    
        # Converter 'residuals' para array unidimensional
        residuals = residuals.flatten()
    
        # Converter X para DataFrame se não for
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_cols)
    
        # Adicionar uma constante a exog se não houver
        exog = X.copy()
        if 'const' not in exog.columns:
            exog = sm.add_constant(exog)
    
        # Remover observações com valores nulos ou infinitos
        mask = (
            ~np.isnan(residuals)
            & ~np.isinf(residuals)
            & ~np.isnan(exog).any(axis=1)
            & ~np.isinf(exog).any(axis=1)
        )
        residuals = residuals[mask]
        exog = exog[mask]
    
        # Verificar se 'residuals' e 'exog' têm o mesmo número de observações
        if residuals.shape[0] != exog.shape[0]:
            print("Erro: 'residuals' e 'exog' têm números diferentes de observações.")
            return
    
        # Verificar multicolinearidade
        rank_exog = np.linalg.matrix_rank(exog)
        num_vars = exog.shape[1]
        if rank_exog < num_vars:
            print("Aviso: Multicolinearidade detectada.")
            vif_data = pd.DataFrame()
            vif_data["feature"] = exog.columns
            vif_data["VIF"] = [variance_inflation_factor(exog.values, i) for i in range(exog.shape[1])]
            print(vif_data)
    
            # Remover variáveis com VIF infinito ou muito alto
            high_vif = vif_data[vif_data['VIF'] > 10]['feature']
            if not high_vif.empty:
                print("Removendo variáveis com VIF alto:")
                print(high_vif)
                exog = exog.drop(columns=high_vif)
    
            # Atualizar rank e número de variáveis
            rank_exog = np.linalg.matrix_rank(exog)
            num_vars = exog.shape[1]
    
            # Verificar novamente se o problema foi resolvido
            if rank_exog < num_vars:
                print("Erro: Ainda há multicolinearidade após remover variáveis.")
                print("Não é possível executar o teste de White devido à multicolinearidade.")
                return
    
        # Continuar com a análise
    
        # Gráfico de dispersão dos resíduos
        plt.figure(figsize=(10, 6))
        plt.scatter(fitted_values, residuals)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Valores Ajustados')
        plt.ylabel('Resíduos')
        plt.title('Gráfico de Dispersão dos Resíduos')
        plt.show()
    
        # Gráfico Q-Q Plot
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot dos Resíduos')
        plt.show()
    
        # Testes de heterocedasticidade
        bp_test = het_breuschpagan(residuals, exog)
        print(f"Teste de Breusch-Pagan: estatística={bp_test[0]}, p-valor={bp_test[1]}")
    
        white_test = het_white(residuals, exog)
        print(f"Teste de White: estatística={white_test[0]}, p-valor={white_test[1]}")
    
        # Teste de autocorrelação
        dw_stat = durbin_watson(residuals)
        print(f"Teste de Durbin-Watson: estatística={dw_stat}")
    
        # Conclusão
        conclusion = "Conclusão: A análise de resíduos é crucial para validar a adequação do modelo."
        if bp_test[1] < 0.05 or white_test[1] < 0.05:
            conclusion += " Há evidências de heterocedasticidade, indicando que a variância dos resíduos não é constante. Pode ser necessário transformar variáveis ou especificar um modelo diferente."
        if dw_stat < 1.5 or dw_stat > 2.5:
            conclusion += " Há evidências de autocorrelação, sugerindo que os resíduos não são independentes. Pode ser necessário um modelo de séries temporais ou adicionar variáveis defasadas."
        if bp_test[1] >= 0.05 and white_test[1] >= 0.05 and 1.5 <= dw_stat <= 2.5:
            conclusion += " Os resíduos parecem ser homocedásticos e não autocorrelacionados, indicando que o modelo está bem especificado."
        self.conclusion_analyze_residuals = conclusion
        print(conclusion)

    def plot_model_comparisons(self):
        """Plota a comparação dos modelos treinados."""
        if not self.best_scores_models:
            print("Nenhum modelo disponível para comparação.")
            return
        plt.figure(figsize=(10, 5))
        plt.bar(self.best_scores_models.keys(), self.best_scores_models.values())
        plt.xlabel('Modelos')
        plt.ylabel('Scores')
        plt.xticks(rotation=45)
        plt.title(f'Comparação de Modelos para Série Temporal - Métrica: {self.metric.upper()}')
        plt.show()
