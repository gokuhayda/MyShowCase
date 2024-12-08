import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller  
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
import pandas as pd
from scipy import stats

from typing import Optional, Dict, Tuple, List

class ARIMAPredictor:
    def __init__(self, df, target_col, train_size=0.8):
        """
        Inicializa o objeto ARIMAPredictor com os dados e configurações básicas.
        """
        self.df = df
        self.target_col = target_col
        self.train_size = train_size
        self.train_df, self.test_df = self._split_data()

    def _split_data(self):
        """
        Divide os dados em conjuntos de treino e teste.
        """
        split_index = int(len(self.df) * self.train_size)
        train_df = self.df.iloc[:split_index]
        test_df = self.df.iloc[split_index:]
        return train_df, test_df

    def _compute_metrics(self, y_true, y_pred, weights=None):
        """
        Calcula as métricas de avaliação.
        """
        if weights is None:
            weights = {'MAE': 0.25, 'RMSE': 0.25, 'MSE': 0.25, 'R2': 0.25}
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        combined_score = sum([weights['MAE'] * mae, weights['RMSE'] * rmse, weights['MSE'] * mse, weights['R2'] * (1 - r2)])
        
        return mse, rmse, mae, r2, combined_score

    def train_and_predict_sarimax(
        self, 
        feature_cols, 
        seasonal=False, 
        seasonal_order=(0, 0, 0, 0), 
        stepwise=True, 
        forecast_horizon=None, 
        disp=False
    ):
        """
        Treina e faz previsões usando o modelo SARIMAX com variáveis exógenas e sazonalidade explícita.

        Parameters:
            feature_cols: Colunas de variáveis exógenas.
            seasonal: Booleano indicando se sazonalidade deve ser considerada.
            seasonal_order: Parâmetros sazonais (P, D, Q, s).
            stepwise: Utilizar ajuste stepwise no auto_arima.
            forecast_horizon: Número de passos futuros a prever.
            disp: Exibir detalhes do ajuste do modelo.

        Returns:
            train_predictions: Previsões no conjunto de treino.
            test_predictions: Previsões no conjunto de teste.
            future_predictions: Previsões futuras (opcional).
            model_fit: Objeto ajustado do modelo.
            train_metrics: Métricas do conjunto de treino.
            test_metrics: Métricas do conjunto de teste.
        """
        # Ajuste do modelo com auto_arima
        auto_model = auto_arima(
            self.train_df[self.target_col],
            exogenous=self.train_df[feature_cols],
            seasonal=seasonal,
            m=seasonal_order[3] if seasonal else 1,  # Define a periodicidade (s)
            stepwise=stepwise,
            suppress_warnings=True,
            error_action="ignore",
            trace=True
        )

        order = auto_model.order  # Parâmetros ARIMA
        seasonal_order_auto = auto_model.seasonal_order  # Parâmetros sazonais detectados

        # Usar ordem sazonal explícita, se fornecida
        final_seasonal_order = seasonal_order if seasonal else (0, 0, 0, 0)

        # Ajuste do modelo SARIMAX com sazonalidade explícita
        model = SARIMAX(
            self.train_df[self.target_col],
            exog=self.train_df[feature_cols],
            order=order,
            seasonal_order=final_seasonal_order
        )
        model_fit = model.fit(disp=disp)  # Ajuste do modelo

        # Previsões
        train_predictions = model_fit.predict(
            start=0, 
            end=len(self.train_df)-1, 
            exog=self.train_df[feature_cols]
        )
        test_predictions = model_fit.predict(
            start=len(self.train_df), 
            end=len(self.df)-1, 
            exog=self.test_df[feature_cols]
        )

        # Previsões futuras, se aplicável
        future_predictions = None
        if forecast_horizon:
            future_exog = self._generate_future_exog(feature_cols, forecast_horizon)
            future_predictions = model_fit.forecast(steps=forecast_horizon, exog=future_exog)

        # Métricas
        train_metrics = self._compute_metrics(self.train_df[self.target_col], train_predictions)
        test_metrics = self._compute_metrics(self.test_df[self.target_col], test_predictions)

        # Plot dos resultados
        self._plot_results(train_predictions, test_predictions, future_predictions, forecast_horizon)

        return train_predictions, test_predictions, future_predictions, model_fit, train_metrics, test_metrics

    def _generate_future_exog(self, feature_cols, forecast_horizon):
        """
        Gera previsões futuras para as variáveis exógenas com base nos dados históricos.
        """
        future_exog = []
        for col in feature_cols:
            # Usa ARIMA para prever os valores futuros de cada variável exógena
            auto_model = auto_arima(
                self.df[col],
                seasonal=False,  # Ajuste conforme necessário
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                trace=False
            )
            order = auto_model.order
            model = ARIMA(self.df[col], order=order)
            model_fit = model.fit()
            future_values = model_fit.predict(start=len(self.df), end=len(self.df) + forecast_horizon - 1)
            future_exog.append(future_values)
        
        # Converte a lista de arrays em um array 2D (colunas correspondem às variáveis exógenas)
        future_exog = np.column_stack(future_exog)
        return future_exog

    def predict_future(self, model_fit, feature_cols=None, future_horizon=60):
        """
        Gera previsões futuras com base em um modelo ajustado.
        
        Parameters:
            model_fit: Modelo ajustado (SARIMAX ou ARIMA).
            feature_cols: Lista de colunas de variáveis exógenas, se aplicável.
            future_horizon: Número de passos futuros para prever.
        
        Returns:
            future_predictions: Previsões futuras para o horizonte especificado.
        """
        future_exog = None
        if feature_cols:
            # Gera valores futuros para variáveis exógenas
            future_exog = self._generate_future_exog(feature_cols, future_horizon)
    
        # Faz previsões futuras com o modelo ajustado
        future_predictions = model_fit.forecast(steps=future_horizon, exog=future_exog)
        return future_predictions

    def _plot_results(self, train_predictions, test_predictions, future_predictions=None, future_horizon=None):
        """
        Plota os resultados das previsões.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(len(self.train_df[self.target_col])), self.train_df[self.target_col], label='Real (Train)', color='orange')
        plt.plot(np.arange(len(self.train_df[self.target_col]), len(self.df)), self.test_df[self.target_col], label='Real (Test)', color='yellow')
        plt.plot(np.arange(len(train_predictions)), train_predictions, label='Prediction (Train)', linestyle='--', color='blue')
        plt.plot(np.arange(len(self.train_df[self.target_col]), len(self.df)), test_predictions, label='Prediction (Test)', linestyle='dashdot', color='purple')
        if future_predictions is not None:
            plt.plot(np.arange(len(self.df), len(self.df) + future_horizon), future_predictions, label='Future Prediction', linestyle='dashdot', color='green')
        plt.xlabel('Index')
        plt.ylabel(self.target_col)
        plt.legend()
        plt.show()

    def analyze_residuals(self, model_fit, dataset=None, X: Optional[np.ndarray] = None, dataset_label="Treinamento"):
        """
        Analisa os resíduos do modelo ajustado em diferentes datasets e gera conclusões baseadas nos testes estatísticos.

        Args:
            model_fit: Modelo ajustado (SARIMAX ou ARIMA).
            dataset (Optional[pd.DataFrame], opcional): Dataset a ser usado para análise de resíduos. Se None, usa o dataset do modelo.
            X (Optional[np.ndarray], opcional): Variáveis exógenas utilizadas no modelo, se aplicável.
            dataset_label (str): Rótulo para o dataset analisado (e.g., "Treinamento", "Validação", "Teste").
        """

        # Verifica o tipo do modelo
        if not isinstance(model_fit, SARIMAXResultsWrapper):
            raise TypeError("O objeto model_fit deve ser uma instância de SARIMAXResultsWrapper.")
        
        # Calcula os resíduos
        if dataset is not None:
            predictions = model_fit.get_prediction(start=dataset.index[0], end=dataset.index[-1], exog=X)
            residuals = dataset[self.target_col] - predictions.predicted_mean
        else:
            residuals = model_fit.resid

        # Q-Q Plot dos resíduos
        plt.figure(figsize=(10, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title(f'Q-Q Plot dos Resíduos ({dataset_label})')
        plt.grid(True)
        plt.show()

        # Preparação para testes estatísticos
        X_with_const = np.column_stack((np.ones(len(X)), X)) if X is not None else np.ones(len(residuals)).reshape(-1, 1)

        # Resultados dos testes estatísticos
        test_results = {}

        # Teste de Breusch-Pagan
        if X_with_const.shape[1] > 1:  # Verifica se há mais de uma coluna
            bp_test = het_breuschpagan(residuals, X_with_const)
            test_results['Breusch-Pagan'] = {'statistic': bp_test[0], 'p_value': bp_test[1]}
        else:
            test_results['Breusch-Pagan'] = "Não aplicável (falta de variáveis exógenas)."

        # Teste de White
        if X_with_const.shape[1] > 1:
            white_test = het_white(residuals, X_with_const)
            test_results['White'] = {'statistic': white_test[0], 'p_value': white_test[1]}
        else:
            test_results['White'] = "Não aplicável (falta de variáveis exógenas)."

        # Teste de Durbin-Watson
        dw_stat = durbin_watson(residuals)
        test_results['Durbin-Watson'] = {'statistic': dw_stat}

        # Teste de Dickey-Fuller Aumentado
        adf_test = adfuller(residuals, autolag='AIC')
        test_results['ADF'] = {'statistic': adf_test[0], 'p_value': adf_test[1]}

        # Teste de Ljung-Box
        ljungbox_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
        test_results['Ljung-Box'] = ljungbox_test.to_dict('records')[0]

        # Exibe os resultados dos testes
        print(f"Resultados dos testes estatísticos ({dataset_label}):")
        for test_name, result in test_results.items():
            if isinstance(result, dict):
                statistic = result.get('statistic', 'N/A')
                p_value = result.get('p_value', 'N/A')
                statistic_str = f"{statistic:.4f}" if isinstance(statistic, (float, int)) else statistic
                p_value_str = f"{p_value:.4f}" if isinstance(p_value, (float, int)) else p_value
                print(f"- {test_name}: estatística={statistic_str}, p-valor={p_value_str}")
            else:
                print(f"- {test_name}: {result}")


        # Conclusões baseadas nos testes
        conclusions = []
        if test_results['ADF']['p_value'] > 0.05:
            conclusions.append("- Os resíduos não são estacionários, sugerindo que o modelo pode não capturar toda a dinâmica dos dados.")
        else:
            conclusions.append("- Os resíduos são estacionários, indicando que o modelo captura bem a dinâmica dos dados.")

        if 'Ljung-Box' in test_results and test_results['Ljung-Box']['lb_pvalue'] < 0.05:
            conclusions.append("- Há evidências de autocorrelação nos resíduos, indicando perda de informações no modelo.")
        else:
            conclusions.append("- Não há evidências de autocorrelação nos resíduos.")

        if isinstance(test_results['Breusch-Pagan'], dict) and test_results['Breusch-Pagan']['p_value'] < 0.05:
            conclusions.append("- Há evidências de heterocedasticidade, indicando variância não constante.")
        else:
            conclusions.append("- Não há evidências de heterocedasticidade, indicando variância constante.")

        if dw_stat < 1.5 or dw_stat > 2.5:
            conclusions.append("- A estatística Durbin-Watson indica evidências de autocorrelação.")
        else:
            conclusions.append("- A estatística Durbin-Watson não indica autocorrelação.")

        # Exibe conclusões
        print("\nConclusões:")
        for conclusion in conclusions:
            print(conclusion)
