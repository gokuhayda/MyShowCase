import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

    def train_and_predict_arima(self, seasonal=False, stepwise=True, forecast_horizon=100):
        """
        Treina e faz previsões usando o modelo ARIMA puro.
        """
        auto_model = auto_arima(
            self.train_df[self.target_col],
            seasonal=seasonal,
            stepwise=stepwise,
            suppress_warnings=True,
            error_action="ignore",
            trace=True
        )
        order = auto_model.order
        model = ARIMA(self.train_df[self.target_col], order=order)
        model_fit = model.fit()

        train_predictions = model_fit.predict(start=0, end=len(self.train_df)-1)
        test_predictions = model_fit.predict(start=len(self.train_df), end=len(self.df)-1)
        future_predictions = model_fit.predict(start=len(self.df), end=len(self.df)+forecast_horizon)

        self._plot_results(train_predictions, test_predictions, future_predictions, forecast_horizon)

        train_metrics = self._compute_metrics(self.train_df[self.target_col], train_predictions)
        test_metrics = self._compute_metrics(self.test_df[self.target_col], test_predictions)
        
        return train_predictions, test_predictions, future_predictions, model, train_metrics, test_metrics

    def train_and_predict_sarimax(self, feature_cols, seasonal=False, stepwise=True, forecast_horizon=None):
        """
        Treina e faz previsões usando o modelo SARIMAX com variáveis exógenas.
        """
        auto_model = auto_arima(
            self.train_df[self.target_col],
            exogenous=self.train_df[feature_cols],
            seasonal=seasonal,
            stepwise=stepwise,
            suppress_warnings=True,
            error_action="ignore",
            trace=True
        )
        order = auto_model.order
        model = SARIMAX(self.train_df[self.target_col], exog=self.train_df[feature_cols], order=order)
        model_fit = model.fit(disp=False)

        train_predictions = model_fit.predict(start=0, end=len(self.train_df)-1, exog=self.train_df[feature_cols])
        test_predictions = model_fit.predict(start=len(self.train_df), end=len(self.df)-1, exog=self.test_df[feature_cols])

        future_predictions = None
        if forecast_horizon:
            # Gera previsões futuras com variáveis exógenas
            future_exog = self._generate_future_exog(feature_cols, forecast_horizon)
            future_predictions = model_fit.forecast(steps=forecast_horizon, exog=future_exog)

        self._plot_results(train_predictions, test_predictions, future_predictions, forecast_horizon)

        train_metrics = self._compute_metrics(self.train_df[self.target_col], train_predictions)
        test_metrics = self._compute_metrics(self.test_df[self.target_col], test_predictions)
        
        return train_predictions, test_predictions, future_predictions, model, train_metrics, test_metrics

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
