import warnings
from itertools import combinations
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

def select_features_based_on_vif(df, feature_cols, target, vif_threshold=5.0, correlation_threshold=0.5):
    def safe_variance_inflation_factor(X, i):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return variance_inflation_factor(X, i)
            except ZeroDivisionError:
                return float('inf')

    best_score = 0
    best_combination = []
    avg_corr = 0

    for r in range(2, len(feature_cols) + 1):
        for combo in combinations(feature_cols, r):
            X = df[list(combo)].copy()
            X.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Remove colunas com valores NaN ou Inf
            cols_with_na = X.columns[X.isna().any()].tolist()
            X.drop(cols_with_na, axis=1, inplace=True)
            combo = [x for x in combo if x not in cols_with_na]

            # Adiciona a constante
            X = add_constant(X)

            # Calcula VIF com tratamento de exceções
            vif_data = []
            for i in range(X.shape[1]):
                vif = safe_variance_inflation_factor(X.values, i)
                vif_data.append(vif)
            
            vif_df = pd.DataFrame({
                'feature': X.columns,
                'VIF': vif_data
            }).drop(index=X.columns.get_loc('const'))  # Remove constante do DataFrame de VIF

            # Verifica se todos os VIFs estão abaixo do limite
            if all(vif_df['VIF'] <= vif_threshold):
                avg_corr = df[list(combo)].corrwith(df[target]).abs().mean()
                if avg_corr > correlation_threshold and avg_corr > best_score:
                    best_score = avg_corr
                    best_combination = combo

    if not best_combination:
        # Caso nenhuma combinação atenda aos critérios, escolhe a variável com maior correlação com o alvo
        target_corr = df[feature_cols].corrwith(df[target]).abs()
        best_var = target_corr.idxmax()
        best_combination = [best_var]
        print(f"The variable '{best_combination[0]}' was chosen because it had the highest absolute correlation with the target.")
    else:
        print(f"The variables {best_combination} were chosen because they have a combination of low multicollinearity (VIF <= {vif_threshold}) and average correlation with target equal {round(avg_corr,4)*100}%).")

    return best_combination
