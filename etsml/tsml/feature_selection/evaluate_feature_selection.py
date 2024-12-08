# tsml/feature_selection/evaluate_feature_selection.py
from tsml.models.time_series_model_trainer import TSModelTrainer
from tsml.utils.fix_params import fix_params
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

def evaluate_feature_selection(df, model, target_col):
    tab = {'n_features': [], 'score': [], 'model': [], 'features': []}
    
    for nft in range(1, len(df.columns)):
        print(f'Number of features: {nft}')
        # try:
        models_params, feature_cols, _, n_features_to_select = fix_params(
            df,
            target_col=target_col,
            n_features_to_select=nft,
            keeping_models=model.best_model_name.lower(),
            new_params={model.best_model_name.lower(): model.best_params}
        )
        test_model = TSModelTrainer(
            models_params,
            df,
            feature_cols,
            target_col=target_col,
            metric='mse',
            n_splits=3,
            n_features_to_select=n_features_to_select
        )
        test_model.train_models()
        
        tab['score'].append(test_model.best_score)
        tab['n_features'].append(nft)
        tab['model'].append(test_model.best_model_name)
        tab['features'].append(test_model.selected_features)

        try:
            test_model.plot_feature_importance()
        except AttributeError:
            print("Feature importance plot is not available for this model type or configuration.")
        # except Exception as e:
        #     print(f"Error occurred at {nft} features: {str(e)}")
        #     continue

    tab = pd.DataFrame(tab)
    
    if tab.empty or tab['score'].isnull().all():
        print("No valid results were obtained during feature selection.")
        return pd.DataFrame(), [], []

    plt.figure(figsize=(10, 6))

    models = tab['model'].unique()
    for model in models:
        model_data = tab[tab['model'] == model]
        sns.lineplot(data=model_data, x='n_features', y='score', label=model, estimator='mean')

    if not tab['score'].isnull().all():
        min_score_row = tab.loc[tab['score'].idxmin()]
        plt.plot(min_score_row['n_features'], min_score_row['score'], 'ro')
        plt.text(
            min_score_row['n_features'], 
            min_score_row['score'], 
            f"Features: {min_score_row['features']}", 
            verticalalignment='bottom', 
            horizontalalignment='right'
        )

    plt.title('Score vs. Number of Features by Model')
    plt.xlabel('Number of Features')
    plt.ylabel('Score')
    plt.legend(title='Model')
    plt.show()

    first_best_features = list(min_score_row['features']) if 'features' in min_score_row else []

    second_best_row = tab[
        (tab['n_features'] < len(first_best_features)) & 
        (tab['score'] > tab['score'].min())
    ].nsmallest(1, 'score')

    if not second_best_row.empty:
        second_best_features = list(second_best_row.iloc[0]['features'])
    else:
        second_best_features = []

    return tab, first_best_features, second_best_features
