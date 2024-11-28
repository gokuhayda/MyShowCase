# tsml

`tsml` (Time Series and Machine Learning) is a Python package designed to simplify and accelerate workflows involving time series analysis, feature engineering, and machine learning. This repository organizes tools, scripts, and notebooks to provide modular, reusable, and efficient solutions for data-driven projects.

## 📂 Datasets

The `tsml` package includes integrated datasets for practical experiments and demonstrations. The datasets are stored in the `datasets/` directory.

### 🚴 Bike Sharing Dataset
This dataset is sourced from the **UCI Machine Learning Repository** and contains information about bike-sharing systems, including:
- **`day.csv`**: Aggregated data for daily usage.
- **`hour.csv`**: Hourly data with detailed information.

### 📚 Dataset Features
The Bike Sharing Dataset includes:
- Weather conditions (temperature, humidity, windspeed).
- Temporal information (season, year, month, hour).
- Usage metrics (count of bikes rented).

#### Source:
The dataset was provided by [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset).

### 🔍 Usage
To load and experiment with the datasets:
```python
import pandas as pd

# Load daily dataset
daily_data = pd.read_csv('tsml/datasets/bike_sharing/day.csv')

# Load hourly dataset
hourly_data = pd.read_csv('tsml/datasets/bike_sharing/hour.csv')

print(daily_data.head())
print(hourly_data.head())

## 📌 Key Features
- **Time Series Analysis**: Tools for preprocessing, modeling, and evaluating time series data.
- **Feature Engineering**: Utilities for generating and selecting meaningful features.
- **Model Training**: Pre-configured pipelines for machine learning models with hyperparameter optimization.
- **Extensibility**: A modular structure allows easy integration of new functionalities.

## 📂 Directory Structure

## 🚀 Installation

To install the `tsml` package:

1. Clone the repository:
   ```bash
   git clone https://github.com/gokuhayda/TechShowCase.git
   cd TechShowCase/tsml

2. Set up the environment and install dependencies:

python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

3. Install the package:

pip install .

📖 Usage

Time Series Model Training

Train and evaluate time series models using pre-configured pipelines:

from tsml.models.time_series_model_trainer import TSModelTrainer

trainer = TSModelTrainer(
    models_params=models_params,
    df=dataframe,
    feature_cols=['feature1', 'feature2'],
    target_col='target',
    metric='mse',
    n_splits=5
)
trainer.train_models()
print(trainer.performance_table)

Feature Engineering

Generate and transform features with feature_engineering.py:

from tsml.feature_engineering.feature_engineering import FeatureEngineering

fe = FeatureEngineering()
transformed_data = fe.fit_transform(data)

Feature Selection

Evaluate and select the best features:

from tsml.feature_selection.evaluate_feature_selection import evaluate_feature_selection

tab, best_features, _ = evaluate_feature_selection(dataframe, model, target_col='target')
print(best_features)

Jupyter Notebooks

Explore the notebooks/ directory for hands-on examples:

curso_forecast_ml_series.ipynb: Forecasting walkthrough.

curso_forecast_feature_engineering.ipynb: Feature engineering tutorial.

📋 License

This project is licensed under the MIT License. See the LICENSE file for details.

🤝 Contributions

Contributions are welcome! To contribute:

1. Fork this repository.


2. Create a feature branch (git checkout -b feature/new-feature).


3. Commit your changes (git commit -m "Add new feature").


4. Push to the branch (git push origin feature/new-feature).


5. Open a pull request.



📫 Connect with Me

Find me on LinkedIn or GitHub for questions, collaboration, or feedback.
 
