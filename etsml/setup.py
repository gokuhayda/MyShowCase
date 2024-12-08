from setuptools import setup, find_packages

requirements = [
    "pandas==1.5.3",
    "numpy>=1.25.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "scikit-learn>=1.3.0",
    "statsmodels>=0.14.0",
    "pmdarima>=2.0.3",
    "catboost>=1.2",
    "xgboost>=1.7.5",
    "lightgbm>=3.3.5",
    "scipy>=1.11.3",
    "joblib>=1.3.1",
    "psutil>=5.9.6",
    "jupyter>=1.0.0",
    "ipykernel>=6.24.0",
]

setup(
    name="tsml",
    version="0.1.0",
    author="Éric Gustavo Reis de Sena",
    author_email="egsena@gmail.com",
    description="Package for feature engineering and machine learning..",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/gokuhayda/tsml",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)

