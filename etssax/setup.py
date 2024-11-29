from setuptools import setup, find_packages

requirements = [
    "numpy>=1.24.4",
    "matplotlib>=3.8.0",
    "pandas>=2.2.0",
    "scikit-learn>=1.5.0",
    "statsmodels>=0.14.0",
    "pmdarima>=2.0.4"
]

setup(
    name="tsSarimax",
    version="1.0.0",
    author="Éric Gustavo Reis de Sena",
    author_email="egsena@gmail.com",
    description="Pacote para previsão de séries temporais usando ARIMA e SARIMAX.",
    long_description="Pacote para previsão de séries temporais com suporte a variáveis exógenas.",
    long_description_content_type="text/markdown",
    url="https://github.com/gokuhayda/arima_predictor",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
