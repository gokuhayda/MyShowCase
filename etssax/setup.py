from setuptools import setup, find_packages

requirements = [
    "anyio==4.6.2.post1",
    "argon2-cffi==23.1.0",
    "beautifulsoup4==4.12.3",
    "Cython==3.0.11",
    "joblib==1.4.2",
    "matplotlib==3.9.2",
    "numpy==1.24.3",
    "pandas==2.2.3",
    "pmdarima==2.0.4",
    "python-dateutil==2.9.0.post0",
    "pytz==2024.2",
    "scikit-learn==1.5.2",
    "scipy==1.13.1",
    "six==1.16.0",
    "statsmodels==0.14.4",
    "threadpoolctl==3.5.0",
    "tzdata==2024.2",
    "urllib3==2.2.3",
]

setup(
    name="tsSarimax",
    version="1.0.6",
    author="Éric Gustavo Reis de Sena",
    author_email="egsena@gmail.com",
    description="Pacote para previsão de séries temporais usando ARIMA e SARIMAX.",
    long_description=(
        "Pacote especializado em previsão de séries temporais com suporte a variáveis exógenas. "
        "Inclui ferramentas para ajuste automático de modelos e métricas de avaliação."
    ),
    long_description_content_type="text/markdown",
    url="https://github.com/gokuhayda/arima_predictor",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8, <3.12",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=23.0",
            "flake8>=6.0",
        ],
    },
)
