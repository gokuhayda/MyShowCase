from setuptools import setup, find_packages

requirements = [
    "numpy>=1.25.0",
    "matplotlib>=3.8.0",
    "pandas>=1.5.3",
    "scikit-learn>=1.3.0",
    "jupyter>=1.0.0", 
    "statsmodels==0.14.0",  
    "pmdarima==2.0.4" 
]

setup(
    name="tsSarimax",
    version="1.0.0",
    author="Éric Gustavo Reis de Sena",
    author_email="egsena@gmail.com",
    description="Pacote para previsão de séries temporais usando ARIMA e SARIMAX.",
    long_description="",
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
