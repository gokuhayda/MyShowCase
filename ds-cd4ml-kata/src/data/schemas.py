"""
Data validation schemas using Pandera
"""
import pandera as pa
from pandera import Column, Check

# Schema para dados RAW
raw_schema = pa.DataFrameSchema({
    "fixed acidity": Column(float, Check.in_range(4, 16)),
    "volatile acidity": Column(float, Check.in_range(0.1, 2)),
    "citric acid": Column(float, Check.in_range(0, 1)),
    "residual sugar": Column(float, Check.in_range(0.5, 16)),
    "chlorides": Column(float, Check.in_range(0, 1)),
    "free sulfur dioxide": Column(float, Check.in_range(1, 100)),
    "total sulfur dioxide": Column(float, Check.in_range(6, 300)),
    "density": Column(float, Check.in_range(0.99, 1.01)),
    "pH": Column(float, Check.in_range(2.5, 4.5)),
    "sulphates": Column(float, Check.in_range(0.3, 2)),
    "alcohol": Column(float, Check.in_range(8, 15)),
    "quality": Column(int, Check.in_range(3, 8)),
}, strict=True, coerce=True)

# Schema para dados PROCESSADOS  
processed_schema = pa.DataFrameSchema({
    "fixed_acidity": Column(float),
    "volatile_acidity": Column(float),
    "citric_acid": Column(float),
    "residual_sugar": Column(float),
    "chlorides": Column(float),
    "free_sulfur_dioxide": Column(float),
    "total_sulfur_dioxide": Column(float),
    "density": Column(float),
    "pH": Column(float),
    "sulphates": Column(float),
    "alcohol": Column(float),
    "quality_binary": Column(int, Check.isin([0, 1])),
}, strict=True, coerce=True)