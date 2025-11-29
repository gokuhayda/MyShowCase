"""
Download Wine Quality dataset from UCI
"""
import pandas as pd
import os

def download_wine_data():
    """Download and save wine quality dataset."""
    
    # URL do dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    
    # Download
    print("📥 Downloading dataset...")
    df = pd.read_csv(url, sep=';')
    
    # Save
    output_path = "data/raw/wine_quality.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"✅ Dataset saved to {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    
    return df

if __name__ == "__main__":
    df = download_wine_data()
    print("\n📊 First rows:")
    print(df.head())