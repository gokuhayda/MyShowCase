"""
ETL Pipeline: Transform raw data to processed features
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))
from schemas import raw_schema, processed_schema

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_raw_data(path: str) -> pd.DataFrame:
    """Load and validate raw data."""
    logger.info(f"📂 Loading data from {path}")
    df = pd.read_csv(path)
    
    # Validate schema
    logger.info("✅ Validating raw data schema...")
    raw_schema.validate(df, lazy=True)
    
    logger.info(f"   Shape: {df.shape}")
    logger.info(f"   Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering."""
    logger.info("🔧 Creating features...")
    
    df_processed = df.copy()
    
    # 1. Limpar nomes de colunas (remover espaços)
    df_processed.columns = df_processed.columns.str.replace(' ', '_')
    
    # 2. Target engineering: qualidade >= 6 = good wine (1), senão bad wine (0)
    df_processed['quality_binary'] = (df_processed['quality'] >= 6).astype(int)
    df_processed = df_processed.drop('quality', axis=1)
    
    # 3. Log transformations (opcional - pode melhorar performance)
    # df_processed['log_sulphates'] = np.log1p(df_processed['sulphates'])
    
    logger.info(f"   Features: {df_processed.shape[1]} columns")
    logger.info(f"   Target distribution:")
    for label, count in df_processed['quality_binary'].value_counts().items():
        pct = count / len(df_processed) * 100
        logger.info(f"      Class {label}: {count} ({pct:.1f}%)")
    
    return df_processed

def save_processed_data(df: pd.DataFrame, path: str):
    """Save and validate processed data."""
    logger.info("💾 Saving processed data...")
    
    # Validate before saving
    logger.info("✅ Validating processed data schema...")
    processed_schema.validate(df, lazy=True)
    
    # Save
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    
    logger.info(f"   Saved to: {path}")
    logger.info(f"   Size: {Path(path).stat().st_size / 1024:.2f} KB")

def main():
    """Run ETL pipeline."""
    logger.info("=" * 60)
    logger.info("🚀 Starting ETL Pipeline")
    logger.info("=" * 60)
    
    # Paths
    raw_path = "data/raw/wine_quality.csv"
    processed_path = "data/processed/wine_features.csv"
    
    # Pipeline
    try:
        df_raw = load_raw_data(raw_path)
        df_processed = create_features(df_raw)
        save_processed_data(df_processed, processed_path)
        
        logger.info("=" * 60)
        logger.info("✅ ETL Pipeline completed successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ ETL Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()