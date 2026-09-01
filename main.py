import numpy as np
import pandas as pd
from src.ml_pipeline import AutomatedMLPipeline
from src.eda_toolkit import EDAToolkit

def run_demo():
    print("🚀 Initializing Python Machine Learning & EDA Demonstration...")
    
    # Generate synthetic dataset
    np.random.seed(42)
    data = {
        'age': np.random.randint(18, 65, size=100),
        'income': np.random.normal(50000, 15000, size=100),
        'department': np.random.choice(['Tech', 'Sales', 'HR', 'Finance'], size=100),
        'purchased': np.random.choice([0, 1], size=100)
    }
    df = pd.DataFrame(data)
    
    # 1. Run EDA Summary
    print("\n📊 Running Exploratory Data Analysis (EDA)...")
    eda = EDAToolkit(df)
    summary = eda.generate_summary()
    print(f"Dataset Shape: {summary['shape']}")
    print(f"Missing Values: {summary['missing_values']}")
    
    # 2. Run Automated ML Pipeline
    print("\n🤖 Training Automated Machine Learning Classifier...")
    pipeline = AutomatedMLPipeline(
        numeric_features=['age', 'income'],
        categorical_features=['department'],
        target_col='purchased'
    )
    metrics = pipeline.train_and_evaluate(df)
    print(f"Model Training Accuracy: {metrics['accuracy']}%")
    print("Evaluation Complete!")

if __name__ == "__main__":
    run_demo()
