import pandas as pd
import numpy as np

class EDAToolkit:
    def __init__(self, df):
        self.df = df

    def generate_summary(self):
        summary = {
            'shape': self.df.shape,
            'missing_values': self.df.isnull().sum().to_dict(),
            'dtypes': self.df.dtypes.astype(str).to_dict(),
            'describe_numeric': self.df.describe().to_dict()
        }
        return summary

    def detect_outliers_iqr(self, column):
        if column not in self.df.columns or not np.issubdtype(self.df[column].dtype, np.number):
            return []
        
        q1 = self.df[column].quantile(0.25)
        q3 = self.df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers = self.df[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)][column].tolist()
        return outliers
