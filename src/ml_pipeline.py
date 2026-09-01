import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

class AutomatedMLPipeline:
    def __init__(self, numeric_features, categorical_features, target_col):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_col = target_col
        self.pipeline = None

    def build_pipeline(self, model_type="random_forest"):
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ]
        )

        if model_type == "random_forest":
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)

        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
        return self.pipeline

    def train_and_evaluate(self, df):
        X = df[self.numeric_features + self.categorical_features]
        y = df[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.pipeline is None:
            self.build_pipeline()

        self.pipeline.fit(X_train, y_train)
        predictions = self.pipeline.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        metrics = {
            'accuracy': round(acc * 100, 2),
            'report': classification_report(y_test, predictions, output_dict=True)
        }
        return metrics
