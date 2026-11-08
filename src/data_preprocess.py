from __future__ import annotations
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.config import numerical_cols

class DataPreProcess:
    def __init__(self, model)-> None:
        self.model_instance = model
        self.preprocessor: ColumnTransformer | None = None
        self.pipeline: Pipeline | None = None
        
    def build_pipeline(self)-> Pipeline:
        self.preprocessor = ColumnTransformer(
            transformers = [
                ("scaling", StandardScaler(), numerical_cols)
            ],
            remainder = "passthrough"
        )
        
        self.pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("model", self.model_instance)
        ])
        
        return self.pipeline