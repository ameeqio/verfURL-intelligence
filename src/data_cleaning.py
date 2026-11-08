from __future__ import annotations
import pandas as pd
from src.config import exclude_features

class DataCleaner:
    def __init__(self, df: pd.DataFrame)-> None:
        self.data = df
        
    def clean(self)-> pd.DataFrame:
        self.data = self.data.drop(columns = exclude_features)
        self.data = self.data.drop_duplicates()
        self.data = self.data.reset_index(drop = True)
        
        return self.data
        