from __future__ import annotations
import pandas as pd
from pathlib import Path

class DataLoader:
    def __init__(self, file_path: Path)-> None:
        self.file_path = file_path
        
    def load_data(self)-> pd.DataFrame:
        if self.file_path.exists():
            return pd.read_csv(self.file_path)
        raise FileNotFoundError(f"specified file path does not exist : {self.file_path}")