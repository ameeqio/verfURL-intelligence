from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from src.data_preprocess import DataPreProcess
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class Models:
    def __init__(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame)-> None:
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.pipeline: Pipeline | None = None
        self.basic_models = {
            "logistic_regression": LogisticRegression(random_state = 42),
            "knn": KNeighborsClassifier(n_neighbors = 5),
            "naive_bayes": GaussianNB(),
            "decision-trees": DecisionTreeClassifier(criterion = "entropy", random_state = 42),
            "svc": SVC(random_state = 42),
            "rf-classifier": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=20, min_samples_leaf=10, max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1),
            "xgb-classifier": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, min_child_weight=5,subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=5, random_state=42, n_jobs=-1,eval_metric='logloss')
        }
        
    def train(self)-> dict[str, dict[str, float]]:
        model_report = {}
        for model_name, model_instance in self.basic_models.items():
            self.pipeline = DataPreProcess(
                model = model_instance
            ).build_pipeline()
            
            self.pipeline.fit(self.X_train, self.y_train)
            
            y_preds: np.array = self.pipeline.predict(self.X_test)
            
            precision: float = round(precision_score(self.y_test, y_preds, pos_label = 0), 4)
            recall: float = round(recall_score(self.y_test, y_preds, pos_label = 0), 4)
            f1score: float = round(f1_score(self.y_test, y_preds, pos_label = 0), 4)
            accuracy: float = round(accuracy_score(self.y_test, y_preds), 4)
            
            model_report[model_name] = {
                "precision": precision,
                "recall": recall,
                "f1-score": f1score,
                "accuracy": accuracy
            }
            
            y_train_preds = self.pipeline.predict(self.X_train)
            y_test_preds = self.pipeline.predict(self.X_test)

            print(f"\nMODEL {model_name} - \n")
            print("Train F1:", f1_score(
                self.y_train, y_train_preds, pos_label=0
            ))

            print("Test F1:", f1_score(
                self.y_test, y_test_preds, pos_label=0
            ))

            print("Train Accuracy:", accuracy_score(
                self.y_train, y_train_preds
            ))

            print("Test Accuracy:", accuracy_score(
                self.y_test, y_test_preds
            ))
            
        return model_report
            
        