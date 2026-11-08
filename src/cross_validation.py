from __future__ import annotations
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_validate
from src.data_preprocess import DataPreProcess
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, make_scorer

class CrossValidate:
    def __init__(self, X: pd.DataFrame, y: pd.Series)-> None:
        self.X = X
        self.y = y
        self.cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
        self.basic_models = {
            "logistic_regression": LogisticRegression(random_state = 42),
            "knn": KNeighborsClassifier(n_neighbors = 5),
            "naive_bayes": GaussianNB(),
            "decision-trees": DecisionTreeClassifier(criterion = "entropy", random_state = 42),
            "svc": SVC(random_state = 42),
            "rf-classifier": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=20, min_samples_leaf=10, max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1),
            "xgb-classifier": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, min_child_weight=5,subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=5, random_state=42, n_jobs=-1,eval_metric='logloss')
        }
        self.pipeline: Pipeline | None = None
        self.scoring = {
            'accuracy': 'accuracy',
            'precision': make_scorer(
                precision_score, pos_label=0
            ),
            'recall': make_scorer(
                recall_score, pos_label=0
            ),
            'f1': make_scorer(
                f1_score, pos_label=0
            )
        }
        
    def cross_validate_models(self)-> pd.DataFrame:
        cv_results = []
        for model_name, model_instance in self.basic_models.items():
            self.pipeline = DataPreProcess(model_instance).build_pipeline()
            
            cv_score = cross_validate(
                estimator = self.pipeline,
                X = self.X,
                y = self.y,
                cv = self.cv,
                scoring = self.scoring, n_jobs = -1
            )
            
            cv_results.append({
                "model": model_name,
                "CV_mean_accuracy" : cv_score["test_accuracy"].mean(),
                "CV_mean_precision" : cv_score["test_precision"].mean(),
                "CV_mean_recall" : cv_score["test_recall"].mean(),
                "CV_mean_f1_score" : cv_score["test_f1"].mean(),
            })
            
        return pd.DataFrame(cv_results)
    
    
    def group_cross_validate(self, train_domains)-> None:
        for model_name, model_instance in self.basic_models.items():
            self.pipeline = DataPreProcess(model_instance).build_pipeline()
                    
            results = cross_validate(
                estimator = self.pipeline,
                X = self.X,
                y = self.y,
                groups = train_domains,
                cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42),
                scoring = self.scoring, n_jobs = -1
            )
                    
            print(f"\n{model_name} - \n")
            for metric in self.scoring:
                scores = results[f'test_{metric}']

                print(
                    f"{metric}: "
                    f"{scores.mean():.5f} ± {scores.std():.5f}"
                )