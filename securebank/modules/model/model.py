
import random
import pandas as pd
import numpy as np
import json
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


class Model():
    def __init__(self, model_type="XGBoost"): # "ensemble model" as default option
        self.model_type = model_type.lower()

        if self.model_type == 'random':
            self.model = None

        # model hyperparams are defined from the best performance in hyperparameter tuning
        elif self.model_type == 'xgboost':
            self.model =  xgb.XGBClassifier(
            # --- Tuned parameters as in the notebook ---
            max_depth=7,
            min_child_weight=1,
            gamma=0.4,
            subsample=0.8,
            colsample_bytree=1.0,
            reg_alpha=0.5,
            reg_lambda=1.0,
            learning_rate=0.02,
            n_estimators=1070,  
            early_stopping_rounds=50,

            # --- Other fixed parameters ---
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42
        )

        else:
            raise ValueError("Invalid model_type. Please choose from 'XGBoost' and 'random'")

    def fit(self, X_train, y_train, X_test, y_test):
        if self.model_type == "xgboost":
           self.model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)], # required for early stopping
                verbose=False
            )
           
    
    def predict(self, input_data):
        if self.model_type == "random":
            return random.randint(0, 1)
        else:
            return self.model.predict(input_data)
        
    def predict_proba(self, input_data):
        if self.model_type == "random":
            return None
        else:
            return self.model.predict_proba(input_data)

    def evaluate(self, X_test, y_test):
        if self.model_type == "random":
            roc_auc = None
        else:
            y_proba = self.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)

        y_pred = self.predict(X_test)
        
        results = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc,
        }

        return results
    
    def get_model_params(self):
        model_params = {'model_type': self.model_type}

        if self.model_type != "random":
            raw_params = self.model.get_params(deep=True) 
            for k, v in raw_params.items():
                try:
                    json.dumps(v)  # test if v is JSON-serializable
                    model_params[k] = v
                except (TypeError, OverflowError):
                    model_params[k] = str(v)  # fallback: stringify

        return model_params

    def save_model(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        
        
if __name__ == "__main__":
    random_model = Model(model_type="random")
    test_df = pd.read_json('securebank/data_sources/test.json', typ='series')
    print(test_df)
    print(random_model.predict(test_df))
