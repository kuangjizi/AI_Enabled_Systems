from flask import Flask, request, jsonify
import numpy as np
import pickle
import time
import uuid
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from modules.data.raw_data_handler import extract, merge_and_clean
from modules.data.feature_engineering import FeatureEngineering
from modules.model.model import Model
from modules.utils.logging import log_request

from global_store import global_state

# === GLOBAL VARIABLES ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUST_INFO_PATH = os.path.join(BASE_DIR, 'data_sources', 'customer_release.csv')
TRANS_INFO_PATH = os.path.join(BASE_DIR, 'data_sources', 'transactions_release.parquet')
FRAUD_INFO_PATH = os.path.join(BASE_DIR, 'data_sources', 'fraud_release.json')

LOG_DIR = os.path.join(BASE_DIR, "logs")
DATA_STORAGE = os.path.join(BASE_DIR, "storage", "datasets")
FEATURE_STORAGE = os.path.join(BASE_DIR, "storage", "feature_pipelines")
MODEL_STORAGE = os.path.join(BASE_DIR, "storage", "models")

# Create the directory if it doesn't exist
for dir in [LOG_DIR, DATA_STORAGE, FEATURE_STORAGE, MODEL_STORAGE]:
    os.makedirs(dir, exist_ok=True) 

app = Flask(__name__)

# Define an endpoint for create_dataset
@app.route('/create_dataset', methods=['GET', 'POST'])
def create_dataset():
    print(">>> create_dataset called") 
    try:
        # Check if file exists
        for path in [CUST_INFO_PATH, TRANS_INFO_PATH, FRAUD_INFO_PATH]:
            if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
                raise FileNotFoundError(f"Required data file {path} is missing or empty. Please ensure the file exists first.")
                
        # Extract the data
        customer_info, transaction_info, fraud_info = extract(CUST_INFO_PATH, TRANS_INFO_PATH, FRAUD_INFO_PATH)
        df = merge_and_clean(customer_info, transaction_info, fraud_info)
        print("Step 1: data loaded")

        # Get the Train and Test Splits
        X, y = df.drop('is_fraud', axis=1), df['is_fraud']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42, stratify=y) 
        print("Step 2: train and test data generated")

        # Transform the features
        fe = FeatureEngineering()
        X_train = fe.fit_transform(X_train) # fit only on training set
        X_test = fe.transform(X_test) # no fit (transform only)

        assert X_train.shape[1] == 41, "Non-expected Features Generated. Double check the `FeatureEngieering` module."
        assert X_test.shape[1] == 41, "Non-expected Features Generated. Double check the `FeatureEngieering` module."
        
        global_state['feature_pipeline_path'] = os.path.join(FEATURE_STORAGE, f"feature_pipeline_{time.strftime('%Y%m%d_%H%M%S')}.pkl")
        fe.save_pipeline(global_state['feature_pipeline_path'])
        print("Step 3: features transformed")


        # Save the data
        X_train, X_test = np.array(X_train), np.array(X_test)
        y_train, y_test = np.array(y_train).ravel().astype(int), np.array(y_test).ravel().astype(int)

        global_state['X_train_path'] = os.path.join(DATA_STORAGE, f"X_train_stratified_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        global_state['y_train_path'] = os.path.join(DATA_STORAGE, f"y_train_stratified_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        global_state['X_test_path'] = os.path.join(DATA_STORAGE, f"X_test_stratified_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        global_state['y_test_path'] = os.path.join(DATA_STORAGE, f"y_test_stratified_{time.strftime('%Y%m%d_%H%M%S')}.txt")

        np.savetxt(global_state['X_train_path'], X_train, delimiter=',')
        np.savetxt(global_state['y_train_path'], y_train, delimiter=',')
        np.savetxt(global_state['X_test_path'], X_test, delimiter=',')
        np.savetxt(global_state['y_test_path'], y_test, delimiter=',')
        print("Step 4: train and test data saved")

        # Output 
        output = {
            "X_train_path": global_state['X_train_path'],
            "X_train_size": X_train.shape,
            "y_train_path": global_state['y_train_path'],
            "y_train_size": y_train.shape,
            "X_test_path": global_state['X_test_path'],
            "X_test_size": X_test.shape,
            "y_test_path": global_state['y_test_path'],
            "y_test_size": y_test.shape,
            "features_names": fe.get_feature_names_out(),
        }

        # Log the request
        log_request(LOG_DIR, '/create_dataset', request, 'success', output)
        return jsonify({"message": f"Datasets created successfully. Saved to: {DATA_STORAGE}"})
    
    except AssertionError as e:
        log_request(LOG_DIR, '/create_dataset', request, 'failed', error=str(e))
        return jsonify(f"Assertion Error caught: {e}"), 400
    
    except FileNotFoundError as e:
        log_request(LOG_DIR, '/create_dataset', request, 'failed', error=str(e))
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        log_request(LOG_DIR, '/create_dataset', request, 'failed', error=str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/train_model', methods=['GET', 'POST'])
def train_model():
    print(">>> train_model called") 
    try:
        # Check if paths exist and files are not empty
        for key in ['X_train_path', 'y_train_path', 'X_test_path', 'y_test_path']:
            path = global_state.get(key)
            if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
                raise FileNotFoundError(f"Required data file {key} is missing or empty. Please execute `/create_dataset` endpoint.")
        
        # Load the data
        X_train = np.loadtxt(global_state['X_train_path'], delimiter=',')
        y_train = np.loadtxt(global_state['y_train_path'], delimiter=',')
        X_test =  np.loadtxt(global_state['X_test_path'], delimiter=',')
        y_test =  np.loadtxt(global_state['y_test_path'], delimiter=',')
        print("Step 1: datasets loaded")

        # Get the model type
        req_json = request.get_json()
        model_type = req_json['model_type'] if req_json and 'model_type' in req_json else "xgboost" # XGBoost model as defaut

        # Instantiate model
        model = Model(model_type=model_type)
        print(f"Step 2: {model_type} model loaded")

        # Train
        train_start = time.time()
        model.fit(X_train, y_train, X_test, y_test)
        train_end = time.time()
        print(f"Step 3: {model_type} model trained")

        # Save the model
        unique_id = str(uuid.uuid4())
        global_state['model_path'] = os.path.join(MODEL_STORAGE, f"{model_type}_{unique_id}.pkl")
        model.save_model(global_state['model_path'])
        print(f"Step 4: {model_type} model saved")

        # Log the request
        model_params = model.get_model_params()
        output = {
            "model_id": unique_id,
            "model_path": global_state['model_path'], 
            "model_params": model_params, 
            "X_train_path": global_state['X_train_path'], 
            "y_train_path": global_state['y_train_path'],
            "training_duration (sec)": round((train_end - train_start), 3),
        }
        log_request(LOG_DIR, '/train_model', request, 'success', output)
        return jsonify({"message": f"Model trained successfully. Saved to: {global_state['model_path']}"})
    
    except FileNotFoundError as e:
        log_request(LOG_DIR, '/train_model', request, 'failed', error=str(e))
        return jsonify({"error": str(e)}), 400
    
    except Exception as e:
        log_request(LOG_DIR, '/train_model', request, 'failed', error=str(e))
        return jsonify({"error": str(e)}), 500

# Define an endpoint for prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print(">>> predict called") 
    try:
        data = request.get_json() # Get JSON file from the request
        if not data:
            raise FileNotFoundError("Missing JSON file from the request.")
        
        if not global_state.get('model_path'):
            raise FileNotFoundError("No existing model. Please create and train model using `/train_model` endpoint.")
        
        if not global_state.get('feature_pipeline_path'):
            raise FileNotFoundError("No existing feature_pipeline. Please create feature pipeline using `/create_dataset` endpoint.")
        
        if not CUST_INFO_PATH or not os.path.exists(CUST_INFO_PATH) or os.path.getsize(CUST_INFO_PATH) == 0:
             raise FileNotFoundError(f"Required data file {CUST_INFO_PATH} is missing or empty. Please ensure the file exists first.") 
        

        trans_info = pd.DataFrame([data]) # Convert single dict to DataFrame
        print("Step 1: file extracted")

        # Extract and merge the data
        cust_info, _, _ = extract(CUST_INFO_PATH)
        X_df = merge_and_clean(cust_info, trans_info)
        print("Step 2: data loaded")
        
        # Get the features
        fe_loaded = pickle.load(open(global_state['feature_pipeline_path'], "rb"))
        X_trans = fe_loaded.transform(X_df) # no fit (transform only)
        assert X_trans.shape[1] == 41, "Non-expected Features Generated. Double check the `FeatureEngieering` module."
        print("Step 3: feature generated")

        # Load the model (default is XGBoost, the best performing one)
        model = pickle.load(open(global_state['model_path'], "rb"))
        print("Step 4: model loaded")

        # Call modules.model.Model.predict(input_data)
        pred_start = time.time()
        prediction = model.predict(X_trans)
        pred_end = time.time()
        print("Step 5: prediction generated")

        # Log the request
        output = {
            "model_path": global_state['model_path'],
            "prediction": prediction.tolist(),
            "model_params": model.get_model_params(), 
            "prediction_duration (sec)": round((pred_end - pred_start), 3),
        }
        log_request(LOG_DIR, '/predict', request, 'success', output)
        return jsonify({"message": "Prediction successful", "prediction": prediction.tolist()})

    except AssertionError as e:
        log_request(LOG_DIR, '/predict', request, 'failed', error=str(e))
        return jsonify(f"Assertion Error caught: {e}"), 400
    
    except FileNotFoundError as e:
        log_request(LOG_DIR, '/predict', request, 'failed', error=str(e))
        return jsonify({"error": str(e)}), 400
    
    except Exception as e:
        log_request(LOG_DIR, '/predict', request, 'failed', error=str(e))
        return jsonify({"error": str(e)}), 500

# Define an endpoint for evaluation
@app.route('/evaluate_model', methods=['GET', 'POST'])
def evaluate_model():
    print(">>> evaluate_model called") 
    try:
        if not global_state.get('model_path'):
            raise FileNotFoundError("No existing model. Please create and train model using `/train_model` endpoint.")

        if not global_state.get('X_test_path') or not global_state.get('y_test_path'):
            raise FileNotFoundError("No test data. Please create datasets using `/create_dataset` endpoint.")
        
        # Load the model (default is XGBoost, the best performing one)
        model = pickle.load(open(global_state['model_path'], "rb"))
        print("Step 1: model loaded")

        # Load the test data
        X_test = np.loadtxt(global_state['X_test_path'], delimiter=',')
        y_test = np.loadtxt(global_state['y_test_path'], delimiter=',')
        print("Step 2: test data loaded")

        # Evaluation
        eval_start = time.time()
        performance = model.evaluate(X_test, y_test)
        eval_end = time.time()
        print("Step 3: performance measured")

        # Log the request
        output = {
            "model_path": global_state['model_path'], 
            "perfomrance": performance,
            "model_params": model.get_model_params(), 
            "X_test_path": global_state['X_test_path'], 
            "y_test_path": global_state['y_test_path'],
            "evaluation_duration (sec)": round((eval_end - eval_start), 3),
        }

        log_request(LOG_DIR, '/evaluate_model', request, 'success', output)
        return jsonify({"message": "Evaluation successful", "performance": performance})

    except FileNotFoundError as e:
        log_request(LOG_DIR, '/evaluate_model', request, 'failed', error=str(e))
        return jsonify({"error": str(e)}), 400
    
    except Exception as e:
        log_request(LOG_DIR, '/evaluate_model', request, 'failed', error=str(e))
        return jsonify({"error": str(e)}), 500

# Test the API
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)