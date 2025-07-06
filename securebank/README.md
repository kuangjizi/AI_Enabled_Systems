# Quick Start
Instructions to start the server (using Docker) and the curl command to predict if the transaction is fraudulent. 

## Follow these steps to start the server with Docker and make predictions with test data:

### **Step 1: Go to the Executables Directory**
Ensure you are at the directory `securebank/executables`. 

``` 
cd securebank/executables
```

---

### **Step 2: Docker Deployment**
Run the following command to deploy the fraud prediction system to Docker:

``` 
sh run_server.sh
```


It would take serveral minutes to finish building, with output information like below.
```
Building Docker image: securebank...
[+] Building 45.5s (9/9) FINISHED                                                                                          ...                                                                                                 
Docker image 'securebank' built successfully.
Running Docker container on port 5000...
ce097b7149df595419cb410160edd884267325bb7adad9ed16b3fc7616b2d1ac
```
---

### **Step 3: Create Datasets**
Run the `create_dataset.sh` file to create training and testing datasets

```bash
sh create_dataset.sh
```
Example Output:
```
"message": "Datasets created successfully. Saved to: /app/storage/datasets"
```

---

### **Step 4: Train the model**
Run the `train_model.sh` file to train model with produced training sets
```
sh train_model.sh
```
Example Output:
```
{
  "message": "Model trained successfully. Saved to: /app/storage/models/xgboost_e012b1a2-8957-46fc-85cb-ff17907dcae1.pkl"
}
```
---

### **Step 5: Make Prediction**
Run the `predict.sh` file to make a prediction using test.json file. 
```
sh predict.sh
```
Example Output:
```
{
  "message": "Prediction successful",
  "prediction": [
    0
  ]
}

```
Here, 0 is given as a False fraud case for the current transaction. 


### [OPTIONAL] **Step 6: Evaluate the model**
Run the `predict.sh` file to evaluate the model using generated test datasets.
```
sh evaluate_model.sh
```
Example Output:
```
{
  "message": "Evaluation successful",
  "performance": {
    "f1": 0.8461842625543693,
    "precision": 0.9570661896243292,
    "recall": 0.7583274273564847,
    "roc_auc": 0.9989119539867208
  }
}
```