#!/bin/bash

# Define the model type
model_type="xgboost"  

# Make the POST request
curl -X POST http://localhost:5000/train_model \
  -H "Content-Type: application/json" \
  -d "{\"model_type\": \"$model_type\"}"