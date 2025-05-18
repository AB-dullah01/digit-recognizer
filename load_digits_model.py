import pickle
import numpy as np
from sklearn.datasets import load_digits

# Load the saved model
try:
    with open('digits_model.pkl', 'rb') as file:
        model = pickle.load(file)
    print("Model loaded successfully!")
    
    # Print model information
    print("\nModel type:", type(model).__name__)
    
    # If it's a sklearn model, print more details
    if hasattr(model, 'get_params'):
        print("\nModel parameters:")
        for param, value in model.get_params().items():
            print(f"{param}: {value}")
    
    # Load a sample digit for prediction
    digits = load_digits()
    sample_digit = digits.data[0].reshape(1, -1)  # Take first digit as example
    
    # Make a prediction
    prediction = model.predict(sample_digit)
    print(f"\nSample prediction for first digit: {prediction[0]}")
    
    # If the model has predict_proba, show probabilities
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(sample_digit)
        print("\nPrediction probabilities for all digits:")
        for digit, prob in enumerate(probabilities[0]):
            print(f"Digit {digit}: {prob:.4f}")

except Exception as e:
    print(f"Error loading or using the model: {str(e)}") 