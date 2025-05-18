from flask import Flask, render_template, request, jsonify, send_file
import joblib
import numpy as np
import cv2
import base64
from PIL import Image
import io
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import os
import tempfile

app = Flask(__name__)

# Load the model
model = joblib.load('digits_model.pkl')

def preprocess_custom_image(image_array):
    # Convert to grayscale if needed
    if len(image_array.shape) == 3:
        image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        image = image_array
    
    # Threshold the image to get binary image
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Invert if background is black
    if np.mean(image[0, 0]) < 127:  # Check corner pixel
        image = 255 - image
    
    # Find bounding box of digit
    coords = cv2.findNonZero(255 - image)  # Invert for findNonZero
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Add padding
        pad = int(max(w, h) * 0.2)  # 20% padding
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(image.shape[1] - x, w + 2*pad)
        h = min(image.shape[0] - y, h + 2*pad)
        # Crop to bounding box
        image = image[y:y+h, x:x+w]
    
    # Make square by padding the shorter dimension
    height, width = image.shape
    if width > height:
        diff = width - height
        pad_top = diff // 2
        pad_bottom = diff - pad_top
        image = cv2.copyMakeBorder(image, pad_top, pad_bottom, 0, 0, 
                                 cv2.BORDER_CONSTANT, value=255)
    elif height > width:
        diff = height - width
        pad_left = diff // 2
        pad_right = diff - pad_left
        image = cv2.copyMakeBorder(image, 0, 0, pad_left, pad_right, 
                                 cv2.BORDER_CONSTANT, value=255)
    
    # Resize to 8x8 (sklearn digits format)
    image = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values to match sklearn digits format (0-16)
    image = (255 - image.astype(np.float64)) / 255.0 * 16
    
    return image.reshape(1, -1)

def preprocess_canvas_image(image_data):
    # Convert base64 image to numpy array
    image = Image.open(io.BytesIO(base64.b64decode(image_data.split(',')[1])))
    image = image.convert('L')  # Convert to grayscale
    image = np.array(image)
    
    # Invert the image (white digits on black background)
    image = 255 - image
    
    # Find bounding box of digit
    coords = cv2.findNonZero(image)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        # Add padding
        pad = 20
        x = max(0, x - pad)
        y = max(0, y - pad)
        w = min(image.shape[1] - x, w + 2*pad)
        h = min(image.shape[0] - y, h + 2*pad)
        # Crop to bounding box
        image = image[y:y+h, x:x+w]
    
    # Resize to 8x8 (sklearn digits format)
    image = cv2.resize(image, (8, 8), interpolation=cv2.INTER_AREA)
    
    # Normalize pixel values to match sklearn digits format (0-16)
    image = image.astype(np.float64) / 255.0 * 16
    
    return image.reshape(1, -1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image data from the request
        image_data = request.json['image']
        
        # Preprocess the image
        processed = preprocess_canvas_image(image_data)
        
        # Make prediction
        prediction = model.predict(processed)[0]
        confidence = None
        
        # Get prediction probability if available
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(processed)[0]
            confidence = float(probs[prediction] * 100)
        
        # Get the processed image for display
        processed_img = processed.reshape(8, 8)
        processed_img = (processed_img * 255 / 16).astype(np.uint8)
        processed_img = cv2.resize(processed_img, (80, 80), interpolation=cv2.INTER_NEAREST)
        
        # Convert processed image to base64
        processed_img_pil = Image.fromarray(processed_img)
        buffered = io.BytesIO()
        processed_img_pil.save(buffered, format="PNG")
        processed_img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'confidence': confidence,
            'processed_image': f'data:image/png;base64,{processed_img_base64}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        # Read image file
        img = Image.open(file.stream)
        img_array = np.array(img)
        
        # Preprocess the image
        processed = preprocess_custom_image(img_array)
        
        # Make prediction
        prediction = model.predict(processed)[0]
        confidence = None
        
        # Get prediction probability if available
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(processed)[0]
            confidence = float(probs[prediction] * 100)
        
        # Get the processed image
        processed_img = processed.reshape(8, 8)
        processed_img = (processed_img * 255 / 16).astype(np.uint8)
        processed_img = cv2.resize(processed_img, (80, 80), interpolation=cv2.INTER_NEAREST)
        
        # Convert processed image to base64
        processed_img_pil = Image.fromarray(processed_img)
        buffered = io.BytesIO()
        processed_img_pil.save(buffered, format="PNG")
        processed_img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Convert original image to base64
        original_buffered = io.BytesIO()
        img.save(original_buffered, format="PNG")
        original_base64 = base64.b64encode(original_buffered.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'confidence': confidence,
            'processed_image': f'data:image/png;base64,{processed_img_base64}',
            'original_image': f'data:image/png;base64,{original_base64}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/evaluate', methods=['GET'])
def evaluate():
    try:
        print("Starting model evaluation...")
        # Load digits dataset
        digits = load_digits()
        X, y = digits.data, digits.target
        print(f"Loaded digits dataset: {X.shape} samples")
        
        # Make predictions on all data
        print("Making predictions...")
        predictions = model.predict(X)
        print("Predictions completed")
        
        # Calculate overall accuracy
        accuracy = (predictions == y).mean() * 100
        print(f"Overall accuracy: {accuracy:.2f}%")
        
        # Calculate per-digit accuracy
        print("Calculating per-digit accuracy...")
        per_digit_accuracy = {}
        for digit in range(10):
            mask = (y == digit)
            if mask.sum() > 0:  # Avoid division by zero
                digit_accuracy = (predictions[mask] == y[mask]).mean() * 100
                per_digit_accuracy[str(digit)] = float(digit_accuracy)
        
        # Get confusion matrix data
        print("Generating confusion matrix...")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y, predictions)
        
        # Create confusion matrix plot
        print("Creating visualization plots...")
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        plt.xticks(np.arange(10))
        plt.yticks(np.arange(10))
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        # Add per-digit accuracy plot
        plt.subplot(1, 2, 2)
        digits_list = list(per_digit_accuracy.keys())
        accuracies = list(per_digit_accuracy.values())
        plt.bar(digits_list, accuracies, color='skyblue')
        plt.title('Per-Digit Accuracy')
        plt.xlabel('Digit')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        
        # Add grid
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        print("Saving plot to buffer...")
        # Convert plot to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100)
        plt.close()
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        
        print("Getting misclassified examples...")
        # Get example misclassifications
        misclassified_examples = {}
        for true_digit in range(10):
            mask = (y == true_digit) & (predictions != true_digit)
            if np.any(mask):
                # Get up to 3 misclassified examples for each digit
                examples = []
                misclassified_indices = np.where(mask)[0][:3]
                for idx in misclassified_indices:
                    # Reshape the data back to 8x8 image
                    img_data = X[idx].reshape(8, 8)
                    # Scale to 0-255 range
                    img_data = (img_data * 255 / 16).astype(np.uint8)
                    # Convert to PIL Image
                    img = Image.fromarray(img_data)
                    # Save to bytes
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    # Convert to base64
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                    
                    example = {
                        'image': img_base64,
                        'predicted': int(predictions[idx]),
                        'shape': (8, 8)
                    }
                    examples.append(example)
                misclassified_examples[str(true_digit)] = examples
        
        print("Preparing response...")
        response = {
            'success': True,
            'accuracy': float(accuracy),
            'per_digit_accuracy': per_digit_accuracy,
            'confusion_matrix': cm.tolist(),
            'plots': f'data:image/png;base64,{plot_base64}',
            'misclassified_examples': misclassified_examples
        }
        print("Evaluation completed successfully")
        return jsonify(response)
        
    except Exception as e:
        import traceback
        error_msg = f"Error in evaluate endpoint: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return jsonify({
            'success': False,
            'error': error_msg
        })

if __name__ == '__main__':
    app.run(debug=True) 