import joblib
import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from PIL import Image, ImageGrab, ImageDraw, ImageTk
import tkinter as tk
from tkinter import filedialog
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

def load_and_examine_model():
    # Load the model
    print("Loading the model...")
    model = joblib.load('digits_model.pkl')
    print("Model loaded successfully!")
    
    # Print model information
    print("\nModel type:", type(model).__name__)
    
    # Print model parameters
    if hasattr(model, 'get_params'):
        print("\nModel parameters:")
        for param, value in model.get_params().items():
            print(f"{param}: {value}")
    
    return model

def test_on_sample_digits(model, num_samples=5):
    # Load digits dataset
    digits = load_digits()
    
    plt.figure(figsize=(15, 3))
    for i in range(num_samples):
        # Get a sample digit
        sample_digit = digits.data[i].reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(sample_digit)
        
        # Get prediction probabilities if available
        prob_str = ""
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(sample_digit)[0]
            confidence = probs[prediction[0]] * 100
            prob_str = f"(Confidence: {confidence:.1f}%)"
        
        # Plot the digit
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(digits.images[i], cmap='gray')
        plt.title(f'Predicted: {prediction[0]}\n{prob_str}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png')
    print("\nSample predictions have been saved to 'sample_predictions.png'")

def evaluate_model(model):
    # Load full digits dataset
    digits = load_digits()
    
    # Make predictions on all data
    predictions = model.predict(digits.data)
    
    # Calculate accuracy
    accuracy = (predictions == digits.target).mean() * 100
    print(f"\nModel accuracy on full dataset: {accuracy:.2f}%")

def preprocess_custom_image(image):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
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

def preprocess_drawing(image):
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
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

def test_custom_image(model):
    # Open file dialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select digit image",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
    )
    
    if not file_path:
        print("No file selected")
        return
    
    # Load and preprocess image
    image = cv2.imread(file_path)
    if image is None:
        print("Error loading image")
        return
    
    processed = preprocess_custom_image(image)
    
    # Create a window to show results
    result_window = tk.Toplevel()
    result_window.title("Prediction Result")
    
    # Show original image
    orig_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_img = Image.fromarray(orig_img)
    # Resize if too large
    if max(orig_img.size) > 300:
        orig_img.thumbnail((300, 300), Image.Resampling.LANCZOS)
    orig_photo = ImageTk.PhotoImage(orig_img)
    
    orig_label = tk.Label(result_window, text="Original Image")
    orig_label.pack()
    orig_canvas = tk.Canvas(result_window, width=orig_img.width, height=orig_img.height)
    orig_canvas.pack()
    orig_canvas.create_image(orig_img.width//2, orig_img.height//2, image=orig_photo)
    
    # Show processed image
    proc_img = processed.reshape(8, 8)
    proc_img = (proc_img * 255 / 16).astype(np.uint8)
    proc_img = cv2.resize(proc_img, (200, 200), interpolation=cv2.INTER_NEAREST)
    proc_img = Image.fromarray(proc_img)
    proc_photo = ImageTk.PhotoImage(proc_img)
    
    proc_label = tk.Label(result_window, text="Processed Image (8x8)")
    proc_label.pack()
    proc_canvas = tk.Canvas(result_window, width=200, height=200)
    proc_canvas.pack()
    proc_canvas.create_image(100, 100, image=proc_photo)
    
    # Make prediction
    prediction = model.predict(processed)[0]
    confidence = ""
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(processed)[0]
        confidence = f"(Confidence: {probs[prediction]*100:.1f}%)"
    
    # Show prediction
    pred_label = tk.Label(result_window, 
                         text=f"Predicted: {prediction} {confidence}",
                         font=('Arial', 14))
    pred_label.pack(pady=10)
    
    # Keep references to prevent garbage collection
    result_window.orig_photo = orig_photo
    result_window.proc_photo = proc_photo
    
    # Wait for window to be closed
    result_window.mainloop()

class DigitDrawer:
    def __init__(self, model):
        self.model = model
        self.root = tk.Tk()
        self.root.title("Draw Digit")
        
        # Create main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10)
        
        # Drawing canvas
        self.canvas_size = 280
        self.canvas = tk.Canvas(self.main_frame, 
                              width=self.canvas_size, 
                              height=self.canvas_size, 
                              bg='white',  # Changed to white background
                              highlightthickness=1,
                              highlightbackground="gray")
        self.canvas.pack(pady=20)
        
        # Bind mouse events
        self.canvas.bind('<B1-Motion>', self.paint)
        self.canvas.bind('<ButtonRelease-1>', self.predict_digit)
        self.canvas.bind('<Button-1>', self.reset_position)
        
        # Create PIL image for drawing
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 'white')  # White background
        self.draw = ImageDraw.Draw(self.image)
        
        # Buttons frame
        self.button_frame = tk.Frame(self.main_frame)
        self.button_frame.pack(pady=10)
        
        # Clear button
        self.clear_btn = tk.Button(self.button_frame, text="Clear", command=self.clear_canvas)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Preview frame
        self.preview_label = tk.Label(self.main_frame, text="Preprocessed Image:")
        self.preview_label.pack()
        self.preview_canvas = tk.Canvas(self.main_frame, width=80, height=80)
        self.preview_canvas.pack()
        
        # Prediction label
        self.pred_label = tk.Label(self.main_frame, 
                                 text="Draw a digit and release to predict",
                                 font=('Arial', 14))
        self.pred_label.pack(pady=10)
        
        self.last_x = None
        self.last_y = None
    
    def reset_position(self, event):
        self.last_x = None
        self.last_y = None
    
    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            # Draw on canvas
            self.canvas.create_line(self.last_x, self.last_y, x, y,
                                  width=30,  # Increased line width
                                  fill='black',  # Changed to black color
                                  capstyle=tk.ROUND,
                                  smooth=tk.TRUE)
            # Draw on PIL image
            self.draw.line([self.last_x, self.last_y, x, y],
                         fill='black',  # Changed to black color
                         width=30)  # Increased line width
        self.last_x = x
        self.last_y = y
    
    def clear_canvas(self):
        # Clear tkinter canvas
        self.canvas.delete("all")
        # Clear PIL image
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 'white')  # White background
        self.draw = ImageDraw.Draw(self.image)
        self.pred_label.config(text="Draw a digit and release to predict")
        # Clear preview
        self.preview_canvas.delete("all")
    
    def show_preview(self, processed_image):
        # Scale the 8x8 image to 80x80 for preview
        preview = cv2.resize(processed_image.reshape(8, 8), (80, 80), interpolation=cv2.INTER_NEAREST)
        preview_img = Image.fromarray((preview * 255 / 16).astype(np.uint8))
        preview_img = ImageTk.PhotoImage(preview_img)
        
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(40, 40, image=preview_img)
        self.preview_canvas.image = preview_img  # Keep a reference
    
    def predict_digit(self, event):
        try:
            # Convert PIL image to numpy array
            img_array = np.array(self.image)
            
            # Preprocess
            processed = preprocess_drawing(img_array)
            
            # Show preview
            self.show_preview(processed)
            
            # Predict
            prediction = self.model.predict(processed)[0]
            confidence = ""
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(processed)[0]
                confidence = f"(Confidence: {probs[prediction]*100:.1f}%)"
            
            self.pred_label.config(text=f"Predicted: {prediction} {confidence}")
            
        except Exception as e:
            self.pred_label.config(text=f"Error: {str(e)}")
            
        self.last_x = None
        self.last_y = None
    
    def run(self):
        self.root.mainloop()

def interactive_mode(model):
    drawer = DigitDrawer(model)
    drawer.run()

if __name__ == "__main__":
    # Load and examine the model
    model = load_and_examine_model()
    
    while True:
        print("\nChoose an option:")
        print("1. Test on sample digits")
        print("2. Test custom image")
        print("3. Interactive drawing mode")
        print("4. Evaluate model")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '1':
            test_on_sample_digits(model)
        elif choice == '2':
            test_custom_image(model)
        elif choice == '3':
            interactive_mode(model)
        elif choice == '4':
            evaluate_model(model)
        elif choice == '5':
            break
        else:
            print("Invalid choice. Please try again.") 