# Digit Recognition Web Application

A web-based digit recognition application that uses machine learning to identify handwritten digits. The application provides real-time prediction, image upload capabilities, and model evaluation features.

## Features

- Interactive drawing canvas for digit input
- Real-time digit prediction
- Image upload support
- Model evaluation dashboard
- Dark/light theme support
- Prediction gallery
- Drawing tools (pen, eraser, brush size, grid)

## Tech Stack

- Backend: Flask
- Frontend: HTML, CSS, JavaScript
- Machine Learning: scikit-learn
- Image Processing: OpenCV, Pillow
- Data Handling: NumPy

## Prerequisites

- Python 3.8 or higher
- pip 

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-name>
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Make sure you're in the project directory and your virtual environment is activated

2. Run the Flask application:
```bash
python app.py
```

3. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

### Drawing Mode
- Use the canvas to draw a digit (0-9)
- Adjust brush size using the slider
- Toggle grid for better alignment
- Use the eraser tool to correct mistakes
- Enable real-time prediction for instant feedback

### Upload Mode
- Upload an image containing a handwritten digit
- View both original and processed images
- See prediction results and confidence scores

### Evaluation Mode
- View model performance metrics
- Examine confusion matrix
- Check per-digit accuracy
- Review misclassified examples

## Project Structure

```
.
├── app.py                 # Flask application
├── digits_model.pkl       # Trained ML model
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Main application template
└── README.md             # Project documentation
```

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 
