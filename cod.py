from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier  # Importing Random Forest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import os

app = Flask(__name__)

# Global variables
# Setting up the vectorizer with a limit of 5000 features for efficiency
vectorizer = TfidfVectorizer(max_features=5000)

# Random Forest model initialized with 100 decision trees
model = RandomForestClassifier(n_estimators=100, random_state=42) 
model_ready = False

def train_model():
    global model_ready
    try:
        # Define local file paths for the datasets
        fake_path = r"C:\Users\USER\Desktop\Project\Data\Fake.csv"
        true_path = r"C:\Users\USER\Desktop\Project\Data\True.csv"
        
        # Load the CSV files into DataFrames
        fake = pd.read_csv(fake_path)
        true = pd.read_csv(true_path)
        
        # Assign labels: 0 for Fake News and 1 for Real News
        fake["label"] = 0
        true["label"] = 1
        
        # Combine datasets and shuffle the rows randomly
        data = pd.concat([fake, true]).sample(frac=1).reset_index(drop=True)
        
        # Split the data into features (X) and target labels (y)
        X = data["text"]
        y = data["label"]
        
        # Split into training (80%) and testing (20%) sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Vectorization: Convert text data into numerical TF-IDF features
        X_train_vect = vectorizer.fit_transform(X_train)
        X_test_vect = vectorizer.transform(X_test)
        
        # Random Forest Model Training
        print("Training Random Forest... this may take a few moments.")
        model.fit(X_train_vect, y_train)
        
        # Model Evaluation: Calculate accuracy and generate classification report
        predictions = model.predict(X_test_vect)
        acc = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        
        print("\n" + "="*35)
        print(f"RANDOM FOREST ACCURACY: {acc * 100:.2f}%")
        print("CLASSIFICATION REPORT:")
        print(report)
        print("="*35 + "\n")
        
        model_ready = True
        return True
    except Exception as e:
        print(f"ERROR DURING TRAINING: {e}")
        return False

# Trigger model training on startup
train_model()

@app.route('/')
def home():
    """Renders the main dashboard page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles news classification requests from the web form."""
    if not model_ready:
        return "Error: The model is not trained or ready yet."
    
    message = request.form.get('news')
    if not message:
        return render_template('index.html', prediction_text='Error: Please enter some text to analyze.')

    # Transform input text and perform prediction
    vect = vectorizer.transform([message])
    prediction = model.predict(vect)
    
    # Map numerical prediction back to string label
    result = "Real News" if prediction[0] == 1 else "Fake News"
    
    return render_template('index.html', prediction_text=f'Result: {result}')

if __name__ == "__main__":
    # Run the Flask development server
    app.run(debug=True)