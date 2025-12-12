import gradio as gr
import re
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

# Load pre-trained models and tokenizer (they're in root directory)
print("Loading models...")
try:
    loaded_lr = joblib.load('models/logreg_tfidf.pkl')
    loaded_svm = joblib.load('models/svm_tfidf.pkl')
    loaded_lstm = load_model('models/lstm_model.keras', compile=False)
    tokenizer = joblib.load('models/tokenizer.pkl')
    print("All models loaded successfully from /models folder!")
    
    # Verify models work
    test_text = "space mission"
    test_clean = re.sub(r'[^a-z\s]', ' ', test_text.lower()).strip()
    test_pred = loaded_lr.predict_proba([test_clean])[0]
    print(f"Test prediction successful: {test_pred}")
    
except Exception as e:
    print(f"Error loading models: {e}")
    print("Available files:", os.listdir('.'))

# Constants (should match training)
MAX_LEN = 200

def clean_text(text):
    """Clean input text"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_text(text, model_choice='LogisticRegression_TFIDF'):
    """Predict using selected model"""
    if not text or len(text.strip()) == 0:
        return "Please enter text", 0.5, model_choice
    
    try:
        text_clean = clean_text(text)
        
        if model_choice == 'LogisticRegression_TFIDF':
            proba = loaded_lr.predict_proba([text_clean])[:,1][0]
            label = 'Space' if proba > 0.5 else 'Medical'
            confidence = proba if label == 'Space' else 1 - proba
            return label, float(confidence), f"{model_choice} (Space prob: {proba:.3f})"
            
        elif model_choice == 'SVM_TFIDF':
            proba = loaded_svm.predict_proba([text_clean])[:,1][0]
            label = 'Space' if proba > 0.5 else 'Medical'
            confidence = proba if label == 'Space' else 1 - proba
            return label, float(confidence), f"{model_choice} (Space prob: {proba:.3f})"
            
        else:  # LSTM
            seq = pad_sequences(tokenizer.texts_to_sequences([text_clean]), 
                               maxlen=MAX_LEN, padding='post', truncating='post')
            proba = loaded_lstm.predict(seq, verbose=0)[0][0]
            label = 'Space' if proba > 0.5 else 'Medical'
            confidence = proba if label == 'Space' else 1 - proba
            return label, float(confidence), f"{model_choice} (Space prob: {proba:.3f})"
            
    except Exception as e:
        return f"Error: {str(e)}", 0.5, model_choice

# Create Gradio interface - REMOVED 'theme' parameter to fix the error
iface = gr.Interface(
    fn=predict_text,
    inputs=[
        gr.Textbox(
            lines=4, 
            placeholder="Enter your text here...\nExample: 'NASA launches new mission to Mars' or 'New study shows breakthrough in cancer treatment'",
            label="Text to Classify"
        ),
        gr.Dropdown(
            ['LogisticRegression_TFIDF', 'SVM_TFIDF', 'LSTM'], 
            value='LogisticRegression_TFIDF',
            label="Select Model"
        )
    ],
    outputs=[
        gr.Textbox(label="Predicted Category"),
        gr.Number(label="Confidence Score"),
        gr.Textbox(label="Model Info")
    ],
    title="🔬 Medical vs Space News Classifier",
    description="""
    Classify text as either **Medical** or **Space** related content.
    Choose from three different models:
    - **LogisticRegression_TFIDF**: Fast, traditional ML model
    - **SVM_TFIDF**: Support Vector Machine with TF-IDF
    - **LSTM**: Deep learning model with embeddings
    """,
    examples=[
        ["NASA launches new telescope to study distant galaxies"],
        ["Clinical trial shows promising results for new cancer treatment"],
        ["Astronauts conduct spacewalk to repair ISS solar panel"],
        ["New study reveals benefits of exercise for heart health"],
        ["Mars rover discovers evidence of ancient water"],
        ["Breakthrough in mRNA vaccine technology"]
    ]
    # Removed: theme=gr.themes.Soft() - this was causing the error
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
