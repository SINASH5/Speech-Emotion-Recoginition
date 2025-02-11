import joblib
from utils.feature_extraction import extract_feature
import os

# Load the trained model
model = joblib.load("models/speech_emotion_model.pkl")

# Predict emotion from a new audio file
def predict_emotion(file_path):
    feature = extract_feature(file_path)
    if feature is None:
        return "Error: Invalid audio file or file could not be processed"
    
    # Ensure feature is a 2D array
    feature = feature.reshape(1, -1)
    
    emotion = model.predict(feature)[0]
    emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
    return emotions[emotion]

# Test with a sample file
file_path = r"C:\Users\ASHISH\OneDrive\Desktop\SpeechEmotionRecognition\data\Actor_02\03-01-02-01-01-01-02.wav"  # Use raw string
if os.path.exists(file_path):
    print(f"Predicted Emotion: {predict_emotion(file_path)}")
else:
    print("File not found!")