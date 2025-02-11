# Speech Emotion Recognition (SER) using RAVDESS Dataset

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Dataset](https://img.shields.io/badge/Dataset-RAVDESS-orange)

A Python-based project to recognize emotions from speech using the **RAVDESS dataset**. This system extracts audio features (MFCC, Chroma, Mel Spectrogram) and trains a machine learning model to classify emotions like **calm, happy, sad, angry**, and more.

## Features
- Emotion classification from speech.
- Uses Librosa for audio feature extraction.
- Trains a Random Forest classifier (accuracy ~75-80%).
- Supports real-time prediction (via PyAudio).

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/SpeechEmotionRecognition.git
   cd SpeechEmotionRecognition


   Install dependencies:

bash
Copy
pip install -r requirements.txt
(Or manually install: pip install librosa soundfile numpy scikit-learn pyaudio)

Download the RAVDESS dataset:

Visit RAVDESS Dataset.

Download and extract the Audio_Speech_Actors_01-24.zip into the data/ folder.

Usage
1. Train the Model
bash
Copy
python train_model.py
This saves the trained model to models/speech_emotion_model.pkl.

2. Predict Emotion from Audio
bash
Copy
python predict.py --file_path "data/Actor_01/03-01-01-01-01-01-01.wav"
Example output:

Copy
Predicted Emotion: calm
3. Real-Time Prediction (Optional)
Run the live emotion detection script (requires microphone):

bash
Copy
python live_prediction.py
Project Structure
Copy
SpeechEmotionRecognition/
├── data/                   # RAVDESS dataset (not included in repo)
├── models/                 # Saved ML models
├── utils/                  # Utility scripts
│   ├── feature_extraction.py
│   └── load_data.py
├── train_model.py          # Train the emotion classifier
├── predict.py              # Predict emotion from a file
├── live_prediction.py      # Real-time emotion detection
├── requirements.txt        # Dependencies
└── README.md
Contributing
Contributions are welcome! Here’s how:

Fork the repository.

Create a new branch: git checkout -b feature/your-feature.

Commit changes: git commit -m "Add your feature".

Push to the branch: git push origin feature/your-feature.

Open a pull request.

License
This project is licensed under the MIT License. See LICENSE for details.