import librosa
import numpy as np
import soundfile as sf

def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    try:
        with sf.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate = sound_file.samplerate

            # Skip files that are too short
            if len(X) < 2048:  # Minimum length for n_fft=2048
                print(f"Skipping {file_name}: File too short")
                return None

            result = np.array([])

            if mfcc:
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                result = np.hstack((result, mfccs))

            if chroma:
                stft = np.abs(librosa.stft(X))
                chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
                result = np.hstack((result, chroma))

            if mel:
                mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
                result = np.hstack((result, mel))

            return result

    except Exception as e:
        print(f"Error processing {file_name}: {e}")
        return None