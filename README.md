# 🗣️ Speech Emotion Recognition System

An end-to-end system that analyzes human speech to recognize and classify emotions using a 1D Convolutional Neural Network (CNN) trained on the CREMA-D dataset.

## 🌟 Features

- **Audio File Analysis**: Upload `.wav` files to detect the emotion of the speaker.
- **Live Recording**: Record audio directly from your microphone (3-second clips) for real-time emotion recognition.
- **Confidence Scoring**: Get a precise confidence percentage for the detected emotion.
- **Probability Visualization**: View a detailed bar chart showing the probability distribution across all emotion classes.

## 🛠️ Technical Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Deep Learning**: [TensorFlow / Keras](https://www.tensorflow.org/)
- **Audio Processing**: [Librosa](https://librosa.org/), [SciPy](https://scipy.org/)
- **Audio Capture**: [SoundDevice](https://python-sounddevice.readthedocs.io/)
- **Dataset**: CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)

## 📂 Project Structure

```text
.
├── app.py                     # Streamlit application for inference
├── Train_Model.ipynb          # Jupyter Notebook for model training & evaluation
├── crema_emotion_model.h5      # Pre-trained 1D CNN model
└── crema_classes.npy           # Emotion class labels
```

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed on your system.

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Algonive_SPEECH-EMOTION-RECOGNITION-SYSTEM
   ```

2. **Install dependencies:**
   ```bash
   pip install streamlit numpy pandas librosa tensorflow scipy sounddevice
   ```
   *Note: On some systems, `sounddevice` may require additional system-level drivers (e.g., PortAudio).*

### Running the Application

To launch the web interface, run the following command in your terminal:

```bash
streamlit run app.py
```

Once the app starts, you can navigate to the provided URL (usually `http://localhost:8501`) in your browser.

## 🧠 How it Works

1. **Feature Extraction**: The system extracts **Mel-frequency cepstral coefficients (MFCCs)** from the audio input. MFCCs represent the short-term power spectrum of sound and are highly effective for speech recognition tasks.
2. **Preprocessing**: Audio is standardized to a 3-second duration and a sample rate of 22,050 Hz.
3. **Inference**: The processed MFCC features are fed into a **1D CNN model**, which outputs the probability of each emotion class.
4. **Classification**: The class with the highest probability is selected as the detected emotion.

## 📈 Model Training

If you wish to retrain the model or experiment with different architectures:
1. Open `Train_Model.ipynb` in Jupyter Lab or Google Colab.
2. Follow the cells to load the CREMA-D dataset, preprocess the audio, and train the CNN.
3. The notebook will export `crema_emotion_model.h5` and `crema_classes.npy`, which are required for `app.py` to function.
