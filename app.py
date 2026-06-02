import os
import streamlit as st
import numpy as np
import pandas as pd
import librosa
from tensorflow.keras.models import load_model
from scipy.io.wavfile import write

st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")

st.title("🗣️ Speech Emotion Recognition System")
st.write("Analyze emotions in speech using a trained CREMA-D 1D CNN model. Choose to either upload an audio file or record live.")

MODEL_PATH = r'crema_emotion_model.h5'
CLASSES_PATH = r'crema_classes.npy'

if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
    st.error("⚠️ Missing Model Files! Please run your Jupyter Notebook training cells first to generate 'crema_emotion_model.h5' and 'crema_classes.npy'.")
    st.stop()

@st.cache_resource
def load_trained_assets():
    model = load_model(MODEL_PATH)
    classes = np.load(CLASSES_PATH, allow_pickle=True)
    return model, classes

model, classes = load_trained_assets()


SAMPLE_RATE = 22050
DURATION = 3         
MAX_TIME_FRAMES = 128  
TEMP_FILENAME = "processed_input.wav"

def process_and_predict(audio_source):
    audio_data, sr = librosa.load(audio_source, duration=3)
    
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40)
    
    if mfccs.shape[1] < MAX_TIME_FRAMES:
        pad_width = MAX_TIME_FRAMES - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :MAX_TIME_FRAMES]
        
    features = np.expand_dims(mfccs, axis=0)
    
    predictions = model.predict(features, verbose=0)
    predicted_idx = np.argmax(predictions)
    
    return classes[predicted_idx], predictions[0]

tab1, tab2 = st.tabs(["📁 Upload Audio File", "🎙️ Record Live Audio"])

with tab1:
    st.subheader("Upload a local .wav file")
    uploaded_file = st.file_uploader("Choose an audio file...", type=["wav"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        
        if st.button("🔍 Predict Emotion from File"):
            with st.spinner("Analyzing uploaded file structure..."):
                try:
                    detected_emotion, probabilities = process_and_predict(uploaded_file)
                    confidence_percentage = probabilities[np.where(classes == detected_emotion)[0][0]] * 100
                    
                    st.markdown("---")
                    st.subheader(f"Detected Emotion: **{detected_emotion.upper()}**")
                    st.progress(int(confidence_percentage))
                    st.write(f"**Confidence Rating:** {confidence_percentage:.2f}%")
                    
                    chart_data = pd.DataFrame(probabilities * 100, index=classes, columns=["Probability (%)"])
                    st.bar_chart(chart_data)
                except Exception as e:
                    st.error(f"❌ An error occurred during file parsing: {e}")

with tab2:
    st.subheader("Record with your microphone")
    if st.button("🔴 Start Recording (3 Seconds)"):
        with st.spinner("🎙️ Recording... Speak now!"):
            import sounddevice as sd
            audio_buffer = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait() 
            
        st.success("✅ Audio captured successfully!")
        
        write(TEMP_FILENAME, SAMPLE_RATE, audio_buffer)
        
        with st.spinner("Analyzing live frequencies..."):
            try:
                detected_emotion, probabilities = process_and_predict(TEMP_FILENAME)
                confidence_percentage = probabilities[np.where(classes == detected_emotion)[0][0]] * 100
                
                if os.path.exists(TEMP_FILENAME):
                    os.remove(TEMP_FILENAME)
                
                st.markdown("---")
                st.subheader(f"Detected Emotion: **{detected_emotion.upper()}**")
                st.progress(int(confidence_percentage))
                st.write(f"**Confidence Rating:** {confidence_percentage:.2f}%")
                
                chart_data = pd.DataFrame(probabilities * 100, index=classes, columns=["Probability (%)"])
                st.bar_chart(chart_data)
                
            except Exception as e:
                st.error(f"❌ An error occurred during live audio processing: {e}")