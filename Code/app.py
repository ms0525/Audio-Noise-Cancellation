import streamlit as st
from utils import *
import streamlit as st
from audiorecorder import audiorecorder
from pathlib import Path
from io import BytesIO
import sounddevice as sd

st.title("Record Noisy Audio")

# Record the audio
audio = audiorecorder("Click to record", "Click to stop recording")

# If audio is recorded then save it
if len(audio) > 0:
    audio.export("rec/audio.wav", format="wav")

if Path("rec/audio.wav").exists():
    st.write('## Recorded Audio')
    outout_audio = predict_audio('rec')
    audio, sr = librosa.load('rec/audio.wav')
    st.audio(audio, sample_rate=sr)
    
if Path("rec.wav").exists():
    st.write('## Cleaned Audio')    
    audio, sr = librosa.load('rec.wav')
    st.audio(audio, sample_rate=sr)

###### the recording part ends here #######

st.markdown("<h1 style='text-align: center;'>OR</h1>", unsafe_allow_html=True)

###### the Uploading part Starts here #######

st.title("Upload Noisy Audio")

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

upload_dir = Path("upload")
upload_dir.mkdir(exist_ok=True)

if uploaded_file is not None:
    file_format = "audio/wav" if uploaded_file.name.endswith(".wav") else "audio/mp3"
    st.write('## Uploaded Audio')  
    st.audio(uploaded_file, format=file_format)

    audio, sr = librosa.load(uploaded_file, sr=None)

    output_file = upload_dir / "audio.wav"
    sf.write(output_file, audio, samplerate=sr)

    st.write(f"Sample Rate: {sr} Hz")
    st.write(f"Audio Length: {len(audio) / sr:.2f} seconds")

    # Predict if file exists
    if Path("upload/audio.wav").exists():
        st.write("Predicting")
        output_audio = predict_audio('upload')
    
    if Path("upload.wav").exists():
        st.write('## Cleaned Audio')    
        audio, sr = librosa.load('upload.wav')
        st.audio(audio, sample_rate=sr)

