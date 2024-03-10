import streamlit as st
from transformers import VitsModel, AutoTokenizer
import torch
import soundfile as sf

# Title and Description
st.title("Text-to-Speech with VitsModel")
st.write("Enter some English text, and I'll generate audio for you!")

# Load Model and Tokenizer
@st.cache_resource  # Cache the model for efficiency
def load_tts_model():
    model = VitsModel.from_pretrained("facebook/mms-tts-eng")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    return model, tokenizer

model, tokenizer = load_tts_model()

# User Input
user_text = st.text_input("Enter your text here:")

# Generate Audio on Button Click
if st.button("Generate Speech"):    
    if not user_text:        
        st.warning("Please enter some text.")    
    else:        
        inputs = tokenizer(user_text, return_tensors="pt")        
        with torch.no_grad():            
            output = model(**inputs).waveform

        # Specify sample rate (assuming it's the correct rate for the model)
        sample_rate = 16000  # Or replace with the correct sample rate for 'facebook/mms-tts-eng'

        # Optionally save to a temporary file (if needed)
        sf.write("temp_audio.wav", output[0].numpy(), sample_rate)  

        # Choose one of the following playback methods:

        # Method 1: Play from temporary file
        st.audio("temp_audio.wav")  

        # Method 2: Play directly with sample rate 
        st.audio(output[0].numpy(), sample_rate=sample_rate) 
