import streamlit as st
import requests
from PIL import Image
from io import BytesIO
from transformers import BlipProcessor, BlipForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel
from gtts import gTTS
import torch

# Load the BLIP model for image captioning
caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load the GPT-2 model for story generation
story_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
story_model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_caption(image):
    # Preprocess the image
    inputs = caption_processor(images=image, return_tensors="pt")

    # Generate caption
    out = caption_model.generate(**inputs)
    caption = caption_processor.decode(out[0], skip_special_tokens=True)
    return caption

def generate_story(caption):
    # Prepare the prompt for story generation
    prompt = f"{caption}. "
    inputs = story_tokenizer(prompt, return_tensors="pt", max_length=50, truncation=True)

    # Generate story
    outputs = story_model.generate(**inputs, max_length=50, num_return_sequences=1)
    story = story_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return story

def text_to_speech(text, filename="output.mp3"):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    return filename

# Streamlit app
st.title("Image Captioning and Story Generation with TTS")

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Generate caption
    caption = generate_caption(image)
    st.write(f"**Generated Caption:** {caption}")

    # Generate story
    story = generate_story(caption)
    st.write(f"**Generated Story:** {story}")

    # Convert story to speech
    audio_file = text_to_speech(story)
    st.audio(audio_file, format='audio/mp3', start_time=0)
