import streamlit as st
from tensorflow import keras
import tensorflow as tf
import numpy as np
from PIL import Image

IMAGE_SIZE = (180, 180)
DEMO_IMAGE = 'kotek.jpg'

def load_model():
    model = keras.models.load_model('dog_cat_model.h5')
    return model

def predict(img, model):
    img = img.resize(IMAGE_SIZE, Image.ANTIALIAS)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = predictions[0]
    return score

def run(model):
    st.title("Cat vs. Dog")
    img_file_buffer = st.file_uploader(label="", type=["png", "jpg", "jpeg"])
    if img_file_buffer is not None:
        img = Image.open(img_file_buffer)
    else:
        img = Image.open(DEMO_IMAGE)
    img.thumbnail((300,300), Image.ANTIALIAS)
    st.image(
        img, use_column_width=False,
    )

    score = int(100*predict(img, model))
    if score > 50:
        caption = f"It's a dog with {score}% probability!"
    else:
        caption = f"It's a cat with {100-score}% probability!"
    st.markdown(f"<h1 style='text-align: center;'>{caption}</h1>", unsafe_allow_html=True)

def main():
    model = load_model()
    run(model)

if __name__ == '__main__':
    main()