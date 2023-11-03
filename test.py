import cv2
import numpy as np
import streamlit as st
from keras.models import model_from_json
import pyttsx3
import pywhatkit
import time
import speech_recognition as sr

page_options = ["Music of your choice", "Music based on your Current emotion"]

json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("facialemotionmodel.h5")

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Create a sidebar with navigation links
selected_page = st.sidebar.selectbox("Select a page", page_options)

# Initialize the TTS engine once
machine = pyttsx3.init()

listener = sr.Recognizer()

def talk(text):
    machine.say(text)
    machine.runAndWait()

def detect_emotion(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image, 1.3, 5)

    for (p, q, r, s) in faces:
        face = gray[q:q + s, p:p + r]
        face = cv2.resize(face, (48, 48))
        img = extract_features(face)
        pred = model.predict(img)
        emotion_label = labels[pred.argmax()]
        return emotion_label

if selected_page == "Music of your choice":
    st.header("Welcome to Amelia, your virtual assistant for music")
    if st.button("Activate Virtual Assistant"):
        st.write("Virtual Assistant is listening. Please speak your instructions.")
        machine.say("Hi, I am Jarvis, your virtual assistant. How can I help you today?")
        time.sleep(2)

        try:
            with sr.Microphone() as source:
                listener.adjust_for_ambient_noise(source)
                st.write("Listening...")
                audio = listener.listen(source, timeout=5)
                st.write("Processing your instruction...")

            user_input = listener.recognize_google(audio)
            st.write(f"User Instruction: {user_input}")

            if "play" in user_input:
                song = user_input.replace("play", "").strip()
                machine.say(f"Playing {song}")
                pywhatkit.playonyt(song)
        except sr.WaitTimeoutError:
            st.write("No audio input. Please try again.")

elif selected_page == "Music based on your Current emotion":
    st.header("Music based on your Current emotion")
    if st.button("Activate Virtual Assistant"):
        # Capture a frame from the webcam
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()

        if ret:
            emotion = detect_emotion(frame)
            st.write(f"Detected Emotion: {emotion}")
        else:
            st.write("Error capturing frame from the webcam.")

        st.write("Virtual Assistant is listening. Please speak your instructions.")
        machine.say("Hi, I am Amelia, your virtual assistant. which page would you like to navigate to?")
        time.sleep(2)  # Wait for the intro message to finish

        try:
            with sr.Microphone() as source:
                listener.adjust_for_ambient_noise(source)
                st.write("Listening...")
                audio = listener.listen(source, timeout=5)
                st.write("Processing your instruction...")

            user_input = listener.recognize_google(audio)
            st.write(f"User Instruction: {user_input}")
            
            # Process user instructions based on emotion (you can add more cases as needed)
            if emotion == 'happy':
                talk("Would you like me to play a song to lighten your mood?")
                talk("Playing song")
                pywhatkit.playonyt('Happy Song')  # You can replace 'Happy Song' with a suitable song name
            elif emotion == 'angry':
                talk("Would you like me to play a song to calm you down?")
                talk("Playing song")
                pywhatkit.playonyt('Calm Down Song')  # Replace with an appropriate song
            # Add more conditions for other emotions
            
        except sr.WaitTimeoutError:
            st.write("No audio input. Please try again.")

    




    


