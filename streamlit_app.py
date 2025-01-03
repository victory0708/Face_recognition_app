import streamlit as st
import os
import sys

try:
    import face_recognition
except ModuleNotFoundError:
    st.warning("Installing face_recognition... Please wait.")
    
    # Install dependencies using pre-built wheels
    os.system(f"{sys.executable} -m pip install cmake dlib numpy Pillow opencv-python-headless")
    os.system(f"{sys.executable} -m pip install face_recognition")

    try:
        import face_recognition
    except ModuleNotFoundError:
        st.error("Failed to install face_recognition. Please restart the app or check dependencies.")
        st.stop()

import numpy as np
import pickle
from PIL import Image
import cv2

FACE_DATA_FILE = 'face_data.pkl'
GRAYSCALE_FOLDER = 'grayscale_images'

def load_face_data():
    if os.path.exists(FACE_DATA_FILE):
        with open(FACE_DATA_FILE, 'rb') as file:
            face_data = pickle.load(file)
    else:
        face_data = {}
    return face_data

def save_face_data(face_data):
    with open(FACE_DATA_FILE, 'wb') as file:
        pickle.dump(face_data, file)

def add_student_face_data(name, new_encodings, images):
    face_data = load_face_data()
    if name in face_data:
        face_data[name].extend(new_encodings)
    else:
        face_data[name] = new_encodings

    save_face_data(face_data)

    if not os.path.exists(GRAYSCALE_FOLDER):
        os.makedirs(GRAYSCALE_FOLDER)

    for i, uploaded_img in enumerate(images):
        img = Image.open(uploaded_img)
        grayscale_img = np.array(img.convert('L'))
        grayscale_image_path = os.path.join(GRAYSCALE_FOLDER, f"{name}_grayscale_{i}.png")
        Image.fromarray(grayscale_img).save(grayscale_image_path)

    st.success(f"Added face data for {name} and saved grayscale images!")

st.title("Face Recognition App")

action = st.radio("Choose an action:", ('None', 'Register a student', 'Delete a student'))

if action == 'Register a student':
    student_name = st.text_input("Student name")
    uploaded_images = st.file_uploader("Upload 10+ images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if student_name and uploaded_images:
        if len(uploaded_images) < 10:
            st.warning("Upload at least 10 images.")
        else:
            new_face_encodings = []
            for uploaded_image in uploaded_images:
                image = np.array(Image.open(uploaded_image))
                face_locations = face_recognition.face_locations(image)
                face_encodings = face_recognition.face_encodings(image, face_locations)

                if len(face_encodings) > 0:
                    new_face_encodings.append(face_encodings[0])
                else:
                    st.warning("No face detected in one image.")
            
            if new_face_encodings:
                add_student_face_data(student_name, new_face_encodings, uploaded_images)

elif action == 'Delete a student':
    face_data = load_face_data()
    if face_data:
        student_name_to_delete = st.selectbox("Select student:", list(face_data.keys()))
        if st.button(f"Delete {student_name_to_delete}"):
            del face_data[student_name_to_delete]
            save_face_data(face_data)
            st.success(f"{student_name_to_delete} deleted.")
    else:
        st.error("No students registered.")
