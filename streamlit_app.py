import streamlit as st
import face_recognition
import numpy as np
import os
import pickle
from PIL import Image
import cv2

# File path for the face data
FACE_DATA_FILE = 'face_data.pkl'
GRAYSCALE_FOLDER = 'grayscale_images'  # Folder to store grayscale images

# Function to load face data
def load_face_data():
    if os.path.exists(FACE_DATA_FILE):
        with open(FACE_DATA_FILE, 'rb') as file:
            face_data = pickle.load(file)
    else:
        face_data = {}  # Initialize as empty dictionary if no file exists
    return face_data

# Function to save face data
def save_face_data(face_data):
    with open(FACE_DATA_FILE, 'wb') as file:
        pickle.dump(face_data, file)

# Function to add a new student's face data
def add_student_face_data(name, new_encodings, images):
    face_data = load_face_data()

    if name in face_data:
        face_data[name].extend(new_encodings)
    else:
        face_data[name] = new_encodings

    save_face_data(face_data)

    # Ensure the grayscale images folder exists
    if not os.path.exists(GRAYSCALE_FOLDER):
        os.makedirs(GRAYSCALE_FOLDER)

    # Save grayscale images for the student
    st.write(f"Grayscale images for {name} are saved in the project folder.")
    for i, uploaded_img in enumerate(images):
        img = Image.open(uploaded_img)
        grayscale_img = np.array(img.convert('L'))  # Convert image to grayscale

        # Save the grayscale image to the folder
        grayscale_image_path = os.path.join(GRAYSCALE_FOLDER, f"{name}_grayscale_{i}.png")
        Image.fromarray(grayscale_img).save(grayscale_image_path)

    st.success(f"Successfully added face data for {name} and saved grayscale images!")

# Streamlit app UI
st.title("Face Recognition App with Multiple Image Upload")

# Option to register new student, delete a student, or test
action = st.radio("What do you want to do?", ('None', 'Register a new student', 'Delete a student'))

if action == 'Register a new student':
    student_name = st.text_input("Enter the student's name")
    
    if student_name:
        st.write(f"Upload images for {student_name}...")
        uploaded_images = st.file_uploader("Upload multiple images (minimum 10)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

        if uploaded_images:
            if len(uploaded_images) < 10:
                st.warning("Please upload at least 10 images.")
            else:
                new_face_encodings = []

                for uploaded_image in uploaded_images:
                    image = Image.open(uploaded_image)
                    st.image(image, caption=f"Uploaded Image", use_column_width=True)
                    image = np.array(image)

                    # Detect face locations and encodings
                    face_locations = face_recognition.face_locations(image)
                    face_encodings = face_recognition.face_encodings(image, face_locations)

                    if len(face_encodings) > 0:
                        new_face_encodings.append(face_encodings[0])
                    else:
                        st.warning(f"No face detected in one of the images. Please upload clear images.")

                if new_face_encodings:
                    add_student_face_data(student_name, new_face_encodings, uploaded_images)

elif action == 'Delete a student':
    face_data = load_face_data()

    if len(face_data) > 0:
        student_name_to_delete = st.selectbox("Select the student to delete:", list(face_data.keys()))

        if st.button(f"Delete {student_name_to_delete}"):
            if student_name_to_delete in face_data:
                del face_data[student_name_to_delete]
                save_face_data(face_data)
                st.success(f"Successfully deleted {student_name_to_delete}.")
            else:
                st.error(f"Student {student_name_to_delete} not found.")
    else:
        st.error("No students found. Please register some students first.")

else:
    # Testing mode
    st.write("Testing Mode: Verify identity using the webcam or uploaded images.")
    face_data = load_face_data()

    if len(face_data) > 0:
        webcam_image = st.camera_input("Capture your face with webcam")

        if webcam_image:
            frame = np.array(Image.open(webcam_image))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if len(face_encodings) == 0:
                st.error("No human face found in the webcam feed.")
            else:
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    matched_name = "Unknown"
                    for name, encodings in face_data.items():
                        face_distances = face_recognition.face_distance(encodings, face_encoding)
                        if len(face_distances) > 0 and np.min(face_distances) < 0.4:
                            matched_name = name
                            break

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(frame, matched_name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                st.image(image, caption="Webcam Feed", use_column_width=True)

    else:
        st.error("No registered students. Please register students first.")

