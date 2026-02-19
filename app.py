import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

st.set_page_config(layout="wide")

st.title("Multi-Modal Behavioral Inconsistency Analyzer")
st.subheader("Blink Detection Module")

start = st.button("Start Camera")

frame_placeholder = st.empty()
blink_counter_placeholder = st.empty()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

camera = cv2.VideoCapture(0)

blink_count = 0
blink_threshold = 0.21
blink_cooldown = 0.3
last_blink_time = 0


def calculate_EAR(landmarks, eye_indices):
    eye = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])

    vertical1 = np.linalg.norm(eye[1] - eye[5])
    vertical2 = np.linalg.norm(eye[2] - eye[4])
    horizontal = np.linalg.norm(eye[0] - eye[3])

    ear = (vertical1 + vertical2) / (2.0 * horizontal)
    return ear


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

if start:
    while True:
        success, frame = camera.read()
        if not success:
            st.error("Camera error")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                left_ear = calculate_EAR(landmarks, LEFT_EYE)
                right_ear = calculate_EAR(landmarks, RIGHT_EYE)

                ear = (left_ear + right_ear) / 2.0

                current_time = time.time()

                if ear < blink_threshold:
                    if current_time - last_blink_time > blink_cooldown:
                        blink_count += 1
                        last_blink_time = current_time

                cv2.putText(frame, f"Blinks: {blink_count}",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

        frame_placeholder.image(frame)

camera.release()
