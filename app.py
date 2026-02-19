import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time

st.set_page_config(layout="wide")

st.title("Multi-Modal Behavioral Inconsistency Analyzer")

mode = st.radio("Select Mode", ["Baseline Capture", "Response Analysis"])

start = st.button("Start Camera")

frame_placeholder = st.empty()
stats_placeholder = st.empty()

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

session_start_time = time.time()

baseline_rate = None


def calculate_EAR(landmarks, eye_indices):
    eye = np.array([(landmarks[i].x, landmarks[i].y) for i in eye_indices])
    vertical1 = np.linalg.norm(eye[1] - eye[5])
    vertical2 = np.linalg.norm(eye[2] - eye[4])
    horizontal = np.linalg.norm(eye[0] - eye[3])
    return (vertical1 + vertical2) / (2.0 * horizontal)


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

        elapsed_time = time.time() - session_start_time
        blink_rate = (blink_count / elapsed_time) * 60 if elapsed_time > 0 else 0

        if mode == "Baseline Capture" and elapsed_time >= 10:
            baseline_rate = blink_rate
            stats_placeholder.success(f"Baseline Blink Rate Captured: {baseline_rate:.2f}")
            break

        if mode == "Response Analysis" and baseline_rate is not None:
            deviation = blink_rate - baseline_rate
            stats_placeholder.markdown(f"""
            ### 📊 Analysis
            - Baseline Rate: **{baseline_rate:.2f}**
            - Current Rate: **{blink_rate:.2f}**
            - Deviation: **{deviation:.2f}**
            """)

        else:
            stats_placeholder.markdown(f"""
            ### 📊 Current Blink Rate
            - **{blink_rate:.2f} per minute**
            """)

        frame_placeholder.image(frame)

camera.release()
