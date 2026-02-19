import streamlit as st
import cv2
import mediapipe as mp

st.set_page_config(layout="wide")

st.title("Multi-Modal Behavioral Inconsistency Analyzer")
st.subheader("Live Face Landmark Tracking")

start = st.button("Start Camera")
stop = st.button("Stop Camera")

frame_placeholder = st.empty()

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

camera = cv2.VideoCapture(0)

if start:
    while True:
        success, frame = camera.read()
        if not success:
            st.error("Failed to capture video")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

        frame_placeholder.image(frame)

        if stop:
            break

camera.release()
