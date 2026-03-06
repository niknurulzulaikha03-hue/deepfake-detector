import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

def extract_landmarks(video_path):

    cap = cv2.VideoCapture(video_path)

    landmarks_all = []
    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        if frame_count % 10 != 0:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(frame)

        if results.multi_face_landmarks:

            face_landmarks = results.multi_face_landmarks[0]

            coords = []

            for lm in face_landmarks.landmark:
                coords.append(lm.x)
                coords.append(lm.y)

            landmarks_all.append(coords)

    cap.release()

    if len(landmarks_all) == 0:
        return None

    return np.mean(landmarks_all, axis=0)
