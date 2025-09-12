import cv2
import mediapipe as mp


"""
# suppress log messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
import logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)  # Suppress MediaPipe logs
"""

# Face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Image
image = cv2.imread('person.jpg')
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Facial landmarks
result = face_mesh.process(rgb_image)
print(result)

# height
height, width, _ = image.shape
print("Height, width", height, width)

# Facial landmark mapping
for facial_landmarks in result.multi_face_landmarks:
  for i in range(0, 468):
    pt1 = facial_landmarks.landmark[i]
    x = (pt1.x * width)
    y = (pt1.y * height)
    cv2.circle(image, (int(x), int(y)), 1, (144, 238, 144), -1)
    print("x, y", x, y)

# Display output
cv2.imshow('Image', image)
cv2.waitKey(0)

"""_references_
  https://www.youtube.com/watch?v=LF7Lgz4_lus
"""
