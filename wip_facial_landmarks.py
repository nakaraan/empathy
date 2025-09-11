import cv2
import mediapipe as mp

# Face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Image
image = cv2.imread('test.jpg')
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Facial landmarks
result = face_mesh.process(rgb_image)
print(result)

for facial_landmarks in result.multi_face_landmarks:
  print(facial_landmarks)


# Display output
cv2.imshow('Image', image)
cv2.waitKey(0)

"""_references_
  https://www.youtube.com/watch?v=LF7Lgz4_lus
"""
