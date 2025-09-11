import cv2
import mediapipe as mp

# face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Image
image = cv2.imread('test.png')

result = face_mesh.process(image)

# display output
cv2.imshow('Image', image)
cv2.waitKey(0)

"""_references_
  https://www.youtube.com/watch?v=LF7Lgz4_lus
"""
