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
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

cam = cv2.VideoCapture(0)
if not cam.isOpened():
  print("Error: Could not open camera")
  exit()

print(type(cam))

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection, mp_face_mesh.FaceMesh() as face_mesh:
  while True:
    ret, image = cam.read()
    if not ret:
      break

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result_detection = face_detection.process(rgb_image)
    result = face_mesh.process(rgb_image)
    height, width, _ = image.shape
    # print("Height, width", height, width)
    
    # Face Detection
    if result_detection.detections:
      for detection in result_detection.detections:
        bboxC = detection.location_data.relative_bounding_box
        x = int(bboxC.xmin * width)
        y = int(bboxC.ymin * height)
        w = int(bboxC.width * width)
        h = int(bboxC.height * height)
        cv2.rectangle(image, (x, y), (x + w, y + h), (144, 238, 144), 3)

    # Face Landmark Detection
    if result.multi_face_landmarks:
      for facial_landmarks in result.multi_face_landmarks:
        for i in range(0, 468):
          pt1 = facial_landmarks.landmark[i]
          x = int(pt1.x * width)
          y = int(pt1.y * height)
          cv2.circle(image, (x, y), 1, (144, 238, 144), -1)
          # print("x, y", x, y) # uncomment to see landmark coordinates

    cv2.imshow('Image', image)
    if cv2.waitKey(1) == ord('q'):
      break

cam.release()
cv2.destroyAllWindows()

"""_references_
  https://www.youtube.com/watch?v=LF7Lgz4_lus
"""
