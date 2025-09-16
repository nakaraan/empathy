import cv2
import mediapipe as mp
from deepface import DeepFace
import matplotlib.pyplot as plt
import time

last_emotion = ""
last_time = 0

FACE_DETECTION = True
LANDMARK_DETECTION = True

"""
# suppress log messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
import logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)  # Suppress MediaPipe logs
"""

# DeepFace models
models = [
    "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace",
    "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet",
    "Buffalo_L",
]

# Face mesh
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

cam = cv2.VideoCapture(0)
if not cam.isOpened():
  print("Error: Could not open camera")
  exit()

print(type(cam))


def mouse_callback(event, x, y, flags, param):
    global FACE_DETECTION, LANDMARK_DETECTION
    # Example: Toggle face detection if you click inside a rectangle
    if event == cv2.EVENT_LBUTTONDOWN:
        # Button 1: Face Detection (top left corner)
        if 10 < x < 110 and 10 < y < 60:
            FACE_DETECTION = not FACE_DETECTION
        # Button 2: Landmark Detection (below button 1)
        if 10 < x < 110 and 70 < y < 120:
            LANDMARK_DETECTION = not LANDMARK_DETECTION

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)


with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection, mp_face_mesh.FaceMesh() as face_mesh:
  while True:
    ret, image = cam.read()
    if not ret:
      break

    dominant = None
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result_detection = face_detection.process(rgb_image)
    result = face_mesh.process(rgb_image)
    height, width, _ = image.shape
    # print("Height, width", height, width) # uncomment for calculated dimensions
    
    # buttons
    cv2.rectangle(image, (10, 10), (110, 60), (200, 200, 200), -1)
    cv2.putText(image, f"FD: {'[X]' if FACE_DETECTION else '[ ]'}", (15, 45), cv2.FONT_HERSHEY_PLAIN, 0.7, (0,0,0), 2)
    cv2.rectangle(image, (10, 70), (110, 120), (200, 200, 200), -1)
    cv2.putText(image, f"LM: {'[X]' if LANDMARK_DETECTION else '[ ]'}", (15, 105), cv2.FONT_HERSHEY_PLAIN, 0.7, (0,0,0), 2)

    # Face Detection
    if FACE_DETECTION and result_detection and result_detection.detections:
      for detection in result_detection.detections:
        bboxC = detection.location_data.relative_bounding_box
        x = int(bboxC.xmin * width)
        y = int(bboxC.ymin * height)
        w = int(bboxC.width * width)
        h = int(bboxC.height * height)
        cv2.rectangle(image, (x, y), (x + w, y + h), (144, 238, 144), 3)
        
        # Crop face region for deepface analysis
        face_img = image[y:y+h, x:x+w]
        if face_img.size > 0:
          try:
            objs = DeepFace.analyze(face_img, "emotion", True)
            for obj in objs:
              emotions = obj['emotion']
              dominant = obj['dominant_emotion']
              # print("Emotion scores:")
              # for emotion, score in emotions.items():
              #  print(f" {emotion}: {score:.2f}%")
              print(f"Dominant emotion: {dominant}")
              # Draw a filled rectangle as background for text for better visibility
              cv2.rectangle(image, (height - 40), (300, height), (255, 255, 255), -1)
              cv2.putText(image, f"Emotion: {dominant}", (10, height - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
          except Exception as e:  
            # print("DeepFace error: ", e) 
            pass
            
    # Face Landmark Detection
    if LANDMARK_DETECTION and result and result.multi_face_landmarks:
      for facial_landmarks in result.multi_face_landmarks:
        for i in range(0, 468):
          pt1 = facial_landmarks.landmark[i]
          x = int(pt1.x * width)
          y = int(pt1.y * height)
          cv2.circle(image, (x, y), 1, (144, 238, 144), -1)
          # print("x, y", x, y) # uncomment to see landmark coordinates
          
    # For persistent emotion display
    if dominant:
        last_emotion = dominant
        last_time = time.time()

    # Draw text if emotion was detected in the last 2 seconds
    if time.time() - last_time < 2 and last_emotion:
        cv2.putText(image, f"Emotion: {last_emotion}", (10, height - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)      
        
    cv2.imshow('Image', image)
    
    if cv2.waitKey(1) == ord('q'):
      break

cam.release()
cv2.destroyAllWindows()

"""_references_
  https://www.youtube.com/watch?v=LF7Lgz4_lus
"""
