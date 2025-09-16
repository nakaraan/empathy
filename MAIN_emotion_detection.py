import cv2
import mediapipe as mp
from deepface import DeepFace
import time

last_emotion = ""
last_time = 0
FACE_DETECTION = True
LANDMARK_DETECTION = True
POSE_DETECTION = True

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open camera")
    exit()

def mouse_callback(event, x, y, flags, param):
    global FACE_DETECTION, LANDMARK_DETECTION, POSE_DETECTION
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 < x < 110 and 10 < y < 60:
            FACE_DETECTION = not FACE_DETECTION
        if 10 < x < 110 and 70 < y < 120:
            LANDMARK_DETECTION = not LANDMARK_DETECTION
        if 10 < x < 110 and 130 < y < 180:
            POSE_DETECTION = not POSE_DETECTION

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', mouse_callback)

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection, \
     mp_face_mesh.FaceMesh() as face_mesh, \
     mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

    while True:
        ret, image = cam.read()
        if not ret:
            break

        dominant = None
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result_detection = face_detection.process(rgb_image)
        result_mesh = face_mesh.process(rgb_image)
        result_pose = pose.process(rgb_image)
        height, width, _ = image.shape

        # Buttons
        cv2.rectangle(image, (10, 10), (110, 60), (200, 200, 200), -1)
        cv2.putText(image, f"FD: {'[X]' if FACE_DETECTION else '[ ]'}", (15, 45), cv2.FONT_HERSHEY_PLAIN, 0.7, (0,0,0), 2)
        cv2.rectangle(image, (10, 70), (110, 120), (200, 200, 200), -1)
        cv2.putText(image, f"LM: {'[X]' if LANDMARK_DETECTION else '[ ]'}", (15, 105), cv2.FONT_HERSHEY_PLAIN, 0.7, (0,0,0), 2)
        cv2.rectangle(image, (10, 130), (110, 180), (200, 200, 200), -1)
        cv2.putText(image, f"Pose: {'[X]' if POSE_DETECTION else '[ ]'}", (15, 165), cv2.FONT_HERSHEY_PLAIN, 0.7, (0,0,0), 2)

        # Face detection + DeepFace
        if FACE_DETECTION and result_detection and result_detection.detections:
            for detection in result_detection.detections:
                bboxC = detection.location_data.relative_bounding_box
                x = int(bboxC.xmin * width)
                y = int(bboxC.ymin * height)
                w = int(bboxC.width * width)
                h = int(bboxC.height * height)
                cv2.rectangle(image, (x, y), (x + w, y + h), (144, 238, 144), 3)

                face_img = image[y:y+h, x:x+w]
                if face_img.size > 0:
                    try:
                        objs = DeepFace.analyze(face_img, actions=["emotion"], enforce_detection=False)
                        dominant = objs[0]['dominant_emotion']
                        cv2.putText(image, f"Emotion: {dominant}", (10, height - 50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
                    except:
                        pass

        # Face landmarks
        if LANDMARK_DETECTION and result_mesh and result_mesh.multi_face_landmarks:
            for facial_landmarks in result_mesh.multi_face_landmarks:
                for i in range(0, 468):
                    pt1 = facial_landmarks.landmark[i]
                    x = int(pt1.x * width)
                    y = int(pt1.y * height)
                    cv2.circle(image, (x, y), 1, (144, 238, 144), -1)

        # Pose detection
        gesture = ""
        if POSE_DETECTION and result_pose.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(image, result_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            lm = result_pose.pose_landmarks.landmark
            shoulder_y = (lm[11].y + lm[12].y) / 2
            hand_y = min(lm[15].y, lm[16].y)
            wrist_x_diff = abs(lm[15].x - lm[16].x)

            # Simple heuristic rules
            if hand_y < shoulder_y:
                gesture = "Excitement (hands raised)"
            elif wrist_x_diff < 0.1:
                gesture = "Defensiveness (arms close)"
            elif shoulder_y > 0.6:
                gesture = "Sadness/Tiredness (slouching)"
            else:
                gesture = "Confidence (upright posture)"

            cv2.putText(image, f"Gesture: {gesture}", (10, height - 20), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)

        cv2.imshow('Image', image)
        if cv2.waitKey(1) == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
