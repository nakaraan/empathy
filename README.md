# EMPATHY S13: Empathetic Computing

**Authors**  
- ALMIN, Wesner III  
- PINEDA, Dencel Angelo  
- SY, Vaughn Marick  
- VALDEZ, Pulvert Gerald  

---

## 📌 Overview
**EMPATHY S13** is a unimodal emotion detection system that interprets facial and body cues from live camera feeds, images, and video files.  
It combines **MediaPipe**, **OpenCV**, and **DeepFace** to capture facial landmarks, detect expressions, and analyze emotional states.  

---

## 📂 Supported Data
- **Images:** `.JPEG`, `.JPG`, `.PNG`  
- **Videos:** `.MP4`, `.HVEC`  
- **Live Feed:** Laptop/USB camera  

---

## ✨ Features
- **Facial Detection:** Bounding boxes around detected faces.  
- **Facial Landmarks:** Up to 468 anchor points mapped in real-time.  
- **Current Emotion:** Classification into:  
  *Happy, Neutral, Surprise, Sad, Angry, Fear, Disgust*  
- **Dominant Emotion:** Emotion with the highest score displayed live.  
- **Body/Gesture Recognition:** Using MediaPipe Pose to interpret posture and gestures as emotional cues:  
  - Confidence → upright posture, open arms  
  - Defensiveness → crossed arms  
  - Excitement/Joy → fast gestures, raised hands  
  - Sadness/Tiredness → slouching, slow or drooping movement  

---

## 🧠 Models & Tools
- **[DeepFace](https://github.com/serengil/deepface_models/releases):** Pre-trained model for facial emotion, age, and gender recognition.  
- **[MediaPipe](https://developers.google.com/mediapipe):** ML framework for face, hand, and body landmark detection.  
- **OpenCV (`cv2`):** Captures live camera feed, processes video frames, and provides GUI overlays.  

---

## ⚙️ Installation
Clone the repository and install dependencies:

```bash
git clone https://github.com/nakaraan/empathy.git
cd empathy-s13
pip install opencv-python mediapipe deepface matplotlib tensorflow protobuf
