import cv2
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)

while True:
  _, img = cam.read()
  cv2.imshow("Face detection", img)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  faces = face_cascade.detectMultiScale(gray, 1.5, 4)
  for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x + w, y + h), (144, 238, 144), 3)
  cv2.imshow("Face detection", img)
  
  key = cv2.waitKey(10)
  if key == 27:
    break
cam.release()
cv2.destroyAllWindows()