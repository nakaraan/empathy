import cv2
import mediapipe as mp

image = cv2.imread("person.jpg")

cv2.imshow("Image", image)
cv2.waitKey(0)