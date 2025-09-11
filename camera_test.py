import numpy as np
from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2

# Start video capture

cam = cv2.VideoCapture(0)

if not cam.isOpened(): 
  print("Error: Could not open camera")
  exit()

print(type(cam))

frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cam.read()
    
    # write frame to output file
    out.write(frame)
    
    # display captured frame
    cv2.imshow('Camera', frame)
    
    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
      break
    
# Release the capture and writer objects
cam.release()
out.release()
cv2.destroyAllWindows()
      
      