
# EMPATHY S13: EMPATHETIC COMPUTING

*ALMIN, Wesner III* •
*PINEDA, Dencel Angelo* •
*SY, Vaughn Marick* •
*VALDEZ, Pulvert Gerald*

## Data
The current working unimodal tool supports photo files (.JPEG, .JPG, .PNG), video files (.MP4, .HVEC), and a live camera feed (laptop camera). 

## Features
The unimodal emotion detection tool employs the extraction of the following:
* **Facial Detection:** Facial extraction through the MediaPipe module, with colored bounding boxes encircling detected faces.
* **Facial Landmarks:** Detection of facial landmarks or "anchor points" that each map to a specific point in the facial structure. Each point will be represented in the live camera feed, with a maximum of 468 possible points.
* **Current Emotion:** Based on detected facial expression and specified sensitivity metrics, the model will gauge the current facial expression with each of the following emotions: *["Happy", "Neutral", "Surprise", "Sad", "Angry", "Fear", "Disgust"]*
* **Dominant Emotion:** The detected emotion with the highest scored match will be displayed on the screen.

## Model/ Tool
Pre-trained models used:

**DeepFace**

*A highly specialized facial detection model that can support detection and processing of facial detection, expressions, similarity, and verification*

Download: https://github.com/serengil/deepface_models/releases

**MediaPipe**

*An interface to easily integrate machine learning solutions to different media-extracted features (face, eye, hand, movement)*

Download: ``` $ pip install mediapipe ```

## How it Works
To properly unimodally interpret live camera facial expressions, the following applications were integrated into the tool:

1. The ```opencv```'s ```cv2``` module library was utilized to capture the live camera feed and turn said captured images and video to modifiable data, by storing said media into an object with methods. Upon displaying captured data using ```cv2.imshow('Image', image)```, an application window showcasing said photo-video inputs will be displayed, albeit with additional tweaks or changes to the data using methods.

* *also provides dynamic UI support to be layered on top of any captured photo-video feed, which will be added to the windowed outputted display once run. Methods such as ```cv2.putText``` and ```cv2.rectangle``` are examples of such UI modification methods.*

* ```cv2.circle``` is used to map a specific dimensionally-specified bounding box around a given x and y coordinate, which is used later in mapping out the facial landmarks. 



2. ```mediapipe``` serves as the backbone for the initial facial detection and facial mesh, separated into two variables for concurrent visual processing. Mediapipe also allows for simplified model selection and modifiable facial detection weights. 

* *The face itself is extracted through converting the live video feed into individual frame images, converted to either colored images or grayscale images based on its applicative use, and cropped and run through different models to extract different features.*

* *Allows for the bounding box encircling the detected face.*

* *Enables facial landmark detection using the ```face_mesh.process()``` method, which maps facial landmark points around the detected face. (Minimum of 1 landmark and maximum of 468 landmarks, with each landmark scaled to the given width and height of the image)*
 


3. ```DeepFace``` takes the cropped face image as input and extracts subfeatures (emotion, age, gender) and uses its model to compare it against trained weights of the same name. 

* ```obj['emotion']``` represents the extracted DeepFace-processed map of emotions from the given face input, along with subsequent confidence weights. 
* *```obj['dominant_emotion']``` is another key pertaining to the emotion with the highest detected confidence weight, which is displayed on the screen and printed on the terminal with ```cv2.putText()``` and ```print()``` respectively.*


4. ```cam.release()``` and ```cv2.destroyAllWindows()``` free up and close the current processes, preventing video RAM overspilling and breaking subsequent attempts at running the program.



## Experience

Developing the program was very rewarding and highlighted the potential in dynamic camera-based unimodal emotion detection systems. 

One of the more difficult parts of developings aid program was learning how to do all of these processes from scratch, primarily relying on Youtube videos and online documentations to cement understanding in the different moving parts of the program. GeeksForGeeks in particular remains an excellent introduction to the process of converting live video into modifiable image data. 

Another difficulty of note was working with the limited CV2 GUI applications, particularly with the font size and display; most especially with regards to the emotion detection. 

What worked well was the integration of facial landmarks, which accurately displayed and mapped all of the landmarks, albeit limited to one person only. 

Overall, the unimodal emotion detection tool is very efficient at interpreting the current emotions, but requires a much more extensive UI display capability and compat. 


## Further implementations
While not directly implemented, the program can easily be expanded to support the following facial interpretation models:
*["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet", "Buffalo_L"]*

Alternatively, it can also be expanded to potentially support any of the DeepFace submodels *["age_model_weights.h5", "arcface_weights.h5", "deepid_keras_weights.h5", "facenet512_weights.h5", "facenet_weights.h5", "facial_expression_model_weights.h5", "gender_model_weights.h5", "openface_weights.h5", "race_model_single_batch.h5", "retinaface.h5"]*

## Artificial Intelligence Usage
Artificial intelligence was primarily used to brainstorm and recommend the potential areas for learning, particularly in recommending articles and potential libraries that could be used (Mostly ```cv2``` applications). All other parts of the program were developed through manual programming and online documentation. 