import cv2 
import mediapipe as mp  

#==========================================================================================================
#Copied from OpenCV for learning purposes.
#
#Open the webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam
#==========================================================================================================

#==========================================================================================================
#Copied from google MediaPipe for learning purposes.
#
#mediaPipe model setup
BaseOptions = mp.tasks.BaseOptions  # Base configuration for MediaPipe models
HandLandmarker = mp.tasks.vision.HandLandmarker  # Hand tracking model class
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions  # Options for the model
VisionRunningMode = mp.tasks.vision.RunningMode  # Defines how the model runs (IMAGE / VIDEO / LIVE STREAM)

#path to your trained/downloaded hand model
model_path = "./hand_landmarker.task"

#configure the hand landmarker
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),  # Load model from file
    running_mode=VisionRunningMode.VIDEO,  # We are processing video frames
    num_hands=2  # Detect up to 2 hands
)
#==========================================================================================================

points = set()

#start MediaPipe model
with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        #read one frame from the webcam
        success, img = cap.read()
        #print(img.shape[0])
        #print(img.shape[1])

        #if frame not read correctly, break loop
        if not success:
            break

        #flip image horizontally (mirror effect like a selfie)
        img = cv2.flip(img, 1)

        #convert image from BGR (OpenCV format) to RGB (MediaPipe format)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #convert NumPy image into MediaPipe Image object
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img_rgb
        )

        #create timestamp in milliseconds (required for VIDEO mode)
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)

        #run hand landmark detection on the current frame
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        #drawing
        #if hands are detected
        if result.hand_landmarks:
            #show number of detected hands on screen
            cv2.putText(
                img,
                f"Hands detected: {len(result.hand_landmarks)}",
                (20, 50),  #position on screen
                cv2.FONT_HERSHEY_SIMPLEX,  #font type
                0.7,  #font size
                (0, 255, 0),  #color (BGR not RGB)
                1  #thickness
            )
            #print(result.hand_landmarks)
            
            hand_vertexes = result.hand_landmarks[0]
            finger_tip = hand_vertexes[4] #thumb tip
            finger_tip = hand_vertexes[8] #index_tip
            points.add((int(finger_tip.x*640), int(finger_tip.y*480)))
            
            
            #only popping up if there are hands.
            #maybe we have to save all the points that were selected?
            #whenever "if not result.hand_landmarkers" (no hands) we can clean the set of points.

            #img[10:20, 5:15] = [0, 0, 255] # paints red point on screen of size_y=10 size_x=10 (10-20, 5-15)
        else:
            points = set() #no hands -> we clean set
        
        #draw points
        print(len(points))
        for p in points:
            cv2.putText(
                img,
                f"*",
                #(y,x),
                (p[0],p[1]),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                (0,0,255),
                2
            )

        #show the webcam frame in a window
        cv2.imshow("Camera", img)

        #exit loop when pressing 'q'
        if cv2.waitKey(1) == ord('q'):
            break


#==========================================================================================================
#Copied from OpenCV for learning purposes.
cap.release()  #release webcam
cv2.destroyAllWindows()  #close all OpenCV windows
#==========================================================================================================