import cv2 as cv                                    #to capture slide and webcam
import os                                           #to add the folder with current program
from cvzone.HandTrackingModule import HandDetector  #to detect hands movement and gestures
import numpy as np                                  #to calculate co-ordinates
import speech_recognition as sr                     #to implement voice controls

# Variables 
 
width, height = 1000, 600
folderPath = r"C:\Users\hp\Desktop\New folder"

cap = cv.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

path_images = sorted(os.listdir(folderPath), key=len)
imgNumber = 0
hs, ws = int(120 * 1.8), int(213 * 1.8)

gestureThreshold = 350
buttonpressed = False
buttoncounter = 0
delaycounter = 20

#variables used in drawing and erasing 
annotations = [[]]
annotationNumber = -1
annotationStart = False
voiceAnnotation = False

#implement zoom functioning
zoomScale = 1.0
zoomCenter = (width // 2, height // 2)
initialDistance = None

# function to detect hands
detector = HandDetector(detectionCon=0.85, maxHands=2)


#function to get voice recognition
def get_voice_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for voice command...")
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print("You said:", text)
            return text.lower()
        except Exception as e:
            print("Could not understand audio:", e)
            return ""


#implment the loop when hands are detected
while True:

    success, img = cap.read()
    img = cv.flip(img, 1)

    pathFullImage = os.path.join(folderPath, path_images[imgNumber])    #retrun the list of images in sorted order from given folder path
    imgCurrent = cv.imread(pathFullImage)

    hands, img = detector.findHands(img)    #detect hands

    cv.line(img, (0, gestureThreshold), (width, gestureThreshold), (255, 0, 0), 5) #threshold line to prevent disturbance

    if hands and not buttonpressed:
        hand = hands[0]
        fingers = detector.fingersUp(hand)  #detect fingertips and return a list 

        #drawing portions
        #  
        x, y = hand['center']
        lmList = hand['lmList']

        xVal = int(np.interp(lmList[8][0], [0, width], [0, width])) #represent index finger
        yVal = int(np.interp(lmList[8][1], [150, height - 150], [0, height])) #represent middle finger
        indexFinger = xVal, yVal

        # When hands are above the threshold line
        if y <= gestureThreshold:

            # Gesture 01 Move to previous slide
            if fingers == [1, 0, 0, 0, 0] and imgNumber > 0:
                buttonpressed = True

                #To remove all drawings when you move to next page
                annotations = [[]]
                annotationNumber = -1
                annotationStart = False
                imgNumber -= 1


            # Gesture 02 Move to next slide
            if fingers == [0, 0, 0, 0, 1] and imgNumber < len(path_images) - 1:
                buttonpressed = True
                annotations = [[]]
                annotationNumber = -1
                annotationStart = False
                imgNumber += 1

            if fingers == [0, 1, 0, 0, 1]:  # Gesture 03 Exit the window
                break
        
        # Gesture 04 Show pointer used for drawing 
        if fingers == [0, 1, 1, 0, 0]:
            cv.circle(imgCurrent, indexFinger, 22, (0, 0, 255), cv.FILLED)

        # Gesture 05 Draw with Indexfinger 
        if fingers == [0, 1, 0, 0, 0]:
            if not annotationStart:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])

            cv.circle(imgCurrent, indexFinger, 22, (0, 0, 255), cv.FILLED)
            annotations[annotationNumber].append(indexFinger)

        else:
            annotationStart = False

        # Gesture 06 Erase last stroke 
        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonpressed = True

        #Gesture 07 Zoom current slide
        if fingers == [1, 1, 0, 0, 0]:  
            x1, y1 = lmList[4][0], lmList[4][1]
            x2, y2 = lmList[8][0], lmList[8][1]
            currentDistance = np.hypot(x2 - x1, y2 - y1)

            if initialDistance is None:
                initialDistance = currentDistance

            zoomFactor = currentDistance / initialDistance
            zoomScale = np.clip(zoomFactor, 0.5, 2.0)
            zoomCenter = ((x1 + x2) // 2, (y1 + y2) // 2)

        else:
            initialDistance = None

    else:
        annotationStart = False

    # process to generate a delay when traversing slides  
    if buttonpressed:
        buttoncounter += 1
        if buttoncounter > delaycounter:
            buttoncounter = 0
            buttonpressed = False

    #draw line when and add it to annottion list  
    for i in range(len(annotations)):
        for j in range(len(annotations[i])):
            if j != 0:
                cv.line(imgCurrent, annotations[i][j - 1], annotations[i][j], (0, 125, 23), 12)

    # resize the slide
    imgCurrent = cv.resize(imgCurrent, (width, height))
    
    #adding webcam on slide
    
    imgSmall=cv.resize(img,(ws,hs))
    h,w,_ =imgCurrent.shape

    imgCurrent[0:hs, w-ws:w]= imgSmall

    #Resize imgCurrent based on zoomScale
    zoomedSlide = cv.resize(imgCurrent, None, fx=zoomScale, fy=zoomScale)

    # Create a black canvas of the original size
    canvas = np.zeros_like(imgCurrent)

    zh, zw = zoomedSlide.shape[:2]
    cx, cy = zoomCenter  # Center of zoom (usually width//2, height//2)

    # Coordinates to crop from zoomed image
    startX = zw // 2 - width // 2
    startY = zh // 2 - height // 2

    # Clamp to valid range
    startX = max(0, startX)
    startY = max(0, startY)
    endX = startX + width
    endY = startY + height

    # Crop the zoomed slide centered
    croppedZoom = zoomedSlide[startY:endY, startX:endX]

    # If cropped area is smaller than canvas (e.g., zoom out too much), pad it
    finalZoom = np.zeros_like(imgCurrent)
    ch, cw = croppedZoom.shape[:2]
    finalZoom[0:ch, 0:cw] = croppedZoom

    imgCurrent = finalZoom

    #Important windows 
    cv.imshow("Image", img)
    cv.imshow("Slide", imgCurrent)

    key = cv.waitKey(1) #waiting key


    #break the loop
    if key == ord('q'):
        break

    # acessing voice controls with by pressing 'v'    
    if key == ord('v'):
        command = get_voice_command()

        if "next slide" in command and imgNumber < len(path_images) - 1:
            imgNumber += 1
            print("Next Slide")

        elif "previous slide" in command and imgNumber > 0:
            imgNumber -= 1
            print("Previous Slide")

        elif "zoom in" in command:
            zoomScale = min(zoomScale + 0.1, 2.0)
            print("Zoom In")

        elif "zoom out" in command:
            zoomScale = max(zoomScale - 0.1, 0.5)
            print("Zoom Out")

        elif "exit" in command or "quit" in command:
            print("Exiting by voice command.")
            break

        elif "start drawing" in command:
            voiceAnnotation = True
            annotationNumber += 1
            annotations.append([])
            print("Started drawing by voice.")

        elif "stop" in command:
            voiceAnnotation = False
            print("Stopped drawing by voice.")

        elif "back" in command:
            if annotationNumber >= 0 and annotations[annotationNumber]:
                annotations[annotationNumber].pop(-1)
                print("Removed last point.")

        elif "erase" in command:
            annotations = [[]]
            annotationNumber = -1
            annotationStart = False
            print("All drawings erased.")

#Release the camera and close all windows   
cap.release()
cap.destroyAllWindows()
