from pickle import FALSE
import cv2 as cv
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from pyparsing import annotations
import speech_recognition as sr  # Add speech recognition


# variables

width, height=1000, 800
folderPath = r"C:\Users\hp\Desktop\New folder"


# camera setup

cap= cv.VideoCapture(0)

cap.set(3,width)
cap.set(4,height)

# get the list of presentation images 

path_images = sorted(os.listdir(folderPath),key=len)


# variables
imgNumber= 0
hs, ws=int(120*1.5),int(213*1.5)
gestureThreshold= 400
buttonpressed=False # used for button delay 
buttoncounter=0
delaycounter=20
annotations=[[]]
annotationNumber= -1
annotationStart=False

zoomScale = 1.0
zoomCenter = (width // 2, height // 2)
initialDistance = None

# hand detector
detector= HandDetector(detectionCon=0.8,maxHands=2)


# listen to voice commands
def get_voice_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening you ...")
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            print("You said:", text)
            return text.lower()
        except Exception as e:
            print("Could not understand audio:", e)
            return ""


while True:
    # import images

    success, img= cap.read()
    img=cv.flip(img,1)

    pathFullImage= os.path.join(folderPath,path_images[imgNumber])
    imgCurrent=cv.imread(pathFullImage)

    hands,img=detector.findHands(img)
    cv.line(img,(0,gestureThreshold),(width,gestureThreshold),(255,0,0),5)

    if hands and buttonpressed==False:
        hand= hands[0]
        fingers= detector.fingersUp(hand)
        x,y=hand['center']
        lmList= hand['lmList'] # landmark list


        # constraits value for easy movement (drawing)
        
        xVal= int(np.interp(lmList[8][0],[0,width],[0,width]))
        yVal= int(np.interp(lmList[8][1],[0,height],[0,height]))

        indexFinger= xVal,yVal

        

        if y<=gestureThreshold: #if hand is align with my face

            # gesture 1  left
            if fingers== [1,0,0,0,0]:
    
                if imgNumber>0:
                    buttonpressed= True

                    annotations=[[]]
                    annotationNumber= -1
                    annotationStart=False

                    imgNumber-=1
                    

            #gesture 2  right
            if fingers== [0,0,0,0,1]:
                if imgNumber<len(path_images)-1:
                    buttonpressed= True

                    annotations=[[]]
                    annotationNumber= -1
                    annotationStart=False
        
                    imgNumber+=1
                    
            #gesture close
            if fingers==[0,1,0,0,1]:
                break
            
        # Gesture 3 show pointer
        if fingers== [0,1,1,0,0]:
            cv.circle(imgCurrent,indexFinger,22,(0,0,255),cv.FILLED)
        
        # gesture 4 draw 
        if fingers==[0,1,0,0,0]:
            if annotationStart is False:
                annotationStart=True
                annotationNumber+=1
                annotations.append([]) # if not added give error as no list will be there to append

            cv.circle(imgCurrent,indexFinger,22,(0,0,255),cv.FILLED)
            annotations[annotationNumber].append(indexFinger)
        else:
            annotationStart=False    

        # gesture 5 erase
        if fingers==[0,1,1,1,0]:
            if annotations:
                annotations.pop(-1)
                annotationNumber-=1
                buttonpressed=True
        
        # Zoom gesture: Thumb and Index Up
        if fingers == [1, 1, 0, 0, 0]:
            x1, y1 = lmList[4][0], lmList[4][1]  # Thumb tip
            x2, y2 = lmList[8][0], lmList[8][1]  # Index tip
            currentDistance = np.hypot(x2 - x1, y2 - y1)

            if initialDistance is None:
                initialDistance = currentDistance

            zoomFactor = currentDistance / initialDistance
            zoomScale = np.clip(zoomFactor, 0.5, 2.0)  # Zoom between 0.5x and 2x
        else:
            initialDistance = None  # Reset when gesture not held
        
    else:
        annotationStart=False 

    #button pressed iterations
    if buttonpressed:
        buttoncounter+=1
        if buttoncounter>delaycounter:
            buttoncounter=0
            buttonpressed=False

    # draw a line freely
    for i in range (len(annotations)):
        for j in range(len(annotations[i])):
            if j!=0:
                cv.line(imgCurrent,annotations[i][j-1],annotations[i][j],(0,125,23),12)

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


    # Add to your while loop after imgCurrent is ready
    # Display Slide Number
    cv.putText(imgCurrent, f"Slide: {imgNumber + 1}/{len(path_images)}", (40, 60), 
            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Zoom Display
    cv.putText(imgCurrent, f"Zoom: {zoomScale:.1f}x", (40, 110), 
            cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Drawing Mode Indicator
    if annotationStart:
        cv.putText(imgCurrent, "Drawing...", (40, 160), 
                cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Voice command prompt
    cv.putText(imgCurrent, "Press 'V' for Voice Command", (40, height - 40), 
            cv.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)



    cv.imshow("Image",img)
    cv.imshow("Slide",imgCurrent)

    
    key=cv.waitKey(1)

    if key== ord('q'):
        break


    # Voice command trigger on pressing 'v'
    if key == ord('v'):
        command = get_voice_command()
        if "next" in command or "next slide" in command or "move to the next slide" in command or "next page" in command :
            if imgNumber < len(path_images) - 1:
                imgNumber += 1
                print("Next Slide")
                annotations = [[]]
                annotationNumber = -1
                annotationStart = False
        elif "previous" in command or "move to the last slide" in command or "last slide" in command or "back" in command or "previous slide" in command or "previous page" in command:
            if imgNumber > 0:
                imgNumber -= 1
                print("Previous Slide")
                annotations = [[]]
                annotationNumber = -1
                annotationStart = False
        elif "zoom in" in command:
            zoomScale = min(zoomScale + 0.1, 2.0)
            print("Zoom In")
        elif "zoom out" in command:
            zoomScale = max(zoomScale - 0.1, 0.5)
            print("Zoom Out")
        elif "undo" in command:
            if annotations and annotationNumber >= 0:
                annotations.pop(annotationNumber)
                annotationNumber -= 1
                print("Undo last stroke")
        elif "erase" in command or "clear" in command or "clear screen" in command or "remove" in command:
            annotations = [[]]
            annotationNumber = -1
            annotationStart = False
            print("Erase all drawing")
        elif "exit" in command or "done for today" in command or "close" in command or "thank you" in command or "bye bye" in command:
            print("Exiting by voice command.")
            break

#Release the camera and close all windows   
cap.release()
cv.destroyAllWindows()
