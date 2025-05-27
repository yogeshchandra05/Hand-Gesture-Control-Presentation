from pickle import FALSE
import cv2 as cv
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from pyparsing import annotations
import re 
import speech_recognition as sr  # Library for speech recognition

# ========== VARIABLE INITIALIZATION ==========
width, height = 1000, 800  # Dimensions for the display window
folderPath = r"C:\Users\Puran\Documents\Presentation\presentation"  # Path to presentation slides

# ========== CAMERA SETUP ==========
cap = cv.VideoCapture(0)  # Initialize webcam
cap.set(3, width)  # Set camera width
cap.set(4, height)  # Set camera height

# ========== SLIDE MANAGEMENT ==========
path_images = sorted(os.listdir(folderPath), key=len)  # Get sorted list of slide images

# ========== PRESENTATION CONTROL VARIABLES ==========
imgNumber = 0  # Current slide index
hs, ws = int(120*1.5), int(213*1.5)  # Webcam overlay dimensions
gestureThreshold = 400  # Y-coordinate threshold for gesture activation
buttonpressed = False  # Flag for button press delay
buttoncounter = 0  # Counter for button delay
delaycounter = 20  # Delay duration
annotations = [[]]  # Stores drawing annotations
annotationNumber = -1  # Current annotation index
annotationStart = False  # Flag for drawing state

# ========== ZOOM CONTROL VARIABLES ==========
zoomScale = 1.0  # Current zoom level
zoomCenter = (width // 2, height // 2)  # Center point for zoom
initialDistance = None  # Initial distance between fingers for zoom

# ========== HAND DETECTION SETUP ==========
detector = HandDetector(detectionCon=0.8, maxHands=2)  # Hand detector with confidence threshold

# ========== VOICE COMMAND FUNCTION ==========
def get_voice_command():
    """Listen for and process voice commands using microphone input"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for voice command...")
        r.adjust_for_ambient_noise(source, duration=1)  # Reduce background noise
        audio = r.listen(source)  # Capture audio input
        try:
            text = r.recognize_google(audio)  # Convert speech to text
            print("You said:", text)
            return text.lower()  # Return lowercase command
        except Exception as e:
            print("Could not understand audio:", e)
            return ""  # Return empty string on error

# ========== MAIN PRESENTATION LOOP ==========
while True:
    # ========== IMAGE PROCESSING ==========
    success, img = cap.read()  # Capture frame from camera
    img = cv.flip(img,1)  # Mirror the image for natural movement

    # ========== LOAD CURRENT SLIDE ==========
    pathFullImage = os.path.join(folderPath, path_images[imgNumber])
    imgCurrent = cv.imread(pathFullImage)

    # ========== HAND DETECTION ==========
    hands, img = detector.findHands(img)  # Detect hands in the frame
    cv.line(img, (0, gestureThreshold), (width, gestureThreshold), (255, 0, 0), 5)  # Draw gesture threshold line

    # ========== GESTURE PROCESSING ==========
    if hands and buttonpressed == False:
        hand = hands[0]  # Get primary hand
        fingers = detector.fingersUp(hand)  # Get finger states (up/down)
        x, y = hand['center']  # Hand center coordinates
        lmList = hand['lmList']  # Landmark positions

        # ========== FINGER POSITION MAPPING ==========
        xVal = int(np.interp(lmList[8][0], [0, width], [0, width]))  # Index finger X position
        yVal = int(np.interp(lmList[8][1], [0, height], [0, height]))  # Index finger Y position
        indexFinger = xVal, yVal  # Index finger coordinates

        # ========== GESTURE ACTIONS ==========
        if y <= gestureThreshold:  # Only process gestures above threshold line
            # GESTURE 1: Thumb up - Previous slide
            if fingers == [1, 0, 0, 0, 0]:
                if imgNumber > 0:
                    buttonpressed = True
                    annotations = [[]]  # Clear annotations
                    annotationNumber = -1
                    annotationStart = False
                    imgNumber -= 1  # Move to previous slide

            # GESTURE 2: Pinky up - Next slide
            if fingers == [0, 0, 0, 0, 1]:
                if imgNumber < len(path_images) - 1:
                    buttonpressed = True
                    annotations = [[]]  # Clear annotations
                    annotationNumber = -1
                    annotationStart = False
                    imgNumber += 1  # Move to next slide

            # GESTURE 3: Middle finger up - Exit presentation
            if fingers == [0, 0, 1, 0, 0]:
                break  # Exit loop

        # GESTURE 4: Index and middle up - Show pointer
        if fingers == [0, 1, 1, 0, 0]:
            cv.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv.FILLED)  # Draw pointer

        # GESTURE 5: Index up - Drawing mode
        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])  # Start new annotation path
            cv.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv.FILLED)  # Draw circle
            annotations[annotationNumber].append(indexFinger)  # Store position
        else:
            annotationStart = False  # Exit drawing mode

        # GESTURE 6: Index, middle, ring up - Erase last annotation
        if fingers == [0, 1, 1, 1, 0]:
            if annotations:
                annotations.pop(-1)  # Remove last annotation
                annotationNumber -= 1
                buttonpressed = True

        # GESTURE 7: Thumb and index up - Zoom control
        if fingers == [1, 1, 0, 0, 0]:
            x1, y1 = lmList[4][0], lmList[4][1]  # Thumb tip
            x2, y2 = lmList[8][0], lmList[8][1]  # Index tip
            currentDistance = np.hypot(x2 - x1, y2 - y1)  # Distance between fingers

            if initialDistance is None:
                initialDistance = currentDistance  # Set initial distance

            zoomFactor = currentDistance / initialDistance  # Calculate zoom factor
            zoomScale = np.clip(zoomFactor, 1.0, 2.5)  # Limit zoom range
        elif fingers == [0, 0, 0, 0, 0]:  # Reset zoom when no fingers are up
             # Reset zoom when gesture ends
            if initialDistance is not None:  # Only reset if we were zooming
                zoomScale = 1.0  # Reset to normal view
                zoomCenter = (width // 2, height // 2)  # Reset center
                initialDistance = None  # Clear the initial distance
    

    else:
        annotationStart = False  # No hands detected - exit drawing mode

    # ========== BUTTON PRESS DELAY HANDLING ==========
    if buttonpressed:
        buttoncounter += 1
        if buttoncounter > delaycounter:
            buttoncounter = 0
            buttonpressed = False

    # ========== DRAW ANNOTATIONS ==========
    for i in range(len(annotations)):
        for j in range(len(annotations[i])):
            if j != 0:
                # Draw line between consecutive points
                cv.line(imgCurrent, annotations[i][j-1], annotations[i][j], (0, 125, 23), 12)

    # ========== SLIDE DISPLAY PROCESSING ==========
    imgCurrent = cv.resize(imgCurrent, (width, height))  # Resize slide to window
    
    # ========== WEBCAM OVERLAY ==========
    imgSmall = cv.resize(img, (ws, hs))  # Resize webcam feed
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w-ws:w] = imgSmall  # Overlay webcam on slide

    # ========== ZOOM PROCESSING ==========
    zoomedSlide = cv.resize(imgCurrent, None, fx=zoomScale, fy=zoomScale)  # Apply zoom
    canvas = np.zeros_like(imgCurrent)  # Create blank canvas

    zh, zw = zoomedSlide.shape[:2]
    cx, cy = zoomCenter  # Get zoom center

    # Calculate crop area for zoomed image
    startX = zw // 2 - width // 2
    startY = zh // 2 - height // 2
    startX = max(0, startX)  # Ensure within bounds
    startY = max(0, startY)
    endX = startX + width
    endY = startY + height

    # Crop and display zoomed area
    croppedZoom = zoomedSlide[startY:endY, startX:endX]
    finalZoom = np.zeros_like(imgCurrent)
    ch, cw = croppedZoom.shape[:2]
    finalZoom[0:ch, 0:cw] = croppedZoom
    imgCurrent = finalZoom

    # ========== STATUS DISPLAY ==========
    # Show current slide number
    cv.putText(imgCurrent, f"Slide: {imgNumber + 1}/{len(path_images)}", (40, 60), 
              cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Show current zoom level
    cv.putText(imgCurrent, f"Zoom: {zoomScale:.1f}x", (40, 110), 
              cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Show drawing mode status
    if annotationStart:
        cv.putText(imgCurrent, "Drawing...", (40, 160), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # Show voice command prompt
    cv.putText(imgCurrent, "Press 'V' for Voice Command", (40, height - 40), 
              cv.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 2)

    # ========== DISPLAY WINDOWS ==========
    cv.imshow("Image", img)  # Show camera feed
    cv.imshow("Slide", imgCurrent)  # Show presentation slide

    # ========== KEYBOARD INPUT HANDLING ==========
    key = cv.waitKey(1)
    
    if key == ord('q'):  # Quit on 'q' key
        break

    # ========== VOICE COMMAND HANDLING ==========
    if key == ord('v'):  # Voice command on 'v' key
        command = get_voice_command()
        
        # Navigation commands
        if "next" in command:
            if imgNumber < len(path_images) - 1:
                imgNumber += 1
                print("Next Slide")
                
        elif "previous" in command or "back" in command:
            if imgNumber > 0:
                imgNumber -= 1
                print("Previous Slide")
                
        # NEW: Slide jump command (e.g., "slide 5")
        elif "slide" in command:
            try:
                # Extract number from command (e.g., "slide 3" -> 3)
                slide_num = int(command.split()[-1]) - 1  # Convert to 0-based index
                if 0 <= slide_num < len(path_images):
                    imgNumber = slide_num
                    print(f"Jumped to Slide {slide_num + 1}")
                else:
                    print("Invalid slide number")
            except (ValueError, IndexError):
                print("Could not parse slide number")
                
        # Zoom commands
        elif "zoom in" in command:
            zoomScale = min(zoomScale + 0.1, 2.0)
            print("Zoom In")
            
        elif "zoom out" in command:
            zoomScale = max(zoomScale - 0.1, 0.5)
            print("Zoom Out")
            
        elif "reset zoom" in command:
            zoomScale = 1.0
            print("Zoom Reset")
            
        # Exit command
        elif "exit" in command or "quit" in command:
            print("Exiting by voice command.")
            break
        # move slide to specific no. 
         elif "slide" in command:
            numbers = re.findall(r'\d+', command)
            if numbers:
                slide_num = int(numbers[0]) - 1  # Convert to 0-based index
                if 0 <= slide_num < len(path_images):
                    imgNumber = slide_num
                    print(f"Jumping to Slide {slide_num + 1}")
                    annotations = [[]]
                    annotationNumber = -1
                    annotationStart = False
                else:
                    print("Slide number out of range.")
            else:
                print("Couldn't detect a slide number.")

# ========== CLEANUP ==========
cap.release()  # Release camera
cv.destroyAllWindows()  # Close all OpenCV windows
