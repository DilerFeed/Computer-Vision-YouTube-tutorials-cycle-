import cv2
import numpy as np
import HandTrackingModule as htm
import time
from pynput.mouse import Button, Controller
import keyboard
#from AppKit import NSScreen    # MacOS
import ctypes                   # Windows

wCam ,hCam = 1280, 720
#wScr, hScr = int(NSScreen.mainScreen().frame().size.width), \          # MacOS
#    int(NSScreen.mainScreen().frame().size.height)                     # MacOS
user32 = ctypes.windll.user32                                           # Windows
wScr, hScr = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)     # Windows
frameReduction = int(wCam * 0.16)

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

mouse = Controller()
smoothening, click_smoother = 7, 0
pLocX, pLocY, cLocX, cLocY = 0, 0, 0, 0
RMB_pressed, LMB_pressed = False, False

pTime = 0
cTime = 0

detector = htm.handDetector(maxHands=1, detectionCon=0.5)

while True:
    # Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    # Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x0, y0 = lmList[4][1:]
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
    
        # Check which fingers are up
        fingers = detector.fingersUp()
        cv2.rectangle(img, (frameReduction, frameReduction), 
                     ((wCam - frameReduction), (hCam - frameReduction)), 
                     (255, 0, 255), 2)
        
        # Only Index Finger: Moving Mode
        if fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0:
        
            # Convert Coordinates
            x3 = np.interp(x1, (frameReduction, (wCam - frameReduction)), (0, wScr))
            y3 = np.interp(y1, (frameReduction, (hCam - frameReduction)), (0, hScr))
        
            # Smoothen Values
            cLocX = pLocX + (x3 - pLocX) / smoothening
            cLocY = pLocY + (y3 - pLocY) / smoothening
            
            # Move Mouse
            mouse.position = ((wScr - cLocX), cLocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            pLocX, pLocY = cLocX, cLocY
            
            # Smoothen Clicks
            if click_smoother >= 30:
                RMB_pressed, LMB_pressed = False, False
            click_smoother += 1
        
        # Both Index and Middle fingers are up: Clicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # Find distance between Index and Middle fingers
            lenghtIM, img, lineInfoIM = detector.findDistance(8, 12, img)
            # Find distance between Middle and Ring fingers
            lenghtMR, img, lineInfoMR = detector.findDistance(12, 16, img)

            # Click RMB if distance between Index and Middle and Ring fingers short
            if RMB_pressed == False and fingers[3] == 1 and lenghtMR < 65 and lenghtIM < 65:
                cv2.circle(img, (lineInfoIM[4], lineInfoIM[5]),
                           15, (0, 255, 255), cv2.FILLED)
                cv2.circle(img, (lineInfoMR[4], lineInfoMR[5]),
                           15, (0, 255, 255), cv2.FILLED)
                mouse.click(Button.right, 1)
                RMB_pressed, LMB_pressed = True, False
            # Click LMB if distance between Index and Middle fingers short
            elif LMB_pressed == False and lenghtIM < 65:
                cv2.circle(img, (lineInfoIM[4], lineInfoIM[5]),
                           15, (0, 255, 0), cv2.FILLED)
                mouse.click(Button.left, 1)
                RMB_pressed, LMB_pressed = False, True
            click_smoother = 0
        
        # Scroll up/down if all fingers are closed
        if fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0:
            cv2.circle(img, (x0, y0), 15, (255, 0, 0), cv2.FILLED)
            if fingers[0] == 0:
                mouse.scroll(0, 2)
            else:
                mouse.scroll(0, -2)
    
    # Frame Rate
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, 
                (255, 0, 255), 3)
    
    # Display
    cv2.imshow("Hand Virtual Mouse", img)
    cv2.waitKey(1)
    
    if keyboard.is_pressed('space'):
        break