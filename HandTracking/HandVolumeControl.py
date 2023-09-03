import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
import osascript

wCam, hCam = 640, 480

cap = cv2.VideoCapture(0)
"""
cap.set(3, wCam)
cap.set(4, hCam)
"""
pTime = 0
target_volume_hud = 400
target_volume = 0

detector = htm.handDetector(detectionCon = 0.7)

while True:
    success, img = cap.read()
    
    img = detector.findHands(img)
    
    lmList = detector.findPosition(img, draw = False)
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        
        lenght = math.hypot(x2 - x1, y2 - y1)
        
        #Hand range 50 - 400
        #Volume range 0 - 100
        target_volume = np.interp(lenght, [50, 400], [0, 100])
        target_volume_hud = np.interp(lenght, [50, 400], [400, 150])
        vol = "set volume output volume " + str(int(target_volume))
        osascript.osascript(vol)
        
        if lenght < 50:
            cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)
        if lenght > 400:
            cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)
            
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(target_volume_hud)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(target_volume)}%', (40, 450), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 3)
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 0, 0), 3)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)