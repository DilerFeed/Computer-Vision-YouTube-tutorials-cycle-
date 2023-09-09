import cv2
import numpy as np
import time
import keyboard
import PoseEstimationModule as pem

cap = cv2.VideoCapture("PoseEstimation/AITrainer/video1.mp4")

pTime = 0
cTime = 0

detector = pem.PoseDetector()
count = 0
dir = 0

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 720))
    #img = cv2.imread("PoseEstimation/AITrainer/test.png")
    #img = cv2.resize(img, (1280, 720))
    
    img = detector.findPose(img, False)
    lmList = detector.findPosition(img, False)
    #print(lmList)
    
    if len(lmList) != 0:
        # Left Arm
        angle = detector.findAngle(img, 11, 13, 15)
        # Right Arm
        #detector.findAngle(img, 12, 14, 16)
        
        per = np.interp(angle, (210, 310), (0, 100)) # video1
        bar = np.interp(angle, (210, 310), (650, 100))
        
        #per = np.interp(angle, (200, 280), (0, 100)) # video2
        #bar = np.interp(angle, (200, 280), (650, 100))
        #print(angle, per)
        
        # Check for the dumbbell curls
        color = (255, 0, 255)
        if per > 95:
            color = (0, 0, 255)
            if dir == 0:
                count += 0.5
                dir = 1
        if per < 5:
            color = (0, 255, 0)
            if dir == 1:
                count += 0.5
                dir = 0
        print(count)
        
        # Draw bar
        cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
        cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
        cv2.putText(img, f'{int(per)}%', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 
                    4, color, 4)
        
        # Draw curl count
        cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 
                    15, (255, 0, 0), 25)
        
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                3, (255, 0, 255), 3)
    
    cv2.imshow("AI Trainer", img)
    cv2.waitKey(2)
    
    if keyboard.is_pressed("space"):
        break