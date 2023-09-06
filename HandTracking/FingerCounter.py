import cv2
import time
import os
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)

folderPath = "HandTracking/FingerImages"
myList = os.listdir(folderPath)
myList.remove('.DS_Store') # For MacOS
myList.sort()
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
    
print(len(overlayList))
pTime = 0

detector = htm.handDetector(maxHands=1, detectionCon=0.7)

tipIds = [4, 8, 12, 16, 20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    fingers = []
    if len(lmList) != 0:
        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # 4 other fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        #print(fingers)
        totalFingers = fingers.count(1)
        #print(totalFingers)
        
        h, w, c = overlayList[totalFingers-1].shape
        img[0:h, 0:w] = overlayList[totalFingers-1]
        
        cv2.rectangle(img, (20, 255), (170, 425), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 
                    10, (255, 0, 0), 25)
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (900, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                3, (255, 255, 255), 3)
    
    cv2.imshow("Finger Counter", img)
    cv2.waitKey(1)