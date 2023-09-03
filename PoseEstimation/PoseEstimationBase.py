import cv2
import mediapipe as mp
import time
import keyboard

cap = cv2.VideoCapture("PoseEstimation/video1.mp4")

pTime = 0
cTime = 0

mpDraw = mp.solutions.drawing_utils
mpPose  = mp.solutions.pose
pose = mpPose.Pose()

while True:
    success, img = cap.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            if id == 0:
                cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)
    
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 3, (255, 0, 255), 3)
    
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    
    if keyboard.is_pressed("space"):
        break