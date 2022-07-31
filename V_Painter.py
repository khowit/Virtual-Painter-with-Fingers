import  cv2
import numpy as np
import Hand_tracking as htm
import  os

brushThickness = 15
eraserThickness = 100

folderPath = "pain"
myList = os.listdir(folderPath)
overlayList = []

for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)

header = overlayList[0]
drawColor = (255,0,255)


cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

detector = htm.handDetector()
xp, yp = 0,0
imgCanvas = np.zeros((720,1280,3), np.uint8)

while True:
    check, frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame)
    
    if len(lmList) != 0:       
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()
        # print(fingers)

        if fingers[1] and fingers[2]:
            xp, yp = 0,0
            print("selection mode")
            if y1 < 125:
                if 250<x1<350:
                    header = overlayList[0]
                    drawColor = (255,0,255)
                elif 450<x1<600:
                    header = overlayList[1]
                    drawColor = (0,0,255)
                elif 650<x1<850:
                    header = overlayList[2]
                    drawColor = (255,0,0)
                elif 900<x1<1000:
                    header = overlayList[3]
                    drawColor = (0,255,0)
                elif 1050<x1<1200:
                    header = overlayList[4]
                    drawColor = (0,0,0)
            cv2.rectangle(frame, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        if fingers[1] and fingers[2]==False:
            cv2.circle(frame, (x1,y1), 15, drawColor, cv2.FILLED)
            print("Drawing")

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0,0,0):
                cv2.line(frame, (xp,yp), (x1,y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp,yp), (x1,y1), drawColor, eraserThickness)
            else:
                cv2.line(frame, (xp,yp), (x1,y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp,yp), (x1,y1), drawColor, brushThickness)
            xp, yp = x1, y1

    gray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, frameInv = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    frameInv = cv2.cvtColor(frameInv, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame,frameInv)
    frame = cv2.bitwise_or(frame,imgCanvas)
    
    frame[0:125,0:1280] = header
    # frame = cv2.addWeighted(frame,0.5,imgCanvas,0.5)

    cv2.imshow("Output", frame)
    cv2.imshow("Canvas", imgCanvas)
    cv2.imshow("CanvasInv", frameInv)
    if cv2.waitKey(1) & 0xFF == ord("e"):
        break


cap.release()
cv2.destroyAllWindows()





