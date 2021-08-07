import cv2
import numpy as np 
import imutils

def main():

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        
        a = frame.shape[0]
        b = frame.shape[1]
        
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    
        lower_Red = np.array([160,100,100])
        upper_Red = np.array([179,255,255])
        
        mask = cv2.inRange(hsv,lower_Red,upper_Red)

        contours = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)

        areas = [cv2.contourArea(c) for c in contours]

        if len(areas) < 1:
            cv2.imshow("Frame",frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue
        else:
            max_index = np.argmax(areas)

        cnt = contours[max_index]
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
        
        x2 = x + int(w/2)
        y2 = y + int(h/2)
        cv2.circle(frame,(x2,y2),4,(0,255,0),-1)
        
        t = abs((x2 - int(b/2)))
        r = abs((y2 - int(a/2)))

        h, w, ch = frame.shape
        cv2.line(frame, (w//2,y2), (x2,y2), (0,255,0),1)
        cv2.line(frame, (x2,h//2), (x2,y2), (0,255,0),1)
        
        cv2.line(frame, (w//2,0), (w//2,h), (255,0,255),1)
        cv2.line(frame, (0,h//2), (w,h//2), (255,0,255),1)

        text = "x: " + str(x2) + ", y: " + str(y2)
        cv2.putText(frame,text,(x2-10,y2-10),cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0), 2)
        
        print("x eksenine uzaklık --> " + str(t) + " y eksenine uzaklık --> " + str(r))
        cv2.imshow("Frame",frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print(__doc__)
    main()
