import numpy as np
import cv2
# define classifier
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)
snap_number = 1

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x, y, w, h) in faces:
    	roi_color = frame[y:y+h, x:x+w]
    	color = (255, 0, 0) #BGR 0-255
    	stroke = 2
    	end_cord_x = x + w
    	end_cord_y = y + h
    	cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)


    if cv2.waitKey(20) & 0xFF == ord('s'):
    	snap_name = str(snap_number) + ".png"
    	cv2.imwrite(snap_name, roi_color)
    	print("snap taken:" + snap_name)
    	snap_number += 1


    # Display the resulting frame
    cv2.imshow('frame',frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break




# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()