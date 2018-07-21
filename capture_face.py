import cv2
import numpy as np 


def main():
	face_cascade = cv2.CascadeClassifier('cascades\\haarcascade_frontalface_default.xml')
	cap = cv2.VideoCapture(0)

	while(True):
		ret,frame = cap.read()
		
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3,minNeighbors= 5)
		
		for (x,y,w,h) in faces: 
		    roi_gray = gray[y:y+h, x:x+w]
		    roi_color = frame[y:y+h, x:x+w]
		    cv2.imwrite("image_detected_face.jpg",roi_color)
		    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

		cv2.imshow('Frame',frame)

		if cv2.waitKey(1)==27:
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()