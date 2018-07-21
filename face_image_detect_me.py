import cv2
import numpy as np 


def main():
	face_cascade = cv2.CascadeClassifier('cascades\\haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('cascades\\haarcascade_eye.xml')
	frame = cv2.imread('images/harshita/3.jpg')

	gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3,minNeighbors= 5)
	for (x,y,w,h) in faces:
		    
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[y:y+h, x:x+w]
		img_item= "my_image.jpg"
		cv2.imwrite(img_item,roi_color)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
		    cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)


	cv2.imshow('Frame',frame)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()