import cv2
import numpy as np 
import pickle


def main():
	face_cascade = cv2.CascadeClassifier('cascades\\haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('cascades\\haarcascade_eye.xml')

	recognizer = cv2.face.LBPHFaceRecognizer_create()
	recognizer.read("face-trainner.yml")

	labels = {}
	labels_og={}

	with open("labels.pickle",'rb') as f:
		labels_og = pickle.load(f)
		labels = {v:k for k,v in labels_og.items()}

	cap = cv2.VideoCapture(0)

	while(True):
		ret,frame = cap.read()
		gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3,minNeighbors= 5)
		for (x,y,w,h) in faces:
		    
		    roi_gray = gray[y:y+h, x:x+w]
		    roi_color = frame[y:y+h, x:x+w]

		    id_,conf = recognizer.predict(roi_gray)
		    if conf>=4 and conf<=85:
		    	# print(id_,labels[id_])
		    	cv2.putText(frame,labels[id_],(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

		    # img_item= "my_image.jpg"
		    # cv2.imwrite(img_item,frame)
		    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		    eyes = eye_cascade.detectMultiScale(roi_gray)
		    for (ex,ey,ew,eh) in eyes:
		    	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
		cv2.imshow('Frame',frame)

		if cv2.waitKey(1)==27:
			break

	cap.release()

	cv2.destroyAllWindows()

if __name__ == '__main__':
	main()