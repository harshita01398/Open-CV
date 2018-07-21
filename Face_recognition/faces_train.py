import os
from PIL import Image
import numpy as np
import cv2
import pickle

def main():

	face_cascade = cv2.CascadeClassifier('cascades\\haarcascade_frontalface_alt2.xml')
	recognizer = cv2.face.LBPHFaceRecognizer_create()

	x_train =[]
	y_labels = []
	label_ids = {}
	cur_id = 0

	base_dir = os.path.dirname(os.path.abspath(__file__))
	img_dir = os.path.join(base_dir,"images")

	for root,dirs,files in os.walk(img_dir):
		for file in files:
			if file.endswith('png') or file.endswith('jpg'):
				path = os.path.join(root,file)
				label = os.path.basename(os.path.dirname(path)).replace(' ','-').lower()

				if not label in label_ids:
					label_ids[label] = cur_id
					cur_id+=1
				id_ =  label_ids[label]
				# print(label,path)
				pil_img = Image.open(path).convert("L")
				size = (550,550)
				final_img = pil_img.resize(size,Image.ANTIALIAS)
				img_ar = np.array(final_img,'uint8')
				# print(img_ar)
				faces = face_cascade.detectMultiScale(img_ar,1.5,5)

				for (x,y,w,h) in faces:
					roi = img_ar[y:y+h,x:x+w]
					x_train.append(roi)
					y_labels.append(id_)

	# print(label_ids)					
	with open('labels.pickle','wb') as f:
		pickle.dump(label_ids,f)

	recognizer.train(x_train,np.array(y_labels))
	recognizer.save("face-trainner.yml")

if __name__ == '__main__':
	main()