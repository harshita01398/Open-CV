import cv2
import numpy as np 

def main():
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))
	cap = cv2.VideoCapture(0)

	if cap.isOpened():
		ret, frame = cap.read()
	else:
		ret = False

	while(ret):
		ret, frame = cap.read()

		hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
		# Blue Color
		low = np.array([100, 50, 50])
		high = np.array([125, 255, 255])
		blue = cv2.inRange(hsv,low,high)
		blue = cv2.dilate(blue, np.ones((3, 3), np.uint8), iterations=2 )
		erodedb = cv2.erode(blue, np.ones((3, 3), np.uint8), iterations=1 )


		#Red color
		# low = np.array([136,87,111])
		# high = np.array([180,255,255])
		low = np.array([155,70,50])
		high = np.array([180, 255, 255])
		red = cv2.inRange(hsv,low,high)
		# red1 = cv2.inRange(hsv,low,np.array([10,255,255]))
		# red = cv2.addWeighted(red1,1.0,red2,1.0,0.0)
		# red = red1|red2
		red = cv2.dilate(red, np.ones((3, 3), np.uint8), iterations=1 )
		erodedr = cv2.erode(red, np.ones((3, 3), np.uint8), iterations=1 )

		#Green Color
		low = np.array([40, 100, 100])
		high = np.array([100, 255, 255])
		green = cv2.inRange(hsv,low,high)
		green = cv2.dilate(green, np.ones((3, 3), np.uint8), iterations=2 )
		erodedg = cv2.erode(green, np.ones((3, 3), np.uint8), iterations=1 )


		# Yellow Color
		low = np.array([20, 124, 123])
		high = np.array([30, 255, 255])
		yellow = cv2.inRange(hsv,low,high)
		yellow = cv2.dilate(yellow, np.ones((8, 8), np.uint8), iterations=1 )
		erodedy = cv2.erode(yellow, np.ones((3, 3), np.uint8), iterations=1 )
        
		output = cv2.bitwise_and(frame,frame,mask = green+blue+red+yellow)

		img, c, h = cv2.findContours(erodedb, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# print(c)
		# cv2.drawContours(output, c, -1, (255, 0, 0), 2)
		for cnt in c:
			M = cv2.moments(cnt)
			if(M['m00']>800):
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])
				text1 = str(cx)+", "+str(cy)
				cv2.putText(output,text1,(cx,cy+20),1,1,(255, 0, 0))
				cv2.putText(output,"blue",(cx,cy-20),1,2,(255, 0, 0)) 
				cv2.drawContours(output, c, -1, (255, 0, 0), 2)

		img, c, h = cv2.findContours(erodedr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# cv2.drawContours(output, c, -1, (0, 0, 255), 2)
		for cnt in c:
			M = cv2.moments(cnt)
			if(M['m00']>800):
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])
				text1 = str(cx)+", "+str(cy)
				cv2.putText(output,text1,(cx,cy+20),1,1,(0, 0, 255))
				cv2.putText(output,"red",(cx,cy-20),1,2,(0, 0, 255))
				cv2.drawContours(output, c, -1, (0, 0, 255), 2)

		img, c, h = cv2.findContours(erodedg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# cv2.drawContours(output, c, -1, (0, 255, 0), 2)
		for cnt in c:
			M = cv2.moments(cnt)
			if(M['m00']>800):
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])
				text1 = str(cx)+", "+str(cy)
				cv2.putText(output,text1,(cx,cy+20),1,1,(0, 255, 0))
				cv2.putText(output,"green",(cx,cy-20),1,2,(0, 255, 0))
				cv2.drawContours(output, c, -1, (0, 255, 0), 2)

		img, c, h = cv2.findContours(erodedy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		# cv2.drawContours(output, c, -1, (0, 255, 255), 2)
		for cnt in c:
			M = cv2.moments(cnt)
			if(M['m00']>800):
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])
				text1 = str(cx)+", "+str(cy)
				cv2.putText(output,text1,(cx,cy+20),1,1,(0, 255, 255))
				cv2.putText(output,"yellow",(cx,cy-20),1,2,(0, 255, 255))
				cv2.drawContours(output, [cnt], 0, (0, 255, 255), 2)
		



		# cv2.imshow("Image Mask",green)
		cv2.imshow("Original feed",frame)
		cv2.imshow("Color Tracking",output)
		out.write(output)

		if cv2.waitKey(1)==27:
			break

	cv2.destroyAllWindows()
	cap.release()

if __name__=="__main__":
	main()
