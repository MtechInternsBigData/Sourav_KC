# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os.path
#from os import path
import csv
import pandas as pd
import matplotlib.pyplot as plt


my_path = '/home/souravkc/Desktop/yolo-object-detection/'
my_file = my_path+"yolo_points.csv"

if os.path.isfile(my_file):
	print("file is present")
	os.remove(my_file)
	print("file is deleted")

#os.remove('')
# if path.exists('/home/souravkc/Desktop/yolo-object-detection/yolo_points_final.csv'):
# 	print("file is present")
# 	.remove('/home/souravkc/Desktop/yolo-object-detection/yolo_points_final.csv')
# 	#remove('/home/souravkc/Desktop/yolo-object-detection/yolo_points_final.csv')

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True,
	help="path to input video")
ap.add_argument("-o", "--output", required=True,
	help="path to output video")
ap.add_argument("-y", "--yolo", required=True,
	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

frame_no = 0

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

# loop over frames from the video file stream

while cv2.waitKey(1) < 0:
	frame_no+=1
	print(frame_no)

	# read the next frame from the file
	(grabbed, frame) = vs.read()

	
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		#cv2.waitKey(3000)
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	
	#print(frame.shape)
	#ff=cv2.CV_CAP_PROP_POS_FRAMES
		
	
	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	x1 = 0
	y1 = 450
	x2 = 1280
	y2 = 960

	frame = frame[y1:y2, x1:x2]

	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			
			classID = np.argmax(scores)
			#print(classID)
			#classID = np.(scores)
			confidence = scores[classID]
			#print(confidence)

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				if LABELS[classID] != 'car':
					continue
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)
				

				#print(centerX,centerY)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
		args["threshold"])

	#print(idxs)
	# lstt=[]
	# lstt.append(idxs)
	# lstt =[sub[0] for sub in lstt]
	# lstt1=[]
	# for sub in lstt:
	# 	lstt1.append(sub[0])
	# lstt=lstt1
	# print(lstt)
	#for i in range(len(lstt)):
		#print("{}:{}".format(i+1,lstt[0][i])
	#print(lstt[0][0])
	#print(lstt[0][2])
	# ensure at least one detection exists
	if len(idxs) > 0:
		# loop
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			#print(x)#,y,w,h)
			#tuple(lst_obj)
			#print(lst_obj)

			
			#with open('yolo_points.csv', 'a') as file_P:
				#writeit = csv.writer(file_P)
        
    			#writeit.writerows(map(lambda x: [x],all_lst))
        
        		#writeit.writerow([x,y,w,h,])



            
            #labell = str(classes[classIDs])

			# draw a bounding box rectangle and label on the frame
			#labell=int(labelsPath[classIDs])
			color = [int(c) for c in COLORS[classIDs[i]]]

			#labell = [int(d) for d in range(0,5) for i in str(classID)]
			#print(classIDs)
			
			#for i in range(0,int(idxs)):
			#frame_no= frame.count() 



			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

			#cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

			#text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			
			# ff=frame.get(CV_CAP_PROP_POS_FRAMES)
			# ff.show()
			text = "{}".format(i)
			cv2.putText(frame,text, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			
			cv2.line(frame,pt1=(0,450),pt2=(1280,450), color=(255,0,0), thickness =1,lineType = 8, shift =0 )
			# frame_no =+ 1
			# print(frame_no)
			#img = cv2.imread('all_input/*.jpg',cv2.IMREAD_COLOR)
            
            #pts = np.array([[535,425],[100,825],[1280,825],[775,425]], np.int32)

            #cv2.polylines(img, [pts], True, (0,255,255), 3)
            



			with open('yolo_points.csv', 'a') as file_P:

				writeit = csv.writer(file_P)
				#writeit.writerow([frame_no, x,y,x,y+h,x+w,y,x+w,y+h,text])
				writeit.writerow([frame_no, x,x,x+w,x+w])

			
			readcsv= pd.read_csv('yolo_points.csv')
			#readcsv.columns= ['frame_no', 'lxmin','lymin','lxmax','lymax','rxmin','rymin','rxmax','rymax','object_id']
			readcsv.columns= ['frame_no', 'lxmin','lxmax','rxmin','rxmax']
			readcsv.to_csv('1233_T.csv',index = False)


	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	# write the output frame to disk
	writer.write(frame)

	#print(x,y,"top-left")
	#print(x+w,y,"top-right")
	#print(x,y+h,"bottom-left")
	#print(x+w
print("[INFO eaning up...")
writer.release()
vs.release()