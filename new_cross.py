import pandas as pd 
import numpy as np
import csv
import cv2,os,sys

df1= pd.read_csv('1233_full.csv') #YOLO_points
df2=pd.read_csv('lst_final_output.csv') #LANE_points

ddf1 = pd.DataFrame(df1)
ddf2 = pd.DataFrame(df2)



lst1=[]

cap = cv2.VideoCapture("/home/souravkc/Desktop/yolo-object-detection/videos/1233.avi")
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

i=0
list1=[]
list2=[]
for i in range(0,len(df2['lxmax'])):
	
	x1 = ddf2['lxmin'][i]
	#print(x1)
	list1.append(x1)
	i+=1
#print(list1)
#s1 = df2['rxmin']
#print(s1)


i1=0
for i1 in range(0,len(df2['lxmin'])):
	x2 = ddf2['rxmin'][i1]
	list2.append(x2)
	i1+=1


def Agg(lst): 
    return sum(lst) / len(lst) 


agg1 = Agg(list1)
agg2 = Agg(list2)

#print(avg1,avg2,list1,list2)



#print(x2)
#print(x1,x2)
dd1=[]
dd2=[]
# fdf = len(df2['lxmin'])
# print(fdf)
#for i in range(len(df1['lxmin'])):
dd1.append(df1['lxmin'].between(agg1, agg2,inclusive = False))
# print(dd1)

dff1= pd.DataFrame(dd1)
dff2= pd.DataFrame(df1)


#dff1.columns=['frame_no', 'lxmin','lxmax','rxmin','rxmax']
#print(dd1)

##################

dff3 = dff1.T
dd = pd.merge(dff2,dff3,left_index=True,right_index=True)
#dd = dff2.join([dff3])
#print(dd)

export_csv = dd.to_csv (r'1233_FINAL_newcross.csv', index = None, header=True) 

#gg = data[data[lxmin_y]]
#print(export_csv)
# with open("1233-FINAL.csv",'w') as file1:
# 	writer = csv.writer(file1)
# 	dd.to_csv(r)
# 	#writer.writerow(dd1)

data = pd.read_csv('1233_FINAL__newcross.csv')#, usecols =["lxmin_y"] )
#print(data)

true = data[data["lxmin_y"]==True]
#print(true)

ff_frames = list(true.frame_no)

frame_no = 1
try:
	os.mkdir("true_frames")
except OSError:
	pass

while(cap.isOpened()):
	ret, frame = cap.read()
	current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
	if ret == True:
		newimg = cv2.resize(frame , None, fx=0.5,fy=0.5)
		cv2.imshow('video',newimg)

		if current_frame in ff_frames:
			framename = "true_frames/" + str(current_frame) + ".jpg"
			cv2.imwrite(framename,frame)

		if cv2.waitKey(20)& 0xFF ==ord('q'):
			break
		frame_no+=1
	else:
		break

cap.release()
cv2.destroyAllWindows()

######################

#gg = data[data[lxmin_y]]
#print(export_csv)
# with open("1233-FINAL.csv",'w') as file1:
# 	writer = csv.writer(file1)
# 	dd.to_csv(r)
# 	#writer.writerow(dd1)


#df1[x3] 
#print(dff)

#ww=len(df1)

#print(ww)
#for i in range(ww):

			#dd2.append("outside-the-highway")
	#print(x3,"outside-the-highway")
			# readcsv= pd.read_csv('1233_full.csv')
			# readcsv.columns= ['frame_no', 'lxmin','lxmax','rxmin','rxmax']
			# readcsv['output']= dd1

			# readcsv.to_csv('1233_final-output.csv',index = False)

			# with open('1233_full.csv', 'w') as writeFile:
			# 	writer = csv.writer(writeFile)
			# 	writer.write(dd1)
			# df1['output'] = pd.Series(np.random.randn(sLength), index=df1.index)
			#writer['output'] = dd1
			#csv.write[dd1]= df1['output']
		
		# elif (x5==0):
		# 	print(x5,"no-vehicle")
'''
for x3 in range(0,260):
	print(x3,"vehicle-on-left")
			#dd2.append("vehicle-on-left")
			# readcsv= pd.read_csv('1233_full.csv')
			# readcsv.columns= ['frame_no', 'lxmin','lxmax','rxmin','rxmax']
			# readcsv['output']= dd1

			# readcsv.to_csv('1233_final-output.csv',index = False)

for x3 in range(260,960):
	print(x3,"lane-change")
		#dd2.append("vehicle-changing-lane")
for x3 in range(960,1300):
	print(x3,"right-vehicle")
		#dd2.append("vehicle-on-right")
	
	# elif (x5==0):
	# 	print("no-vehicle")
		#dd2.append("it's-on-the-line")
		
		#print(dd2)
		#ddd1= dd1
		#print(tuple(ddd1))
#dd1.append(dd2)
'''