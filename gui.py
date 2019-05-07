#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtCore import Qt,QCoreApplication,QSize, QRectF
from PyQt5.QtGui import QPixmap,QIcon
from PyQt5.QtWidgets import (QLabel,QMainWindow,QAction,QMessageBox,QGraphicsScene,QPushButton, QVBoxLayout,
                             QGraphicsView,QWidget,QApplication,QScrollBar, QToolButton, QGridLayout, QHBoxLayout,QFileDialog)
from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from scipy.signal import find_peaks_cwt

from PIL import Image, ImageQt, ImageEnhance
import time
import numpy as np
import cv2
import pandas as pd
import math


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0,255)):
    red=img[:,:,0]
    # 1) Convert to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    sobelx = cv2.Sobel(red, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobelx = np.absolute(sobelx)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return sxbinary

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    red=img[:,:,0]
    # Apply the following steps to img
    # 1) Convert to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(red, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(red, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Calculate the magnitude 
    magnitude = np.sqrt(sobelx*sobelx + sobely*sobely)
    # 5) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*magnitude/np.max(magnitude))
    # 6) Create a binary mask where mag thresholds are met
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    # 7) Return this mask as your binary_output image
    return sxbinary

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    red=img[:,:,0]
    # Apply the following steps to img
    # 1) Convert to grayscale
    #gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(red, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(red, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    direction = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    sbinary = np.zeros_like(direction)
    sbinary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return sbinary

def hls_thresh(img,thresh=(0,255)):
    hls=cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    s_channel=hls[:,:,2]
    
    #Combine them
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def hsv_thresh(img,thresh=(0,255)):
    hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    v_channel=hsv[:,:,2]
    
    #Combine them
    binary_output = np.zeros_like(v_channel)
    binary_output[(v_channel > thresh[0]) & (v_channel <= thresh[1])] = 1
    return binary_output

def combo_thresh(img):
    x_thresholded=abs_sobel_thresh(img,orient='x',sobel_kernel=3,thresh=(12,120))
    y_thresholded=abs_sobel_thresh(img,orient='y',sobel_kernel=3,thresh=(25,100))
    hls_thresholded=hls_thresh(img,thresh=(100,255))
    hsv_thresholded=hsv_thresh(img,thresh=(50,255))
    
    binary_output = np.zeros_like(x_thresholded)
    binary_output[((hsv_thresholded ==1) & (hls_thresholded ==1)) | ((x_thresholded ==1) & (y_thresholded ==1))] = 1
    return binary_output

def region_of_interest(img, vertices):
    # Define a blank matrix that matches the image height/width.
    mask = np.zeros_like(img)
    
    imshape = img.shape    
    if len(imshape) > 2:
        channel_count = img.shape[2]  
        match_mask_color = (255,) * channel_count
    else:
        match_mask_color = 255
      
    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)
    
    # Returning the image only where mask pixels match
    cropped_output = cv2.bitwise_and(img, mask)
    return cropped_output

def draw_lines(img, lines, color=(0, 255, 0), thickness=3):
    
    # If there are no lines to draw, exit.
    if lines is None:
        return img

    # Create a blank image that matches the original in size.
    line_image = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype = np.uint8,
    )
    # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)
    # Merge the image with the lines onto the original.
    img = cv2.addWeighted(img, 0.8, line_image, 1.0, 0.0)
    
    # Return the modified image.
    return img

def make_mask(img):
    height,width = img.shape[:2]
#    polygons = np.array([[(0,height - 160), (1280,height - 160),(780,450),(750,450),(750,0),(570,0),(570,450),(500,450)]]) # define ROI
    polygons = np.array([[(0,height - 80), (width,height - 80), (950,int(height/2)), (350,int(height/2))]])
#    inner_poly = np.array([[(300,height - 140), (1000,height - 140), (720,450), (590,450)]])
    mask = np.zeros_like(img) # create mask with same dimension as image
    cv2.fillPoly(mask, polygons,(255,255,255)) # overlay ROI polygon on mask
#    cv2.fillConvexPoly(mask, inner_poly, 0)
    masked_image = cv2.bitwise_and(img,mask) # and image and the mask to get masked image
    return masked_image

def change_perspective(img):
    height,width = img.shape[:2]
    bot_width = 0.76
    mid_width = .08
    height_pct = .45
    bottom_trim = .9
    offset = width*.25

    src = np.float32([[width*(.5 - mid_width/2), height*height_pct], [width*(.5 + mid_width/2), height*height_pct],\
   [width*(.5 + bot_width/2), height*bottom_trim], [width*(.5 - bot_width/2), height*bottom_trim]])
    dst = np.float32([[offset, 0], [height - offset, 0], [height - offset, width], [offset, width]])
  # set fixed transforms based on image size

  # create a transformation matrix based on the src and destination points
    M = cv2.getPerspectiveTransform(src, dst)

  #transform the image to birds eye view given the transform matrix
    warped = cv2.warpPerspective(img, M, (width, int(height + offset)))
    return warped

def process_image(img):
    
    blur_image = cv2.GaussianBlur(img,(5,5),0)
    combo_binary_thresh = combo_thresh(blur_image)
    final_thresh = np.dstack((combo_binary_thresh,combo_binary_thresh,combo_binary_thresh))*255
    height,width = img.shape[:2]
    #region_of_interest_vertices = [(-100, height-80),(width /2, height * 0.3),(width+100, height-80),]
    #region_of_interest_vertices = [(0, height),(width, height),(width, height/2),(0,height/2)]
    
    
    gray = cv2.cvtColor(final_thresh,cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray,100, 125)
    
    #cropped_image = region_of_interest(canny_image,np.array([region_of_interest_vertices], np.int32))
    
    cropped_image = make_mask(canny_image)
    lines = cv2.HoughLinesP(cropped_image,rho=6,theta=np.pi / 60,threshold=120,lines=np.array([]),minLineLength=40,maxLineGap=100)

    line_image = draw_lines(img, lines)
    
    #Group the lines into left and right groups
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            try:
                slope = (int(y2 - y1)) / (int(x2 - x1)) # <-- Calculating the slope.
            except ZeroDivisionError:
                slope = 0    
            
            if math.fabs(slope) < 0.5: # <-- Only consider extreme slope
                continue
            if slope <= 0: # <-- If the slope is negative, left group.
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else: # <-- Otherwise, right group.
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])
    
    
#    if all([len(left_line_x), len(left_line_y), len(right_line_x), len(right_line_y)]) is True:
        
    min_y = int(height * 0.45) # <-- Just below the horizon
    
    max_y = height # <-- The bottom of the image
    
    poly_left = np.poly1d(np.polyfit(
        left_line_y,
        left_line_x,
        deg=1
    ))
    
    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))
    
    poly_right = np.poly1d(np.polyfit(
        right_line_y,
        right_line_x,
        deg=1
    ))
    
    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))

    blank = np.zeros_like(img)
    cv2.line(blank, (left_x_start, max_y), (left_x_end, min_y), (0,0,255), 5)
    cv2.line(blank, (right_x_start, max_y), (right_x_end, min_y), (255,0,0), 5)
    
    theta1 = math.atan((max_y - min_y)/(left_x_start - left_x_end))*180/math.pi
    theta2 = math.atan((max_y - min_y)/(right_x_start - right_x_end))*180/math.pi
#        theta3 = math.atan(0)*180/math.pi
#    print(theta3 - theta2)
#    print(theta3 - theta1)
    
    cv2.fillPoly(blank, np.int_([[(left_x_end, min_y),(left_x_start, max_y),(right_x_start, max_y), (right_x_end, min_y)]]), (0, 255, 0))
    
    outimg = cv2.addWeighted(blank,0.3, img,1,1)
    
#    else:
#    theta1 = -75
#    theta2 = 75
#    outimg = np.copy(img)
    return [outimg,theta1,theta2,canny_image]



class TestWidget(QMainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.status = 0
        self.statusBar()
        self.scene1 = QGraphicsScene()
        self.scene2 = QGraphicsScene()
        self.view1 = QGraphicsView(self.scene1)
        self.view2 = QGraphicsView(self.scene2)
#        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowMaximizeButtonHint)
#        self.setWindowFlags(self.windowFlags() | QtCore.Qt.CustomizeWindowHint)

        self.setWindowIcon(QIcon('download123.jpg'))
        self.setWindowTitle('Cut_In Event Identification')
        
        self.label = QLabel()
        self.label.setText("")
        self.label.setDisabled(True)
        self.label2 = QLabel()
        self.label2.setText("")
        self.label2.setDisabled(True)
        
        self.fileopen = False
        self.openFile = QAction(QIcon('input.jpeg'), 'Open File', self)
        self.openFile.setShortcut('Ctrl+O')
        self.openFile.setStatusTip('Open File')
        self.openFile.triggered.connect(self.OpenDialog)

        self.Play_on =False

        self.Play = QToolButton()
        self.Play.setIcon(QIcon('play.png'))
        self.Play.setIconSize(QSize(150,150))
        self.Play.setStyleSheet('border-radius: 4px;')
        self.Play.setShortcut('Space')
        self.Play.clicked.connect(self.Play_Action)

#        self.load_input = QToolButton()
#        self.load_input.setIcon(QIcon('icons\video_clip.png'))
#        self.load_input.setText("load input")
#        self.load_input.setIconSize(QSize(50,50))
#        self.load_input.setStyleSheet("border-radius: 4px;")
#        self.load_input.setShortcut('Ctrl+I')
#        self.load_input.clicked.connect(self.load_input_video)
        
        toolbar = self.addToolBar('Quick Access')
        toolbar.addAction(self.openFile)
        
        
        self.indicator = QToolButton()
        self.indicator.setIcon(QIcon('go.png'))
#        self.indicator.setText("load output")
        self.indicator.setIconSize(QSize(150,150))
        self.indicator.setStyleSheet("border-radius: 4px;")
        self.indicator.setShortcut('Ctrl+O')
        #self.indicator.clicked.connect(self.indicator_action)
        self.indicator.setDisabled(True)


        #dynamic_canvas = FigureCanvas(Figure(figsize=(7, 4)))
        
        Vlayout = QVBoxLayout()
        Hlayout = QHBoxLayout()
        buttonlayout = QHBoxLayout()
        viewlayout =  QHBoxLayout()
        vstacklayout = QVBoxLayout()

        # xlayout = QVBoxLayout()
        
        viewlayout.setAlignment(Qt.AlignCenter)
        buttonlayout.setAlignment(Qt.AlignCenter)
        #xlayout.setAlignment(Qt.AlignCenter)
        
        Hlayout.addWidget(self.indicator)
        Hlayout.addWidget(self.label)
        
        Hlayout.addWidget(self.Play)

        vstacklayout.addLayout(Hlayout)
        vstacklayout.addWidget(self.label2)
        #vstacklayout.addWidget(dynamic_canvas)
        vstacklayout.addStretch()
        
        viewlayout.addWidget(self.view1)
        viewlayout.addLayout(vstacklayout)
        
        #xlayout.addLayout(Hlayout)
        # xlayout.addWidget(self.Play)

#        viewlayout.addStretch(1)
        
        
#        buttonlayout.addWidget(self.load_input)
        
        Vlayout.addLayout(viewlayout)
#        Vlayout.addLayout(Hlayout)
        
        window = QWidget()
        window.setLayout(Vlayout)
        self.setCentralWidget(window)
        
        # self._dynamic_ax = dynamic_canvas.figure.subplots()
        
        # self._dynamic_ax.set(xlabel='frames (num)', ylabel='magnitude',title='lane plot')
        # self._dynamic_ax.grid()
        # self._timer = dynamic_canvas.new_timer(100, [(self._update_canvas, (), {})])
        
        self.setLayout(Vlayout)
      
    def _update_canvas(self):
        new_th = []
        for (i,t) in enumerate(self.theta_diff):
            new_th.append(int(t*math.sin(2*math.pi*10*i/1000)+time.time()))
            
        self._dynamic_ax.clear()
        self._dynamic_ax.set(xlabel='frames (num)', ylabel='magnitude',title='lane plot')
        self._dynamic_ax.grid(True)
        t = np.arange(self.total_frames)
        
        self._dynamic_ax.plot(t, new_th,'g')
        self._dynamic_ax.figure.canvas.draw()
        
    def OpenDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '', 'Video Files (*.avi *.mp4);; Image Files (*.jpg *.png);; CSV Files (*.csv)')[0]
        print(fname)
        lst1=[]

        df = pd.read_csv('/home/souravkc/Desktop/yolo-object-detection/1233_FINAL__newcross.csv')
        #print(df)
        df1 = pd.DataFrame(df)
        df2 = df1[df1['lxmin_y']][['frame_no']]
        self.dict_df = {}
        for i, row in df1.iterrows():
            self.dict_df[row['frame_no']] = row['lxmin_y']
        #print(df2) #=='True']
        #df3 = pd.DataFrame(df2)
        #lst1.append(df3)
        lst1 = df2['frame_no'].values.tolist()
        #print(lst1)

        lst1 = set(lst1)
        lst1 = list(lst1)
        
        if ('.avi' in fname) or ('.mp4' in fname):
            #call a function to process video
            self.process_video(fname)
            
        elif ('.jpg' in fname) or ('.png' in fname):
            #call a function prosess image
            print("calling image fn")
        else:
            pass
    
    def indicator_action(self):
       
        i=0
        z=0
        while(i!=3001):
            #print(i)
            #print(lst1[z])
            if(lst1[z]== i):
            # print(i)
            # print(lst1[z])
            #print('1')
                #if self.data.lxmin_y == 'True':

                self.indicator.setIcon(QIcon('go.png'))
                self.indicator.setIconSize(QSize(150,150))
                z+=1
                i=0
            else:
                
                self.indicator.setIcon(QIcon('download.png'))
                self.indicator.setIconSize(QSize(150,150))
                i=i+1
        # if self.status == 0:
        #     self.indicator.setIcon(QIcon('download.png'))
        #     self.indicator.setIconSize(QSize(64,64))
        # else:
        #     self.indicator.setIcon(QIcon('stop.png'))
        #     self.indicator.setIconSize(QSize(64,64))
        
    def load_output_video(self):
        print("output")

    def display_image_scene1(self, img):
        npimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(npimg)
        self.scene1.clear()
        self.w, self.h = pil_image.size
        self.imgQ = ImageQt.ImageQt(pil_image)  # we need to hold reference to imgQ, or it will crash
        pixMap = QPixmap.fromImage(self.imgQ)
        self.scene1.addPixmap(pixMap)
        self.view1.fitInView(QRectF(0, 0, self.w, self.h), Qt.KeepAspectRatio)
        self.scene1.update()
        
    def display_image_scene2(self, img):
        npimg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(npimg)
        self.scene2.clear()
        self.w, self.h = pil_image.size
        self.imgQ = ImageQt.ImageQt(pil_image)  # we need to hold reference to imgQ, or it will crash
        pixMap = QPixmap.fromImage(self.imgQ)
        self.scene2.addPixmap(pixMap)
        self.view1.fitInView(QRectF(0, 0, self.w, self.h), Qt.KeepAspectRatio)
        self.scene2.update()
    
    def closeEvent(self, event):
        self.reply = QMessageBox.question(self, 'Confirm Exit',
            "Are you sure to Quit?", QMessageBox.Yes | 
            QMessageBox.No, QMessageBox.No)

        if self.reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()


    def Play_Action(self):
        if self.fileopen == True:
            self.Play_on = not self.Play_on
            if self.Play_on is True:
                self.Play.setStatusTip("Pause")
                self.Play.setIcon(QIcon('pause.jpeg'))
            else:
                self.Play.setStatusTip("Play")
                self.Play.setIcon(QIcon('play.png'))
                
    def Pause_video(self):
        self.Play_on = False
        self.stop.setDisabled(False)
        self.Play.setStatusTip("Play")
        self.Play.setIcon(QIcon('play.png'))

    def close_application(self,path):
        import time
        cap = cv2.VideoCapture(path) 
   
# Check if camera opened successfully 
        if (cap.isOpened()== False):  
            print("Error opening video  file") 
   
# Read until video is completed 
        while(cap.isOpened()): 
      
  # Capture frame-by-frame 
            ret, frame = cap.read() 
            if ret is True:
                self.label.setDisabled(False)
                self.indicator.setDisabled(False)
                self.label2.setDisabled(False) 
   
    # Display the resulting frame 
                #cv2.imshow('Frame', frame)
                self.display_image_scene1(frame) 
   
    # Press Q on keyboard to  exit 
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
   
  # Break the loop 
            else:
#                print("Done Processing :)   ")
                self.indicator.setDisabled(True)
                self.label.setDisabled(True)
                self.label2.setDisabled(True)
                  
                break
   
# When everything done, release  
# the video capture object 
        cap.release() 
   
# Closes all the frames 
        cv2.destroyAllWindows()            

    def process_video(self,path):
        self.Play.setDisabled(False)

        cap = cv2.VideoCapture(path)
        import time

        self.reply = QMessageBox.No
#        video_writer_set = False
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#        print("video file    " + input_video_path)
#        sys.stdout.write("\r")
#         angle_data = []
#         self.theta_diff = np.zeros(self.total_frames)
# #        self.theta_diff = []
        
        oldstatus = self.status
        import json
        #print(json.dumps(self.dict_df, indent=4))
        while cap.isOpened():
            #self._timer.start()
            self.fileopen = True
            #ret,frame = cap.read()
            
            #framecpy = np.copy(frame)

            if self.Play_on == True:
                ret , frame = cap.read()

                framecpy = np.copy(frame)

                if ret is True:
                    self.label.setDisabled(False)
                    self.indicator.setDisabled(False)
                    self.label2.setDisabled(False)
    #                if video_writer_set is False:
    #                    out = cv2.VideoWriter("out.avi",cv2.VideoWriter_fourcc(*'DIVX'), 30, (frame.shape[1], frame.shape[0]))
    #                    video_writer_set = True
                    present_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    #cap.set(cv2.CV_CAP_PROP_FPS, 30)
                    #cv2.waitKey(100)
                    self.display_image_scene1(frame)
                    # cv2.waitKey(200000)
                    try:
                        if not self.dict_df[present_frame]:
                            self.indicator.setIcon(QIcon('go.png'))
                            self.indicator.setIconSize(QSize(150,150))
                            self.label2.setText("   Free_From_Cut-In")
                            #time.sleep(0.01)
                            
                        else:
                            self.indicator.setIcon(QIcon('download.png'))
                            self.indicator.setIconSize(QSize(140,140))
                            self.label2.setText("   Cut-In_Car")
                            #time.sleep(0.01)

                    except Exception as e:
                            #print("except")
                            pass

                    if cv2.waitKey(30) == ord('q'):
                        break
                    #cv2.waitKey(20000)
                    #try :

    #                     colored_image,angle1,angle2,ximg = process_image(framecpy)
    #                     cnt2 += 1
    #                     if cnt2 > 5:
    #                         self.status = 0
    #                         self.label2.setText("FREE_LANE")
    # #                        cv2.putText(framecpy, "Not changing", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 3, cv2.LINE_AA)
    #                         cnt = 0
    #                     else:
    #                         pass
                        
    #                     present_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    #                     self.theta_diff[present_frame] = angle1+angle2
    #                     self.display_image_scene1(frame)
    # #                    self._update_canvas()
    # #                    out.write(colored_image)
    #                     #angle_data.append([present_frame,angle1,angle2,self.status])
                        
    #                 except TypeError:
    #                     cnt += 1
    #                     if cnt>5:
    #                         self.status = 1
    #                         self.label2.setText("INCOMING")
    # #                        cv2.putText(framecpy, "Changing lane", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3, cv2.LINE_AA)
    #                         cnt2=0
    #                     else:
    #                         pass
                        
    #                     present_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    #                     self.theta_diff[present_frame] = 150
    #                     self.display_image_scene1(frame)
    # #                    self._update_canvas()
    # #                    out.write(frame)
    #                     #angle_data.append([present_frame,-75,75,self.status])
    #                     pass
    #                 except ZeroDivisionError:
    #                     cnt += 1
    #                     if cnt>5:
    #                         self.status = 1
    #                         self.label2.setText("INCOMING")
    # #                        cv2.putText(framecpy, "Changing lane", (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3, cv2.LINE_AA)
    #                         cnt2=0
    #                     else:
    #                         pass
                        
    #                     present_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    #                     self.theta_diff[present_frame] = 150
    #                     self.display_image_scene1(frame)
    # #                    self._update_canvas()
    # #                    out.write(frame)
    #                     #angle_data.append([present_frame,-75,75,self.status])
    #                     pass
    # #                if cv2.waitKey(1) == ord('q'):
    # #                    break  
    #             else:
    # #                print("Done Processing :)   ")
    #                 self.indicator.setDisabled(True)
    #                 self.label.setDisabled(True)
    #                 self.label2.setDisabled(True)
    #                 break
                
                self.label.setText("frames: "+str(present_frame) + "\\" + str(self.total_frames))
            
            if self.reply == QMessageBox.Yes:
                cap.release()
                #out.release()
                cv2.destroyAllWindows()
                break

            #if self.status is not oldstatus:
            #    self.indicator.click()
            
            oldstatus = self.status
            QCoreApplication.processEvents()
            
            #csv_data = pd.DataFrame(angle_data, columns = ["frame_no","theta_left","theta_right","status"])
            #csv_data.to_csv('angle_data.csv')
        
        cap.release()
        #out.release()
        cv2.destroyAllWindows()

    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = TestWidget()
#    widget.resize(700, 750)
#    widget.showFullScreen()
    widget.showMaximized()
    widget.show()

    sys.exit(app.exec_())
