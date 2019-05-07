
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle
import sys
import glob
import csv
import io
import pandas as pd
import os
import matplotlib.pyplot as plt


from IPython.display import clear_output
clear_output(wait = True)


my_path = '/home/souravkc/Desktop/lane/'
my_file = my_path+"lst_final_output.csv"

if os.path.isfile(my_file):
    print("file is present")
    os.remove(my_file)
    print("file is deleted")

def plot_images(original, modified, title):
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(original)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(modified, cmap='gray')
    ax2.set_title(title, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


# In[4]:


def undistort(img):
    #Load pickle
    dist_pickle = pickle.load( open( "calibration_pickle.p", "rb" ) )
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]

    # Undistort sample image
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    return undistorted

# undistort it
#undistorted = undistort(image)

# printing out some stats and plotting
#plot_images(image, undistorted, 'Undistorted Image')


# In[5]:


#GET BINARY IMAGE WITH GRADIENT
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
    
# Run the function
#grad_binary = abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(20,100))
# Plot the result
#plot_images(image, grad_binary, 'Thresholded Gradient')


# In[6]:


#GRADIENT WITH MAGNITUDE
# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
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
    
# Run the function
#mag_binary = mag_thresh(image, sobel_kernel=7, mag_thresh=(50, 100))

# Plot the result
#plot_images(image, mag_binary, 'Thresholded Magnitude')


# In[7]:


#Gradient with direction
# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
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

# Run the function
#dir_binary = dir_threshold(image, sobel_kernel=29, thresh=(1.1, 1.3))

# Plot the result
#plot_images(image, dir_binary, 'Thresholded Grad. Dir.')


# In[8]:


# Combined different thresholding techniques
def combined_thresh(img):
    # Choose a Sobel kernel size
    ksize = 21

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize, thresh=(5,100))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize, thresh=(10,120))
    mag_binary = mag_thresh(img, sobel_kernel=7, mag_thresh=(10, 150))
    dir_binary = dir_threshold(img, sobel_kernel=15, thresh=(0.8, 1.0))
    
    #Combine them
    combined2 = np.zeros_like(dir_binary)
    combined2[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    return combined2
    
# Run the function
#gr_combined = combined_thresh(image)

# Plot the result
#plot_images(image, gr_combined, 'Gradient Thresholded Combined')


# In[9]:


#COLOR THRESHOLDING
def hls_thresh(img,thresh=(0,255)):
    hls=cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
    s_channel=hls[:,:,2]
    
    #Combine them
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


# In[10]:


def hsv_thresh(img,thresh=(0,255)):
    hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    v_channel=hsv[:,:,2]
    
    #Combine them
    binary_output = np.zeros_like(v_channel)
    binary_output[(v_channel > thresh[0]) & (v_channel <= thresh[1])] = 1
    return binary_output


# In[11]:


def combo_thresh(img):
    x_thresholded=abs_sobel_thresh(img,orient='x',sobel_kernel=3,thresh=(12,120))
    y_thresholded=abs_sobel_thresh(img,orient='y',sobel_kernel=3,thresh=(25,100))
    hls_thresholded=hls_thresh(img,thresh=(100,255))
    hsv_thresholded=hsv_thresh(img,thresh=(50,255))
    
    
    #Combine them
    binary_output = np.zeros_like(x_thresholded)
    binary_output[((hsv_thresholded ==1) & (hls_thresholded ==1)) | ((x_thresholded ==1) & (y_thresholded ==1))] = 1
    return binary_output

# Run the function
#combined_thresh = combo_thresh(image)

# Plot the result
#plot_images(image, combined_thresh, 'Gradient + Color Combined')
    


# In[12]:


def change_perspective(img):
    img_size = (img.shape[1], img.shape[0])
    height,width=img.shape[:2]

#    bot_width = .76
 #   mid_width = .08
 #   height_pct = .45
  #  bottom_trim = .935
   # offset = img_size[0]*0.25

    bot_width = .76
    mid_width = .08
    height_pct = .45
    bottom_trim = .935
    offset = img_size[0]*0.25

    src = np.float32([[img.shape[1]*(.5 - mid_width/2), img.shape[0]*height_pct], [img.shape[1]*(.5 + mid_width/2), img.shape[0]*height_pct],   [img.shape[1]*(.5 + bot_width/2), img.shape[0]*bottom_trim], [img.shape[1]*(.5 - bot_width/2), img.shape[0]*bottom_trim]])
    dst = np.float32([[offset, 0], [img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]], [offset, img_size[1]]])
 
      # set fixed transforms based on image size

      # used to test that src points matched line
      # cv2.fillConvexPoly(img, src.astype('int32'), 1)
    # plt.imshow(img)
    # plt.title('img_lines')
    # plt.show()

      # create a transformation matrix based on the src and destination points
    M = cv2.getPerspectiveTransform(src, dst)

      #transform the image to birds eye view given the transform matrix
    warped = cv2.warpPerspective(img, M, (img_size[0], img_size[1]))
    # plt.imshow(warped)
    # plt.title('warped')
    # plt.show()
    return warped

# Run the function
#warped_img= change_perspective(combined_thresh)

# Plot the result
#plot_images(img, warped_img, 'Warped Image')


# In[13]:
lsst1=[]
lsst2=[]

def lr_curvature(binary_warped):
    
    #abc=self.draw_on_road()

  # Assuming you have created a warped binary image called "binary_warped"
  # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[600:,:], axis=0)
  # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    #print(out_img.shape)
  # Find the peak of the left and right halves of the histogram
  # These will be the starting point for the left and right lines

    # plt.plot(out_img)
    # plt.title('histo')
    # plt.show()

    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    #print(leftx_base,rightx_base,midpoint)
    
    # Choose the number of sliding windows
    nwindows = 50
  # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
  # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
  # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
  # Set the width of the windows +/- margin
    margin = 80
  # Set minimum number of pixels found to recenter window
    minpix = 50
  # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

  # Step through the windows one by one
    for window in range(nwindows):
      # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
      # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,140,0), 2)
      # print('rectangle 1', (win_xleft_low,win_y_low),(win_xleft_high,win_y_high))
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,140,0), 2)
        cv2.imwrite('sliding_win.jpg',out_img)
      # print('rectangle 2', (win_xright_low,win_y_low), (win_xright_high,win_y_high))
      # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
      # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
      # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

  # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

  # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

  # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
  # At this point, you're done! But here is how you can visualize the result as well:
  # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [30, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 30]
    #plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    #plt.show()
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_lft_last = pts_left[0][959]
    pts_lft_first = pts_left[0][0]

    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts_right_first = pts_right[0][0]
    pts_right_last = pts_right[0][959]
    #print('pts left', pts_left.shape, 'pts right', pts_right.shape)
    pts = np.hstack((pts_left, pts_right))
    
    lsst1.append(pts_lft_last[0])
    lsst1.append(pts_lft_first[0])
    lsst1.append(pts_right_first[0])
    lsst1.append(pts_right_last[0])
    #print(pts_lft_last,"left last")
    #print(pts_lft_first,"lft first")

    #print(lsst1,"lsst1")
    #print(len(left_fitx),len(right_fitx))
    #plt.show()
    #print(right_fitx)
#for left_bottom_x&y    
   
    lst_left=[]
    pt1 = pts_left
    lst_left.append(pt1[0])
    #lst_left.append(pts_left)

    #pt2 = tuple(map(tuple, pt1))
    #pt3 = pt2[0][0]
    #tuple(lst_left[0][0])
    #print(lst_left)
    #lst_left = print(lst_left[0][0][0])
    
#for bottom_x first position
    lst_left_b_x=[]
    pt2 = lst_left[0]
    lst_left_b_x.append(pt2)
    #print(lst_left_b_x)

#for bottom_y second position
    lst_left_b_y=[]
    pt3 = lst_left[0][1]
    lst_left_b_y.append(pt3)
    #print(lst_left_b_y)





#for right    
    lst_right=[]
    pt4 = pts_right
    lst_right.append(pt4[0][1])
    #print(lst_right)

#for right_x
    lst_right_u_x=[]
    pt5= lst_right[0][0]
    lst_right_u_x.append(pt5)
    #print(lst_right_u_x)

#for right y
    lst_right_u_y=[]
    pt6= lst_right[0][1]
    lst_right_u_y.append(pt6)
    #print(lst_right_u_y)



    left_x_min = max(lst_left_b_x)
    left_y_min = max(lst_left_b_y)
    #print(left_x_min,left_y_min)
    


    left_x_max = min(lst_left_b_x)
    left_y_max = min(lst_left_b_y)
    #print(left_x_max,left_y_max)

    right_x_min = max(lst_right_u_x)
    right_y_min = max(lst_right_u_y)
    #print(right_x_min, right_y_min)

    right_x_max = min(lst_right_u_x)
    right_y_max = min(lst_right_u_y)
    #print(right_x_max,right_y_max)

    lf1=left_x_min
    lf2=left_y_min
    lf3=left_x_max
    lf4=left_y_max
    #print(lf1,lf2)

    rg1=right_x_min
    rg2=right_y_min
    rg3=right_x_max
    rg4=right_x_max
##############################################
    new_left_x_min = min(left_fitx)
    new_left_x_max = max(left_fitx)
    #print(new_left_x_min , new_left_x_max)

    new_right_x_min = min(right_fitx)
    new_right_x_max = max(right_fitx)
    #print(new_right_x_min , new_right_x_max)    
    
    min_y = min(ploty)
    max_y = max(ploty)
    #print(min_y,max_y)

    llxmin=new_left_x_min
    llymin= min_y
    llxmax= new_left_x_max
    llymax= max_y 
    

    rrxmin=new_right_x_min
    rrymin= min_y
    rrxmax= new_right_x_max
    rrymax = max_y



    #ss1=pd.DataFrame(data={"lxmin": [lf1], "lymin": [lf2],"lxmax": [lf3],"lymax": [lf4], "rxmin": [rg1],"rymin": [rg2],"rxmax": [rg3],"rymax": [rg4]},index=True)
    #print(ss1)
    #print('---------------------------------------------')

    
#    all_lst.append(left_x_min) #+left_y_min+left_x_max+left_y_max+right_x_min+right_y_min+right_x_max+right_y_max
#    all_lst.append(left_y_min)
#    all_lst.append(left_x_max)
#    all_lst.append(left_y_max)
#    all_lst.append(right_x_min)
#    all_lst.append(right_y_min)
#    all_lst.append(right_x_max)
#    all_lst.append(right_y_max)

    
    #all_lst=[]
    #all_lst.append(df) 
    #print(all_lst)
    #print('---------------------------------------------')
    #concating
    #masterdf = pd.concat(all_lst, ignore_index= True)
    #masterdf.to_csv("PT_points.csv", sep=' ',index=False)    
    #print(masterdf)



    #print(all_lst)

    #print(left_x_min, left_y_min)
    #print(left_x_max,left_y_max)
    #print(right_x_min,right_y_min)
    #print(right_x_max,right_y_max)

    #res = [x, y, z, ....]

    #csvfile = "PT_points.csv"
    


    #headers = ['lxmin','lymin','lxmax','lymax','rxmin','rymin','rxmax','rymax']

    #all_lst.insert(0,['lxmin','lymin','lxmax','lymax','rxmin','rymin','rxmax','rymax'])
    
    

    

    #all_list.append([lf1,lf2,lf3,lf4,rg1,rg2,rg3,rg4])

    #with open('PT_points.csv', 'a') as file_P:

        #writeit = csv.writer(file_P)

        #writeit.writerow([llxmin,llymin,llxmax,llymax,rrxmin,rrymin,rrxmax,rrymax])

    
    #with open('file.csv',newline='') as f:
     #   r = csv.reader(f)
      #  r.writerows(0,['lxmin','lymin','lxmax','lymax','rxmin','rymin','rxmax','rymax'])
      #  data = [[lf1,lf2,lf3,lf4,rg1,rg2,rg3,rg4]]
    #with open('file.csv','a',newline='') as f:
     #   w = csv.writer(f)
        
      #  w.writerows(data)

    #csvfile.writer(0,['lxmin','lymin','lxmax','lymax','rxmin','rymin','rxmax','rymax'])    
    #    writeit.writerows(map(lambda x: [x],all_lst))
        
        
    #writeit.insert(0,['lxmin','lymin','lxmax','lymax','rxmin','rymin','rxmax','rymax'])
    #with open(csvfile, 'a') as output:
    #   f_csv = csv.DictWriter(output,headers)
        
        #f_csv.writerows(all_lst)


    #    writer = csv.writer(output,delimiter=',')
        #writer.writeheader()
    #    writer.writerows(map(lambda x: [x],all_lst))
    #csvFile.close()


    return pts[-1]
        #for val in pts_left:
    

    #    writer.writerow([val])




    #ppt= pts_left.tolist()
    #my_df = pd.DataFrame(ppt)
#    my_df.to_csv(csvfile, index=False, header=False)
#    print(my_df)


    #plt.show(pts_left[ptr:]) 
	#plt.show(pts_left[0:ptr])
    
   


    #cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
   

    
    
    
#lane_polygon = lr_curvature(warped_img)




def draw_on_road(img, warped,pts):
  #create img to draw the lines on
    #plt.imshow(img)
    #plt.show()
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

  #draw the lane onto the warped blank img
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    img_size = (img.shape[1], img.shape[0])
    height,width=img.shape[:2]

    bot_width = .76
    mid_width = .08
    height_pct = .45
    bottom_trim = .935
    offset = img_size[0]*.25


    
    src = np.float32([[img.shape[1]*(.5 - mid_width/2), img.shape[0]*height_pct], [img.shape[1]*(.5 + mid_width/2), img.shape[0]*height_pct],   [img.shape[1]*(.5 + bot_width/2), img.shape[0]*bottom_trim], [img.shape[1]*(.5 - bot_width/2), img.shape[0]*bottom_trim]])
    dst = np.float32([[offset, 0], [img_size[0] - offset, 0], [img_size[0] - offset, img_size[1]], [offset, img_size[1]]])
    blank = np.zeros_like(img)
    dif = src - dst
    #print('difference \n',dif)
    
    
    
    dif0 = dif[0][0]
    #lst_dif.append(dif0)
    dif1 = dif[1][0]
    dif2 = dif[2][0]
    dif3 = dif[3][0]


    lsst2.append(dif0)
    lsst2.append(dif1)
    lsst2.append(dif2)
    lsst2.append(dif3)
    
    #print(lsst2,"lsst2_out")

    #out_value(dif0,dif1)

    #cv2.fillConvexPoly(blank, src, 1)
    # plt.imshow(blank)
    # plt.title('lines')
    # plt.show()
    Minv = cv2.getPerspectiveTransform(dst, src)

    #print("source_points \n",src)
    #print("dst_points \n",dst)
    #print(Minv)
  #warp the blank back oto the original image using inverse perspective matrix
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))

  #combine the result with the original
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    #plt.imshow(newwarp)
    #plt.title('result')
    #plt.show()
    #plt.imshow(result)
    #plt.show()
    #print('result shape', result.shape)
    

    #points = np.int_([pts])
    #print(points)
    # push_P = points.tolist()
    # print(push_P)

    #plt.imshow(result)
    #plt.show()
    return result
    
lsst_final= []
#print(lsst1, lsst2)






#out_img= draw_on_road(image,warped_img,lane_polygon)

# Plot the result
#plot_images(image, out_img, 'Output Image')

def process_image(img):
    #print("yes")
    blur = cv2.GaussianBlur(img,(5,5),0) # smoothen out the image

    combo_image = combo_thresh(blur)
    # plt.imshow(combo_image, cmap='gray')
    # plt.title('combo_image')
    # plt.show()
    # #print(combo_image.shape,'shape of combo')

    #  masked_image = make_masked_image(combo_image)
    #  plt.imshow(masked_image, cmap='gray')
    #  plt.title('masked_image')
    #  plt.show()
    #  print(masked_image.shape,'shape of masked')

    warped_image = change_perspective(combo_image)

    # plt.imshow(warped_image)
    # plt.title('warped_image')
    # plt.show()
    #cv2.imshow(warped_image)
    #plt.imshow(warped_image, cmap='gray')
    #  plt.title('warped_image')
    #plt.show()
    #  print(warped_image.shape,'shape of warped')
      #all_list.insert(0,['lxmin','lymin','lxmax','lymax','rxmin','rymin','rxmax','rymax'])
    lane_curv = lr_curvature(warped_image)
    result = draw_on_road(img, warped_image, lane_curv)
    #cv2.putText(result, full_text, (200, 100), cv2.FONT_HERSHEY_COMPLEX, 1, 255)

      # sci.imsave('./output_images/5_final.jpg', result)
    return result

    
# In[17]:
class Lane():
  def __init__(self):
    #if line was deteced in last iteration
    self.curve = {'full_text': ''}


if __name__ == '__main__':
    # images = get_file_images('test_images')
    # show_images(images)
    # #set video variables
    # proj_output = 'output2.mp4'
    # clip1 = VideoFileClip('project_video.mp4')
    
    #cascade_src = 'cars.xml'
    input_video_path = r"1233_T.mp4"
    #all_lst=[]
    cap = cv2.VideoCapture(input_video_path)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("video file    "+input_video_path)
    sys.stdout.write("\r")

    while cap.isOpened():
        ret,frame = cap.read()

        if ret is True:
                 
            try:
                #blur = cv2.GaussianBlur(frame,(5,5),0)
                #undistorted_img = undistort(blur)
                #gray_img=cv2.cvtColor(undistorted_img,cv2.COLOR_RGB2GRAY)
                #combined_thresh_img = combo_thresh(undistorted_img)
                #final_thresh_img=np.dstack((combined_thresh_img,combined_thresh_img,combined_thresh_img))*255
                
                #warped_imgg=change_perspective(combined_thresh_img)
                #lane_curv=lr_curvature(warped_imgg)
                #output_img= draw_on_road(frame,warped_imgg,lane_curv)

                colored_image = process_image(frame)
                output_img =np.copy(colored_image)
                #print(lane_curv)
                # plt.imshow(colored_image)
                # plt.title('colored_image')
                # plt.show()
    

                present_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                #dez=[]
                
                
                
                for i in range(len(lsst1)):
                    val = lsst1[i] - lsst2[i]
                            
                    lsst_final.append(val)
                
                #lsst_final.insert(0,['lxmin','lxmax','rxmin','rxmax'])

                with open('lst_final.csv','a') as readcsv1:
                    #csv.insert(0,['lxmin','lxmax','rxmin','rxmax'])
                    writeit = csv.writer(readcsv1)
                    #lsst_final.insert(0,['lxmin','lxmax','rxmin','rxmax'])
                    writeit.writerow(lsst_final)

                lsst1.clear()
                lsst2.clear()
                #print(lsst_final)
                # a1=lsst_final[0]
                # a2=lsst_final[1]
                # a3=lsst_final[2]
                # a4=lsst_final[3]
                lsst_final.clear()
                # df = pd.DataFrame(data={'lxmin':a1,'lxmax':a2,'rxmin':a3,'rxmax':a4})
                # #print(df)
                # print(dez.append(df))

                # masterde = pd.concat(dez, ignore_index=True)
                # #print(masterde)
                # masterde.to_csv('pt1.csv', sep=',',index=False)
                # #print(lsst_final,"final out")
                
                
                #with open('lst_final.csv','a') as readcsv1:
                    #writeit = csv.writer(readcsv1)
                #    writeit.writerow(lsst_final)
                
                
                #sys.os.mkdir('output/1233_new')
                cv2.imwrite('out_put/1233_T/frame_'+str(int(present_frame))+'.jpg',output_img)
                cv2.imshow("output",cv2.resize(output_img),None,fx=0.8,fy=0.8)    
                print("processing..."+ str(round(100*present_frame/total_frames,2)),"%", end = '\r',flush = True)
                clear_output(wait=True)
                cv2.cvtColor(output_img,cv2.COLOR_BGR2GRAY)
            except TypeError:
                cv2.imwrite('error_frame.jpg',frame)
                pass
            
            #out.write(output_img)
            
           
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            print("Done Processing :)   ")
            break
    
# with open('lst_final.csv','a') as readcsv1:
#     writeit = csv.writer(readcsv1)
#     writeit.writerow(lsst_final)
        
    #all_lst.insert(0,['lxmin','lymin','lxmax','lymax','rxmin','rymin','rxmax','rymax'])
    #with open('PT_points.csv', 'w') as file_P:
    #    writeit = csv.writer(file_P)
    #    writeit.writerows(all_lst)
    

    #readcsv= pd.read_csv('PT_points.csv')#,sep='\t', names = ['lxmin','lymin','lxmax','lymax','rxmin','rymin','rxmax','rymax'])
    
    #rdcsv = pd.DataFrame(readcsv)
    #print(lsst_final,"final out")
    readcsv= pd.read_csv('lst_final.csv')
    readcsv.columns= ['lxmax','lxmin','rxmin','rxmax']
    readcsv.to_csv('1233_Tfinal.csv',index = False)
    
    
    #     #readcsv.columns= ['lxmin','lymin','lxmax','lymax','rxmin','rymin','rxmax','rymax']
        #readcsv.to_csv(lsst_final,index = False)
    
    #print(rdcsv)
    

    cap.release()
    #out.release()
    cv2.destroyAllWindows()