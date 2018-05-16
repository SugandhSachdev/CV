
# Importing Important Libraries.
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

original_image = cv2.imread("21.png",0) # Saving image in a variable named original_image.
plt.figure()
plt.imshow(original_image , cmap = 'gray') # Showing image.
plt.xticks([]), plt.yticks([])
plt.figure()
temp = cv2.imread("temp1.jpg",0) # Showing template.
plt.imshow(temp, cmap = 'gray')
plt.xticks([]), plt.yticks([])



def correlation(original_image , temp):
    # Initializing correlation image:
    y = original_image.shape[0]
    x = original_image.shape[1]
    image_correlation = np.zeros((y - temp.shape[0] + 1 , x - temp.shape[1] + 1))
    x_correlation = image_correlation.shape[1]
    y_correlation = image_correlation.shape[0]
   
    i=0
    
    while i < x_correlation:
        j=0
        while j < y_correlation and i < x_correlation:
            start_horizontal = i 
            end_horizontal = i + temp.shape[1]
            start_vertical = j
            end_vertical = j + temp.shape[0]
        
            temporary = original_image[start_vertical:end_vertical , start_horizontal:end_horizontal]
            temporary = temporary - np.mean(temporary)
            conv = np.sum(np.multiply(temporary , temp))
        
            image_correlation[j , i] = conv
            j = j+1
            if conv >= 260200: 
                i = i+temp.shape[1] #shift left if a possible peak
                j=0            
        i=i+1

            
    return image_correlation


image_correlation = correlation(original_image, temp) 

def num_fingers(image_correlation,thresh):
    
    x_correlation = image_correlation.shape[1]
    y_correlation = image_correlation.shape[0]
    img_thres = np.zeros((y_correlation,x_correlation))
    t_max = np.max(image_correlation)
    img_thres [image_correlation >= thresh*t_max] = 1 #threshold image
    #print 0.5*t_max
    sz_x = 30
    sz_y = temp.shape[0]
    img_vote = np.zeros((y_correlation,x_correlation)) #initialize voting grid
    for i in range(0,(x_correlation-sz_x),sz_x):
        for j in range(0,(y_correlation-sz_y),sz_y):
            start_horizontal = i
            end_horizontal = i + sz_x
            start_vertical = j
            end_vertical = j + sz_y
            mid_x = np.floor(start_horizontal + sz_x/2).astype(int)
            mid_y = np.floor(start_vertical + sz_y/2).astype(int)
            temporary = img_thres[start_vertical:end_vertical, start_horizontal:end_horizontal]
            conv = np.sum(temporary)
            
            img_vote[mid_y, mid_x] = conv #set total window votes to the middle of window
            
    v_max =  np.max(img_vote)
    [y,x] = np.nonzero(img_vote>0.1*v_max)
    x = x.tolist()
    x_set = set(x)
    for i in range (len(x_set)): #draw circles on the votes, not accurate position due to windowing
        x_t = x_set.pop()
        index = x.index(x_t)
        
        cv2.circle(img_thres, (x_t, y[index]), 10, (255, 0, 0), 1)
    plt.figure()
    plt.imshow(img_thres,cmap='gray')
    plt.xticks([]), plt.yticks([])

    n_fingers = len(set(x))
    return n_fingers
n_fingers = num_fingers(image_correlation,0.5)
if n_fingers > 5:
    n_fingers = num_fingers(image_correlation,0.7) #increase the threshold if number greater than 5
print n_fingers
plt.show()

