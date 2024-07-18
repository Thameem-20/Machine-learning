# Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
import cv2

#------------------------------------
#1. DCT
im_gray = cv2.imread("/home/niranjan/Downloads/Talks2022/NMAMIT_Apr2022/NMAMIT_Python/NCodes/Ndata/med_image1.png", 0)
himg1, wimg1 = im_gray.shape
print('height, width', himg1, wimg1)
imf = np.float32(im_gray)/255.0  # float conversion/scale
im_gray = np.float32(im_gray)/255.0 
dst = cv2.dct(imf)           # the dct
cv2.imshow("DCT of the Xray image", dst) #
cv2.waitKey(4000)
recon_dct = np.zeros([himg1, wimg1], dtype='float')
recon_dct[0:himg1//10, 0:wimg1//10] = dst[0:himg1//10, 0:wimg1//10]
dst1 = cv2.idct(recon_dct) # the idct
res = np.hstack((im_gray, im_gray + (50/255.0))) #dst1
cv2.imshow('Inverse DCT trasnformed image', res)
cv2.waitKey(40000)
cv2.destroyAllWindows()

#------------------------------------
#2. Histogram Equalization
# of a image using cv2.equalizeHist()
im_gray = cv2.imread("/home/niranjan/Downloads/Talks2022/NMAMIT_Apr2022/NMAMIT_Python/NCodes/Ndata/med_image1.png", 0)
equ = cv2.equalizeHist(im_gray)
  
# stacking images side-by-side
res = np.hstack((im_gray, equ))
  
# show image input vs output
cv2.imshow('equalized Xray image', res)
  
cv2.waitKey(4000)
cv2.destroyAllWindows()

#------------------------------------
#3. Pseudocoloring
im_gray = cv2.imread("/home/niranjan/Downloads/Talks2022/NMAMIT_Apr2022/NMAMIT_Python/NCodes/Ndata/Echo-1(1).jpg")
himg1, wimg1, channel_length = im_gray.shape
print('Echo-1(1) height, width, channel_length', himg1, wimg1, channel_length)
im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_JET)
plt.subplot(2,1,1)
plt.imshow(im_gray) #
plt.title("Echocardiogram image Echo-1(1)")
plt.subplot(2,1,2)
plt.imshow(im_color) #
plt.title("Pseudocolored image")
plt.show()

#------------------------------------
#4. Image Filtering: Gaussian Blurring
# You can change the kernel size
med_image = cv2.imread('/home/niranjan/Downloads/Talks2022/NMAMIT_Apr2022/NMAMIT_Python/NCodes/Ndata/med_image.bmp')
himg1, wimg1, channel_length = med_image.shape
print('med_image.bmp height, width, channel_length', himg1, wimg1, channel_length)
med_img = cv2.cvtColor(med_image, cv2.COLOR_BGR2GRAY)

gausBlur = cv2.GaussianBlur(med_img, (5,5),0) 
kernel = np.array([[0, -1, 0],
                   [-1, 5,-1],
                   [0, -1, 0]])
image_sharp = cv2.filter2D(med_img, ddepth=-1, kernel=kernel)

# stacking images side-by-side
res = np.hstack((med_img, gausBlur, image_sharp))
cv2.imshow('CT Image, Smoothed, Sharpened', res)
cv2.waitKey(4000)
cv2.destroyAllWindows()

#------------------------------------
#5. LMS Filter
X = []
Y = []
with open('/home/niranjan/Downloads/Talks2022/NMAMIT_Apr2022/NMAMIT_Python/NCodes/Ndata/foetal_ecg.dat', 'r') as datafile:
	plotting = csv.reader(datafile, delimiter='\t')
	for ROWS in plotting:
		X.append(float(ROWS[1]))
		Y.append(float(ROWS[7]))

plt.subplot(2,1,1)
plt.plot(X)
plt.title('Abdominal ECG')
plt.subplot(2,1,2)
plt.plot(Y)
plt.title('Maternal ECG')
plt.show()

#Y : reference/secondary, maternal;  X : primary, abdominal
mu=0.0000001
length=len(Y)
print('length',length)
filter = np.zeros([length+1, 1], dtype='float')
filter[1]=4
error = np.zeros([length+1, 1], dtype='float')

for i in range(1, length):
      error[i]=np.array(X[i])-filter[i]*np.array(Y[i])
      filter[i+1]=filter[i]+2*mu*error[i]*np.array(Y[i])

#plt.plot(X, 'r')
#plt.plot(filter, 'b')
plt.plot(error[501:length], 'g')
plt.title('LMS Filter Output')
plt.show()

plt.subplot(3,1,1)
plt.plot(X)
plt.title('Abdominal ECG')
plt.subplot(3,1,2)
plt.plot(Y)
plt.title('Maternal ECG')
plt.subplot(3,1,3)
signal = np.zeros((length, 1), dtype = "float")
signal[501:length] = error[501:length]
plt.plot(signal)
plt.title('LMS Filter Output')
plt.show()

#------------------------------------
#5. Segmentation
im_gray = cv2.imread("/home/niranjan/Downloads/Talks2022/NMAMIT_Apr2022/NMAMIT_Python/NCodes/Ndata/maxresdefault.jpg") #Echo-1(1)
twoDimage =  im_gray.reshape((-1,3))
twoDimage = np.float32(twoDimage)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
attempts=10

ret,label,center=cv2.kmeans(twoDimage,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
result_image = res.reshape((im_gray.shape))

#plt.axis('off')
#plt.imshow(result_image)
im_gray1 = cv2.resize(im_gray, (640, 360), interpolation = cv2.INTER_LINEAR)
result_image1 = cv2.resize(result_image, (640, 360), interpolation = cv2.INTER_LINEAR)
result_image1 = cv2.cvtColor(result_image1, cv2.COLOR_BGR2GRAY)
im_gray1 = cv2.cvtColor(im_gray1, cv2.COLOR_BGR2GRAY)
res1 = np.hstack((im_gray1, result_image1))
cv2.imshow('Segmented Histopathology Image', res1)
cv2.waitKey(40000)
cv2.destroyAllWindows()


