# -*- coding: utf-8 -*-
"""
Created on Tue May 23 17:09:39 2023

@author: Usuario
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 10:08:44 2022

@author: Usuario
trackinbg pp 2um in all the visible area
"""




import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import trackpy as tp
import pandas as pd
from tqdm import tqdm
import pims
import shutil
from skimage.morphology import extrema
from scipy import ndimage

def fond_m(image,prim_frame,inten=255):
    pas = len(image)//50
    test = [cv2.threshold(x,inten,255,cv2.THRESH_TRUNC)[1] for x in image[prim_frame::pas]]
    fond= np.mean(test, axis=0)
    del test
    #fond=np.array(fond,dtype=np.uint8)
    return fond


def crop_ajustable(img,x,y,t):#fait un crop
    ymax,xmax=img.shape
    return img[max([y-t,0]):min([y+t,ymax]),max([x-t,0]):min([x+t,xmax])]



def mask_circle(shape,X,Y,R1):
    mask = np.zeros(shape,dtype=np.uint8)
    mask = cv2.circle(mask, (int(X),int(Y)), int(R1), 1, -1)
    return mask


def ext(L):#crée une liste ordonnée de l'ensemble des éléments dans une série pandas
    return sorted(list(set(L)))


def approx_ultimate_points(bw):
    bw_dist = ndimage.distance_transform_edt(bw)
    bw_maxima = extrema.h_maxima(bw_dist, 0.5)
    bw_maxima = ndimage.binary_dilation(bw_maxima)
    bw_ultimate = np.bitwise_and(bw, bw_maxima)
    # Reduce all ultimate point regions to a single point
    # (Here, we take the centroids - this wouldn't be robust for weird shapes with holes,
    # but should be ok here)
    lab_ultimate, n = ndimage.label(bw_ultimate)
    #com = ndimage.measurements.center_of_mass(lab_ultimate, lab_ultimate, index=range(1, n+1))
    return lab_ultimate



@pims.pipeline
def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


os.chdir('F:')


tol_M=1.3 #1.3 factor threshold from the average

k=40

#%%
path="EXP1_2023_03_22_17_49_30.avi"
video=pims.PyAVVideoReader(path)
frame_count = int(len(video)-1)
fps=video.frame_rate
print(frame_count,fps)
video=gray(video)
cx,cy,cz=video.frame_shape

background=fond_m(video,50)


Test=True
invert_image=True


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(1,1))
#pour faire un test



#%% Precise tracking with crop
size_gaussian=5
frame_list=range(frame_count)
#frame_list=range(35000,40000)
data=pd.DataFrame()
if(Test): frame_list=np.sort(np.random.randint(0,frame_count,100))
#frame_list=[1012]
i=0
j=0
#mkdir


for frame in tqdm(frame_list[::1]):#
    print_image=False
    img1=video[round(frame)]
    img1=img1-background
    img1=img1-np.min(img1)
    img1=img1/np.max(img1)*255
    result=img1.astype(np.uint8)[::,50:500]
    if invert_image : result=cv2.bitwise_not(result)
    result=cv2.GaussianBlur(result,(size_gaussian,size_gaussian),0)
    img2=result.copy()
# =============================================================================
#     plt.figure()
#     plt.imshow(result)
# =============================================================================
    M=int(np.mean(result))
    ret,result=cv2.threshold(result,int(M*tol_M),255,cv2.THRESH_BINARY)
# =============================================================================
#     plt.figure()
#     plt.imshow(result)
# =============================================================================
    #result = cv2.erode(result, kernel,iterations=3)
    #result=cv2.dilate(result, kernel,iterations=1)

    result=approx_ultimate_points(result)
    result[result>0]=1
# =============================================================================
#     plt.figure()
#     plt.imshow(result)
# =============================================================================
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(result.astype(np.uint8), None, None, None, 8, cv2.CV_16U)
# =============================================================================
#     plt.figure()
#     plt.imshow(result)
# =============================================================================
    h=pd.DataFrame(np.array([centroids[1:,0],centroids[1:,1],stats[1:,cv2.CC_STAT_AREA]]).T,
           columns=('x','y','area'))
    plt.figure()
    plt.imshow(img2)
    plt.plot(h.x,h.y, color='r',marker='.',linestyle='')
    plt.title(frame)
    plt.show()

        


