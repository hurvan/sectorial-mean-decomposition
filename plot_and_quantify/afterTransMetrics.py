#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 08:48:02 2021

@author: jonas
"""
import tifffile
import os
import sys
from skimage.measure import regionprops
import numpy as np
import matplotlib.pylab as plt 
import pyvistaqt as pvq
import pyvista as pv
from skimage.morphology import ball
import mahotas as mh
import matplotlib



def generateMetrics(preProps, postProps):
    
    cellDict = {}
    
    for preRegion in preProps:
        noMatch = False
        preId = preRegion.label
        for i, postRegion in enumerate(postProps):
            
            if postRegion.label == preId:
                break
            
            elif i == len(postProps)-1:
                noMatch = True
        if noMatch:
            continue
        postId = postRegion.label
        
        cellDict[preId] = {}
        
        cellDict[preId]["volumeFrac"] = postRegion.area / preRegion.area
        cellDict[preId]["pinConcFrac"] = ( postRegion.intensity_image.sum() / postRegion.area ) / ( preRegion.intensity_image.sum() / preRegion.area )

        cellDict[preId]["volumePairs"] = [postRegion.area , preRegion.area]
        cellDict[preId]["pinConcPairs"] = [postRegion.intensity_image.sum() / postRegion.area ,  preRegion.intensity_image.sum() / preRegion.area ]

        print(cellDict[preId]["volumeFrac"], cellDict[preId]["pinConcFrac"])


    return cellDict







if __name__ == "__main__":
    preSegPath = r'/home/jonas/projects/allImages/seg/C2-210305-PIN1GFP-MMdsRed-WT_flowers-SAM3-FM1_predictions_multicut-refined.tif'
    prePinPath = r'/home/jonas/projects/allImages/pin/C1-210305-PIN1GFP-MMdsRed-WT_flowers-SAM3-FM1.tif'
    
    postSegPath = r'/home/jonas/projects/allImages/120hSources/transSegData_210305_31.tif'
    postPinPath = r'/home/jonas/projects/allImages/120hSources/transPinData_210305_31.tif'
    
    
    tPins = [r"/home/jonas/projects/allImages/132hSource/transPinData_201112_11.tif",
             r"/home/jonas/projects/allImages/132hSource/transPinData_201112_21.tif"]
    
    tSegs = [r"/home/jonas/projects/allImages/132hSource/transSegData_201112_11.tif",
             r"/home/jonas/projects/allImages/132hSource/transSegData_201112_21.tif"]
    
    
    
    preSegPathList = [r'/home/jonas/projects/allImages/seg/C2-201112-PIN_GFP-MM_dsRed-WT-SAM1-FM1_predictions_multicut-refined.tif',
                      r'/home/jonas/projects/allImages/seg/C2-201112-PIN_GFP-MM_dsRed-WT-SAM2-FM1_predictions_multicut-refined.tif']
   
    prePinPathList = [r'/home/jonas/projects/allImages/pin/C1-201112-PIN_GFP-MM_dsRed-WT-SAM1-FM1.tif',
                      r'/home/jonas/projects/allImages/pin/C1-201112-PIN_GFP-MM_dsRed-WT-SAM2-FM1.tif']
    
    perfectLine = []
    pre = []
    post = []
    
    font = {'family' : 'normal',
    # 'weight' : 'bold',
    'size'   : 14}

    matplotlib.rc('font', **font)
    
    for i in range(2):
        print(i)
        
        preSegPath = preSegPathList[i]
        prePinPath = prePinPathList[i]
        postSegPath = tSegs[i]
        postPinPath = tPins[i]
    
        preSeg = tifffile.imread(preSegPath).astype(np.int32)
        prePin = tifffile.imread(prePinPath).astype(np.int32)
        postSeg = tifffile.imread(postSegPath).astype(np.int32)
        postPin = tifffile.imread(postPinPath).astype(np.int32)
        
        # wallBall = ball(2)
        
        # preBord = mh.borders(preSeg, Bc=wallBall)
        # postBord = mh.borders(postSeg, Bc=wallBall)
        
        # prePin *= preBord.astype(np.int32)
        # postPin *= postBord.astype(np.int32)
        preProps = regionprops(preSeg, prePin)
        postProps = regionprops(postSeg, postPin)
        
        
        metricDict = generateMetrics(preProps, postProps)
        
    
        
    
        
        for j, key in enumerate(metricDict.keys()):
            # plt.plot(metricDict[key]["pinConcPairs"][0], metricDict[key]["pinConcPairs"][1], 'r+')
            # perfectLine.append(metricDict[key]["pinConcPairs"][1])
            
            # plt.plot(metricDict[key]["volumePairs"][0], metricDict[key]["volumePairs"][1], 'r+')
            pre.append(metricDict[key]["volumePairs"][1])
            post.append(metricDict[key]["volumePairs"][0])
            perfectLine.append(metricDict[key]["volumePairs"][1])
                
            

    

    for x, y in zip(post, pre):
        plt.plot(x, y, 'r+')
        
    plt.plot(perfectLine, perfectLine, 'b', label='y=x')
    
    # plt.xlabel("Post Transformation Volume")
    plt.xlabel("Post Transformation Volume")
    plt.ylabel("Pre Transformation Volume")        
            
            
    # p = pvq.BackgroundPlotter()
    # p.add_volume(postPin, opacity='linear', resolution=[1,1,1],shade=True,diffuse=1.5, specular=1.)
    # p.show()
            
            
            
            
    
    # for key in metricDict.keys():
    #     plt.plot(metricDict[key]["volumePairs"][0], metricDict[key]["volumePairs"][1], 'r+')
    
    # for i, val in enumerate(metricDict.values()):
    #     plt.plot(i,  val["pinConcFrac"], 'r+')
    
    
    # lookFor = 1553
    # for postRegion in postProps:
    #     if postRegion.label == lookFor:
    #         break
        
    # for preRegion in preProps:
    #     if preRegion.label == lookFor:
    #         break
    
    # plotImg = (postRegion.image).astype(np.int16)
    # plotImg2 = (preRegion.image).astype(np.int16)
    
    # p = pvq.BackgroundPlotter()
    # p.add_volume(plotImg2, opacity='linear', resolution=[1,1,1],shade=True,diffuse=1.5, specular=1.)
    # p.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



