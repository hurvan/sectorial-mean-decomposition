#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 20:57:31 2021

@author: jonas
"""

import numpy as np
import tifffile
import pyvistaqt as pvq
import pyvista as pv
import matplotlib.pyplot as plt
import mahotas as mh
from imgmisc import get_resolution



def readData(path):
    with open(path, "r") as f:
        rawLines = f.readlines()
        timeStepList = []
        for i, line in enumerate(rawLines):
            line = line.strip()
            rawLineList = line.split(' ')
            
            if i == 0:
                nrOfTimesteps = int(rawLineList[0])
            elif i == 1:
                nrOfCells = int(rawLineList[0])
                nrOfVars = int(rawLineList[1])
                coords = []
                vols = []
                vals = []
                neighs = []
            else:
                if len(rawLineList) == nrOfVars:
                    coords.append( [float(rawLineList[0]), float(rawLineList[1]), float(rawLineList[2])] )
                    vols.append(float(rawLineList[3]))
                    vals.append([ float(d)  for d in rawLineList[4:-1]])
                    neighs.append(float(rawLineList[-1]))
                if (i-1) % (nrOfCells+2) == 0:
                    timeStepList.append([coords, vols, vals, neighs])
                    coords = []
                    vols = []
                    vals = []
                    neighs = []
    return timeStepList


def build3dImage(image, data):
    newImg = np.zeros_like(image).astype(np.float32)
    
    for i, val in enumerate(data[:,1]):
        newImg[image==(i+1)] = val
        print(i+1)
    
    return newImg

def plotData():
    return 0










if __name__ == "__main__":
    # dataFiles = [r"/home/jonas/projects/Organism/examples/tutorials/132h/11_132.data",
    #              r"/home/jonas/projects/Organism/examples/tutorials/132h/11_132_creation.data",
    #              r"/home/jonas/projects/Organism/examples/tutorials/132h/11_raw.data",
    #              r"/home/jonas/projects/Organism/examples/tutorials/132h/11_raw_creation.data",
    #              r"/home/jonas/projects/Organism/examples/tutorials/132h/21_132.data",
    #              r"/home/jonas/projects/Organism/examples/tutorials/132h/21_132_creation.data",
    #              r"/home/jonas/projects/Organism/examples/tutorials/132h/21_raw.data",
    #              r"/home/jonas/projects/Organism/examples/tutorials/132h/21_raw_creation.data",
    #              r"/home/jonas/projects/Organism/examples/tutorials/132h/mean_21_11_132.data",
    #              r"/home/jonas/projects/Organism/examples/tutorials/132h/mean_21_11_132_creation.data"]
    
    dataFiles = [r"/home/jonas/projects/allImages/selfCompFinal/11_180_360.data",
                r"/home/jonas/projects/allImages/selfCompFinal/11_225_45.data",
                r"/home/jonas/projects/allImages/selfCompFinal/11_forcomp_raw.data"]
    
    # dataFiles = [r"/home/jonas/projects/allImages/compareFlower1_2/11_21.data",
    #              r"/home/jonas/projects/allImages/compareFlower1_2/21_11.data"]
    
    
    # targetImagePath = r"/home/jonas/projects/allImages/seg/C2-201112-PIN_GFP-MM_dsRed-WT-SAM1-FM1_predictions_multicut-refined.tif"
    
    targetImagePath = r"/home/jonas/projects/allImages/seg/C2-201112-PIN_GFP-MM_dsRed-WT-SAM1-FM1_predictions_multicut-refined.tif"
    resolution = get_resolution(targetImagePath)
    targetImage = tifffile.imread(targetImagePath) 
    
    # targetImagePath = r"/home/jonas/projects/allImages/seg/C2-201112-PIN_GFP-MM_dsRed-WT-SAM1-FM1_predictions_multicut-refined.tif"
    # resolution = (0.19999999999999998, 0.20756646174321786, 0.20756646174321786)
    # targetImage = tifffile.imread(targetImagePath) 
    
    # targetImagePath = r"/home/jonas/projects/allImages/seg/C2-201112-PIN_GFP-MM_dsRed-WT-SAM2-FM1_predictions_multicut-refined.tif"
    # resolution = (0.19999999999999998, 0.20756646174321786, 0.20756646174321786)
    # targetImage = tifffile.imread(targetImagePath) 

    
    dataList = readData(dataFiles[2])
    finalData = np.array(dataList[-1][2])
    
    # borders = mh.borders(targetImage)
    
    newImg = build3dImage(targetImage, finalData)
    
    # plt.imshow(newImg[:,:,350] )
    
    
    p = pvq.BackgroundPlotter()
    p.set_background("white")
    p.add_volume(newImg, opacity='linear', resolution=resolution,shade=True,diffuse=1.5, specular=1.)
    p.show()
    
    
    



