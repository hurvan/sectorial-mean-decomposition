#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 07:59:06 2021

@author: jonas
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

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
                    
                    
                    # print(i, rawLineList)

    return timeStepList


if __name__ == "__main__":
    # outPath = r"/home/jonas/projects/Organism/examples/tutorials/jonasTest/out.data"
    
    paths = [r"/home/jonas/projects/allImages/compareDecomp/raw.data",
             r"/home/jonas/projects/allImages/compareDecomp/180_360.data",
             r"/home/jonas/projects/allImages/compareDecomp/90_180.data",
             r"/home/jonas/projects/allImages/compareDecomp/45_90.data",
             r"/home/jonas/projects/allImages/compareDecomp/225_45.data",
             r"/home/jonas/projects/allImages/compareDecomp/1125_225.data"]
    
    
    startList = []
    endList = []
    
    for path in paths:
         timeStepList = readData(path)
         startAuxinData = np.array(timeStepList[0][2])
         finalAuxinData = np.array(timeStepList[-1][2])
         
         startList.append(startAuxinData)
         endList.append(finalAuxinData)
         
         
        
    # rawSim = startList[0]
    # errorsStart = []
    # stdsStart = []
    # for sim in startList[1::]:
    #     diff = rawSim - sim
    #     error = np.mean(diff[:,1]**2)
    #     std = np.std(diff[:,1]**2)
    #     errorsStart.append(error)
        
    rawSim = endList[0]
    errorsFinal = []
    stdsFinal = []
    for sim in endList[1::]:
        diff = rawSim - sim
        error = np.mean(diff[:,1]**2)
        std = np.std(diff[:,1])
        errorsFinal.append(error)
        stdsFinal.append(std)
         
         
    X = [r"$180^{\circ} , 360^{\circ}$", r"$90^{\circ}, 180^{\circ}$", r"$45^{\circ}, 90^{\circ}$", r"$22.5^{\circ}, 45^{\circ}$", r"$11.25^{\circ}, 22.5^{\circ}$"]
    
    # fig = plt.figure()
    # ax = fig.add_axes([0,0,1,1])
    # ax.bar(X , errorsFinal)
    
    # matplotlib.rc('xtick', labelsize=11) 
    # matplotlib.rc('ytick', labelsize=11) 
    
    font = {'family' : 'normal',
            # 'weight' : 'bold',
            'size'   : 12}

    matplotlib.rc('font', **font)
    
    plt.bar(X, errorsFinal, color='red', edgecolor='k')
    # ax.bar(X + 0.5, errorsFinal, color = 'g', width = 0.5)
    
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Interval Size In Degrees")
         
    plt.show()

    # diff = np.array(timeStepList[-1][2]) - np.array(timeStepList[0][2])
    
    # error = np.sqrt(np.mean(diff[:,1]**2))
    # print(error)
    # plt.plot(diff[:,1]**2)
    
    