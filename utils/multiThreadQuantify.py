import os
import copy
import numpy as np
import tifffile as tiff
from skimage.morphology import ball
from skimage.measure import regionprops
from skimage.transform import rescale

# import pyvistaqt as pvq
# import pyvista as pv

from imgmisc import listdir, to_uint8, autocrop, cut, get_resolution, mkdir, symlink

from img2org.quantify import find_neighbors
from img2org.quantify import get_l1
from img2org.quantify import get_wall
from img2org.quantify import area_between_walls
import mahotas as mh
from img2org.quantify import get_l1, get_layers


import sys
import multiprocessing
from multiprocessing import Pool, cpu_count


def spaceMatcher(image, spacing, order=0):
    factor = spacing[1] / spacing[0]
    newImage = rescale(image, [factor, 1., 1.], order=order, preserve_range=True)
    changeFactor = newImage.shape[0] / image.shape[0]
    newSpacing = (spacing[0]*changeFactor, spacing[1], spacing[2])
    print(f"Old spacing: {spacing}")
    print(f"New spacing: {newSpacing}")
    return newImage, newSpacing
    
    
    


def padding(array, shape):
    xx, yy, zz = shape
    h = array.shape[0]
    w = array.shape[1]
    d = array.shape[2]
    a = (xx - h) // 2
    aa = xx - a - h
    b = (yy - w) // 2
    bb = yy - b - w
    c = (zz - d) // 2
    cc = zz - c - d
    return np.pad(array, pad_width=((a, aa), (b, bb), (c, cc)), mode='constant')
        
def cleaner(cellDict):
    newCellDict = copy.deepcopy(cellDict)
    markedForRemoval = []
    for key in cellDict.keys():
        neighs = cellDict[key]["neighbors"]
        for neigh in neighs:
            try:
                neighsNeighs = cellDict[neigh]["neighbors"]
                if key not in neighsNeighs:
                    print(key, neigh)
            except:
                index = list(newCellDict[key]["neighbors"]).index(neigh)
                newCellDict[key]["neighbors"] = np.delete(newCellDict[key]["neighbors"], index)
                newCellDict[key]["a_ij"] = np.delete(newCellDict[key]["a_ij"], index)
                newCellDict[key]["P_ij"] = np.delete(newCellDict[key]["P_ij"], index)
                newCellDict[key]["P_ji"] = np.delete(newCellDict[key]["P_ji"], index)
                print('Does not exist', key, neigh, index)
    return newCellDict   



def computeRegion(cellId, coords, volume, total_pin, seg_img, pin_img, neighs, wall_selem=ball(2), resolution=(1.,1.,1.)):
    cellDict = {}
    # Retrieve and compute simple information
    cellDict['cellId'] = cellId
    cellDict['coords'] = coords
    cellDict['volume'] = volume
    cellDict['total_PIN'] = total_pin

    cell_neighs = copy.copy(neighs[cellId]) # will be modified, so need a copy
    a_ij, P_ij, P_ji = [], [], []
    for neigh in cell_neighs:
        print(cellId, neigh)
        # Compute the wall and corresponding PIN. NOTE: the wall depth is
        # always at least 2 for the computation of the wall, whereas the 
        # PIN in the extreme case of wall_depth=1 is then acquired correctly
        # by filtering out unwanted PIN above. 
        this_wall, that_wall, pij, pji = get_wall(seg_img, cell1=cellId, cell2=neigh, 
                                                  intensity_image=pin_img, selem=wall_selem)
        area = area_between_walls(this_wall, that_wall, resolution)
    
        # Remove the neighbour if the contact area doesn't exist. Otherwise,
        # add data to lists.
        if area == 0: 
            neighs[cellId] = neighs[cellId][neighs[cellId] != neigh]
            # neighs[neigh] = neighs[neigh][neighs[neigh] != region]
            continue
        else:
            a_ij.append(area)
            P_ij.append(pij)
            P_ji.append(pji)

    # Add the values to the entire collection
    cellDict["neighbors"] = neighs[cellId]
    cellDict["a_ij"] = a_ij
    cellDict["P_ij"] = P_ij
    cellDict["P_ji"] = P_ji
    return cellDict
        
        
def batch_compute(segImg, pinImg, nThreads, wall_selem, resolution):
    regProps = regionprops(segImg, pinImg)
    neighs = find_neighbors(segImg, background=0, connectivity=1)
    shape = segImg.shape
    
    batchList = []
    batch = []
    for region in regProps:
        batch.append(region)
        if len(batch) == nThreads:
            batchList.append(batch)
            batch = []

    resultList = []
    for i, batch in enumerate(batchList):
        print("Batch number "+str(i+1)+" out of "+str(len(batchList)))
        num_cores = len(batch)
        pool = multiprocessing.Pool(num_cores)
        workers = []
        for region in batch:
            cellId = region.label
            
            size = 100
            
            cent = np.array(region.centroid).astype(np.int32)
            # print(cent)
            xmin, ymin, zmin, xmax, ymax, zmax = max(cent[0]-size, 0), max(cent[1]-size, 0), max(cent[2]-size, 0), min(cent[0]+size, shape[0]-1), min(cent[1]+size, shape[1]-1), min(cent[2]+size, shape[2]-1)
            # print(xmin, ymin, zmin, xmax, ymax, zmax)
            segZoom = segImg[xmin:xmax, ymin:ymax, zmin:zmax]
            pinZoom = pinImg[xmin:xmax, ymin:ymax, zmin:zmax]

            
            coords = np.multiply(region.centroid, resolution)
            volume = np.multiply(region.image.sum(), np.prod(resolution))
            total_pin = region.intensity_image.sum()
            workers.append(pool.apply_async(computeRegion, args=(cellId, coords, volume, total_pin, segZoom, pinZoom, neighs, wall_selem, resolution)))
        final_result = [worker.get() for worker in workers]
        for result in final_result:
            resultList.append(result)
        print("Finished")
    return resultList
        
def reorganizeLabels(segImage):
    newImage = np.copy(segImage)
    regProps = regionprops(segImage)
    
    labelDict = {}
    for i, region in enumerate(regProps):
        currLabel = region.label
        newLabel = i+1
        if newLabel == currLabel:
            continue
        labelDict[currLabel] = newLabel
        print(currLabel, newLabel)
        newImage[segImage==currLabel] = newLabel
    return newImage, labelDict
        

def writeToFiles(batchList,path, initName, neighName):
    all_props = {}
    
    for cellDict in batchList:
        cellId = cellDict["cellId"] 
        all_props[cellId] = {}
        all_props[cellId]["coords"] = cellDict["coords"] 
        all_props[cellId]["volume"] = cellDict["volume"]
        all_props[cellId]["total_PIN"] = cellDict["total_PIN"]
        all_props[cellId]["neighbors"] = cellDict["neighbors"]
        all_props[cellId]["a_ij"] = cellDict["a_ij"]
        all_props[cellId]["P_ij"] = cellDict["P_ij"] 
        all_props[cellId]["P_ji"] = cellDict["P_ji"]
        
    
    
    cleanProps = cleaner(all_props)
    
    all_props = cleanProps
    
    # Compute normalised PIN values
    total_PIN = np.array([r['total_PIN'] / r['volume'] for r in all_props.values()])
    membrane_PIN = [np.array(val["P_ij"]) / val["a_ij"] for val in all_props.values()] 
    membrane_AUX = np.array([np.array([sum(val["P_ij"]) / sum(val["a_ij"])] * len(val["a_ij"])) if len(val["a_ij"]) > 0 else np.array([]) for val in all_props.values()])

    # Calculate or assign init values based on the above
    init_total_PIN = total_PIN  
    init_auxin = total_PIN 
    init_total_AUX = total_PIN 
    init_membrane_PIN = membrane_PIN 
    init_membrane_AUX = membrane_AUX 
    
    # Normalize Added by Jonas
    normalisation_factor = np.nanmean([np.nansum(pp) for pp in total_PIN])
    init_auxin = init_auxin / normalisation_factor 
    init_total_PIN = init_total_PIN / normalisation_factor 
    init_total_AUX = init_total_AUX / normalisation_factor 
    init_membrane_PIN = init_membrane_PIN  / normalisation_factor 
    init_membrane_AUX = init_membrane_AUX / normalisation_factor 
    
    # Write init file
    init = str(len(all_props)) + " 9" 
    neigh = str(len(all_props)) + " 4"
    # cellPos = 0
    # listIndex = 0
    for cell_id in all_props.keys():

        # Cell variables
        init += "\n" + str(all_props[cell_id]["coords"][0])
        init += " " + str(all_props[cell_id]["coords"][1])
        init += " " + str(all_props[cell_id]["coords"][2])
        init += " " + str(all_props[cell_id]["volume"])
        init += " " + str(init_total_PIN[cell_id-1])
        init += " " + str(init_total_AUX[cell_id-1])
        init += " " + str(init_auxin[cell_id-1])
        init += " " + str(0) 
        init += " " + str(0) 
        # init += " " + str(all_props[cell_id]["L1"]) 
        # init += " " + str(all_props[cell_id]["L2"]) 
        
        # Membrane variables
    
        n_neighs = len(all_props[cell_id]["neighbors"])
        neigh += "\n" + str(cell_id-1) 
        neigh += " " + str(n_neighs)
    
        for ii in range(n_neighs):
            neigh += " " + str(all_props[cell_id]["neighbors"][ii] - 1)
        for ii in range(n_neighs):
            neigh += " " + str(all_props[cell_id]["a_ij"][ii])
        for ii in range(n_neighs):
            neigh += " " + str(init_membrane_PIN[cell_id-1][ii])
        for ii in range(n_neighs):
            neigh += " " + str(init_membrane_AUX[cell_id-1][ii])
        for jj in range(n_neighs): # Fluxes placeholder
            neigh += " " + str(0)
        
        # listIndex += 1
        # # if not stepped:
        # cellPos += 1
        
        
            


    init_filepath = to_raw(path+'/'+initName+'.init')
    neigh_filepath = to_raw(path+'/'+neighName+'.neigh')

    with open(init_filepath, 'w') as f1:
        f1.write(init)
    with open(neigh_filepath, 'w') as f2:
        f2.write(neigh)

def to_raw(string):
    return fr"{string}"

if __name__ == "__main__":
    sourceList = [r"/home/jonas/projects/allImages/seg/C2-201112-PIN_GFP-MM_dsRed-WT-SAM1-FM1_predictions_multicut-refined.tif",
                  r"/home/jonas/projects/allImages/seg/C2-201112-PIN_GFP-MM_dsRed-WT-SAM2-FM1_predictions_multicut-refined.tif",
                  r"/home/jonas/projects/allImages/seg/C2-201112-PIN_GFP-MM_dsRed-WT-SAM4-FM1_predictions_multicut-refined.tif",
                  r"/home/jonas/projects/allImages/seg/C2-201112-PIN_GFP-MM_dsRed-WT-SAM5-FM1_predictions_multicut-refined.tif",
                  r"/home/jonas/projects/allImages/seg/C2-201112-PIN_GFP-MM_dsRed-WT-SAM5-FM2_predictions_multicut-refined.tif",
                  r"/home/jonas/projects/allImages/seg/C2-201112-PIN_GFP-MM_dsRed-WT-SAM5-FM3_predictions_multicut-refined.tif",
                  r"/home/jonas/projects/allImages/seg/C2-201112-PIN_GFP-MM_dsRed-WT-SAM6-FM1_predictions_multicut-refined.tif",
                  r"/home/jonas/projects/allImages/seg/C2-210305-PIN1GFP-MMdsRed-WT_flowers-SAM2-FM2_predictions_multicut-refined.tif",
                  r"/home/jonas/projects/allImages/seg/C2-210305-PIN1GFP-MMdsRed-WT_flowers-SAM3-FM1_predictions_multicut-refined.tif",
                  r"/home/jonas/projects/allImages/seg/C2-210305-PIN1GFP-MMdsRed-WT_flowers-SAM6-FM2_predictions_multicut-refined.tif",
                  r"/home/jonas/projects/allImages/seg/C2-210305-PIN1GFP-MMdsRed-WT_flowers-SAM6-FM1_predictions_multicut-refined.tif",
                  r"/home/jonas/projects/allImages/seg/C2-210305-PIN1GFP-MMdsRed-WT_flowers-SAM2-FM1_predictions_multicut-refined.tif"]
    
    pinList = [r"/home/jonas/projects/allImages/pin/C1-201112-PIN_GFP-MM_dsRed-WT-SAM1-FM1.tif",
                  r"/home/jonas/projects/allImages/pin/C1-201112-PIN_GFP-MM_dsRed-WT-SAM2-FM1.tif",
                  r"/home/jonas/projects/allImages/pin/C1-201112-PIN_GFP-MM_dsRed-WT-SAM4-FM1.tif",
                  r"/home/jonas/projects/allImages/pin/C1-201112-PIN_GFP-MM_dsRed-WT-SAM5-FM1.tif",
                  r"/home/jonas/projects/allImages/pin/C1-201112-PIN_GFP-MM_dsRed-WT-SAM5-FM2.tif",
                  r"/home/jonas/projects/allImages/pin/C1-201112-PIN_GFP-MM_dsRed-WT-SAM5-FM3.tif",
                  r"/home/jonas/projects/allImages/pin/C1-201112-PIN_GFP-MM_dsRed-WT-SAM6-FM1.tif",
                  r"/home/jonas/projects/allImages/pin/C1-210305-PIN1GFP-MMdsRed-WT_flowers-SAM2-FM2.tif",
                  r"/home/jonas/projects/allImages/pin/C1-210305-PIN1GFP-MMdsRed-WT_flowers-SAM3-FM1.tif",
                  r"/home/jonas/projects/allImages/pin/C1-210305-PIN1GFP-MMdsRed-WT_flowers-SAM6-FM2.tif",
                  r"/home/jonas/projects/allImages/pin/C1-210305-PIN1GFP-MMdsRed-WT_flowers-SAM6-FM1.tif",
                  r"/home/jonas/projects/allImages/pin/C1-210305-PIN1GFP-MMdsRed-WT_flowers-SAM2-FM1.tif"]
    
    targetList = [r"/home/jonas/projects/allImages/atlas/segmented_tiffs/10h_segmented.tif",
                 r"/home/jonas/projects/allImages/atlas/segmented_tiffs/40h_segmented.tif",
                 r"/home/jonas/projects/allImages/atlas/segmented_tiffs/96h_segmented.tif",
                 r"/home/jonas/projects/allImages/atlas/segmented_tiffs/120h_segmented.tif",
                 r"/home/jonas/projects/allImages/atlas/segmented_tiffs/132h_segmented.tif"]
    
    mappedSegList = [r'/home/jonas/projects/allImages/120hSources/mappedSegData_210305_22_highres.tif',
                     r'/home/jonas/projects/allImages/120hSources/mappedSegData_210305_31_highres.tif',
                     r'/home/jonas/projects/allImages/120hSources/mappedSegData_201112_41_highres.tif'
                     r'/home/jonas/projects/allImages/120hSources/mappedSegData_210305_62_highres.tif']
    
    mappedPinList = [r'/home/jonas/projects/allImages/120hSources/mappedPinData_210305_22_highres.tif',
                     r'/home/jonas/projects/allImages/120hSources/mappedPinData_210305_31_highres.tif',
                     r'/home/jonas/projects/allImages/120hSources/mappedPinData_201112_41_highres.tif',
                     r'/home/jonas/projects/allImages/120hSources/mappedPinData_210305_62_highres.tif']
    
    targetResList = [(0.12621046353913226, 0.2415319, 0.2415319),
                    (0.12797625374470484, 0.2415319, 0.2415319),
                    (0.16652491797820784, 0.2361645, 0.2361645),
                    (0.17078415345851233, 0.2361645, 0.2361645),
                    (0.12529532381630637, 0.1735086, 0.1735086)]    
        
    sourceResList = [(0.19999999999999998, 0.20756646174321786, 0.20756646174321786), 
                     (0.19999999999999998, 0.20756646174321786, 0.20756646174321786), 
                     (0.19999999999999998, 0.231375702080646, 0.231375702080646), 
                     (0.19999999999999998, 0.10664973922005766, 0.10664973922005766),
                     (0.19999999999999998, 0.13437085549892572, 0.13437085549892572),
                     (0.19999999999999998, 0.09745106032113438, 0.09745106032113438),
                     (0.19999999999999996, 0.06310707427778371, 0.06310707427778371),
                     (0.25      , 0.42071381/2, 0.42071381/2),
                     (0.25     , 0.4378858/2, 0.4378858/2),
                     (0.25, 0.18869677820449887, 0.18869677820449887),
                     (0.25, 0.17264853272876005, 0.17264853272876005),
                     (0.25, 0.15884092787927093, 0.15884092787927093)]



    compPins = [r"/home/jonas/projects/allImages/compareDecomp/rawPin_210305_22.tif",
                r"/home/jonas/projects/allImages/compareDecomp/selfComposed_210305_22_180_360.tif",
                r"/home/jonas/projects/allImages/compareDecomp/selfComposed_210305_22_90_180.tif",
                r"/home/jonas/projects/allImages/compareDecomp/selfComposed_210305_22_45_90.tif",
                r"/home/jonas/projects/allImages/compareDecomp/selfComposed_210305_22_225_45.tif",
                r"/home/jonas/projects/allImages/compareDecomp/selfComposed_210305_22_1125_225.tif"]



    # im1 = tiff.imread(r'/home/jonas/projects/allImages/120hSources/mappedPinData_210305_22_highres.tif').astype(np.int32)
    # im2 = tiff.imread(r'/home/jonas/projects/allImages/120hSources/mappedPinData_210305_31_highres.tif').astype(np.int32)
    # im3 = tiff.imread(r'/home/jonas/projects/allImages/120hSources/mappedPinData_201112_41_highres.tif').astype(np.int32)
    # im4 = tiff.imread(r'/home/jonas/projects/allImages/120hSources/mappedPinData_210305_62_highres.tif').astype(np.int32)

    
    # # stds = np.std( np.array([im1, im2, im3, im4]), axis=0 )
    # means = np.mean( np.array([ im1, im2,im3,  im4]), axis=0 )

    

    # dataIndex = 9
    resIndex = 0
    compIndex = 0
    segPath = sourceList[resIndex] #r'/home/jonas/projects/allImages/120hSources/mappedSegData_210305_22_highres.tif' #targetList[dataIndex]
    pinPath = pinList[resIndex]
    
    # segImg = tiff.imread(r"/home/jonas/projects/allImages/seg/C2-210305-PIN1GFP-MMdsRed-WT_flowers-SAM4-FM1_ROTATED180_predictions_multicut-refined.tif").astype(np.int32)
    # pinImg = tiff.imread(r'/home/jonas/projects/allImages/pin/C1-201112-PIN_GFP-MM_dsRed-WT-SAM4-FM1_ROTATED180.tif').astype(np.int32)
    
    
    

    
    resolution = sourceResList[resIndex] #get_resolution(r'/home/jonas/projects/allImages/SAMimages/segmented_refined/131007-PIN_GFP-acyl_YFP-plant_3-0h_stackreg_crop_wiener_C0_predictions_multicut-refined.tif')#sourceResList[resIndex]
    segImg = tiff.imread(segPath).astype(np.int16)
    pinImg = tiff.imread(pinPath).astype(np.uint16)
    
    segImg, labelDict = reorganizeLabels(segImg)
        
    pinImg, newRes = spaceMatcher(pinImg, resolution)
    segImg, newRes = spaceMatcher(segImg, resolution)
    
    resolution = newRes
    
    
    if compIndex == 0:
        segShape = segImg.shape[0] + 30, segImg.shape[1] + 30, segImg.shape[2] + 30
        segImg = padding(segImg, segShape)
        pinImg = padding(pinImg, segShape)
    else:
        segShape = segImg.shape[0] + 30, segImg.shape[1] + 30, segImg.shape[2] + 30
        segImg = padding(segImg, segShape)

    
    batchList = batch_compute(segImg.astype(np.int16), pinImg.astype(np.uint16), 24, ball(2), resolution)

    name = '11_forcomp_raw'
    
    writeToFiles(batchList, name, name)
    
    
    # norms = means/(stds+1.)
    # norms[norms>5]=5
        
    p = pvq.BackgroundPlotter()
    p.add_volume(segImg, opacity='linear', resolution=[1,1,1],shade=True,diffuse=1.5, specular=1.)
    p.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    """
    
    
    # segPath = r'/home/jonas/projects/concentrationApproach/padSegAtlas.tif'
    # pinPath = r'/home/jonas/projects/concentrationApproach/recompPinPadAtlas.tif'
    # segPath = r'/home/jonas/projects/elastixRegi/transSegData.tif'
    # pinPath = r'/home/jonas/projects/elastixRegi/transPinData.tif'
    
    # segPath = r'/home/jonas/projects/allImages/120hSources/mappedSegData_210305_22.tif'
    # pinPath = r'/home/jonas/projects/allImages/120hSources/mappedPinData_210305_22.tif'
    
    segPath = r'/home/jonas/projects/allImages/120hSources/mappedSegData_210305_22_highRes.tif'
    # pinPath = r'/home/jonas/projects/allImages/120hSources/mappedPinData_210305_22.tif'
    pinPath = r'/home/jonas/projects/allImages/120hSources/mappedPinData_210305_22_highRes.tif'
    
    # segPath = r'/home/jonas/projects/allImages/seg/C2-210305-PIN1GFP-MMdsRed-WT_flowers-SAM2-FM2_predictions_multicut-refined.tif'
    # pinPath = r'/home/jonas/projects/allImages/pin/C1-210305-PIN1GFP-MMdsRed-WT_flowers-SAM2-FM2.tif'
    
    segImg = tiff.imread(segPath).astype(np.int32)
    segImg, labelDict = reorganizeLabels(segImg)
    pinImg = tiff.imread(pinPath).astype(np.int32)
    # pinImg2 = tiff.imread(pinPath2).astype(np.int32)
    # resolution = get_resolution(r'/home/jonas/projects/allImages/pin/C1-201112-PIN_GFP-MM_dsRed-WT-SAM2-FM1.tif')
    # resolution = (0.2361645, 0.2361645, 0.17078415345851233)
    resolution = (0.17078415345851233,0.2361645, 0.2361645)

    # resolution = (0.25, 0.42071381, 0.42071381)
    
    # meanPinImg = (pinImg + pinImg2)/2
    
    batchList = batch_compute(segImg, pinImg, 24, ball(2), resolution)
    
    # resMod = (0.17078415345851233,0.2361645, 0.2361645)
    
    

    
    
    
    
    
    
    
    
    
    
    # p = pvq.BackgroundPlotter()
    # p.add_volume(meanPinImg, opacity='linear', resolution=[1,1,1],shade=True,diffuse=1.5, specular=1.)
    # p.show()   
    
    
    
    
    
    
    
    # print(sys.argv)
    # if len(sys.argv) < 2:
    #     n_cores = cpu_count() - 1
    # else:
    #     n_cores = int(sys.argv[1])
    
    
    # print(f'Using {n_cores} cores')
    # args = np.arange(len(seg_files))
    # p = Pool(n_cores)
    # p.map(quantify_single, args)
    # p.close()
    # p.join()
    """