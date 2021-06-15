import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from skimage import measure
from skimage.measure import regionprops
from skimage.morphology import ball
from img2org.quantify import find_neighbors
from img2org.quantify import get_l1
from img2org.quantify import get_wall
from img2org.quantify import area_between_walls
import mahotas as mh
from img2org.quantify import get_l1, get_layers
from scipy import ndimage
from scipy.spatial.distance import cdist
from imgmisc import get_resolution, get_layers
import tifffile
import tifffile as tiff
# import pyvistaqt as pvq
# import pyvista as pv
import multiprocessing
import copy
import pickle

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

def asCartesian(rthetaphi):
    #takes list rthetaphi (single coord)
    r       = rthetaphi[0]
    theta   = rthetaphi[1]* np.pi/180 # to radian
    phi     = rthetaphi[2]* np.pi/180
    x = r * np.sin( theta ) * np.cos( phi )
    y = r * np.sin( theta ) * np.sin( phi )
    z = r * np.cos( theta )
    return np.array([x,y,z])

def asSpherical(xyz):
    #takes list xyz (single coord)
    x       = xyz[0]
    y       = xyz[1]
    z       = xyz[2]
    r       =  np.sqrt(x*x + y*y + z*z)
    theta   =  np.arccos(z/r)*180/ np.pi #to degrees
    phi     =  np.arctan2(y,x)*180/ np.pi
    return np.array([r,theta,phi])

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec

def projectTensorOnVector2(tensor, vector):
    vector = np.array(vector)
    newVector = np.zeros(vector.shape)
    scales = [0.]
    tempVecs = []
    overlaps = []
    
    for row in tensor:
        angle = angle_between(row, vector)
        if angle > 90:
            continue
        rowScale = np.linalg.norm(row)
        v_norm = np.sqrt(sum(vector**2))   
        scale = (np.dot(row, vector)/v_norm**2)
        overlap = (np.dot(row/rowScale, vector)/v_norm**2)
        proj_of_row_on_v = scale*vector
        newVector += proj_of_row_on_v
        tempVecs.append(proj_of_row_on_v)
        scales.append(scale)
        overlaps.append(overlap)
        
    maxOverlap = np.argmax(overlaps)
    newVector = scales[maxOverlap]*vector
    return newVector




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
        

def writeToFiles(batchList,totalPinList,path, initName, neighName): #totalPinList
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
    if totalPinList == None:
        total_PIN = np.array([r['total_PIN'] / r['volume'] for r in all_props.values()])
    else:
        total_PIN = np.array(totalPinList) # Already concentration
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
        


    init_filepath = path+initName+'.init'
    neigh_filepath = path+neighName+'.neigh'

    with open(init_filepath, 'w') as f1:
        f1.write(init)
    with open(neigh_filepath, 'w') as f2:
        f2.write(neigh)


def calculateOverlapFastWeighted(segSource, segTarget):
    fullRegprop = regionprops(segTarget, segSource)
    targetL1, targetL2 = get_layers(segTarget, bg=0, depth=2)
    sourceL1, sourceL2 = get_layers(segSource, bg=0, depth=2)
    cellDict = {}
    
    for region in fullRegprop:
        targetId = region.label
        print("Overlap key: "+str(targetId))

        overlapDict = {}
        
        targetVolume = region.area
        zoomRegprop = regionprops(region.intensity_image)
        
        overlapList = []
        sourceIdList = []
        
        for zoomRegion in zoomRegprop:
            sourceId = zoomRegion.label
            sourceVolume = zoomRegion.area
            
            if targetId in targetL1 and sourceId in sourceL1:
                print('L1L1')
                weight = 10.
            elif targetId in targetL2 and sourceId in sourceL2:
                print('L2L2')
                weight = 10.
            # elif (targetId not in targetL1  or targetId not in targetL2) and (sourceId not in sourceL1 or sourceId not in sourceL2):
            #     weight = 10.
            else:
                weight = 1.
            
            overlapList.append(sourceVolume / targetVolume * weight)
            sourceIdList.append(sourceId)
            
            
        if np.sum(overlapList) != 0:
            overlapList = np.array(overlapList) / np.sum(overlapList)
            
        markedForRemoval = []
        for i, overlap in enumerate(overlapList):
            if overlap < 0.05:
                markedForRemoval.append(i)
                
        
        if len(overlapList) > len(markedForRemoval):
            overlapList = np.delete(overlapList, markedForRemoval)
            sourceIdList = np.delete(sourceIdList, markedForRemoval)

                
        if np.sum(overlapList) != 0:
            overlapList = np.array(overlapList) / np.sum(overlapList)
            
        for ind, over in zip(sourceIdList, overlapList):
            overlapDict[ind] = over
            
        cellDict[targetId] = overlapDict

    return cellDict


def calculateOverlapFast(segSource, segTarget):
    fullRegprop = regionprops(segTarget, segSource)
    cellDict = {}
    
    for region in fullRegprop:
        targetId = region.label
        print("Overlap key: "+str(targetId))

        overlapDict = {}
        
        targetVolume = region.area
        zoomRegprop = regionprops(region.intensity_image)
        
        overlapList = []
        sourceIdList = []
        
        for zoomRegion in zoomRegprop:
            sourceId = zoomRegion.label
            sourceVolume = zoomRegion.area
            overlapList.append(sourceVolume / targetVolume)
            sourceIdList.append(sourceId)
            
            
        if np.sum(overlapList) != 0:
            overlapList = np.array(overlapList) / np.sum(overlapList)
            
        for ind, over in zip(sourceIdList, overlapList):
            overlapDict[ind] = over
            
        cellDict[targetId] = overlapDict
            

    
    return cellDict

        

def calculateOverlap(segSource, segTarget, errorAccept):
    segSourceProps = regionprops(segSource)
    segTargetProps = regionprops(segTarget)
    overlapDict = {}
    
    cellDict1 = {}
    cellDict2 = {}
    
    for region in segSourceProps:
        cellId1 = region.label
        cellDict1[cellId1] = {}
        cellDict1[cellId1]["center"] = region.centroid
    
    for region in segTargetProps:
        cellId2 = region.label
        cellDict2[cellId2] = {}
        cellDict2[cellId2]["center"] = region.centroid
        
        
    centroids1 = np.array([np.array(r["center"]) for r in cellDict1.values()])
    centroids2 = np.array([np.array(r["center"]) for r in cellDict2.values()])
    
    distMat = cdist(centroids2, centroids1)



    for i, key in enumerate(cellDict2.keys()):
        row = distMat[i]
        indicies = range(len(row))
        row, indicies = zip(*sorted(zip(row, indicies)))        
        row = list(row)[0:20]
        indicies = list(indicies)[0:20]  
        cellDict2[key]["closest"] = [list(cellDict1.keys())[j] for j in indicies]
        
        
    
    
    for key in cellDict2.keys():
        print("Overlap key: "+str(key))

        totalOverlap = 0.
        overlapList = []
        overlapIndexList = []
        for i, neighKey in enumerate(cellDict2[key]["closest"]):
            mask1 = np.where(segSource==neighKey, 1, 0)
            mask2 = np.where(segTarget==key, 1, 0)
            
            overlap = np.logical_and(mask1, mask2)
            
            overlapFrac = np.sum(overlap).astype(np.float32) / np.sum(mask2)
            
            totalOverlap += overlapFrac
            
            
            if overlapFrac > 0:
                overlapList.append(overlapFrac)
                overlapIndexList.append(neighKey)
            
            if totalOverlap >= errorAccept:
                break
            if i > 5 and totalOverlap == 0.:
                ### If no overlap, take the nearest neighbour.
                overlapList.append(1.)
                overlapIndexList.append(cellDict2[key]["closest"][0])
                totalOverlap = 1.
                cellDict2[key]["noOverlap"] = True
                break

        if "noOverlap" not in cellDict2[key]:
            cellDict2[key]["noOverlap"] = False
    
                
        #normalize
        overlapList = np.array(overlapList)/totalOverlap
        print(overlapList)
        overlapDict = {}
        for ind, over in zip(overlapIndexList, overlapList):
            overlapDict[ind] = over
            
        cellDict2[key]["overlap"] = overlapDict
        
        

    return cellDict2

def genSphere(shape, radius, position):
    # assume shape and position are both a 3-tuple of int or float
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    semisizes = (radius,) * 3

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        # this can be generalized for exponent != 2
        # in which case `(x_i / semisize)`
        # would become `np.abs(x_i / semisize)`
        arr += (x_i / semisize) ** 2

    # the inner part of the sphere will have distance below 1
    return arr <= 1.0

def getUnitSphere(spacing1=10, spacing2=20):
    phi = np.arange(0, 180+spacing1, spacing1)
    theta = np.arange(0, 360+spacing2, spacing2)
    height = 0
    width = 0
    vertList = []
    fcList = []
    while height+1 < len(phi):
        while width+1 < len(theta):
            # print(width, len(theta))
            verts = []
            p = phi[height]
            t = theta[width]
            verts.append( asCartesian([1., t, p]) )
            height += 1
            p = phi[height]
            for i in range(2):
                t = theta[width]
                verts.append( asCartesian([1., t, p]) )
                width += 1
            height -= 1
            width -= 1
            p = phi[height]
            t = theta[width]
            verts.append( asCartesian([1., t, p]) )
            verts.append(verts[0])
            verts = np.array(verts)
            fc = [np.mean(verts[0:4,0]), np.mean(verts[0:4,1]), np.mean(verts[0:4,2])  ]
            fc /= np.linalg.norm(fc)
            vertList.append(verts)
            fcList.append(fc)

        height += 1
        width = 0

    fcList = np.array(fcList)
    vertList = np.array(vertList)
    
    phiLim = []
    for i, p in enumerate(phi):
        if i==0:
            pass
        else:
            phiLim.append([phi[i-1], p])
            
    thetaLim = []
    for i, t in enumerate(theta):
        if i==0:
            pass
        else:
            thetaLim.append([theta[i-1], t])
    
    return fcList, vertList, phiLim, thetaLim

def getInfoSphericalDecompositionNoPin(segImage,decompositionAngles=(45,90), resolution=(1,1,1)):
    regProps = regionprops(segImage)
    cellDict = {}
    
    shape = segImage.shape
    for region in regProps:
        cellId = region.label
        print("Decomposing: " + str(cellId))

        # if cellId >= 5: ############## REMOVE
        #     break

        ### filter out and crop target cell
        imageInd = np.where(segImage == cellId)
        imPoints = np.swapaxes(imageInd,0,1)
        
        if len(imPoints) < 5:
            print('skipping '+str(cellId))
            continue

        cellDict[cellId] = {}
        
        xmax, ymax, zmax = np.max(imPoints, axis=0)
        xmin, ymin, zmin = np.min(imPoints, axis=0)
        xmax, ymax, zmax = np.min([xmax+5, shape[0]-1]), np.min([ymax+5, shape[1]-1]), np.min([zmax+5, shape[2]-1])
        xmin, ymin, zmin = np.max([xmin-5, 0]), np.max([ymin-5, 0]), np.max([zmin-5, 0]) 
        segZoom = segImage[xmin:xmax, ymin:ymax, zmin:zmax]
        tarShape = segZoom.shape[0] + 10, segZoom.shape[1] + 10, segZoom.shape[2] + 10
        segZoom = padding(segZoom, tarShape)
        segZoom = np.where(segZoom == cellId, 1, 0)        
        wallBall = ball(2)
        areaBall = ball(1)
        border = mh.border(segZoom, 1, 0, Bc=wallBall) * segZoom
        areaBorder = mh.border(segZoom, 1, 0, Bc=areaBall) * segZoom
        
        cellDict[cellId]["segmentation"] = segZoom
        
        cellDict[cellId]["globalCenter"] = np.multiply(region.centroid, resolution)
        cellDict[cellId]["localCenter"] = np.multiply(ndimage.center_of_mass(segZoom), resolution)
        
        cellDict[cellId]["volume"] = np.multiply(region.image.sum(), np.prod(resolution))

        
        if np.isnan(cellDict[cellId]["localCenter"][0]):
            cellDict.pop(cellId)
            continue
        
        cm = ndimage.center_of_mass(segZoom)
        # try:
        pinDict = decomposeCellNoPin(segZoom*border ,cm, areaBorder, decompositionAngles, resolution)
        
        sectorAreas = []
        
        for key in pinDict.keys():
            sectorAreas.append(pinDict[key]["area"])

        cellDict[cellId]["sectorAreas"] = np.array(sectorAreas)
        
        # except:
        #     tensor = np.array([[0.,0.,0.]])
        #     print("Failed getting tensor for source cell: "+str(cellId))
        # cellDict[cellId]["tensor"] = tensor

    return cellDict

def getInfoSphericalDecomposition(segImage, pinImage,decompositionAngles=(45,90), resolution=(1,1,1)):
    regProps = regionprops(segImage, pinImage)
    cellDict = {}

    shape = segImage.shape

    for region in regProps:
        cellId = region.label
        print("Decomposing: " + str(cellId))

        # if cellId >= 20: ############## REMOVE
        #     break

        ### filter out and crop target cell
        imageInd = np.where(segImage == cellId)
        imPoints = np.swapaxes(imageInd,0,1)
        
        
        
        if len(imPoints) < 5:
            print('skipping '+str(cellId))
            continue

        cellDict[cellId] = {}

        
        xmax, ymax, zmax = np.max(imPoints, axis=0)
        xmin, ymin, zmin = np.min(imPoints, axis=0)
        xmax, ymax, zmax = np.min([xmax+5, shape[0]-1]), np.min([ymax+5, shape[1]-1]), np.min([zmax+5, shape[2]-1])
        xmin, ymin, zmin = np.max([xmin-5, 0]), np.max([ymin-5, 0]), np.max([zmin-5, 0]) 
        segZoom = segImage[xmin:xmax, ymin:ymax, zmin:zmax]
        pinZoom = pinImage[xmin:xmax, ymin:ymax, zmin:zmax]     
        tarShape = segZoom.shape[0] + 10, segZoom.shape[1] + 10, segZoom.shape[2] + 10
        segZoom = padding(segZoom, tarShape)
        pinZoom = padding(pinZoom, tarShape)
        segZoom = np.where(segZoom == cellId, 1, 0)        
        wallBall = ball(2)
        areaBall = ball(1)
        border = mh.border(segZoom, 1, 0, Bc=wallBall) * segZoom
        areaBorder = mh.border(segZoom, 1, 0, Bc=areaBall) * segZoom
        pinZoom *= border
        
        cellDict[cellId]["segmentation"] = segZoom
        cellDict[cellId]["pin"] = pinZoom
        
        cellDict[cellId]["globalCenter"] = np.multiply(region.centroid, resolution)
        cellDict[cellId]["localCenter"] = np.multiply(ndimage.center_of_mass(segZoom), resolution)
        
        cellDict[cellId]["volume"] = np.multiply(region.image.sum(), np.prod(resolution))
        cellDict[cellId]["totalPin"] = region.intensity_image.sum()
        
        cellDict[cellId]["totalWallPin"] = np.sum(pinZoom) 
        
        if np.isnan(cellDict[cellId]["localCenter"][0]):
            cellDict.pop(cellId)
            continue
        
        cm = ndimage.center_of_mass(segZoom)
        # try:
        pinDict, pinImageTemp = decomposeCell(segZoom*border, pinZoom,cm, areaBorder, decompositionAngles, resolution)
        
        sectorPins = []
        sectorMeanPins = []
        sectorAreas = []
        sectorSampels = []
        
        for key in pinDict.keys():
            sectorAreas.append(pinDict[key]["area"])
            sectorPins.append(pinDict[key]["pin"])
            sectorMeanPins.append(pinDict[key]["meanPin"])
            sectorSampels.append(pinDict[key]["nSampels"])
        
        cellDict[cellId]["sectorPins"] = np.array(sectorPins)
        cellDict[cellId]["sectorMeanPins"] = np.array(sectorMeanPins)
        cellDict[cellId]["sectorAreas"] = np.array(sectorAreas)
        cellDict[cellId]["sectorSampels"] = np.array(sectorSampels)
        
        # except:
        #     tensor = np.array([[0.,0.,0.]])
        #     print("Failed getting tensor for source cell: "+str(cellId))
        # cellDict[cellId]["tensor"] = tensor

    return cellDict

def decomposeCellNoPin(segZoom, sourceArea, sourceMeanPin, sourceSampels, cm, areaBorder,decompositionAngles=(45,90) , resolution=(1,1,1)):
    ang1, ang2 = decompositionAngles
    
    ### group coordinates to fcVectors

    rawCoords = np.swapaxes(np.array(np.where(segZoom==1)),0,1)
    
    fcList, vertList, phiLim, thetaLim = getUnitSphere(ang1, ang2)
    
    segments = [np.zeros_like(segZoom).astype(np.bool_) for i in range(len(fcList))]
    
    faceColors = np.arange(0,len(fcList))
    pinDict = {}
    for col in faceColors:
        pinDict[col] = {}
    for x, y, z in rawCoords:
        xx, yy, zz = np.array([x,y,z])-cm
        r, theta, phi = asSpherical([xx,yy,zz])
        phi += 180
        theta, phi = phi, theta
        foundMatch = False
        colInd = 0
        for i, tLim in enumerate(thetaLim):
            for j, pLim in enumerate(phiLim):
                if tLim[0] < theta < tLim[1] and pLim[0] < phi < pLim[1]:
                    segZoom[x,y,z] = faceColors[colInd]
                    segments[colInd][x,y,z] = True
                    foundMatch = True
                    break
                colInd += 1
            if foundMatch:
                break

    
    ### Compute area of grouped coordinates and normalize PIN sum to concentration
    resolution = tuple(resolution)  # in case list object was given

    pinImage = np.zeros_like(segZoom).astype(np.float32)

    # tensor = []
    for i, segment in enumerate(segments):
        dPoints = len(np.swapaxes(np.array(np.where(segment==True)),0,1))
        sA = sourceArea[i]
        sMP = sourceMeanPin[i]
        sN = sourceSampels[i]
        segmentA = np.logical_and(segment,areaBorder.astype(np.bool_))
        
        if sN == 0:
            try:
                verts, faces, _, _ = measure.marching_cubes(segmentA, level=0.5, spacing=resolution, allow_degenerate=False) #spacing=resolution,
                area = measure.mesh_surface_area(verts, faces)
                pinDict[i]["area"] = area
                pinDict[i]["meanPin"] = np.array(0.) + 1e-6
            except:
                pinDict[i]["area"] = 0
                pinDict[i]["meanPin"] = np.array(0.) + 1e-6

        elif np.sum(segmentA.astype(np.int32)) != 0:
            verts, faces, _, _ = measure.marching_cubes(segmentA, level=0.5, spacing=resolution, allow_degenerate=False) #spacing=resolution,
            area = measure.mesh_surface_area(verts, faces)
            pinDict[i]["area"] = area
            try:
                val = (area * sMP * sN) / (sA * dPoints)
            except:
                val = np.array(0.) + 1e-6
            
            if np.isnan(val) or val <= 0:
                pinDict[i]["meanPin"] = np.array(0.) + 1e-6
            else:
                pinDict[i]["meanPin"] = val
            
        else:
            pinDict[i]["area"] = 0.
            pinDict[i]["meanPin"] = np.array(0.) + 1e-6

            
        # tensor.append(pinDict[i]["pinConcVec"])
        

        pinImage += segment.astype(np.float32)*(pinDict[i]["meanPin"]).astype(np.float32)
        

    return pinDict, pinImage

def decomposeCellNoPin2(segZoom, cm, areaBorder,decompositionAngles=(45,90) , resolution=(1,1,1)):
    ang1, ang2 = decompositionAngles
    
    ### group coordinates to fcVectors

    rawCoords = np.swapaxes(np.array(np.where(segZoom==1)),0,1)
    
    fcList, vertList, phiLim, thetaLim = getUnitSphere(ang1, ang2)
    
    segments = [np.zeros_like(segZoom).astype(np.bool_) for i in range(len(fcList))]
    
    faceColors = np.arange(0,len(fcList))
    pinDict = {}
    for col in faceColors:
        pinDict[col] = {}
    for x, y, z in rawCoords:
        xx, yy, zz = np.array([x,y,z])-cm
        r, theta, phi = asSpherical([xx,yy,zz])
        phi += 180
        theta, phi = phi, theta
        foundMatch = False
        colInd = 0
        for i, tLim in enumerate(thetaLim):
            for j, pLim in enumerate(phiLim):
                if tLim[0] < theta < tLim[1] and pLim[0] < phi < pLim[1]:
                    segZoom[x,y,z] = faceColors[colInd]
                    segments[colInd][x,y,z] = True
                    foundMatch = True
                    break
                colInd += 1
            if foundMatch:
                break

    
    ### Compute area of grouped coordinates and normalize PIN sum to concentration
    resolution = tuple(resolution)  # in case list object was given

    # pinImage = np.zeros_like(segZoom).astype(np.float32)

    # tensor = []
    for i, segment in enumerate(segments):

        segmentA = np.logical_and(segment,areaBorder.astype(np.bool_))
        if np.sum(segmentA.astype(np.int32)) != 0:
            verts, faces, _, _ = measure.marching_cubes(segmentA, level=0.5, spacing=resolution, allow_degenerate=False) #spacing=resolution,
            area = measure.mesh_surface_area(verts, faces)
            pinDict[i]["area"] = area
            
        else:
            pinDict[i]["area"] = 0.
            
        # tensor.append(pinDict[i]["pinConcVec"])
        

        # pinImage += segment.astype(np.float32)*(pinDict[i]["meanPin"]).astype(np.float32)
        

    return pinDict

def decomposeCell(segZoom, pinZoom, cm, areaBorder,decompositionAngles=(45,90) , resolution=(1,1,1)):
    ang1, ang2 = decompositionAngles
    
    ### group coordinates to fcVectors

    rawCoords = np.swapaxes(np.array(np.where(segZoom==1)),0,1)
    
    fcList, vertList, phiLim, thetaLim = getUnitSphere(ang1, ang2)
    
    segments = [np.zeros_like(segZoom).astype(np.bool_) for i in range(len(fcList))]
    
    faceColors = np.arange(0,len(fcList))
    pinDict = {}
    for col in faceColors:
        pinDict[col] = {}
    for x, y, z in rawCoords:
        xx, yy, zz = np.array([x,y,z])-cm
        r, theta, phi = asSpherical([xx,yy,zz])
        phi += 180
        theta, phi = phi, theta
        foundMatch = False
        colInd = 0
        for i, tLim in enumerate(thetaLim):
            for j, pLim in enumerate(phiLim):
                if tLim[0] < theta < tLim[1] and pLim[0] < phi < pLim[1]:
                    segZoom[x,y,z] = faceColors[colInd]
                    segments[colInd][x,y,z] = True
                    foundMatch = True
                    break
                colInd += 1
            if foundMatch:
                break

    
    ### Compute area of grouped coordinates and normalize PIN sum to concentration
    resolution = tuple(resolution)  # in case list object was given

    pinImage = np.zeros_like(segZoom).astype(np.float32)

    # tensor = []
    for i, segment in enumerate(segments):
        dPoints = len(np.swapaxes(np.array(np.where(segment==True)),0,1))
        pin = np.sum(segment.astype(np.int32)*pinZoom)
        if dPoints != 0:
            meanPin = pin/dPoints
        else:
            meanPin = 1e-6
        segmentA = np.logical_and(segment,areaBorder.astype(np.bool_))
        if np.sum(segmentA.astype(np.int32)) != 0:
            verts, faces, _, _ = measure.marching_cubes(segmentA, level=0.5, spacing=resolution, allow_degenerate=False) #spacing=resolution,
            area = measure.mesh_surface_area(verts, faces)
            pinDict[i]["area"] = area
            pinDict[i]["pin"] = pin
            pinDict[i]["meanPin"] = meanPin +1e-6
            pinDict[i]["nSampels"] = dPoints
            pinDict[i]["pinConc"] = pin/area
            pinDict[i]["pinConcVec"] = pin/area * np.array(fcList[i])
            
        else:
            pinDict[i]["area"] = 0.
            pinDict[i]["pin"] = 0.
            pinDict[i]["nSampels"] = 0
            pinDict[i]["meanPin"] = np.array(0.) + 1e-6
            pinDict[i]["pinConc"] = np.array(0.)
            pinDict[i]["pinConcVec"] = 0. * np.array(fcList[i])
            
        # tensor.append(pinDict[i]["pinConcVec"])
        pinImage += segment.astype(np.float32)*(pinDict[i]["meanPin"]).astype(np.float32)
        

    return pinDict, pinImage #, np.array(tensor)


def computeTargetPinSameGeo(sourceDict, targetDict):

    for key in sourceDict.keys():

        sectorMeanPins = []
        # sectorPins = []
        
        sourceAreas = sourceDict[key]["sectorAreas"]
        # sourcePins = sourceDict[key]["sectorPins"]
        sourceMeanPins = sourceDict[key]["sectorMeanPins"]
        targetAreas = targetDict[key]["sectorAreas"]
        
        for sA, sMP, tA in zip(sourceAreas, sourceMeanPins, targetAreas):
            
            sectorMeanPins.append( (tA*sMP) / sA )
            
        
        targetDict[key]["sectorMeanPins"] = np.array(sectorMeanPins)
        
    return targetDict
    

def recomposeSame(segImage, pinImage, decompositionAngles=(45,90), resolution=(1,1,1)):
    newPinImage = np.zeros_like(segImage)
    regProps = regionprops(segImage, intensity_image=pinImage)
    cellDict = {}
    shape = segImage.shape

    for region in regProps:
        cellId = region.label
        print("Recomposing: " + str(cellId))

        # if cellId >= 5: ############## REMOVE
        #     break

        ### filter out and crop target cell
        imageInd = np.where(segImage == cellId)
        imPoints = np.swapaxes(imageInd,0,1)
        
        if len(imPoints) < 5:
            print('skipping '+str(cellId))
            continue

        cellDict[cellId] = {}
        
        xmax, ymax, zmax = np.max(imPoints, axis=0)
        xmin, ymin, zmin = np.min(imPoints, axis=0)
        xmax, ymax, zmax = np.min([xmax+5, shape[0]-1]), np.min([ymax+5, shape[1]-1]), np.min([zmax+5, shape[2]-1])
        xmin, ymin, zmin = np.max([xmin-5, 0]), np.max([ymin-5, 0]), np.max([zmin-5, 0]) 
        segZoom = segImage[xmin:xmax, ymin:ymax, zmin:zmax]
        pinZoom = pinImage[xmin:xmax, ymin:ymax, zmin:zmax]       
        tarShape = segZoom.shape[0] + 10, segZoom.shape[1] + 10, segZoom.shape[2] + 10
        segZoom = padding(segZoom, tarShape)
        pinZoom = padding(pinZoom, tarShape)
        segZoom = np.where(segZoom == cellId, 1, 0)        
        wallBall = ball(2)
        areaBall = ball(1)
        border = mh.border(segZoom, 1, 0, Bc=wallBall) * segZoom
        areaBorder = mh.border(segZoom, 1, 0, Bc=areaBall) * segZoom
        pinZoom *= border
        
        cellDict[cellId]["segmentation"] = segZoom
        cellDict[cellId]["pin"] = pinZoom
        
        cellDict[cellId]["globalCenter"] = np.multiply(region.centroid, resolution)
        cellDict[cellId]["localCenter"] = np.multiply(ndimage.center_of_mass(segZoom), resolution)
        
        cellDict[cellId]["volume"] = np.multiply(region.image.sum(), np.prod(resolution))
        cellDict[cellId]["totalPin"] = region.intensity_image.sum()
        
        cellDict[cellId]["totalWallPin"] = np.sum(pinZoom) 
        
        if np.isnan(cellDict[cellId]["localCenter"][0]):
            cellDict.pop(cellId)
            continue
        
        cm = ndimage.center_of_mass(segZoom)
        # try:
        pinDict, newPin = decomposeCell(segZoom*border, pinZoom,cm, areaBorder, decompositionAngles, resolution)
        
        newShape = newPin.shape
        xLim = xmin+newShape[0]-10
        yLim = ymin+newShape[1]-10
        zLim = zmin+newShape[2]-10
        
        if np.isnan(np.sum(newPin)):
            print('WARNING NANANANANA')
        if xLim < 0 or yLim < 0 or zLim < 0 or np.sum(newPin) < 0:
            print(xLim, yLim, zLim, np.sum(newPin))
            return -1, -1
        
        newPinImage[xmin : xmin+newShape[0]-10,  ymin : ymin+newShape[1]-10,  zmin : zmin+newShape[2]-10] += newPin.astype(np.int32)[5:-5, 5:-5, 5:-5]
        # newPinImage[xmin : xmin+newShape[0],  ymin : ymin+newShape[1],  zmin : zmin+newShape[2]] += newPinImage.astype(np.int32)

        
        
        # mergeSeg = np.where(segImage==cellId, 1, 0)
        # bigBorder = mh.border(mergeSeg, 1, 0, Bc=wallBall) * mergeSeg
        # segBord = mergeSeg*bigBorder
        # newPinImage[segBord == 1] = newPin[newPin != 0].flatten()
        
        sectorAreas = []
        
        for key in pinDict.keys():
            sectorAreas.append(pinDict[key]["area"])

        cellDict[cellId]["sectorAreas"] = np.array(sectorAreas)
        
        # except:
        #     tensor = np.array([[0.,0.,0.]])
        #     print("Failed getting tensor for source cell: "+str(cellId))
        # cellDict[cellId]["tensor"] = tensor

    return cellDict, newPinImage

def recomposeOther(segImage, decompDictSource, overlapDict, decompositionAngles=(45,90), resolution=(1,1,1)):
    newPinImage = np.zeros_like(segImage).astype(np.float64())
    regProps = regionprops(segImage)
    cellDict = {}
    shape = segImage.shape

    for region in regProps:
        cellId = region.label
        try:
            overlaps = [o for o in overlapDict[cellId].values()]
        except:
            continue
        print("Recomposing: " + str(cellId))

        # if cellId >= 5: ############## REMOVE
        #     break

        ### filter out and crop target cell
        imageInd = np.where(segImage == cellId)
        imPoints = np.swapaxes(imageInd,0,1)
        
        if len(imPoints) < 5:
            print('skipping '+str(cellId))
            continue

        cellDict[cellId] = {}
        
        xmax, ymax, zmax = np.max(imPoints, axis=0)
        xmin, ymin, zmin = np.min(imPoints, axis=0)
        xmax, ymax, zmax = np.min([xmax+5, shape[0]-1]), np.min([ymax+5, shape[1]-1]), np.min([zmax+5, shape[2]-1])
        xmin, ymin, zmin = np.max([xmin-5, 0]), np.max([ymin-5, 0]), np.max([zmin-5, 0]) 
        segZoom = segImage[xmin:xmax, ymin:ymax, zmin:zmax]
        tarShape = segZoom.shape[0] + 10, segZoom.shape[1] + 10, segZoom.shape[2] + 10
        segZoom = padding(segZoom, tarShape)
        segZoom = np.where(segZoom == cellId, 1, 0)        
        wallBall = ball(2)
        areaBall = ball(1)
        border = mh.border(segZoom, 1, 0, Bc=wallBall) * segZoom
        areaBorder = mh.border(segZoom, 1, 0, Bc=areaBall) * segZoom
        
        cellDict[cellId]["segmentation"] = segZoom

        cellDict[cellId]["globalCenter"] = np.multiply(region.centroid, resolution)
        cellDict[cellId]["localCenter"] = np.multiply(ndimage.center_of_mass(segZoom), resolution)
        
        cellDict[cellId]["volume"] = np.multiply(region.image.sum(), np.prod(resolution))
        
        if np.isnan(cellDict[cellId]["localCenter"][0]):
            cellDict.pop(cellId)
            continue
        
        cm = ndimage.center_of_mass(segZoom)

        
        overlaps = [o for o in overlapDict[cellId].values()]
        sourceIds = [r for r in overlapDict[cellId].keys()]    
        try:
            overlaps, sourceIds = zip(*sorted(zip(overlaps, sourceIds), reverse=True))
        except:
            cellDict[cellId]["pinConc"] = 0.
            continue
        
        # mergeSeg = np.where(segImage==cellId, 1, 0)
        # bigBorder = mh.border(mergeSeg, 1, 0, Bc=wallBall) * mergeSeg
        # segBord = mergeSeg*bigBorder
        
        
        pinConc = 0.
        
        #### ITERATE OVER ALL OVERLAPS
        for i, sourceId in enumerate(sourceIds):
            overlap = overlaps[i]
            print('Mapping from source cell: '+str(sourceId)+' with overlap: '+str(overlap))
            
            ############ JUST TAKE THE CLOSEST FOR NOW
            # try:
                
            try:
                sourceAreas = decompDictSource[sourceId]["sectorAreas"]
                sourceMeanPins = decompDictSource[sourceId]["sectorMeanPins"]
                sourceSampels = decompDictSource[sourceId]["sectorSampels"]
                
                pinConc += (decompDictSource[sourceId]["totalPin"] / decompDictSource[sourceId]["volume"]) * overlap
            except:
                print('There was an issue getting the source cell: '+str(sourceId))
                continue
            try:
                pinDict, newPin = decomposeCellNoPin(segZoom*border, sourceAreas, sourceMeanPins,sourceSampels ,cm, areaBorder, decompositionAngles, resolution)
                newPin *= overlap
                
                newShape = newPin.shape
                xLim = xmin+newShape[0]-10
                yLim = ymin+newShape[1]-10
                zLim = zmin+newShape[2]-10
                
                if np.isnan(np.sum(newPin)):
                    print('WARNING NANANANANA')
                if xLim < 0 or yLim < 0 or zLim < 0 or np.sum(newPin) < 0:
                    print(xLim, yLim, zLim, np.sum(newPin))
                    return -1, -1
            
            
                newPinImage[xmin : xmin+newShape[0]-10,  ymin : ymin+newShape[1]-10,  zmin : zmin+newShape[2]-10] += newPin.astype(np.int32)[5:-5, 5:-5, 5:-5]
            except:
                print("Failed {sourceId}")
            # newPinImage[xmin : xmin+newShape[0],  ymin : ymin+newShape[1],  zmin : zmin+newShape[2]] += newPinImage.astype(np.int32)

            
            # newPinImage[segBord == 1] += newPin[newPin != 0].flatten()
            # break
            
            
            # except:
            #     print('There was no data for source cell: '+str(sourceId))
        # cellDict[cellId]["pinDict"] = pinDict
        cellDict[cellId]["pinConc"] = pinConc
        # sourceCellDict = sourceDict[sourceId]
        # sourceAreas = [r["area"] for r in sourceCellDict.values()]
        # sourceMeanPins = [r["meanPin"] for r in sourceCellDict.values()]
        # pinDict, newPin = decomposeCellNoPin(segZoom*border, sourceAreas, sourceMeanPin ,cm, areaBorder, decompositionAngles, resolution)
        
        # mergeSeg = np.where(segImage==cellId, 1, 0)
        # bigBorder = mh.border(mergeSeg, 1, 0, Bc=wallBall) * mergeSeg
        # segBord = mergeSeg*bigBorder
        # newPinImage[segBord == 1] = newPin[newPin != 0].flatten()
        
        # sectorAreas = []
        
        # for key in pinDict.keys():
        #     sectorAreas.append(pinDict[key]["area"])

        # cellDict[cellId]["sectorAreas"] = np.array(sectorAreas)
        
        # except:
        #     tensor = np.array([[0.,0.,0.]])
        #     print("Failed getting tensor for source cell: "+str(cellId))
        # cellDict[cellId]["tensor"] = tensor

    return cellDict, newPinImage


def decomposeSphere():
    sphere = genSphere((100,100,100),40,(50,50,50)).astype(np.int32)
    cm = (50,50,50)
    
    
    rawCoords = np.swapaxes(np.array(np.where(sphere==1)),0,1)
    
    fcList, vertList, phiLim, thetaLim = getUnitSphere(45/4, 90/4)
    
    segments = [np.zeros_like(sphere).astype(np.bool_) for i in range(len(fcList))]
    
    
    
    faceColors = np.arange(0,len(fcList))
    faceColors = [np.random.randint(105,len(fcList)+1) for i in range(len(fcList))]
    pinDict = {}
    for col in faceColors:
        pinDict[col] = {}
    for x, y, z in rawCoords:
        xx, yy, zz = np.array([x,y,z])-cm
        r, theta, phi = asSpherical([xx,yy,zz])
        phi += 180
        theta, phi = phi, theta
        foundMatch = False
        colInd = 0
        for i, tLim in enumerate(thetaLim):
            for j, pLim in enumerate(phiLim):
                if tLim[0] < theta <= tLim[1] and pLim[0] < phi <= pLim[1]:
                    sphere[x,y,z] =  faceColors[colInd]
                    segments[colInd][x,y,z] = True
                    foundMatch = True
                    break
                colInd += 1
            if foundMatch:
                break
    
    # final = np.sum(np.array(segments).astype(np.int8), axis=0)
    
    
    p = pvq.BackgroundPlotter()
    p.add_volume(sphere, opacity="linear", resolution=[1,1,1],shade=True,diffuse=1.5, specular=1.)
    p.show()

def visOverlap(sourceImage, targetImage):
    overlapDict = calculateOverlapFast(sourceImage, targetImage)
    
    compositeImage = np.zeros_like(sourceImage).astype(np.float32)
    
    overlapDict69 = overlapDict[69]
    
    overlaps = [o for o in overlapDict69.values()]
    sourceIds = [r for r in overlapDict69.keys()]    
    
    overlaps, sourceIds = zip(*sorted(zip(overlaps, sourceIds), reverse=True))
    

    compositeImage[sourceImage==sourceIds[0]] = 1
        
    compositeImage[targetImage==69] += 2
    shape = compositeImage.shape
    imageInd = np.where(compositeImage > 0)
    imPoints = np.swapaxes(imageInd,0,1)
    xmax, ymax, zmax = np.max(imPoints, axis=0)
    xmin, ymin, zmin = np.min(imPoints, axis=0)
    xmax, ymax, zmax = np.min([xmax+5, shape[0]-1]), np.min([ymax+5, shape[1]-1]), np.min([zmax+5, shape[2]-1])
    xmin, ymin, zmin = np.max([xmin-5, 0]), np.max([ymin-5, 0]), np.max([zmin-5, 0]) 
    compositeImage = compositeImage[xmin:xmax, ymin:ymax, zmin:zmax]
    
    p = pvq.BackgroundPlotter()
    p.add_volume(compositeImage, opacity="linear", resolution=[1,1,1],shade=True,diffuse=1.5, specular=1.)
    p.show()
        
    
    
def recomposeMultiCore(targetImage, decompDictSource, overlapDict, decompositionAngles=(45,90), resolution=(1,1,1), nCores=1):
    targetImage = targetImage.astype(np.int16)

    for key in decompDictSource.keys():
        try:
            del decompDictSource[key]["pin"]
        except:
            pass
        try:
            del decompDictSource[key]["segmentation"]
        except:
            pass

    nrOfCells = len(overlapDict)
    cellsPerThread = int(np.ceil(nrOfCells/nCores))
    overlapDictList = []
    for n in range(nCores):
        splitDict = dict(list(overlapDict.items())[n*cellsPerThread:(n+1)*cellsPerThread])
        overlapDictList.append(splitDict)
    
    
    
    pool = multiprocessing.Pool(nCores)
    workers = []
    for batchOverlapDict in overlapDictList:
        workers.append(pool.apply_async(recomposeOther, args=(targetImage, decompDictSource, batchOverlapDict, decompositionAngles, resolution)))

    final_result = [worker.get() for worker in workers]
    
    finalDictList = []
    
    for i, res in enumerate(final_result):
        if i == 0:
            finalPinImage = res[1]
            finalDictList.append(res[0])
        else:
            finalPinImage += res[1]
            finalDictList.append(res[0])
            
    all_props = {}
    
    for cellDict in finalDictList:
        for key in cellDict.keys():
            # cellId = cellDict["cellId"] 
            all_props[key] = {}
            all_props[key]["coords"] = cellDict[key]["globalCenter"] 
            all_props[key]["volume"] = cellDict[key]["volume"]
            all_props[key]["pinConc"] = cellDict[key]["pinConc"]

        
        
        
    return finalPinImage, all_props

def save_obj(obj, name ):
    with open(r'/home/jonas/projects/obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(r'/home/jonas/projects/obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

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
    
    transSegList = [r'/home/jonas/projects/allImages/120hSources/transSegData_210305_22.tif',
                    r'/home/jonas/projects/allImages/120hSources/transSegData_210305_31.tif',
                    r'/home/jonas/projects/allImages/120hSources/transSegData_201112_41.tif',
                    r'/home/jonas/projects/allImages/120hSources/transSegData_210305_62.tif']
    
    transPinList = [r'/home/jonas/projects/allImages/120hSources/transPinData_210305_22.tif',
                    r'/home/jonas/projects/allImages/120hSources/transPinData_210305_31.tif',
                    r'/home/jonas/projects/allImages/120hSources/transPinData_201112_41.tif',
                    r'/home/jonas/projects/allImages/120hSources/transPinData_210305_62.tif']
    
    compSegs = [r"/home/jonas/projects/allImages/compareFlower1_2/transSegData_11_21.tif",
                r"/home/jonas/projects/allImages/compareFlower1_2/transSegData_21_11.tif"]
    
    compPins = [r"/home/jonas/projects/allImages/compareFlower1_2/transPinData_11_21.tif",
                r"/home/jonas/projects/allImages/compareFlower1_2/transPinData_21_11.tif",]
    
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
                     (0.25     , 0.42071381/2, 0.42071381/2),
                     (0.25     , 0.4378858/2, 0.4378858/2),
                     (0.25, 0.18869677820449887, 0.18869677820449887),
                     (0.25, 0.17264853272876005, 0.17264853272876005),
                     (0.25, 0.15884092787927093, 0.15884092787927093)]
    


    # pinImg = tifffile.imread(pinList[1])
    # pinRes = sourceResList[1]
    
    # maxVal = np.max(pinImg)
    
    # pinImg[pinImg<800] = 0
    # pinImg[pinImg>maxVal*0.7] = maxVal*0.7
    # pinImg = ndimage.gaussian_filter(pinImg, 0.5)

    # p = pvq.BackgroundPlotter()
    # p.set_background("white")
    # p.add_volume(pinImg, opacity='linear', resolution=pinRes,shade=True,diffuse=1.5, specular=1.)
    # p.show()



    # tarId = 1
    # sorId = 0
    # # resId = 1
    


    # sourcePath =  r"/home/jonas/projects/allImages/132hSource/transSegData_201112_21.tif"  #sourceList[sorId] #r"/home/jonas/projects/allImages/SAMimages/transformed/transSegData_13_1.tif" #sourceList[sorId] #r'/home/jonas/projects/allImages/quantifySimilar/transSegData_201112_11.tif'
    # pinPath =  r"/home/jonas/projects/allImages/132hSource/transPinData_201112_21.tif" #pinList[sorId] #r"/home/jonas/projects/allImages/SAMimages/transformed/transPinData_13_1.tif" #pinList[sorId] #r'/home/jonas/projects/allImages/quantifySimilar/transPinData_201112_11.tif' 
    # targetPath = targetList[tarId]#r"/home/jonas/projects/allImages/SAMimages/segmented_refined/131007-PIN_GFP-acyl_YFP-plant_1-0h_stitched_stackreg_crop_wiener_C0_predictions_multicut-refined.tif" #sourceList[tarId] #targetList[tarId]
    
    # # sourceResolution = get_resolution(targetPath) #sourceResList[sorId]
    # targetResolution = targetResList[tarId]
    # sourceResolution = targetResolution
    
    # pinPath = r"/home/jonas/projects/allImages/SAMimages/transformed/transPinData_3_to_1.tif"
    # sourcePath = r"/home/jonas/projects/allImages/SAMimages/transformed/transSegData_3_to_1.tif"
    # targetPath = r'/home/jonas/projects/allImages/SAMimages/segmented_refined/131007-PIN_GFP-acyl_YFP-plant_3-0h_stackreg_crop_wiener_C0_predictions_multicut-refined.tif'
    
    pinPath = pinList[0]
    sourcePath = sourceList[0]
    targetPath = sourceList[0]
    
    # targetResolution = sourceResList[0]
    sourceResolution = sourceResList[0]
    
    
    sourceImage = tifffile.imread(sourcePath).astype(np.int32)
    pinImage = tifffile.imread(pinPath).astype(np.int32)
    # targetImage = tifffile.imread(targetPath).astype(np.int32)
    
    # if np.min(targetImage) == 1:
    #     targetImage -= 1
    if np.min(sourceImage) == 1:
        sourceImage -= 1
        
    # targetShape = targetImage.shape[0] + 30, targetImage.shape[1] + 30, targetImage.shape[2] + 30
    # targetImage = padding(targetImage, targetShape)
    
    sourceImage, labelDict = reorganizeLabels(sourceImage)
    all_props, finalPinImage = recomposeSame(sourceImage, pinImage, decompositionAngles=(22.5,45), resolution=sourceResolution)
    batchList = batch_compute(sourceImage, finalPinImage, 24, ball(2), sourceResolution)

    name = '11_225_45'
    path = r'/home/jonas/projects/allImages/selfCompFinal/'
    
    writeToFiles(batchList, None, path, name, name)    
    
    
    
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    w1 = pool.apply_async(calculateOverlapFast, args=(sourceImage, targetImage))
    w2 = pool.apply_async(getInfoSphericalDecomposition, args=(sourceImage, pinImage,(45/2,90/2), sourceResolution))
    workers = [w1, w2]
    final_result = [worker.get() for worker in workers]
    overlapDict = final_result[0]
    decompDictSource = final_result[1]
    

    finalPinImage, all_props = recomposeMultiCore(targetImage, decompDictSource, overlapDict, decompositionAngles=(45/2,90/2), resolution=targetResolution, nCores=5)
    # cellDict, finalPinImage = recomposeOther(targetImage, decompDictSource, overlapDict, decompositionAngles=(45/4,90/4), resolution=targetResolution)

    tifffile.imsave(r'/home/jonas/projects/allImages/compareFlower1_2/mappedSegData_21_11.tif', targetImage.astype(np.uint16))
    tifffile.imsave(r'/home/jonas/projects/allImages/compareFlower1_2/mappedPinData_21_11.tif', finalPinImage.astype(np.uint16))
    
    # im1 = tifffile.imread(r"/home/jonas/projects/allImages/132hSource/mappedPinData_11_132_225_45.tif")
    # im2 = tifffile.imread(r"/home/jonas/projects/allImages/132hSource/mappedPinData_21_132_225_45.tif")
    
    # im_sum = im1+im2
    
    # p = pvq.BackgroundPlotter()
    # p.add_volume(finalPinImage, opacity='linear', resolution=targetResolution,shade=True,diffuse=1.5, specular=1.)
    # p.show()


    try:
        save_obj(all_props, "all_props")
        tifffile.imsave(r'/home/jonas/projects/obj/tempImg.tif', finalPinImage.astype(np.uint16))
    except:
        all_props = load_obj("all_props")
        finalPinImage = tifffile.imread(r'/home/jonas/projects/obj/tempImg.tif')
    

    pinConcList = [v["pinConc"] for v in all_props.values()]

    segImg, labelDict = reorganizeLabels(targetImage)
    
    # img2 = tifffile.imread(r'/home/jonas/projects/allImages/132hSource/mappedPinData_11_132_225_45.tif')

    # finalPinImage += img2
    
    # finalPinImage = finalPinImage/2.

    del targetImage
    del pinImage
    del sourceImage
    del all_props
    
    batchList = batch_compute(segImg, finalPinImage, 24, ball(2), targetResolution)

    
    name = '21_11'
    path = r'/home/jonas/projects/allImages/compareFlower1_2/'
    
    writeToFiles(batchList, pinConcList, path, name, name)
    



    
    
    """
    
    # sourceImage = tiff.imread(r'/home/jonas/projects/allImages/SAMimages/segmented_refined/131007-PIN_GFP-acyl_YFP-plant_3-0h_stackreg_crop_wiener_C0_predictions_multicut-refined.tif').astype(np.int32)
    # pinImage = tiff.imread(r'/home/jonas/projects/allImages/SAMimages/raw/131007-PIN_GFP-acyl_YFP-plant_3-0h_stackreg_crop_wiener_C1.tif').astype(np.int32)
    

    # sourceResolution = get_resolution(r'/home/jonas/projects/allImages/SAMimages/segmented_refined/131007-PIN_GFP-acyl_YFP-plant_3-0h_stackreg_crop_wiener_C0_predictions_multicut-refined.tif')
    
    sourcePath = transSegList[1]
    pinPath = transPinList[1]
    
    sourceResolution = sourceResList[8]

    sourceImage = tifffile.imread(sourcePath).astype(np.int32)
    pinImage = tifffile.imread(pinPath).astype(np.int32)
    
    if np.min(sourceImage) == 1:
        sourceImage -= 1
        
    sourceShape = sourceImage.shape[0] + 30, sourceImage.shape[1] + 30, sourceImage.shape[2] + 30
    sourceImage = padding(sourceImage, sourceShape)
    pinImage = padding(pinImage, sourceShape)
    
    # wallBall = ball(2)
    # border = (mh.borders(sourceImage, Bc=wallBall) ).astype(np.uint16)
    # pinBords = pinImage * border
    # p = pvq.BackgroundPlotter()
    # p.add_volume(pinImage, opacity='linear', resolution=[1,1,1],shade=True,diffuse=1.5, specular=1.)
    # p.show()
    
    
    cellDict, newPinImage = recomposeSame(sourceImage, pinImage, (45/4, 90/4), resolution=sourceResolution)
    # tifffile.imsave(r'/home/jonas/projects/allImages/120hSources/transBorders_210305_31.tif', pinBords.astype(np.uint16))
    tifffile.imsave(r'/home/jonas/projects/allImages/120hSources/selfComposed_210305_31_highRes.tif', newPinImage.astype(np.uint16))


    # pinConcList = [v["pinConc"] for v in cellDict.values()]

    # sourceImage, labelDict = reorganizeLabels(sourceImage)


    
    # batchList = batch_compute(sourceImage.astype(np.int16), newPinImage.astype(np.uint16), 24, ball(1), sourceResolution)

    
    # name = 'SAM_plant3_1125_225'
    
    # writeToFiles(batchList, name, name)
    """



    """
    plotImage = tifffile.imread(pinList[-2])
    plotImage[plotImage < 500] = 0
    
    p = pvq.BackgroundPlotter()
    p.add_volume(plotImage, opacity='linear', resolution=[1,1,1],shade=True,diffuse=1.5, specular=1.)
    p.show()
    """

    
    """
    for key in decompDictSource.keys():
        sampels = decompDictSource[key]["sectorSampels"]
        areas = decompDictSource[key]["sectorAreas"]
        fracs = areas/sampels
        for f in fracs:
            plt.plot(key, f, 'r+')
    """
    
    """
    
    
    
    from skimage import transform
    # segPath = r'/home/jonas/projects/allImages/seg/C2-201112-PIN_GFP-MM_dsRed-WT-SAM5-FM1_predictions_multicut-refined.tif'
    # pinPath = r'/home/jonas/projects/allImages/pin/C1-201112-PIN_GFP-MM_dsRed-WT-SAM5-FM1.tif'
    
    # targetPath = r'/home/jonas/projects/allImages/seg/C2-201112-PIN_GFP-MM_dsRed-WT-SAM5-FM1_predictions_multicut-refined.tif'
    # # tarPinPath = r'/home/jonas/projects/allImages/pin/C1-201112-PIN_GFP-MM_dsRed-WT-SAM5-FM1.tif'
    # sourcePath = r'/home/jonas/projects/registration/transformedSeg.tif'
    # pinPath = r'/home/jonas/projects/registration/transformedPin.tif'
    
    targetPath = r'/home/jonas/projects/allImages/atlas/segmented_tiffs/120h_segmented.tif'
    # tarPinPath = r'/home/jonas/projects/allImages/pin/C1-201112-PIN_GFP-MM_dsRed-WT-SAM5-FM1.tif'
    sourcePath = r'/home/jonas/projects/allImages/120hSources/transSegData_210305_22.tif'
    pinPath = r'/home/jonas/projects/allImages/120hSources/transPinData_210305_22.tif'
    
    
    
    
    
    
    sourceImage = tifffile.imread(sourcePath).astype(np.int32)
    pinImage = tifffile.imread(pinPath).astype(np.int32)
    targetImage = tifffile.imread(targetPath).astype(np.int32) - 1
    
    targetResolution = (0.2361645, 0.2361645, 0.17078415345851233)
    # targetResolution = get_resolution(targetPath)
    # sourceResolution = (0.25     , 0.4378858, 0.4378858)
    sourceResolution = (0.25      , 0.42071381, 0.42071381)
    # sourceResolution = get_resolution(r'/home/jonas/projects/allImages/pin/C1-201112-PIN_GFP-MM_dsRed-WT-SAM2-FM1.tif')
    pinImage = transform.resize(pinImage, sourceImage.shape, preserve_range=True).astype(np.int32)
    
    
    # sourceShape = sourceImage.shape[0] + 10, sourceImage.shape[1] + 10, sourceImage.shape[2] + 10
    # sourceImage = padding(sourceImage, sourceShape)
    # pinImage = padding(pinImage, sourceShape)
    targetShape = targetImage.shape[0] + 30, targetImage.shape[1] + 30, targetImage.shape[2] + 30
    targetImage = padding(targetImage, targetShape)
    
    
    # calculateOverlapFast(sourceImage, targetImage)
    
    
    
        ### MULTI CORE 
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)
    w1 = pool.apply_async(calculateOverlapFast, args=(sourceImage, targetImage))
    w2 = pool.apply_async(getInfoSphericalDecomposition, args=(sourceImage, pinImage,(45/4,90/4), sourceResolution))
    workers = [w1, w2]
    final_result = [worker.get() for worker in workers]
    overlapDict = final_result[0]
    decompDictSource = final_result[1]
    
    
        ### SINGLE CORE
    # overlapDict = calculateOverlap(sourceImage, targetImage, 0.99)
    # decompDictSource = getInfoSphericalDecomposition(sourceImage, pinImage,decompositionAngles=(45/2.,90/2.), resolution=sourceResolution)
    
    
    cellDict, finalPinImage = recomposeOther(targetImage, decompDictSource, overlapDict, decompositionAngles=(45/4,90/4), resolution=targetResolution)
    
    
    
    
    # cellDict, newPinImage = recomposeSame(segImage, pinImage, decompositionAngles=(10,20), resolution=resolution)
        
    # finalPinImage = finalPinImage.astype(np.int32)
    
    tifffile.imsave(r'/home/jonas/projects/allImages/120hSources/mappedSegData_210305_22_highRes.tif', targetImage.astype(np.uint16))
    tifffile.imsave(r'/home/jonas/projects/allImages/120hSources/mappedPinData_210305_22_highRes.tif', finalPinImage.astype(np.uint16))
    
    ### Decompose Source
    
    # decompDictSource = getInfoSphericalDecomposition(segImage, pinImage,decompositionAngles=(45,90), resolution=resolution)
    # decompDictTarget = getInfoSphericalDecompositionNoPin(segImage,decompositionAngles=(45,90), resolution=resolution)


    # decompDictTarget2 = computeTargetPinSameGeo(decompDictSource, decompDictTarget)
    
    ### Map concentrations to walls
    
    ### Extract PIN and Neighbours
    
    ### Save file
    
    
    
    
    
    # print(decompDictSource[4]["sectorMeanPins"] / decompDictSource[4]["sectorAreas"] )
    
    # print(decompDictTarget[1]["sectorMeanPins"] / decompDictTarget[1]["sectorAreas"])
    
    
    
    
    imageInd = np.where(sourceImage == 140)
    imPoints = np.swapaxes(imageInd,0,1)


    shape = targetImage.shape
    xmax, ymax, zmax = np.max(imPoints, axis=0)
    xmin, ymin, zmin = np.min(imPoints, axis=0)
    xmax, ymax, zmax = np.min([xmax+5, shape[0]-1]), np.min([ymax+5, shape[1]-1]), np.min([zmax+5, shape[2]-1])
    xmin, ymin, zmin = np.max([xmin-5, 0]), np.max([ymin-5, 0]), np.max([zmin-5, 0]) 
    segZoom = sourceImage[xmin:xmax, ymin:ymax, zmin:zmax]
    pinZoom = pinImage[xmin:xmax, ymin:ymax, zmin:zmax]       
    tarShape = segZoom.shape[0] + 10, segZoom.shape[1] + 10, segZoom.shape[2] + 10
    segZoom = padding(segZoom, tarShape)
    pinZoom = padding(pinZoom, tarShape)
    segZoom = np.where(segZoom == 140, 1, 0)        
    wallBall = ball(2)
    areaBall = ball(1)
    border = mh.border(segZoom, 1, 0, Bc=wallBall) * segZoom
    areaBorder = mh.border(segZoom, 1, 0, Bc=areaBall) * segZoom
    pinZoom *= border
    
    cm = ndimage.center_of_mass(segZoom)
    
    
    # pinDict = decomposeCellNoPin2(segZoom*border,cm, areaBorder, (45,90), targetResolution)
    
    pinDict, newPinImage = decomposeCell(segZoom*border, pinZoom, cm, areaBorder, (45/4,90/4), targetResolution)


    
    sourceAreas = [r["area"] for r in pinDict.values()]
    sourceMeanPins = [r["meanPin"] for r in pinDict.values()]
    sourceSampels = [r["nSampels"] for r in pinDict.values()]
    
    pinDict2, newPin = decomposeCellNoPin(segZoom*border, sourceAreas, sourceMeanPins, sourceSampels, cm, areaBorder,(45/4,90/4), targetResolution)

    mergeImg = np.zeros_like(sourceImage).astype(np.int32)
    
    newShape = newPinImage.shape
    mergeImg[xmin : xmin+newShape[0]-10,  ymin : ymin+newShape[1]-10,  zmin : zmin+newShape[2]-10] += newPinImage.astype(np.int32)[5:-5, 5:-5, 5:-5]

    # mergeSeg = np.where(sourceImage==354, 1, 0)
    # bigBorder = mh.border(mergeSeg, 1, 0, Bc=wallBall) * mergeSeg
    # segBord = mergeSeg*bigBorder
    # segBord[segBord == 1] = newPinImage[newPinImage != 0].flatten()
    
    
    tempImage = ndimage.gaussian_filter(finalPinImage, 0.25)
    
    p = pvq.BackgroundPlotter()
    p.add_volume(finalPinImage, opacity='linear', resolution=[1,1,1],shade=True,diffuse=1.5, specular=1.)
    p.show()
    
    
    
    
    
    
    
    imageInd = np.where(sourceImage == 480)
    imPoints = np.swapaxes(imageInd,0,1)


    shape = targetImage.shape
    xmax, ymax, zmax = np.max(imPoints, axis=0)
    xmin, ymin, zmin = np.min(imPoints, axis=0)
    xmax, ymax, zmax = np.min([xmax+5, shape[0]-1]), np.min([ymax+5, shape[1]-1]), np.min([zmax+5, shape[2]-1])
    xmin, ymin, zmin = np.max([xmin-5, 0]), np.max([ymin-5, 0]), np.max([zmin-5, 0]) 
    segZoom = sourceImage[xmin:xmax, ymin:ymax, zmin:zmax]
    pinZoom = pinImage[xmin:xmax, ymin:ymax, zmin:zmax]       
    tarShape = segZoom.shape[0] + 10, segZoom.shape[1] + 10, segZoom.shape[2] + 10
    segZoom = padding(segZoom, tarShape)
    pinZoom = padding(pinZoom, tarShape)
    segZoom = np.where(segZoom == 480, 1, 0)        
    wallBall = ball(2)
    areaBall = ball(1)
    border = mh.border(segZoom, 1, 0, Bc=wallBall) * segZoom
    areaBorder = mh.border(segZoom, 1, 0, Bc=areaBall) * segZoom
    pinZoom *= border
    
    cm = ndimage.center_of_mass(segZoom)
    
    pinDict2, mappedPin = decomposeCellNoPin(segZoom*border, sourceAreas, sourceMeanPins, sourceSampels, cm, areaBorder,(10,20), targetResolution)
    
    
    
    # path = r'/home/jonas/projects/allImages/newImages/210305-PIN1GFP-MMdsRed-WT_flowers-SAM2-FM2.lsm'
    
    # im = tifffile.imread(path)
    
    # print(im.shape)
    
    # im1 = im[0,:,0,:,:]
    # im2 = im[0,:,1,:,:]
    
    from img2org.quantify import get_layers

    layers = get_layers(targetImage, 0, 2)    

    
    newImage = np.zeros_like(targetImage)
    
    for ind in layers[0]:
        newImage[targetImage == ind] = 1

    
    p = pvq.BackgroundPlotter()
    p.add_volume(finalPinImage, opacity='linear', resolution=[1,1,1],shade=True,diffuse=1.5, specular=1.)
    p.show()
    
    
    
    
    
    
    
    
    
    
    pass
    """

