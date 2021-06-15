import numpy as np
import SimpleITK as sitk
import tifffile
import pyvistaqt as pvq
import pyvista as pv
from imgmisc import get_resolution
from skimage import filters
from scipy import ndimage
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib
import ants

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

def registerImages(sourceImage, sourceResolution, targetImage, targetResolution, iterationNumbers=15000, do_translation=True, do_rigid=True, do_affine=True, do_bspline=True, fixFile=None, moveFile=None):
    fixedImage = sitk.GetImageFromArray(targetImage.astype(np.float32), isVector=False)
    movingImage = sitk.GetImageFromArray(sourceImage.astype(np.float32), isVector=False)
    
    fixedImage.SetSpacing(targetResolution)
    movingImage.SetSpacing(sourceResolution)
        
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(fixedImage)
    elastixImageFilter.SetMovingImage(movingImage)
    

    

    
    parameterMapVector = sitk.VectorOfParameterMap()
    
    if do_translation:
        translation = sitk.GetDefaultParameterMap('translation')
        translation["MaximumNumberOfIterations"] = [str(iterationNumbers)]
        translation["NumberOfResolutions"] = [str(6)]
        # # translation["FixedImagePyramidSchedule"] = ["256 256 256 128 128 128"]
        # # translation["MovingImagePyramidSchedule"] = ["256 256 256 128 128 128"]
        parameterMapVector.append(translation)
    
    if do_rigid:
        rigid = sitk.GetDefaultParameterMap('rigid')
        rigid["MaximumNumberOfIterations"] = [str(iterationNumbers)]
        rigid["NumberOfResolutions"] = [str(6)]
        parameterMapVector.append(rigid)
    
    if do_affine:
        affine = sitk.GetDefaultParameterMap('affine')
        affine["MaximumNumberOfIterations"] = [str(iterationNumbers)]
        affine["NumberOfResolutions"] = [str(8)]
        parameterMapVector.append(affine)
    
    if do_bspline:
        bSpline = sitk.GetDefaultParameterMap('bspline')
        bSpline["MaximumNumberOfIterations"] = [str(iterationNumbers*2)]
        bSpline["NumberOfResolutions"] = [str(1)]
        bSpline['ImagePyramidSchedule'] = ['4', '4', '4']
        bSpline['FinalGridSpacingInPhysicalUnits'] = ['10.0', '10.0', '10.0']
        bSpline['GridSpacingSchedule'] = ['10.0' '10.0' '10.0']
        bSpline['FinalBSplineInterpolationOrder'] = ['0']
        parameterMapVector.append(bSpline)
    
    elastixImageFilter.SetParameterMap(parameterMapVector)
    
    if fixFile != None or moveFile != None:
        elastixImageFilter.SetParameter("Registration","MultiMetricMultiResolutionRegistration")  #MultiMetricMultiResolutionRegistration #MultiResolutionRegistration
        # elastixImageFilter.SetParameter("Registration","AdvancedImageToImageMetric" )
        elastixImageFilter.SetParameter( "Metric", ("AdvancedMattesMutualInformation", "CorrespondingPointsEuclideanDistanceMetric")) #, AdvancedMattesMutualInformation  AdvancedKappaStatistic AdvancedMeanSquares "CorrespondingPointsEuclideanDistanceMetric",NormalizedMutualInformation
        elastixImageFilter.SetParameter("Metric0Weight", "0.95")
        elastixImageFilter.SetParameter("Metric1Weight", "0.05")
    
    
    ## sitk.PrintParameterMap(sitk.GetDefaultParameterMap("bspline"))
    if do_bspline:
        pMap = elastixImageFilter.GetParameterMap()
        pMap[-1]["Metric"] = ["AdvancedMattesMutualInformation", "TransformBendingEnergyPenalty"]
        pMap[-1]["Metric0Weight"] = ["1.0"]
        pMap[-1]["Metric1Weight"] = ["1.0"]
        elastixImageFilter.SetParameterMap(pMap)    
    
    
    elastixImageFilter.LogToFileOn()  
    elastixImageFilter.SetOutputDirectory(r"/home/jonas/projects/midCritFullProject")
    
    if fixFile != None:
        elastixImageFilter.SetFixedPointSetFileName(fixFile)
    if moveFile != None:
        elastixImageFilter.SetMovingPointSetFileName(moveFile)
    
    try:
        elastixImageFilter.Execute()
        return elastixImageFilter
    except:
        print('Could not perform registration')
        return None



def transformImage(image, resolution, imageFilter):
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(resolution)
    transformParameterMap = imageFilter.GetTransformParameterMap()
    for tp in transformParameterMap:
        tp["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
    transformix = sitk.TransformixImageFilter()
    transformix.SetTransformParameterMap(transformParameterMap)
    transformix.SetMovingImage(image)
    transformix.Execute()
    return (sitk.GetArrayFromImage(transformix.GetResultImage())).astype(np.uint16)

def transformImageDeform(image, resolution, imageFilter):
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(resolution)
    transformParameterMap = imageFilter.GetTransformParameterMap()[-1]
    transformParameterMap["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
    transformix = sitk.TransformixImageFilter()
    transformix.SetTransformParameterMap(transformParameterMap)
    transformix.SetMovingImage(image)
    transformix.Execute()
    return (sitk.GetArrayFromImage(transformix.GetResultImage())).astype(np.uint16)


def getDeformationField(image, resolution, imageFilter):
    image = sitk.GetImageFromArray(image)
    image.SetSpacing(resolution)
    transformParameterMap = imageFilter.GetTransformParameterMap()
    for tp in transformParameterMap:
        tp["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
    transformix = sitk.TransformixImageFilter()
    transformix.SetTransformParameterMap(transformParameterMap)
    transformix.ComputeDeformationFieldOn()
    transformix.SetMovingImage(image)
    transformix.Execute()
    return (sitk.GetArrayFromImage(transformix.GetResultImage())).astype(np.uint16), transformix.GetDeformationField()

    

def computeError(img1, img2):
    return  np.sqrt(np.mean((img1-img2)**2))


def readFinalMetric():
    with open('elastix.log', 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "Final metric value  =" in line:
                line = line.split("Final metric value  =")[-1]
                line = line.replace(" ", "")
                line = line.strip()
                finalMetric = float(line)
    return finalMetric


def prepRegiImage(image):
    image = np.array(np.where(image>0,1.,0.))
    xx = image.shape[0] + 30
    yy = image.shape[1] + 30
    zz = image.shape[2] + 30
    image = padding(image, (xx, yy, zz))
    mask = np.copy(image)
    image = ndimage.distance_transform_edt(image)
    # image /= np.max(image)
    # image = 1 - image
    image *= mask
   
    return image
    
def prepTransImage(image):
    xx = image.shape[0] + 30
    yy = image.shape[1] + 30
    zz = image.shape[2] + 30
    image = padding(image, (xx, yy, zz))
    return image


def segPlot(image, resolution):
    from imgmisc import find_neighbors, rand_cmap

    seg_img = image

    # Identify cell neighbours and label so that adjacent cells don't have the

    # same value

    neighs = find_neighbors(seg_img, background=0)

    mapping = np.full(len(neighs), -1)

    for cell_id, cell_neighs in neighs.items():
        arr = np.append(cell_id, cell_neighs)

        labelled = mapping[arr-1] != -1


        label = 1

        while True:

            if label not in mapping[arr[labelled] - 1]:

                mapping[cell_id - 1] = label

                break

            label += 1


    vis_img = np.zeros_like(seg_img)

    for vv, label in enumerate(np.unique(seg_img)[1:]):

        vis_img[seg_img == label] = mapping[vv]

    # Plot!

    # Full

    cmap = 'Spectral'

    p = pvq.BackgroundPlotter() #off_screen=True, notebook=False

    p.add_volume(vis_img, cmap=cmap, opacity=[

                 0] + [1] * 254, show_scalar_bar=False, shade=True, diffuse=.9, resolution=resolution)

    # p.view_yz()

    p.set_background('white')

    # p.screenshot(f'{OUTPUT_DIR}/{os.path.splitext(os.path.basename(fname))[0]}-segmentation_top.png',

    #              transparent_background=True, window_size=[2000, 2000])

    p.show()

def reorganizeLabels(segImage):
    newImage = np.copy(segImage)
    regProps = regionprops(segImage)
    
    labelDict = {}
    for i, region in enumerate(regProps):
        currLabel = region.label
        newLabel = i+1
        labelDict[currLabel] = newLabel
        print(currLabel, newLabel)
        newImage[segImage==currLabel] = newLabel
    return newImage, labelDict


def errorFunctionBinary(moveImage, moveRes, fixedImage, fixedRes):
    if np.max(moveImage) > 1:
        moveImage = np.where(moveImage>0.5, 1, 0)
    if np.max(fixedImage) > 1:
        fixedImage = np.where(fixedImage>0.5, 1, 0)
        
    # error = np.sqrt(  np.mean(  (moveImage-fixedImage)**2  )  )
    fixedImage = ants.from_numpy(fixedImage.astype(np.float32), spacing=fixedRes)
    moveImage = ants.from_numpy(moveImage.astype(np.float32), spacing=moveRes)
    
    error = np.abs(ants.image_mutual_information(moveImage, fixedImage))
    
    # origMoveImage = np.where(origMoveImage>1, 1, 0)
    
    # moveVol = np.multiply(np.sum(origMoveImage), np.prod(moveRes))
    # fixedVol = np.multiply(np.sum(fixedImage), np.prod(fixedRes))
    return error

def errorFunction(moveImage, moveRes, fixedImage, fixedRes):
        
    # error = np.sqrt(  np.mean(  (moveImage-fixedImage)**2  )  )
    fixedImage = ants.from_numpy(fixedImage.astype(np.float32), spacing=fixedRes)
    moveImage = ants.from_numpy(moveImage.astype(np.float32), spacing=moveRes)
    
    error = np.abs(ants.image_mutual_information(moveImage, fixedImage))
    
    # origMoveImage = np.where(origMoveImage>1, 1, 0)
    
    # moveVol = np.multiply(np.sum(origMoveImage), np.prod(moveRes))
    # fixedVol = np.multiply(np.sum(fixedImage), np.prod(fixedRes))
    return error

def findMatchesFull(movingList, movingResList, fixedList, fixedResList):

    errorList = []
    matchingPairs = []
    
    for moveInd, movePath in enumerate(movingList):
        
        moveRes = tuple(np.array(movingResList[moveInd])*10.)
        moveImagePath = movingList[moveInd]
        moveImage = tifffile.imread(moveImagePath)
        if np.min(moveImage) == 1:
            moveImage -= 1
        # segImage = prepTransImage(moveImage)
        moveImage = prepRegiImage(moveImage)
        
        
        for fixedInd, fixPath in enumerate(fixedList):

        
            fixRes = tuple(np.array(fixedResList[fixedInd])*10.)
            fixImgPath = fixedList[fixedInd] 
            fixImg = tifffile.imread(fixImgPath)
            if np.min(fixImg) == 1:
                fixImg -= 1
            # fixedImage = prepTransImage(fixImg)
            fixImg = prepRegiImage(fixImg)
    

            
            
            try:
                transformation = registerImages(moveImage, moveRes, fixImg, fixRes, iterationNumbers=15000, do_translation=False,  do_rigid=False, do_affine=True, do_bspline=False)
                
                
                resImage = transformation.GetResultImage()
                # resRes = resImage.GetSpacing()
                resImageNumpy = sitk.GetArrayFromImage(resImage)
                # resImageNumpy = ants.from_numpy(resImageNumpy.astype(np.float32), spacing=resRes)
                
                # error = np.abs(ants.image_mutual_information(resImageNumpy, fixedImageAnts))
        
                error = errorFunction(resImageNumpy, fixRes, fixImg, fixRes)
                errorList.append(error)
            except:
                print(f'FAILED: {moveInd} and {fixedInd}')
                errorList.append(0.)
            
            matchingPairs.append((moveInd, fixedInd))
                
        
       
    for ind, pair in enumerate(matchingPairs):
        i, j = pair

        print(f"Match below with score: {errorList[ind]}")
        print(movingList[i])
        print(fixedList[j])
        print('--')
        
    return matchingPairs, errorList



def findMatchesVolume(movingList, movingResList, fixedList, fixedResList):
    
    matchingPairs = []
    score = []
    nScore = []
    
    for i, moving in enumerate(movingList):
        movePath = moving
        moving = tifffile.imread(moving)
        if np.min(moving) == 1:
            moving -= 1
        moveRes = movingResList[i]
        moveVol = np.multiply(np.sum( np.where(moving>0.5, 1, 0) ), np.prod(moveRes))
        
        for j, fixed in enumerate(fixedList):
            if movePath == fixed:
                continue
            fixed = tifffile.imread(fixed)
            if np.min(fixed) == 1:
                fixed -= 1
            fixedRes = fixedResList[j]
            fixedVol = np.multiply(np.sum( np.where(fixed>0.5, 1, 0) ), np.prod(fixedRes))
            
            tempScore = 3 - np.abs(1 - moveVol/fixedVol)
            tempNScore = 1 - np.abs(1 - np.max(moving) / np.max(fixed))
            
            if tempScore < 0:
                tempScore = 0.
                
            if tempNScore < 0:
                tempNScore = 0.
            
            score.append(  tempScore )
            nScore.append(tempNScore)
            matchingPairs.append((i,j))
            print(i, j, tempScore, tempNScore)
    
    return matchingPairs, score, nScore


def findMatches(movingList, movingResList, fixedList, fixedResList):
    
    matchingPairs = []
    
    for i, moving in enumerate(movingList):
        movePath = moving
        moving = tifffile.imread(moving)
        if np.min(moving) == 1:
            moving -= 1
        moveRes = movingResList[i]
        moveVol = np.multiply(np.sum( np.where(moving>1, 1, 0) ), np.prod(moveRes))
        score = []
        index = []
        for j, fixed in enumerate(fixedList):
            if movePath == fixed:
                continue
            fixed = tifffile.imread(fixed)
            if np.min(fixed) == 1:
                fixed -= 1
            fixedRes = fixedResList[j]
            fixedVol = np.multiply(np.sum( np.where(fixed>1, 1, 0) ), np.prod(fixedRes))
            
            score.append(np.abs(1 - moveVol/fixedVol))
            index.append(j)


        score, index = zip(*sorted(zip(score, index)))
        
        
        score = score[0:3]
        index = index[0:3]
        
        # for sco, ind in zip(score, index):
        #     print(f"Match below with score: {score}")
        #     print(movingList[i])
        #     print(fixedList[ind])
        #     print('--')
        #     matchingPairs.append((i,ind))
        
        # bestMatch = np.argmin(score)
        
        for sco, ind in zip(score, index):
            if sco < 0.5:
                print(f"Match below with score: {sco}")
                print(movingList[i])
                print(fixedList[ind])
                print('--')
                matchingPairs.append((i,ind))
            else:
                print(f"No match came even close for {movingList[i]} in terms of volume")
            
            
    
    errorList = []
    
    for i, j in matchingPairs:
        
        fixedInd = j
        moveInd = i
    
        fixRes = tuple(np.array(fixedResList[fixedInd])*10.)
        fixImgPath = fixedList[fixedInd] 
        fixImg = tifffile.imread(fixImgPath)
        if np.min(fixImg) == 1:
            fixImg -= 1
        # fixedImage = prepTransImage(fixImg)
        fixImg = prepRegiImage(fixImg)


        moveRes = tuple(np.array(movingResList[moveInd])*10.)
        moveImagePath = movingList[moveInd]
        moveImage = tifffile.imread(moveImagePath)
        if np.min(moveImage) == 1:
            moveImage -= 1
        # segImage = prepTransImage(moveImage)
        moveImage = prepRegiImage(moveImage)
        
        
        try:
            transformation = registerImages(moveImage, moveRes, fixImg, fixRes, iterationNumbers=15000, do_translation=True,  do_rigid=True, do_affine=False, do_bspline=False)
            
            
            resImage = transformation.GetResultImage()
            # resRes = resImage.GetSpacing()
            resImageNumpy = sitk.GetArrayFromImage(resImage)
            # resImageNumpy = ants.from_numpy(resImageNumpy.astype(np.float32), spacing=resRes)
            
            # error = np.abs(ants.image_mutual_information(resImageNumpy, fixedImageAnts))
    
            error = errorFunction(resImageNumpy, moveRes, fixImg, fixRes)
            errorList.append(error)
        except:
            print(f'FAILED: {i} and {j}')
            errorList.append(666.)
        
       
    for ind, pair in enumerate(matchingPairs):
        i, j = pair

        print(f"Match below with score: {errorList[ind]}")
        print(movingList[i])
        print(fixedList[j])
        print('--')
        
    return matchingPairs, errorList


def compVolume(fixList, fixRestList, moveList, moveRestList):
    volMat = []
    for move, moveRes in zip(moveList, moveRestList):
        row = []
        moveImg = tifffile.imread(move).astype(np.int16)
        if np.min(moveImg) == 1:
            moveImg -= 1
        movingVol = np.multiply(np.sum( np.where(moveImg>0.5, 1, 0) ), np.prod(moveRes))
        
        for fix, fixRes in zip(fixList, fixRestList):
            fixImg = tifffile.imread(fix).astype(np.int16)
            if np.min(fixImg) == 1:
                fixImg -= 1
            fixedVol = np.multiply(np.sum( np.where(fixImg>0.5, 1, 0) ), np.prod(fixRes))
            # row.append(fixedVol)
            volMetric = 1 - abs(movingVol - fixedVol) / fixedVol
            if volMetric < 0:
                volMetric = 0.
            row.append(volMetric)
            
        volMat.append(row)
        
    # plt.imshow(volMat)
    # plt.show()
        
    return np.array(volMat)
    


def genGrid(size, spacing, zSlize=None):
    grid = np.zeros(size)
    for i, sli in enumerate(grid):
        for j, row in enumerate(sli):
            for k, val in enumerate(row):
                
                if i == 0 or j == 0 or k == 0:
                    continue
                elif i == size[0]-1 or j == size[1]-1 or k == size[2]-1:
                    continue
                
                if zSlize == None:
                    if i%spacing==0 and j%spacing==0 and k%spacing!=0:
                        grid[i-1:i+1,j-1:j+1,k-1:k+1] = 1
                    elif i%spacing==0 and j%spacing!=0 and k%spacing==0:
                        grid[i-1:i+1,j-1:j+1,k-1:k+1] = 1
                    elif i%spacing!=0 and j%spacing==0 and k%spacing==0:
                        grid[i-1:i+1,j-1:j+1,k-1:k+1] = 1
                    elif i%spacing==0 and j%spacing==0 and k%spacing==0:
                        grid[i-1:i+1,j-1:j+1,k-1:k+1] = 1
                        
                else:
                    if i==zSlize and j%spacing==0 and k%spacing!=0:
                        grid[i-1:i+1,j-1:j+1,k-1:k+1] = 1
                    elif i==zSlize and j%spacing!=0 and k%spacing==0:
                        grid[i-1:i+1,j-1:j+1,k-1:k+1] = 1
                    # elif i%spacing!=0 and j%spacing==0 and k%spacing==0:
                    #     grid[i-1:i+1,j-1:j+1,k-1:k+1] = 1
                    elif i==zSlize and j%spacing==0 and k%spacing==0:
                        grid[i-1:i+1,j-1:j+1,k-1:k+1] = 1
  
    return grid


if __name__ == "__main__":
    
    samList = [r"/home/jonas/projects/allImages/SAMimages/segmented_refined/131007-PIN_GFP-acyl_YFP-plant_1-0h_stitched_stackreg_crop_wiener_C0_predictions_multicut-refined.tif",
               r"/home/jonas/projects/allImages/SAMimages/segmented_refined/131007-PIN_GFP-acyl_YFP-plant_3-0h_stackreg_crop_wiener_C0_predictions_multicut-refined.tif",
               r"/home/jonas/projects/allImages/SAMimages/segmented_refined/131007-PIN_GFP-acyl_YFP-plant_8-0h_stackreg_crop_wiener_C0_predictions_multicut-refined.tif",
               r"/home/jonas/projects/allImages/SAMimages/segmented_refined/131007-PIN_GFP-acyl_YFP-plant_10-0h_stackreg_crop_wiener_C0_predictions_multicut-refined.tif",
               r"/home/jonas/projects/allImages/SAMimages/segmented_refined/131007-PIN_GFP-acyl_YFP-plant_13-0h_stackreg_crop_wiener_C0_predictions_multicut-refined.tif"]
                    
    samPinList = [r"/home/jonas/projects/allImages/SAMimages/raw/131007-PIN_GFP-acyl_YFP-plant_1-0h_stitched_stackreg_crop_wiener_C1.tif",
                  r"/home/jonas/projects/allImages/SAMimages/raw/131007-PIN_GFP-acyl_YFP-plant_3-0h_stackreg_crop_wiener_C1.tif",
                  r"/home/jonas/projects/allImages/SAMimages/raw/131007-PIN_GFP-acyl_YFP-plant_8-0h_stackreg_crop_wiener_C1.tif",
                  r"/home/jonas/projects/allImages/SAMimages/raw/131007-PIN_GFP-acyl_YFP-plant_10-0h_stackreg_crop_wiener_C1.tif",
                  r"/home/jonas/projects/allImages/SAMimages/raw/131007-PIN_GFP-acyl_YFP-plant_13-0h_stackreg_crop_wiener_C1.tif"]
                    
                    
    
    sourcePointList = [r"/home/jonas/projects/allImages/seg/20_11_top.txt",
                       r"/home/jonas/projects/allImages/seg/20_21_top.txt",
                       r"/home/jonas/projects/allImages/seg/20_41_top.txt",
                       r"/home/jonas/projects/allImages/seg/20_51_top.txt",
                       r"/home/jonas/projects/allImages/seg/20_52_top.txt",
                       r"/home/jonas/projects/allImages/seg/20_53_top.txt",
                       r"/home/jonas/projects/allImages/seg/20_61_top.txt",
                       r"/home/jonas/projects/allImages/seg/21_22_top.txt",
                       r"/home/jonas/projects/allImages/seg/21_31_top.txt",
                       r"/home/jonas/projects/allImages/seg/21_62_top.txt",
                       r"/home/jonas/projects/allImages/seg/21_61_top.txt",
                       r"/home/jonas/projects/allImages/seg/21_21_top.txt"]
    
    
    movingList = [r"/home/jonas/projects/allImages/seg/C2-201112-PIN_GFP-MM_dsRed-WT-SAM1-FM1_predictions_multicut-refined.tif",
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
    
    fixedList = [r"/home/jonas/projects/allImages/atlas/segmented_tiffs/10h_segmented.tif",
                 r"/home/jonas/projects/allImages/atlas/segmented_tiffs/40h_segmented.tif",
                 r"/home/jonas/projects/allImages/atlas/segmented_tiffs/96h_segmented.tif",
                 r"/home/jonas/projects/allImages/atlas/segmented_tiffs/120h_segmented.tif",
                 r"/home/jonas/projects/allImages/atlas/segmented_tiffs/132h_segmented.tif"]
    
    fixedResList = [(0.12621046353913226, 0.2415319, 0.2415319),
                    (0.12797625374470484, 0.2415319, 0.2415319),
                    (0.16652491797820784, 0.2361645, 0.2361645),
                    (0.17078415345851233, 0.2361645, 0.2361645),
                    (0.12529532381630637, 0.1735086, 0.1735086)]    
        
    movingResList = [(0.19999999999999998, 0.20756646174321786, 0.20756646174321786), 
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
    
    
    
    
    allFixedResList =  [(0.14758457626467922, 0.2415319, 0.2415319),
                        (0.12621046353913226, 0.2415319, 0.2415319),
                        (0.15790386612892207, 0.2415319, 0.2415319),
                        (0.12444464790545198, 0.2415319, 0.2415319),
                        (0.10747233031601376, 0.2415319, 0.2415319),
                        (0.12797625374470484, 0.2415319, 0.2415319),
                        (0.13403487751019727, 0.2361645, 0.2361645),
                        (0.1555041065897905, 0.2361645, 0.2361645),
                        (0.1320346916307907, 0.2361645, 0.2361645),
                        (0.13020055970792704, 0.2361645, 0.2361645),
                        (0.1467801122071224, 0.2361645, 0.2361645),
                        (0.1694201689855073, 0.1445905, 0.1445905),
                        (0.16652491797820784, 0.2361645, 0.2361645),
                        (0.16686897911175463, 0.2355364, 0.2355364),
                        (0.14148094899105437, 0.181665, 0.181665),
                        (0.17078415345851233, 0.2361645, 0.2361645),
                        (0.15680538143629197, 0.2372188, 0.2372188),
                        (0.12529532381630637, 0.1735086, 0.1735086)]
    
    allFixedList = [r"/home/jonas/projects/allImages/segmentation_tiffs/0h_segmented.tif",
                    r"/home/jonas/projects/allImages/segmentation_tiffs/10h_segmented.tif",
                    r"/home/jonas/projects/allImages/segmentation_tiffs/18h_segmented.tif",
                    r"/home/jonas/projects/allImages/segmentation_tiffs/24h_segmented.tif",
                    r"/home/jonas/projects/allImages/segmentation_tiffs/32h_segmented.tif",
                    r"/home/jonas/projects/allImages/segmentation_tiffs/40h_segmented.tif",
                    r"/home/jonas/projects/allImages/segmentation_tiffs/48h_segmented.tif",
                    r"/home/jonas/projects/allImages/segmentation_tiffs/57h_segmented.tif",
                    r"/home/jonas/projects/allImages/segmentation_tiffs/64h_segmented.tif",
                    r"/home/jonas/projects/allImages/segmentation_tiffs/72h_segmented.tif",
                    r"/home/jonas/projects/allImages/segmentation_tiffs/81h_segmented.tif",
                    r"/home/jonas/projects/allImages/segmentation_tiffs/88h_segmented.tif",
                    r"/home/jonas/projects/allImages/segmentation_tiffs/96h_segmented.tif",
                    r"/home/jonas/projects/allImages/segmentation_tiffs/104h_segmented.tif",
                    r"/home/jonas/projects/allImages/segmentation_tiffs/112h_segmented.tif",
                    r"/home/jonas/projects/allImages/segmentation_tiffs/120h_segmented.tif",
                    r"/home/jonas/projects/allImages/segmentation_tiffs/128h_segmented.tif",
                    r"/home/jonas/projects/allImages/segmentation_tiffs/132h_segmented.tif"]


    # volMat = compVolume(fixedList[1::], fixedResList[1::], movingList[0:7], movingResList[0:7])
    
    # fig = plt.figure(figsize=(16, 12))
    # ax = fig.add_subplot(111)
    # im = ax.imshow(volMat)

    
    # # plt.imshow(volMat)
    # ax.set_xticks(range(4))
    # ax.set_yticks(range(7))
    # ax.set_xticklabels(['40h', '96h', '120h', '132h'])
    # ax.set_yticklabels(['Flower 1', 'Flower 2', 'Flower 3', 'Flower 4', 'Flower 5', 'Flower 6', 'Flower 7'])

    
    # for (j,i),label in np.ndenumerate(volMat):
    #     ax.text(i,j,round(label,3),ha='center',va='center')



    # font = {'family' : 'normal',
    # # 'weight' : 'bold',
    # 'size'   : 18}
    
    # matplotlib.rc('font', **font)
    
    # plt.show()

    # ind = 1
    # im = tifffile.imread(movingList[ind])
    # res = movingResList[ind]

    # segPlot(im, res)


    # h104 = tifffile.imread(allFixedList[17])
    # h104 = np.where(h104>1, 1, 0)
    # hRes = allFixedResList[17]
    # img1 = tifffile.imread(movingList[7])
    # img1 = np.where(img1>0, 1, 0) 
    # img1Res = movingResList[7]
    # p = pvq.BackgroundPlotter()
    # p.add_volume(h104, opacity='linear', resolution=hRes,shade=True,diffuse=1.5, specular=1.)
    # p.add_volume(img1, opacity='linear', resolution=img1Res,shade=True,diffuse=1.5, specular=1.)
    # p.show()
    
    # for i, moving in enumerate(movingList):
    #     movePath = moving
    #     moving = tifffile.imread(moving)
    #     nrCells = np.max(moving)
    #     if np.min(moving) == 1:
    #         moving -= 1
    #     moveRes = movingResList[i]
    #     moveVol = np.multiply(np.sum( np.where(moving>0.5, 1, 0) ), np.prod(moveRes))
        
    #     print(moveVol/nrCells)

    # samResList = [get_resolution(fname) for fname in samList]

    # x, y = findMatchesFull(movingList[0:2], 
                            # movingResList[0:2], 
                            # fixedList[-1::], 
                            # fixedResList[-1::])
    
    # xVol, yVol, yN = findMatchesVolume(movingList[0:5]+movingList[7::],
    #                     movingResList[0:5]+movingResList[7::],
    #                     allFixedList[4::], 
    #                     allFixedResList[4::])

    # mat = np.zeros( (x[-1][0]+1, x[-1][1]+1) )
    
    # for i, pair in enumerate(x):
    #     score = y[i]
    #     xx, yy = pair
    #     mat[xx, yy] = score
        
    # plt.imshow(mat)


    fixedInd = 1
    moveInd = 0

    movePoint = r"/home/jonas/projects/allImages/pointFolder/20_11_points.txt"
    fixedPoint = r"/home/jonas/projects/allImages/pointFolder/20_21_points.txt"

    
    fixRes = tuple(np.array(movingResList[fixedInd])*10.)
    # fixRes = tuple(np.array(get_resolution(samList[1]))*10.)
    fixImgPath = movingList[fixedInd] 
    # fixImgPath = samList[1]
    fixImg = tifffile.imread(fixImgPath)
    if np.min(fixImg) == 1:
        fixImg -= 1
        
    # imsize = fixImg.shape
    # imcent = tuple(np.array(ndimage.center_of_mass(fixImg)).astype(np.int32))
    # radius = np.max(imcent) * 0.75
    # sphere = (genSphere(imsize, radius, imcent)).astype(np.int32)
    
    fixImg = prepRegiImage(fixImg)
    
    
    
    
    moveRes = tuple(np.array(movingResList[moveInd])*10.)
    # moveRes = tuple(np.array(get_resolution(samList[3]))*10.)
    moveImagePath = movingList[moveInd]
    # moveImagePath = samList[3]
    moveImage = tifffile.imread(moveImagePath)
    if np.min(moveImage) == 1:
        moveImage -= 1
        
    # imsize = moveImage.shape
    # imcent = tuple(np.array(ndimage.center_of_mass(moveImage)).astype(np.int32))
    # radius = np.max(imcent) * 0.75
    # sphere = (genSphere(imsize, radius, imcent)).astype(np.int32)
    
    moveImage = prepRegiImage(moveImage)
    

    
    
    transformation = registerImages(moveImage, moveRes, fixImg, fixRes, iterationNumbers=15000, do_translation=True, do_rigid=True, do_affine=True, do_bspline=True, fixFile=fixedPoint, moveFile=movePoint) #, fixFile=fixedPoint, moveFile=movePoint
    
    pinImagePath = pinList[moveInd]
    # pinImagePath = samPinList[3]
    pinImage = tifffile.imread(pinImagePath)
    pinImage = prepTransImage(pinImage)
    
    segImagePath = movingList[moveInd]
    # segImagePath = samList[3]
    segImage = tifffile.imread(segImagePath)
    segImage = prepTransImage(segImage)
    
    
    # resImage = transformation.GetResultImage()
    # finalSpacing = resImage.GetSpacing()
    newPinImage = transformImage(pinImage, moveRes, transformation)
    newSegImage = transformImage(segImage, moveRes, transformation)
    
    # grid = genGrid(segImage.shape, 25, 125)
    # newGrid = transformImage(grid, moveRes, transformation)
    
    # newSegImage, deformField = getDeformationField(segImage, moveRes, transformation)
    

    # segPlot(moveImage, moveRes)

    
    # plotFixed = tifffile.imread(fixImgPath) -1
    # plotFixed = prepTransImage(plotFixed)
    
    # error, moveVol, fixedVol = errorFunctionBinary(newSegImage, segImage, moveRes, plotFixed, fixRes)


    # tifffile.imsave(r'/home/jonas/projects/allImages/132hSource/transPinData_201112_21.tif', newPinImage.astype(np.uint16))
    # tifffile.imsave(r'/home/jonas/projects/allImages/132hSource/transSegData_201112_21.tif', newSegImage.astype(np.uint16))



    tifffile.imsave(r'/home/jonas/projects/allImages/compareFlower1_2/transPinData_11_21.tif', newPinImage.astype(np.uint16))
    tifffile.imsave(r'/home/jonas/projects/allImages/compareFlower1_2/transSegData_11_21.tif', newSegImage.astype(np.uint16))

    # tifffile.imsave(r"/home/jonas/projects/allImages/SAMimages/transformed/transPinData_3_to_1.tif", newPinImage.astype(np.uint16))
    # tifffile.imsave(r"/home/jonas/projects/allImages/SAMimages/transformed/transSegData_3_to_1.tif", newSegImage.astype(np.uint16))


    # resImage = transformation.GetResultImage()
    # newRes = resImage.GetSpacing()
    
    # resImageNumpy = np.where(resImageNumpy>0.1, 1, 0)
    # plotFixed = np.where(fixImg>0.1, 1, 0)
    
    
    plotFixed = tifffile.imread(fixImgPath) 
    plotFixed = np.where(plotFixed>0, 2, 0)
    plotFixed = prepTransImage(plotFixed)
    
    newSegPlot = np.where(newSegImage>0.5, 1, 0)
    # newSegPlot = np.copy(newSegImage)
    
    # newSegPlot[newSegImage>1] += 200
    
    # diff = np.abs( plotFixed - newSegPlot )
    
    # test = tifffile.imread(allFixedList[14])
    
        
    # fixedImageAnts = ants.from_numpy(fixImg.astype(np.float32), spacing=fixRes)
    # resImage = transformation.GetResultImage()
    # resRes = resImage.GetSpacing()
    # resImageNumpy = sitk.GetArrayFromImage(resImage)
    # resImageNumpy = ants.from_numpy(resImageNumpy.astype(np.float32), spacing=resRes)
    
    # error = np.abs(ants.image_mutual_information(resImageNumpy, fixedImageAnts))
    # print(error)
    
    # daSum = (plotFixed+newSegPlot).astype(np.float32)
    
    # daSumMask = np.where(daSum == 1, 0.1, 1)
    
    # daSum *= daSumMask
    
    fixedVol = np.multiply(np.sum( np.where(plotFixed>0.5, 1, 0) ), np.prod(fixRes))
    movingVol = np.multiply(np.sum( np.where(segImage>0.5, 1, 0) ), np.prod(moveRes))
    newMovingVol = np.multiply(np.sum( np.where(newSegPlot>0.5, 1, 0) ), np.prod(fixRes))
    
    print(fixedVol / movingVol)
    print(fixedVol / newMovingVol)
    
    # img = tifffile.imread(r"/home/jonas/projects/allImages/mem/C2-201112-PIN_GFP-MM_dsRed-WT-SAM2-FM1.tif")
    
    # img[img<1000]=0
    
    
    daSum = (plotFixed+newSegPlot).astype(np.float32)
    daSum[daSum==1] = 0.2
    
    # im = tifffile.imread(samList[1])
    # fixRes = get_resolution(samList[1])
    
    # imsize = im.shape
    # # imcent = tuple((np.array(im.shape)/2).astype(np.int32))
    # imcent = tuple(np.array(ndimage.center_of_mass(im)).astype(np.int32))
    # radius = np.max(imcent) * 0.6
    
    # sphere = (genSphere(imsize, radius, imcent)).astype(np.int32)
    
    p = pvq.BackgroundPlotter()
    # p.set_background("black")
    # p.add_volume(grid, opacity='linear', resolution=[1,1,1],shade=True,diffuse=1.5, specular=1.)
    # p.add_volume(newGrid, opacity='linear', resolution=moveRes,shade=True,diffuse=1.5, specular=1.)
    p.add_volume(daSum, opacity='linear', resolution=fixRes,shade=True,diffuse=1.5, specular=1.)
    p.show()


    # reOrgSeg = reorganizeLabels(segImage)

    # segPlot(reOrgSeg[0] ,(1,1,1))





"""



    score = []
    combo = []
    
    
    fixRes = fixedResList[3]
    fixImgPath = fixedList[3] 
    fixImg = tifffile.imread(fixImgPath)
    if np.min(fixImg) == 1:
        fixImg -= 1
    fixImg = prepRegiImage(fixImg)
    
    for i, moving in enumerate(movingList):
        moveRes = movingResList[i]
        moveImage = tifffile.imread(moving)
        moveImage = prepRegiImage(moveImage)
        
        output = registerImages(moveImage, moveRes, fixImg, fixRes, iterationNumbers=5000, do_affine=False, do_bspline=False)
        if output == None:
            finalScore = 0
        else:
            finalScore = readFinalMetric()
        
        score.append(finalScore)
        combo.append((i,3))
        
        print(i, 3, finalScore)



"""








"""





fixPath = r'/home/jonas/projects/allImages/atlas/segmented_tiffs/120h_segmented.tif'
#movePath = r'/home/jonas/projects/allImages/seg/C2-201112-PIN_GFP-MM_dsRed-WT-SAM2-FM1_predictions_multicut-refined.tif'
#pinPath = r'/home/jonas/projects/allImages/pin/C1-201112-PIN_GFP-MM_dsRed-WT-SAM2-FM1.tif'

# movePath = r"/home/jonas/projects/allImages/seg/C2-210305-PIN1GFP-MMdsRed-WT_flowers-SAM2-FM2_predictions_multicut-refined.tif"
# pinPath = r"/home/jonas/projects/allImages/pin/C1-210305-PIN1GFP-MMdsRed-WT_flowers-SAM2-FM2.tif"

movePath = r"/home/jonas/projects/allImages/seg/C2-210305-PIN1GFP-MMdsRed-WT_flowers-SAM3-FM1_predictions_multicut-refined.tif"
pinPath = r"/home/jonas/projects/allImages/pin/C1-210305-PIN1GFP-MMdsRed-WT_flowers-SAM3-FM1.tif"

fixedImgO = tifffile.imread(fixPath).astype(np.float32) - 1
movingImgO = tifffile.imread(movePath).astype(np.float32)

# fixRes = (0.1735086, 0.1735086, 0.12529532381630637)
fixRes = (0.2361645, 0.2361645, 0.17078415345851233)
# moveRes = get_resolution(movePath)
moveRes = (0.25     , 0.4378858, 0.4378858)
#pinRes = get_resolution(pinPath)
pinRes = moveRes

fixRes = tuple(np.array(fixRes)*10.)
moveRes = tuple(np.array(moveRes)*10.)
pinRes = tuple(np.array(pinRes)*10.)

fixedImgBin = np.array(np.where(fixedImgO>0,1.,0.))
movingImgBin = np.where(movingImgO>0,1.,0.)



xxM = movingImgBin.shape[0] + 30
yyM = movingImgBin.shape[1] + 30
zzM = movingImgBin.shape[2] + 30

xxF = fixedImgBin.shape[0] + 30
yyF = fixedImgBin.shape[1] + 30
zzF = fixedImgBin.shape[2] + 30

# xx, yy, zz = np.max([xxM, xxF]), np.max([yyM, yyF]), np.max([zzM, zzF])


fixedImgBin = padding(fixedImgBin, (xxF, yyF, zzF))
movingImgBin = padding(movingImgBin, (xxM, yyM, zzM))

fixedImgBin = ndimage.distance_transform_edt(fixedImgBin)
movingImgBin = ndimage.distance_transform_edt(movingImgBin)

fixedImgBin /= np.max(fixedImgBin)
movingImgBin /= np.max(movingImgBin)

# fixedImgBin = padding(fixedImgBin, (664, 730, 730))
# movingImgBin = padding(movingImgBin, (664, 730, 730))


# fixedImgO_smooth = filters.gaussian(fixedImgBin, sigma=15)
# movingImgO_smooth = filters.gaussian(movingImgBin, sigma=15)
# fixedImgO_smooth = np.where(fixedImgO_smooth > 0.3, 1, 0).astype(np.float32)
# movingImgO_smooth = np.where(movingImgO_smooth > 0.3, 1, 0).astype(np.float32)


del fixedImgO
del movingImgO
# del fixedImgBin
# del movingImgBin


# fixedImage = sitk.GetImageFromArray(fixedImgO_smooth, isVector=False)
# movingImage = sitk.GetImageFromArray(movingImgO_smooth, isVector=False)

fixedImage = sitk.GetImageFromArray(fixedImgBin.astype(np.float32), isVector=False)
movingImage = sitk.GetImageFromArray(movingImgBin.astype(np.float32), isVector=False)


fixedImage.SetSpacing(fixRes)
# fixedImage.SetOrigin((0.,0.,0.))
# fixedImage.SetDirection(tuple(np.eye(3).flatten()))

movingImage.SetSpacing(moveRes)
# movingImage.SetOrigin((0.,0.,0.))
# movingImage.SetDirection(tuple(np.eye(3).flatten()))

# reference_distance_map = sitk.SignedMaurerDistanceMap(fixedImage, squaredDistance=False, useImageSpacing=True)



iterationNumbers = 15000

elastixImageFilter = sitk.ElastixImageFilter()
elastixImageFilter.SetFixedImage(fixedImage)
elastixImageFilter.SetMovingImage(movingImage)

elastixImageFilter.SetParameter("Registration","MultiResolutionRegistration")
elastixImageFilter.SetParameter( "Metric", ("AdvancedMeanSquares")) #, AdvancedKappaStatistic AdvancedMeanSquares "CorrespondingPointsEuclideanDistanceMetric",
# elastixImageFilter.SetParameter("Metric0Weight", "0.0")
elastixImageFilter.LogToFileOn()
# elastixImageFilter.LogToConsoleOn()


translation = sitk.GetDefaultParameterMap('translation')
translation["MaximumNumberOfIterations"] = [str(iterationNumbers)]
translation["NumberOfResolutions"] = [str(6)]
# # translation["FixedImagePyramidSchedule"] = ["256 256 256 128 128 128"]
# # translation["MovingImagePyramidSchedule"] = ["256 256 256 128 128 128"]

rigid = sitk.GetDefaultParameterMap('rigid')
rigid["MaximumNumberOfIterations"] = [str(iterationNumbers)]
rigid["NumberOfResolutions"] = [str(6)]

affine = sitk.GetDefaultParameterMap('affine')
affine["MaximumNumberOfIterations"] = [str(iterationNumbers)]
affine["NumberOfResolutions"] = [str(8)]

bSpline = sitk.GetDefaultParameterMap('bspline')
bSpline["MaximumNumberOfIterations"] = [str(iterationNumbers*2)]
bSpline["NumberOfResolutions"] = [str(1)]
bSpline['ImagePyramidSchedule'] = ['4', '4', '4']
bSpline['FinalGridSpacingInPhysicalUnits'] = ['10.0', '10.0', '10.0']
bSpline['GridSpacingSchedule'] = ['10.0' '10.0' '10.0']
bSpline['FinalBSplineInterpolationOrder'] = ['0']

parameterMapVector = sitk.VectorOfParameterMap()
parameterMapVector.append(translation)
parameterMapVector.append(rigid)
parameterMapVector.append(affine)
parameterMapVector.append(bSpline)

elastixImageFilter.SetParameterMap(parameterMapVector)

# elastixImageFilter.SetParameter("MaximumNumberOfIterations" , str(iterationNumbers))
# elastixImageFilter.SetParameter("NumberOfResolutions" , str(4))

# MaximumNumberOfSamplingAttempts



# parameterMapVector = sitk.VectorOfParameterMap()
# parameterMapVector.append(sitk.GetDefaultParameterMap("translation"))
# parameterMapVector.append(sitk.GetDefaultParameterMap("affine"))
# # parameterMapVector.append(sitk.GetDefaultParameterMap("bspline"))
# elastixImageFilter.SetParameterMap(parameterMapVector)



# elastixImageFilter.SetFixedPointSetFileName("/home/jonas/projects/elastixRegi/fixPoints.txt")
# elastixImageFilter.SetMovingPointSetFileName("/home/jonas/projects/elastixRegi/movePoints.txt")


elastixImageFilter.Execute()


resImage = elastixImageFilter.GetResultImage()

resImageNumpy = sitk.GetArrayFromImage(resImage)

diff = np.abs( fixedImgBin - resImageNumpy)




# elastixImageFilter2 = sitk.ElastixImageFilter()
# elastixImageFilter2.SetFixedImage(fixedImage)
# elastixImageFilter2.SetMovingImage(resImage)

# elastixImageFilter2.SetParameter("Registration","MultiResolutionRegistration")
# elastixImageFilter2.SetParameter( "Metric", ("AdvancedKappaStatistic")) #, "CorrespondingPointsEuclideanDistanceMetric",
# # elastixImageFilter2.SetParameter("Metric0Weight", "0.0")
# elastixImageFilter2.LogToFileOn()
# # elastixImageFilter2.LogToConsoleOn()
# # elastixImageFilter2.SetParameterMap(sitk.GetDefaultParameterMap('translation'))
# # elastixImageFilter2.SetParameterMap(sitk.GetDefaultParameterMap('rigid'))
# # elastixImageFilter2.AddParameterMap(sitk.GetDefaultParameterMap('affine'))
# elastixImageFilter2.AddParameterMap(sitk.GetDefaultParameterMap('bspline'))

# elastixImageFilter2.SetParameter("MaximumNumberOfIterations" , str(5000))
# elastixImageFilter2.SetParameter("NumberOfResolutions" , str(2))

# elastixImageFilter2.Execute()



# summed = fixedImgO_smooth + resImageNumpy

# summed[summed <0] = 0

# resImageNumpy[resImageNumpy<0]=0






### FINAL TRANSFORMS

pinImg = tifffile.imread(pinPath).astype(np.float32)
pinImg = tifffile.imread(movePath).astype(np.float32)

xxM = pinImg.shape[0] + 30
yyM = pinImg.shape[1] + 30
zzM = pinImg.shape[2] + 30

pinImg = padding(pinImg, (xxM, yyM, zzM))

pinImg = sitk.GetImageFromArray(pinImg)
pinImg.SetSpacing(moveRes)


transformParameterMap = elastixImageFilter.GetTransformParameterMap()
for tp in transformParameterMap:
    tp["ResampleInterpolator"] = ["FinalNearestNeighborInterpolator"]
transformix = sitk.TransformixImageFilter()

transformix.SetTransformParameterMap(transformParameterMap)
transformix.SetMovingImage(pinImg)

transformix.Execute()


transformedPinImg1 = (sitk.GetArrayFromImage(transformix.GetResultImage())).astype(np.uint16)


# tifffile.imsave(r'/home/jonas/projects/allImages/120hSources/transPinData_210305_31.tif', transformedPinImg1.astype(np.uint16))
tifffile.imsave(r'/home/jonas/projects/allImages/120hSources/transSegData_210305_31.tif', transformedPinImg1.astype(np.uint16))
###


plot = np.copy(resImageNumpy)
plot[plot<0] = 0


plot = np.where(resImageNumpy>0.1, 1, 0).astype(np.int8)
plotF = np.where(fixedImgBin>0.1, 1, 0).astype(np.int8)


pinPath = r'/home/jonas/projects/allImages/120hSources/mappedSegData_210305_31.tif'
pinImg = tifffile.imread(pinPath)

# pinImg[pinImg<1000] = 0
# pinImg[pinImg>25535] = 25535

p = pvq.BackgroundPlotter()
p.add_volume( pinImg , opacity='linear', resolution=[1,1,1],shade=True,diffuse=1.5, specular=1.)
p.show()





"""





































