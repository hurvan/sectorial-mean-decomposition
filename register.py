import sys
import os
pwd = os.getcwd()
sys.path.append(pwd+"/registration")
from elastixRegi import *










if __name__ == "__main__":
    
    
    ###### Fill out registration parameters and files (Edit here)
    
    ### Fill in where the source data is
    path_to_seg_source = ""
    path_to_data_source = ""
    resolution_for_source = (1,1,1)
    
    ### Fill in where the target data is
    path_to_seg_target = ""
    resolution_for_target = (1,1,1)
    
    ### Paths to save data
    path_to_registered_seg = ""
    path_to_registered_data = ""
    
    ### If using points, fill out the paths to the .txt files
    movePoint = ""
    fixedPoint = ""

    


    ###### Start registration procedure
    
    ### Load and prep Images

    fixRes = tuple(np.array(resolution_for_target)*10.) # For some reason it helped to increase the resolution by a factor of 10 during the registration
    fixImg = tifffile.imread(path_to_seg_target)
    if np.min(fixImg) == 1:
        fixImg -= 1   
    fixImg = prepRegiImage(fixImg)
    
    moveRes = tuple(np.array(resolution_for_source)*10.) # For some reason it helped to increase the resolution by a factor of 10 during the registration
    moveImage = tifffile.imread(path_to_seg_source)
    if np.min(moveImage) == 1:
        moveImage -= 1   
    moveImage = prepRegiImage(moveImage)
    
    
    ### Perform registration and find the transformation, if not using pointfiles, pass "None" as arguments to fixFile and MoveFile
    transformation = registerImages(moveImage, moveRes, fixImg, fixRes, iterationNumbers=15000, do_translation=True, do_rigid=True, do_affine=True, do_bspline=True, fixFile=fixedPoint, moveFile=movePoint)
    
    
    ### Load and prepare images to transform
    dataImage = tifffile.imread(path_to_data_source)
    dataImage = prepTransImage(dataImage) 
    
    segImage = tifffile.imread(path_to_seg_source)
    segImage = prepTransImage(segImage)
    
    ### Transform the images
    newPinImage = transformImage(dataImage, moveRes, transformation)
    newSegImage = transformImage(segImage, moveRes, transformation)
    
    ### Save registered images
    tifffile.imsave(path_to_registered_seg, newSegImage.astype(np.uint16))
    tifffile.imsave(path_to_registered_data, newPinImage.astype(np.uint16))

    