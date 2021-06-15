import sys
import os
pwd = os.getcwd()
sys.path.append(pwd+"/temp")
sys.path.append(pwd+"/utils")
from multiThreadQuantify import *










if __name__ == "__main__":
    ###### Fill out quantification parameters and files (Edit here)
    
    ### Fill in where the source data is
    path_to_mapped_seg = ""
    path_to_mapped_data = ""
    resolution = (1,1,1)
    
    ### Paths to save data
    path_to_quantified_data = ""
    name = "" # Name of the final organism .init and .neigh files
    
    ### Number of cores to use and angular intervals to use (in degrees)
    quantification_cores = 24   # Watch out for RAM usage when using a lot of cores
    
    ### Ball size for wall_depth
    wall_depth = 2
    
    
    
    ###### Run quantification 
    
    ### Load data
    segImg = tiff.imread(path_to_mapped_seg).astype(np.int32)
    pinImg = tiff.imread(path_to_mapped_data).astype(np.int32)
    assert np.min(segImg) == 0   # Background should be 0
    
    
    ### Run mapping procedure
    max_cores = multiprocessing.cpu_count()
    assert decomposition_cores <= max_cores  
    assert segImg.shape == pinImg.shape # Images need to have the same shape, 
                                        # 15voxel padding is added to all sides 
                                        # from the registration. Using the padding function
                                        # in multiThreadQuantify.py can help solve this by 
                                        # setting the final shape to shape+30


    
    segImg, labelDict = reorganizeLabels(segImg) # Ensure that the cell ids dont have holes, example: 1,2,3,7,14,15...
    
    ### Quantify the data
    batchList = batch_compute(segImg, pinImg, quantification_cores, ball(wall_depth), resolution)

    ### Write data as organism files
    writeToFiles(batchList,path_to_quantified_data, name, name)
    
    
    
    
    
    
    
    