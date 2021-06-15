import sys
import os
pwd = os.getcwd()
sys.path.append(pwd+"/temp")
sys.path.append(pwd+"/utils")
from decomposeCells import *










if __name__ == "__main__":
    
    ###### Fill out mapping parameters and files (Edit here)
    
    ### Fill in where the source data is
    path_to_registered_seg_source = ""
    path_to_registered_data_source = ""
    resolution_for_registered_source = (1,1,1)

    ### Fill in where the target data is
    path_to_seg_target = ""
    resolution_for_target = (1,1,1)

    ### Paths to save data
    path_to_mapped_data = ""
    path_to_mapped_seg = ""   # might be redudant
    
    ### Number of cores to use and angular intervals to use (in degrees)
    decomposition_cores = 7   # Watch out for RAM usage when using a lot of cores
    angular_intervals = (45/2, 90/2)
    
    
    
    ###### Run mapping (Should not need to be edited, except for datatype of mapped_data)
    
    ### Load data
    seg_source = tifffile.imread(path_to_registered_seg_source).astype(np.int32)
    data_source = tifffile.imread(path_to_registered_data_source).astype(np.int32)
    seg_target = tifffile.imread(path_to_seg_target).astype(np.int32)
    
    assert np.min(seg_source) == 0   # Background should be 0
    assert np.min(seg_target) == 0   # Background should be 0
    
    
    ### Run mapping procedure
    max_cores = multiprocessing.cpu_count()
    assert decomposition_cores <= max_cores     
    # target_shape = seg_target.shape[0] + 30, seg_target.shape[1] + 30, seg_target.shape[2] + 30
    # seg_target = padding(seg_target, target_shape)
    assert seg_source.shape == seg_target.shape # Images need to have the same shape, 
                                                # 15voxel padding is added to all sides 
                                                # from the registration. Using the padding function
                                                # in decomposeCells.py can help solve this by 
                                                # setting the final shape to shape+30 as seen above
            
    overlapDict = calculateOverlapFast(seg_source, seg_target) # Calculate the cell-to-cell overlaps
    decompDictSource = getInfoSphericalDecomposition(seg_source, data_source, 
                                                     angular_intervals, 
                                                     resolution_for_registered_source) # Could be improved for multi-threading

    mapped_data, all_props = recomposeMultiCore(seg_target, decompDictSource, overlapDict, 
                                               decompositionAngles=angular_intervals, 
                                               resolution=resolution_for_target, 
                                               nCores=decomposition_cores)


    tifffile.imsave(path_to_mapped_seg, seg_target.astype(np.uint16))  #
    tifffile.imsave(path_to_mapped_data, mapped_data.astype(np.uint16))  #



























