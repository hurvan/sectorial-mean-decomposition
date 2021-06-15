# sectorial-mean-decomposition
Repo for the method developed in my master thesis





### register.py

#### Dependencies

- SimpleElastix
- imgmisc
- skimage
- scipy
- tifffile

#### How to use

In the file you can edit the paths to images and also input their resolution. You also write the paths to where you want the registered images to be saved.
If you have points to use as landmarks they can be specified here as well, more info regarding the strucutre of the point files here: https://simpleelastix.readthedocs.io/PointBasedRegistration.html
Note: The registration adds 15 voxels of padding to each side of the image.

### smd.py

#### Dependencies

- imgmisc
- skimage
- scipy
- tifffile
- img2org
- mahotas

#### How to use

In the file you can edit the paths to images and also input their resolution. You also write the paths to where you want the registered images to be saved.
The segmentation data for the source and target image need to have their background as 0, and they need to have the same shape. A padding function (commented out in the code) can be used to add padding such that the shape matches.

If working with large images you might have to limit the number of cores used since the multi-threading consume quite a lot of memory.

If you find strange artifacts in the mapped data it might be due to the datatype of the saved images, which currently are saved as unsigned 16bit intergers, this can easily be modified. Example: mapped_data.astype(np.uint16) -> mapped_data.astype(np.int32)

### quantify.py

#### Dependencies

- imgmisc
- skimage
- tifffile
- img2org
- mahotas
#### How to use

In the file you can edit the paths to images and also input their resolution. You also write the paths to where you want the registered images to be saved.
The segmentation data for the source image need to have the background as 0, and the segmentation and data image need to have the same shape. A padding function (commented out in the code) can be used to add padding such that the shape matches.

The multi-threading splits the images in 200x200x200 chunks centered around the cell that is being quantified, if the cells you are working with are close to that size, the chunk size should be increased (will consume more memory). The chunk size is hard-coded in "multiThreadQuantify.py" in the "batch_compute" function.

The quantification script will save the data in a format that is to be used for the Organism software (https://gitlab.com/slcu/teamHJ/Organism)




