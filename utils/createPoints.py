import matplotlib.pyplot as plt
import numpy as np
import tifffile
import pyvistaqt as pvq
import pyvista as pv
import time




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


class IndexTracker3:
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices  = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def on_scroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()




    def onclick(self, event):
        if event.xdata != None and event.ydata != None:
            xx, yy, = int(np.round(event.xdata)), int(np.round(event.ydata))
            # print(self.X[self.ind][yy][xx])
            print(self.ind,xx,yy)
            # img[img==img[self.ind][yy][xx]] = 0

class IndexTracker2:
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows,self.slices, cols  = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, self.ind, :])
        self.update()

    def on_scroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, self.ind, :])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()




    def onclick(self, event):
        if event.xdata != None and event.ydata != None:
            xx, yy, = int(np.round(event.xdata)), int(np.round(event.ydata))
            # print(self.X[self.ind][yy][xx])
            print(xx,self.ind,yy)
            # img[img==img[self.ind][yy][xx]] = 0


class IndexTracker:
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        self.slices, rows, cols  = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[self.ind, :, :])
        self.update()

    def on_scroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind, :, :])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()




    def onclick(self, event):
        if event.xdata != None and event.ydata != None:
            xx, yy, = int(np.round(event.xdata)), int(np.round(event.ydata))
            # print(self.X[self.ind][yy][xx])
            print(xx,yy,self.ind)
            # img[img==img[self.ind][yy][xx]] = 0
            
        
        
        
class IndexTracker3D:
    def __init__(self, ax1,ax2,ax3, X):
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3
        # ax1.set_title('use scroll wheel to navigate images')

        self.X1 = X
        self.slices1, rows1, cols1  = X.shape
        self.ind1 = self.slices1//2
        
        self.X2 = X
        rows2, self.slices2, cols2  = X.shape
        self.ind2 = self.slices2//2
        
        self.X3 = X
        rows3, cols3, self.slices3  = X.shape
        self.ind3 = self.slices3//2

        self.im1 = ax1.imshow(self.X1[self.ind1, :, :])
        self.im2 = ax2.imshow(self.X2[:,self.ind2, :])
        self.im3 = ax3.imshow(self.X3[:, :,self.ind3])
        self.update1()
        self.update2()
        self.update3()

    def on_scroll(self, event):
        
        # print("%s %s" % (event.button, event.step))
        if event.inaxes in [self.ax1]:
            if event.button == 'up':
                self.ind1 = (self.ind1 + 1) % self.slices1
            else:
                self.ind1 = (self.ind1 - 1) % self.slices1
            if abs(self.timer - time.time()) > 0.05:
                self.update1()
            
        elif event.inaxes in [self.ax2]:
            if event.button == 'up':
                self.ind2 = (self.ind2 + 1) % self.slices2
            else:
                self.ind2 = (self.ind2 - 1) % self.slices2
            if abs(self.timer - time.time()) > 0.05:
                self.update2()
            
        elif event.inaxes in [self.ax3]:
            if event.button == 'up':
                self.ind3 = (self.ind3 + 1) % self.slices3
            else:
                self.ind3 = (self.ind3 - 1) % self.slices3
            if abs(self.timer - time.time()) > 0.05:
                self.update3()

    def update1(self):
        self.im1.set_data(self.X1[self.ind1, :, :])
        self.ax1.set_ylabel('slice %s' % self.ind1)
        self.im1.axes.figure.canvas.draw()
        self.timer = time.time()
        
    def update2(self):
        self.im2.set_data(self.X2[:,self.ind2, :])
        self.ax2.set_ylabel('slice %s' % self.ind2)
        self.im2.axes.figure.canvas.draw()
        self.timer = time.time()
        
    def update3(self):
        self.im3.set_data(self.X3[:, :, self.ind3])
        self.ax3.set_ylabel('slice %s' % self.ind3)
        self.im3.axes.figure.canvas.draw()
        self.timer = time.time()




    def onclick(self, event):
        
        if event.inaxes in [self.ax1]:
        
            if event.xdata != None and event.ydata != None:
                xx, yy, = int(np.round(event.xdata)), int(np.round(event.ydata))
                # print(self.X[self.ind][yy][xx])
                print(xx,yy,self.ind1)
                # img[img==img[self.ind][yy][xx]] = 0
                
                self.ind3, self.ind2, self.ind1 = xx, yy, self.ind1
                self.update2()
                self.update3()  
            
        elif event.inaxes in [self.ax2]:
            
            if event.xdata != None and event.ydata != None:
                xx, yy, = int(np.round(event.xdata)), int(np.round(event.ydata))
                # print(self.X[self.ind][yy][xx])
                print(xx,self.ind2,yy)
                # img[img==img[self.ind][yy][xx]] = 0
                
                self.ind3, self.ind2, self.ind1 = xx,self.ind2,yy
                self.update1()
                self.update3()


        
        elif event.inaxes in [self.ax3]:
            
            if event.xdata != None and event.ydata != None:
                xx, yy, = int(np.round(event.xdata)), int(np.round(event.ydata))
                # print(self.X[self.ind][yy][xx])
                print(self.ind3,xx,yy)
                # img[img==img[self.ind][yy][xx]] = 0
                
                self.ind3, self.ind2, self.ind1 = self.ind3,xx,yy
        
                self.update1()
                self.update2()
        
        
        
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
            
    
    
    
    
    # img = tifffile.imread(allFixedList[17])
    img = tifffile.imread(r"/home/jonas/projects/allImages/mem/C2-210305-PIN1GFP-MMdsRed-WT_flowers-SAM6-FM2.tif")
    targetShape = img.shape[0] + 30, img.shape[1] + 30, img.shape[2] + 30
    img = padding(img, targetShape)
    img = img.astype(np.uint16)
        

    

    
    
    # ax = plt.gca()
    # fig = plt.gcf()
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    
    # implot = ax.imshow(img[sliceInd,:,:])
            
    # img[img>0] += 1000
    
    tracker = IndexTracker3D(ax1,ax2, ax3, img)
    
    
    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    fig.canvas.mpl_connect('button_press_event', tracker.onclick)

    
    plt.show()
    
    ### 132h
    # points = [[401, 209, 585], #7
    #           [361, 591, 248], #1
    #           [344, 409, 465], #4
    #           [345, 503, 274], #2
    #           [370, 239, 520], #6
    #           [176, 318, 331], #5
    #           [537, 286, 281]] #3
    
    
    ### 20_11
    # points = [[311, 516, 251], #1
    #           [326, 446, 228], #2
    #           [498, 324, 124], #3
    #           [330, 326, 291], #4
    #           [140, 264, 101], #5
    #           [327, 133, 162], #6
    #           [330, 110, 133]] #7
    
    ### 20_21
    # points = [[163, 208, 210],
    #           [193, 244, 167],
    #           [175, 468, 113],
    #           [362, 327, 268],
    #           [472, 228, 46],
    #           [446, 455, 178],
    #           [486, 478, 173]]
    
    ### 21_31
    points = [[411, 218, 161]]
    
    for pt in points:
        img[pt[2]-5:pt[2]+5, pt[1]-5:pt[1]+5, pt[0]-5:pt[0]+5] = 50000
    
    
    img = tifffile.imread(movingList[8])
    res = movingResList[8]
    
    p = pvq.BackgroundPlotter()
    p.add_volume(img, opacity='linear', resolution=res,shade=True,diffuse=1.5, specular=1.)
    # p.add_volume(img, opacity='linear', resolution=(1,1,1),shade=True,diffuse=1.5, specular=1.)
    p.show()



"""    from skimage import data
    from skimage import transform
    from skimage.feature import (match_descriptors, corner_harris,
                                 corner_peaks, ORB, plot_matches)
    from skimage.color import rgb2gray
    import matplotlib.pyplot as plt




    descriptor_extractor = ORB(n_keypoints=200)

    descriptor_extractor.detect_and_extract(img)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors
    
    descriptor_extractor.detect_and_extract(imgm)
    keypoints2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors"""

































