import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import math

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

width = 1920
height = 1080
GroundTruth=np.array([60,102,48])

path = r'C:\Users\khai\PYWORKSPACE\IMAGE_CALIB'
image_name = 'cpt_60_floor'
image_shape = (height, width)


pca = PCA(n_components=2)

def rotate_to_illumination(Illumination):
    Vdirect = np.array([1, 1, 1]) # Target direction
    Xu      = np.array([1, 0, 0]) # X axis unit vector #GroundTruth if not PCA

    theta   = math.acos (np.dot(Illumination, Vdirect)/(np.linalg.norm(Vdirect)*np.linalg.norm(Illumination)))
    Uaxis   = np.cross(Vdirect, Illumination)/(np.sin(theta)*np.linalg.norm(Vdirect)*np.linalg.norm(Illumination))

    Identify = np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])
                         
    UcrossMatrix = np.array([[0, -1*Uaxis[2], Uaxis[1]],
                            [Uaxis[2], 0, -1*Uaxis[0]],
                            [-1*Uaxis[1], Uaxis[0], 0]])
                            
    UouterMatrix = np.outer(Uaxis, Uaxis)

    RotAboutUaxisMatrix = math.cos(theta)*Identify + math.sin(theta)*UcrossMatrix + (1-math.cos(theta))*UouterMatrix
    return RotAboutUaxisMatrix


if not os.path.isdir(path):
    print("No such a directory: {}".format(path))
    exit(1)

with open(image_name+'.raw', "rb") as rawimg:
    img = np.fromfile(rawimg, dtype='>u2', count=width * height)
    img = img.reshape(image_shape)
    img = img.astype('u2')
    colimg = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB)

    RVector = colimg[:,:,0].flatten() - 120
    GVector = colimg[:,:,1].flatten() - 120
    BVector = colimg[:,:,2].flatten() - 120
    
    RGBVector = np.vstack((RVector,GVector,BVector)).T
    RGBUnsaturated = RGBVector#np.delete(RGBVector, np.where(RGBVector>4095)[0], axis=0)

    #Preprocessing for PCA 
    RGBAverageVector = np.mean(RGBUnsaturated, axis=0)
    distanceFromZero = np.dot(RGBUnsaturated, RGBAverageVector)/(np.linalg.norm(RGBAverageVector)*np.linalg.norm(RGBUnsaturated, axis=1))
    distanceFromZeroSorted = np.sort(distanceFromZero)
    rangeDistance=distanceFromZeroSorted.shape[0]*8/100
    rangeDistance = int(rangeDistance)
    maxLowRangeDistance = np.max(distanceFromZeroSorted[0:rangeDistance])
    minHighRangeDistance = np.min(distanceFromZeroSorted[distanceFromZeroSorted.shape[0]-rangeDistance: distanceFromZeroSorted.shape[0]])

    FilteredRGBVector = np.delete(RGBUnsaturated, np.where((distanceFromZero>maxLowRangeDistance) &(distanceFromZero<minHighRangeDistance)), axis=0)
    print(distanceFromZeroSorted.shape[0], distanceFromZeroSorted)

    #x = StandardScaler().fit_transform(FilteredRGBVector)
    principalComponents = pca.fit_transform(FilteredRGBVector)
    estimatedIlluDirect = pca.components_[0]
    print("PCA mean:\n", pca.mean_)
    print("PCA eigenvector \n", pca.components_)
    print("Estimated illumination :\n", estimatedIlluDirect)
    rotMatToIllumination = rotate_to_illumination(estimatedIlluDirect)
    ProjectedRGB = np.dot(RGBVector, rotMatToIllumination)
    ProjectedRGB = np.clip(ProjectedRGB, 0, 4095) #Saturated value clip

    ProjectedR = ProjectedRGB[:,0]
    ProjectedG = ProjectedRGB[:,1]
    ProjectedB = ProjectedRGB[:,2]

"""
#PLOT Original RGB space
fig = plt.figure()
ax = plt.gca(projection='3d')
ax.plot(RVector, GVector, BVector, 'bo', zorder=1)
ax.quiver(pca.mean_[0], pca.mean_[1], pca.mean_[2], pca.components_[0,0]*2048,pca.components_[0,1]*2048, pca.components_[0,2]*2048, color='red', zorder=2)
ax.quiver(pca.mean_[0], pca.mean_[1], pca.mean_[2], pca.components_[1,0]*2048,pca.components_[1,1]*2048, pca.components_[1,2]*2048, color='green', zorder=2)
ax.quiver(pca.mean_[0], pca.mean_[1], pca.mean_[2], pca.components_[2,0]*2048,pca.components_[2,1]*2048, pca.components_[2,2]*2048, color='yellow', zorder=2)

plt.show()

"""



"""
#Plot 2 principal component
fig, ax = plt.subplots(1,1, figsize=(6,6))
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
ax.plot(principalComponents[:,0], principalComponents[:,1],'bo', zorder=1)
pca_space_mean = np.mean(principalComponents, axis=0)
ax.quiver(pca_space_mean[0], pca_space_mean[1], 4096, pca_space_mean[1], color='red', zorder=2)
ax.quiver(pca_space_mean[0], pca_space_mean[1], pca_space_mean[1], 4096, color='green', zorder=2)
plt.show()
"""

"""
#Plot projected RGB space
fig = plt.figure()
ax = plt.gca(projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)

ax.plot(RVector, GVector, BVector, 'ro', zorder=1)
ax.plot(ProjectedR, ProjectedG, ProjectedB, 'go', zorder=1)
#ax.plot(principalComponents[:,0], principalComponents[:,1], principalComponents[:,2], 'bo', zorder=1)

plt.show()

"""
ProcessedImageR = np.reshape(ProjectedR, (-1, width))
ProcessedImageG = np.reshape(ProjectedG, (-1, width))
ProcessedImageB = np.reshape(ProjectedB, (-1, width))
RestoreImage = np.dstack([ProcessedImageR, ProcessedImageG, ProcessedImageB])
RestoreImage = RestoreImage.astype('u2')
RestoreImage = RestoreImage>>4
print (RestoreImage.shape)
CorrectedImage = (RestoreImage).astype('u1')
CorrectedImage = adjust_gamma(CorrectedImage, 2.2)
cv2.imwrite(image_name+"_output_pca.jpg", CorrectedImage)
#cv2.imshow("Corrected image", CorrectedImage)
#cv2.waitKey(0)


