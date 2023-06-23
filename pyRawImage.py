import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

width = 1920
height = 1080

path = r'C:\Users\khai\PYWORKSPACE\IMAGE_CALIB'
image_name = 'test.jpeg'
image_shape = (height, width)


pca = PCA(n_components=3)


if not os.path.isdir(path):
    print("No such a directory: {}".format(path))
    exit(1)

with open("cpt_30_test_pattern.raw", "rb") as rawimg:
    img = np.fromfile(rawimg, dtype='>u2', count=width * height)
    img = img.reshape(image_shape)
    img = img.astype('u2')
    colimg = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB)

    RVector = colimg[:,:,0].flatten()
    GVector = colimg[:,:,1].flatten()
    BVector = colimg[:,:,2].flatten()
    
    
    RVector = RVector[0:40]
    GVector = GVector[0:40]
    BVector = BVector[0:40]
    RGBVector = np.vstack((RVector,GVector,BVector)).T

    #x = StandardScaler().fit_transform(RGBVector)
    #principalComponents = pca.fit_transform(x)
    
    principalComponents = pca.fit_transform(RGBVector)
    print(principalComponents)
    print(pca.mean_)
    print(pca.components_)
    ProjectedRGB = RGBVector.dot(pca.components_.T)
    print(ProjectedRGB-principalComponents)
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




