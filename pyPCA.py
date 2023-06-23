import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import math


Vdirect = np.array([1, 1, 1]) # Target direction
Xu      = np.array([1, 0, 0]) # X axis unit vector

theta   = math.acos (np.dot(Xu, Vdirect)/(np.linalg.norm(Vdirect)*np.linalg.norm(Xu)))
print("Theta=\n", theta)
Uaxis   = np.cross(Vdirect, Xu)/(np.sin(theta)*np.linalg.norm(Vdirect)*np.linalg.norm(Xu))
print("Uaxis=\n", Uaxis)


Identify = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])
                     
UcrossMatrix = np.array([[0, -1*Uaxis[2], Uaxis[1]],
                        [Uaxis[2], 0, -1*Uaxis[0]],
                        [-1*Uaxis[1], Uaxis[0], 0]])
                        
print("Ucross=\n", UcrossMatrix)
UouterMatrix = np.outer(Uaxis, Uaxis)
print("Uouter=\n", UouterMatrix)

RotAboutUaxisMatrix = math.cos(theta)*Identify + math.sin(theta)*UcrossMatrix + (1-math.cos(theta))*UouterMatrix
print ("Rotation matrix=\n",RotAboutUaxisMatrix)



#RGBVector = np.array([[1, 2, 4], [2, 3, 9], [3, 4, 11.2], [4, 5, 15], [5, 6, 22]])
RGBVector = np.array([[1.1, 1.2, 0.9], [2, 2, 2], [3, 3.4, 3.3], [4, 4, 4], [5, 5, 5]])
RVector = RGBVector[:,0]
GVector = RGBVector[:,1]
BVector = RGBVector[:,2]



pca = PCA(n_components=3)

principalComponents = pca.fit_transform(RGBVector)
print(principalComponents)
print(pca.mean_)
print(pca.components_)

transformMatrix = pca.components_.T
transformMatrix = transformMatrix.dot(RotAboutUaxisMatrix)
ProjectedRGB = RGBVector.dot(transformMatrix)

ProjectedR = ProjectedRGB[:,0]
ProjectedG = ProjectedRGB[:,1]
ProjectedB = ProjectedRGB[:,2]
print(ProjectedRGB)

#Plot projected RGB space
fig = plt.figure()
ax = plt.gca(projection='3d')

ax.set_xlabel('R', fontsize = 15)
ax.set_ylabel('G', fontsize = 15)
ax.set_zlabel('B', fontsize = 15)
ax.quiver(pca.mean_[0], pca.mean_[1], pca.mean_[2], pca.mean_[0]+pca.components_[0,0], pca.mean_[1]+pca.components_[0,1], pca.mean_[2]+pca.components_[0,2], color='red', zorder=2)
ax.plot(RVector, GVector, BVector, 'bo', zorder=1)
ax.plot(ProjectedR, ProjectedG, ProjectedB, 'go', zorder=1)
ax.plot(principalComponents[:,0], principalComponents[:,1], principalComponents[:,2], 'yo', zorder=1)

plt.show()




