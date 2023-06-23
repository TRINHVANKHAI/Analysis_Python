import numpy as np
import matplotlib.pyplot as plt


plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 8)

#N=500, mean=0, sd = 1  ;  xinput = [x0,x1,x2,x3... , x499]
xinput = np.random.normal (0, 1, 500)

#N=500, mean=0, sd = 1  ;  yinput = [y0,y1,y2,y3... , y499]
yinput = np.random.normal (0, 1, 500)

#VSTACK   Z.shape() = 2, 500
#  [x0,x1,x2,x3... , x499]
#  [y0,y1,y2,y3... , y499]
#  X = Z.T


#X shape = 500, 2
#
# [x0, y0]
# [x1, y1]
# [x2, y2]
# [x3, y3]
# [..., ...]
# [x499, y499]
Zinput = np.vstack((xinput, yinput))
XDS = Zinput.T





def Covariance(inputx, inputy):
  # (1) Calculate mean of two random variables
  xbar, ybar = inputx.mean(), inputy.mean()
  N = len(inputx)
  # (2) Calculate SUM (x - xbar)*(y -ybar)
  return np.sum(  (inputx-xbar)*(inputy-ybar) ) / (N-1)





def printCovarianceMatrix(inputDataSet):
  x = inputDataSet[:,0]
  y = inputDataSet[:,1]

  cov__x_x = Covariance(x, x)
  cov__x_y = Covariance(x, y)
  cov__y_x = Covariance(y, x)
  cov__y_y = Covariance(y, y)
  cov__matrix = [[cov__x_x, cov__x_y], [cov__y_x, cov__y_y]]
  print(cov__matrix)


printCovarianceMatrix(XDS)


  
fig, ax = plt.subplots(1,1, figsize=(6,6))
ax.plot(xinput,yinput, 'bo')

ax.set_title('Scatter plot moi tuong quan giua x and y')
ax.set_xlabel('x')
ax.set_ylabel('y')

plt.tight_layout()
plt.show()

