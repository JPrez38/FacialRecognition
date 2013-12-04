import numpy as np
import scipy as sp
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.offsetbox as offsetbox

def visualize(scores, faces):
  """
  The function for visualization part, 
  which put the image at the coordinates given by their coefficients of 
  the first two principal components (with translation and scaling).

  scores: n x 2 array, where each row contains the first 2 principal component scores of each face
  faces: n x 4096 array
  """
  pc_min, pc_max = np.min(scores, 0), np.max(scores, 0)
  pc_scaled = (scores - pc_min) / (pc_max - pc_min)  
  fig, ax = plt.subplots()
  for i in range(len(faces)):
    imagebox = offsetbox.OffsetImage(faces[i, :].reshape(64,64).T, cmap=plt.cm.gray, zoom=0.5)
    box = offsetbox.AnnotationBbox(imagebox, pc_scaled[i, 0:2])
    ax.add_artist(box)
  plt.show()

# Example code starts from here
# Load the data set
faces = sp.genfromtxt('faces.csv', delimiter=',')

# Example for displaying the first face, which may help you how the data set presents
#plt.imshow(faces[0, :].reshape(64, 64).T, cmap=plt.cm.gray)
#plt.show()


# Your code starts from here ....
import time
import copy
# a. Randomly display a face
import random
randIndex = random.randint(0,len(faces))
plt.imshow(faces[randIndex,:].reshape(64,64).T, cmap=plt.cm.gray)
#plt.show()

# b. Compute and display the mean face
mean = [0.0] * len(faces[0])
for image in faces:
  for featureIndex,val in enumerate(image):
    mean[featureIndex] += float(val)

for ind,meanValue in enumerate(mean):
  mean[ind] = float(meanValue) / float(len(faces))

plt.imshow(np.matrix(mean).reshape(64,64).T, cmap=plt.cm.gray)
#plt.show()

# c. Centralize the faces by substracting the mean
newFaces = copy.deepcopy(faces)

for index,face in enumerate(newFaces):
  newFaces[index] = [faceVal - meanVal for faceVal,meanVal in zip(face,mean)]

# d. Perform SVD (you may find scipy.linalg.svd useful)
U, S, V = linalg.svd(newFaces)


# e. Show the first 10 priciple components
for i in range(0,10):
  plt.imshow(V[i,:].reshape(64,64).T, cmap=plt.cm.gray)
  #plt.show()

# f. Visualize the data by using first 2 principal components using the function "visualize"
scores = [[]] * 30
visFaces = np.matrix([[0] * 4096] * 30)

for i in range(30):
  ran = random.randint(0,len(faces)-1)
  visFaces[i, :] = faces[ran, :]
  score = np.dot(newFaces[ran],V[0:2,:].T)
  # score0 = np.dot(facesCentered[ran],Vh[0,:])
  # score1 = np.dot(facesCentered[ran],Vh[1,:])

  scores[i] = score

visualize(scores,visFaces)

# g. Plot the proportion of variance explained
# STUDENT CODE TODO

# h. Face reconstruction using 5, 10, 25, 50, 100, 200, 300, 399 principal components
# STUDENT CODE TODO

# i. Plot the reconstruction error for k = 5, 10, 25, 50, 100, 200, 300, 399 principal components
#    and the sum of the squares of the last n-k (singular values)
#    [extra credit]
# STUDENT CODE TODO
