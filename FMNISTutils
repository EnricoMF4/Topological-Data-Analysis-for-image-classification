import gudhi.representations
from gtda.images import HeightFiltration
from gtda.images import RadialFiltration
from gtda.images import DensityFiltration

import pickle
from skimage.feature import canny
from skimage import filters
from skimage import morphology
from scipy.ndimage import distance_transform_bf
from gudhi.tensorflow import LowerStarSimplexTreeLayer
import tensorflow as tf
import gudhi as gd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import fashion_mnist
from sklearn.preprocessing import MinMaxScaler
from gudhi.representations import (DiagramSelector, Clamping, Landscape, Silhouette, BettiCurve, ComplexPolynomial,\
  TopologicalVector, DiagramScaler, BirthPersistenceTransform,\
  PersistenceImage, PersistenceWeightedGaussianKernel, Entropy, \
  PersistenceScaleSpaceKernel, SlicedWassersteinDistance,\
  SlicedWassersteinKernel, PersistenceFisherKernel, WassersteinDistance)
from sklearn.ensemble import RandomForestClassifier
from gudhi import CubicalComplex
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import scipy
from scipy import ndimage
import PIL
from persim import plot_diagrams
from sklearn.preprocessing import StandardScaler



# FUNCTIONS

indicator_fraud = False   # indicator if fraudenthial complex has been computed yet or not
fraudenthial = 0    # variable that will contain the fraudenthial complex

# compute fraudenthial complex

def fraud(n):

    st = gd.SimplexTree()

    for i in range(0,n-1):
        for j in range(0,n-1):
            st.insert([j*n+i,j*n+i+1,(j+1)*n+i])
            st.insert([j*n+i+1,(j+1)*n+i,(j+1)*n+i+1])

    return(st)


indicator_neigh = False   # indicator if neighbor complex has been computed yet or not
neighbor = 0    # variable that will contain the fraudenthial complex

def neigh(n):

    st = gd.SimplexTree()

    for i in range(0,n-1):
        for j in range(0,n-1):
            st.insert([j*n+i,j*n+i+1,(j+1)*n+i])
            st.insert([j*n+i+1,(j+1)*n+i,(j+1)*n+i+1])
            st.insert([j*n+i,j*n+i+1,(j+1)*n+i+1])
            st.insert([j*n+i,(j+1)*n+i,(j+1)*n+i+1])

    return(st)

# input: image 2D of nxn pixels, written as a numpy array,
# groups of homology for which we want the persistence diagram
# output: list with the numpay arrays of the points of the
# persistence diagram for the different homologies groups

def lower_star_persistence(imm,filtration_values,homology_dimensions=[0,1]):

  persistence_values=[]
  n=np.shape(imm)[0]

  global indicator_neigh
  global neighbor

  # compute fraudenthial complex

  st = gd.SimplexTree()
  if indicator_neigh == False:
      neighbor = neigh(n)
      indicator_neigh = True

  # create the vector of the function values

  filtration_values=np.reshape(filtration_values,(1,n*n))
  F = tf.Variable(filtration_values.tolist()[0], dtype=tf.float32, trainable=True)

  sl = LowerStarSimplexTreeLayer(simplextree=neighbor, homology_dimensions=homology_dimensions)

  for i in homology_dimensions:
    persistence_values.append((sl.call(F))[i][0].numpy())

  # insert the always present 0-class

  persistence_values[0]= np.vstack([persistence_values[0], np.array([[np.min(filtration_values), np.max(filtration_values)]], dtype=np.float32)])

  return(persistence_values)



def pow(n):
  return lambda x: np.power(x[1]-x[0],n)


# input: persistence interval, resolution of vector
# output: vectorized silhouette

def vect(persistence_values,n,m=1):

  if persistence_values.size == 0 :
    #persistence_values = np.array([[0.,0.01]]) ####
    return(np.zeros(n))
    #return(0.9*np.ones(n))

  proc1 = DiagramSelector(use=True, point_type="finite")
  proc2 = DiagramScaler(use=True, scalers=[([0,1], MinMaxScaler())])
  proc3 = DiagramScaler(use=True, scalers=[([1], Clamping(maximum=1))])
  DD = proc3(proc2(proc1(persistence_values)))

  return(np.array(Silhouette(resolution=n, weight=pow(m))(DD)))



# it take as input a list per of couples (as lists) of persistence values

def betticurve(per, finesse):

   min_per = np.min(np.array([per[i][0] for i in range(len(per))]))
   max_per = np.max(np.array([per[i][1] for i in range(len(per))]))

   bc = BettiCurve(predefined_grid=np.linspace(min_per, max_per, finesse))
   return(bc(per))



# function to plot the PCA of some data taken as numpay vectors all inside a list

def pca_fun(data, label, n_components=2):

   pca = PCA(n_components=2)
   principal_components = pca.fit_transform(data)

   principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
   principal_df['Label'] = label

   plt.figure(figsize=(8, 6))

   classes = np.unique(label)
   colors = ['r', 'g', 'b', 'c', 'm', 'y','k','beige','bisque']

   for class_label, color in zip(classes, colors):
       subset = principal_df[principal_df['Label'] == class_label]
       plt.scatter(subset['PC1'], subset['PC2'], label=class_label, color=color)

   plt.xlabel('Principal Component 1')
   plt.ylabel('Principal Component 2')
   plt.title('PCA plot')
   plt.grid()
   plt.show()



# function to perform a RF on the data splitted by training and test set

def classifier_rf(data, label, n_train, n_trees):

  # n_train = number of training data per class we use (we use the reimanders for test)
  if n_train>n_data: n_train = n_data-10

  data_train = data[0:n_train]
  label_train = label[0:n_train]

  data_test = data[n_train:n_data]
  label_test = label[n_train:n_data]

  # Then I build a random forest of n_trees trees

  forest=RandomForestClassifier(n_estimators=n_trees,
                              random_state=0)

  # that I train on train data:
  forest.fit(data_train, label_train)

  # I use such alg. to predict label for test data
  y_pred = forest.predict(data_test)

  accuracy = accuracy_score(label_test, y_pred)
  print("Accuracy:", accuracy)

  # plot the confusion matrix

  cm = confusion_matrix(label_test, y_pred)
  ConfusionMatrixDisplay(confusion_matrix=cm).plot();
