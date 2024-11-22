import gudhi.representations

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from time import time
from scipy import sparse
from scipy.sparse.linalg import lsqr, cg, eigsh
import argparse
import trimesh
from IPython.display import display
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
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
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



# (It takes a mesh of triangles ([0,1,2],[2,5,6],...) made on n points
# and filtration_values that is a nxn numpay matrix that for each point
# of the mesh assignes a value.
# It returns the lower star filtration made on the complex given by the mesh)

# It takes as input the total complex on which I want to compute le lower-star filtration,
# given as a list or numpay array: it must contain the "significative" simpleces of such complex.

# filtration_values is a list or numpay array of length n, with n = number of
# 0-simpleces of the total complex, such that filtration_values[i] = value of a given
# function (filtration) on the i-th vertex of the total complex.

# It returns a list representing the persistence values for the lower-star filtration
# computed o such total complex with such filtration_values

def lower_star_persistence(complex, filtration_values):

  #max_value_filtration = np.max(np.array(filtration_values))

  # n is the number of 0-simpleces of the complex
  n = np.shape(filtration_values)[0]

  # create the lower_star filtration for the complex:

  st = gudhi.SimplexTree()
  # insert the significative simpleces of the complex
  for simplex in complex:

    st.insert(simplex)
  # and assign to each 0-simplex the relative lower-star filtration value
  for i in range(n):

    st.assign_filtration([i], float(filtration_values[i]))
  # compute the value of the induced lower-star filtration for each simplex of the complex
  _ = st.make_filtration_non_decreasing()

  # => we have the lower-star filtration, on which we compute persistence

  per = st.persistence()

  return per #persistence_pairs




def pow(n):
  return lambda x: np.power(x[1]-x[0],n)


# input: persistence interval, resolution of vector
# output: vectorized silhouette

def vect1(persistence_diagram,n,max_value_filtration,m=1):

  # Converti i dati di persistenza in un formato compatibile con GUDHI
  persistence_pairs = []
  for dim, (birth, death) in persistence_diagram:
      if death == float('inf'):
          death = float(max_value_filtration)  # Sostituisci inf con un valore molto grande
      persistence_pairs.append([birth, death])

  # Separare i diagrammi di persistenza per dimensione
  persistence_0d = np.array([pair for dim, pair in zip([dim for dim, _ in persistence_diagram], persistence_pairs) if dim == 0])
  persistence_1d = np.array([pair for dim, pair in zip([dim for dim, _ in persistence_diagram], persistence_pairs) if dim == 1])

  # Calcola la silhouette per i diagrammi di persistenza 0-dimensionali
  silhouette = gd.representations.Silhouette(resolution=n, weight=pow(m))
  silhouette_0d = silhouette.fit_transform([persistence_0d])

  # Calcola la silhouette per i diagrammi di persistenza 0-dimensionali
  silhouette = gd.representations.Silhouette(resolution=n, weight=pow(m))
  silhouette_1d = silhouette.fit_transform([persistence_1d])

  return silhouette_0d, silhouette_1d


# function to plot the PCA of some data taken as numpay vectors all inside a list

def pca_fun(data, label, n_components=2):

   pca = PCA(n_components=2)
   principal_components = pca.fit_transform(data)

   principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
   principal_df['Label'] = label

   plt.figure(figsize=(8, 6))

   classes = np.unique(label)
   colors = ['b','r','aquamarine', 'g','k','gray','lightblue','m','yellow','orange','brown','lightgreen','#301934', 'c']#'bisque']#'beige'

   for class_label, color in zip(classes, colors):
       subset = principal_df[principal_df['Label'] == class_label]
       plt.scatter(subset['PC1'], subset['PC2'], label=class_label, color=color)

   plt.xlabel('Principal Component 1')
   plt.ylabel('Principal Component 2')
   plt.title('PCA plot')
   plt.grid()
   plt.show()





# HEAT KERNEL SIGNATURE


def get_cotan_laplacian(VPos, ITris, anchorsIdx = [], anchorWeights = 1):
    """
    Quickly compute sparse Laplacian matrix with cotangent weights and Voronoi areas
    by doing many operations in parallel using NumPy

    Parameters
    ----------
    VPos : ndarray (N, 3)
        Array of vertex positions
    ITris : ndarray (M, 3)
        Array of triangle indices
    anchorsIdx : list
        A list of vertex indices corresponding to the anchor vertices
        (for use in Laplacian mesh editing; by default none)
    anchorWeights : float


    Returns
    -------
    L : scipy.sparse (NVertices+anchors, NVertices+anchors)
        A sparse Laplacian matrix with cotangent weights
    """
    N = VPos.shape[0]
    M = ITris.shape[0]
    #Allocate space for the sparse array storage, with 2 entries for every
    #edge for eves ry triangle (6 entries per triangle); one entry for directed
    #edge ij and ji.  Note that this means that edges with two incident triangles
    #will have two entries per directed edge, but sparse array will sum them
    I = np.zeros(M*6)
    J = np.zeros(M*6)
    V = np.zeros(M*6)

    #Keep track of areas of incident triangles and the number of incident triangles
    IA = np.zeros(M*3)
    VA = np.zeros(M*3) #Incident areas
    VC = 1.0*np.ones(M*3) #Number of incident triangles

    #Step 1: Compute cotangent weights
    for shift in range(3):
        #For all 3 shifts of the roles of triangle vertices
        #to compute different cotangent weights
        [i, j, k] = [shift, (shift+1)%3, (shift+2)%3]
        dV1 = VPos[ITris[:, i], :] - VPos[ITris[:, k], :]
        dV2 = VPos[ITris[:, j], :] - VPos[ITris[:, k], :]
        Normal = np.cross(dV1, dV2)
        #Cotangent is dot product / mag cross product
        NMag = np.sqrt(np.sum(Normal**2, 1))
        cotAlpha = np.sum(dV1*dV2, 1)/NMag
        I[shift*M*2:shift*M*2+M] = ITris[:, i]
        J[shift*M*2:shift*M*2+M] = ITris[:, j]
        V[shift*M*2:shift*M*2+M] = cotAlpha
        I[shift*M*2+M:shift*M*2+2*M] = ITris[:, j]
        J[shift*M*2+M:shift*M*2+2*M] = ITris[:, i]
        V[shift*M*2+M:shift*M*2+2*M] = cotAlpha
        if shift == 0:
            #Compute contribution of this triangle to each of the vertices
            for k in range(3):
                IA[k*M:(k+1)*M] = ITris[:, k]
                VA[k*M:(k+1)*M] = 0.5*NMag

    #Step 2: Create laplacian matrix
    L = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    #Create the diagonal by summing the rows and subtracting off the nondiagonal entries
    L = sparse.dia_matrix((L.sum(1).flatten(), 0), L.shape) - L
    #Scale each row by the incident areas TODO: Fix this
    """
    Areas = sparse.coo_matrix((VA, (IA, IA)), shape=(N, N)).tocsr()
    Areas = Areas.todia().data.flatten()
    Areas[Areas == 0] = 1
    Counts = sparse.coo_matrix((VC, (IA, IA)), shape=(N, N)).tocsr()
    Counts = Counts.todia().data.flatten()
    RowScale = sparse.dia_matrix((3*Counts/Areas, 0), L.shape)
    L = L.T.dot(RowScale).T
    """

    #Step 3: Add anchors
    L = L.tocoo()
    I = L.row.tolist()
    J = L.col.tolist()
    V = L.data.tolist()
    I = I + list(range(N, N+len(anchorsIdx)))
    J = J + anchorsIdx
    V = V + [anchorWeights]*len(anchorsIdx)
    L = sparse.coo_matrix((V, (I, J)), shape=(N+len(anchorsIdx), N)).tocsr()
    return L

def get_umbrella_laplacian(VPos, ITris, anchorsIdx = [], anchorWeights = 1):
    """
    Quickly compute sparse Laplacian matrix with "umbrella weights" (unweighted)
    by doing many operations in parallel using NumPy

    Parameters
    ----------
    VPos : ndarray (N, 3)
        Array of vertex positions
    ITris : ndarray (M, 3)
        Array of triangle indices
    anchorsIdx : list
        A list of vertex indices corresponding to the anchor vertices
        (for use in Laplacian mesh editing; by default none)
    anchorWeights : float


    Returns
    -------
    L : scipy.sparse (NVertices+anchors, NVertices+anchors)
        A sparse Laplacian matrix with umbrella weights
    """
    N = VPos.shape[0]
    M = ITris.shape[0]
    I = np.zeros(M*6)
    J = np.zeros(M*6)
    V = np.ones(M*6)

    #Step 1: Set up umbrella entries
    for shift in range(3):
        #For all 3 shifts of the roles of triangle vertices
        #to compute different cotangent weights
        [i, j, k] = [shift, (shift+1)%3, (shift+2)%3]
        I[shift*M*2:shift*M*2+M] = ITris[:, i]
        J[shift*M*2:shift*M*2+M] = ITris[:, j]
        I[shift*M*2+M:shift*M*2+2*M] = ITris[:, j]
        J[shift*M*2+M:shift*M*2+2*M] = ITris[:, i]

    #Step 2: Create laplacian matrix
    L = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    L[L > 0] = 1
    #Create the diagonal by summing the rows and subtracting off the nondiagonal entries
    L = sparse.dia_matrix((L.sum(1).flatten(), 0), L.shape) - L

    #Step 3: Add anchors
    L = L.tocoo()
    I = L.row.tolist()
    J = L.col.tolist()
    V = L.data.tolist()
    I = I + list(range(N, N+len(anchorsIdx)))
    J = J + anchorsIdx
    V = V + [anchorWeights]*len(anchorsIdx)
    L = sparse.coo_matrix((V, (I, J)), shape=(N+len(anchorsIdx), N)).tocsr()
    return L




def get_laplacian_spectrum(VPos, ITris, K):
    """
    Given a mesh, to compute first K eigenvectors of its Laplacian
    and the corresponding eigenvalues
    Parameters
    ----------
    VPos : ndarray (N, 3)
        Array of points in 3D
    ITris : ndarray (M, 3)
        Array of triangles connecting points, pointing to vertex indices
    K : int
        Number of eigenvectors to compute
    Returns
    -------
    (eigvalues, eigvectors): a tuple of the eigenvalues and eigenvectors
    """
    L = get_cotan_laplacian(VPos, ITris)
    (eigvalues, eigvectors) = eigsh(L, K, which='LM', sigma = 0)
    return (eigvalues, eigvectors)


def get_heat(eigvalues, eigvectors, t, initialVertices, heatValue = 100.0):
    """
    Simulate heat flow by projecting initial conditions
    onto the eigenvectors of the Laplacian matrix, and then sum up the heat
    flow of each eigenvector after it's decayed after an amount of time t
    Parameters
    ----------
    eigvalues : ndarray (K)
        Eigenvalues of the laplacian
    eigvectors : ndarray (N, K)
        An NxK matrix of corresponding laplacian eigenvectors
        Number of eigenvectors to compute
    t : float
        The time to simulate heat flow
    initialVertices : ndarray (L)
        indices of the verticies that have an initial amount of heat
    heatValue : float
        The value to put at each of the initial vertices at the beginning of time

    Returns
    -------
    heat : ndarray (N) holding heat values at each vertex on the mesh
    """
    N = eigvectors.shape[0]
    I = np.zeros(N)
    I[initialVertices] = heatValue
    coeffs = I[None, :].dot(eigvectors)
    coeffs = coeffs.flatten()
    coeffs = coeffs*np.exp(-eigvalues*t)
    heat = eigvectors.dot(coeffs[:, None])
    return heat

def get_hks(VPos, ITris, K, ts):
    """
    Given a triangle mesh, approximate its curvature at some measurement scale
    by recording the amount of heat that remains at each vertex after a unit impulse
    of heat is applied.  This is called the "Heat Kernel Signature" (HKS)

    Parameters
    ----------
    VPos : ndarray (N, 3)
        Array of points in 3D
    ITris : ndarray (M, 3)
        Array of triangles connecting points, pointing to vertex indices
    K : int
        Number of eigenvalues/eigenvectors to use
    ts : ndarray (T)
        The time scales at which to compute the HKS

    Returns
    -------
    hks : ndarray (N, T)
        A array of the heat kernel signatures at each of N points
        at T time intervals
    """
    L = get_cotan_laplacian(VPos, ITris)
    (eigvalues, eigvectors) = eigsh(L, K, which='LM', sigma = 0)
    res = (eigvectors[:, :, None]**2)*np.exp(-eigvalues[None, :, None]*ts.flatten()[None, None, :])
    return np.sum(res, 1)





