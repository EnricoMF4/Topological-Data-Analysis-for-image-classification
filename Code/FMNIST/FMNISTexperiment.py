from FMNISTutils import *


# EXTRACT DATA

(train_X, train_y), (test_X, test_y) = fashion_mnist.load_data()

# we use only n images
n_data = 500
n_label = 10
data = train_X[0:n_data]
label = train_y[0:n_data]
name_label = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Visualize the data
plt.figure(figsize=(6, 6))

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(data[i], cmap='gray')
    plt.xlabel(name_label[label[i]])

plt.show()


# 2) CONVERT INTO BINARY IMAGES WELL


# We apply padding + median filter + binarization to our images
padding = lambda ima : np.pad(ima, ((2,2), (2,2)), 'constant', constant_values=0)
sq=morphology.rectangle(3, 3, dtype='uint8')
median = lambda ima : filters.median(ima, sq)
binarization = lambda ima : 255*(ima>5)
inverter = lambda ima : np.max(np.float32(ima))-ima
form_pipeline = lambda ima : (inverter(binarization(median(padding(ima)))))

data_form = np.array(list(map(form_pipeline, data)))

# Visualize the well binarizied images

plt.figure(figsize=(6, 6))

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(data_form[i])
    plt.xlabel(name_label[label[i]])

plt.show()


# 3) CONVERT INTO EDGES BINARIZED IMAGES


edger = lambda ima : canny(image=ima, low_threshold=20, high_threshold=100)
filt_taxi = lambda ima : distance_transform_bf(ima, metric='taxicab')
edge_pipeline = lambda ima : inverter(edger(binarization(median(padding(ima)))))

data_edge = np.array(list(map(edge_pipeline, data)))

# Visualize the edges binarizied images

plt.figure(figsize=(6, 6))

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(data_edge[i])
    plt.xlabel(name_label[label[i]])

plt.show()


# APPROFONDIMENTO: PROCESS OF BINARIZATION/EDGE


selected_images=[None]*6
selected_images[0]=data[0]
selected_images[1]=padding(data[0])
selected_images[2]=median(padding(data[0]))
selected_images[3]=binarization(median(padding(data[0])))
selected_images[4]=edger(binarization(median(padding(data[0]))))
selected_images[5]=inverter(edger(binarization(median(padding(data[0])))))

plt.figure(figsize=(10, 5))

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(selected_images[i], cmap='gray')

plt.show()


# COMPUTE PERSISTENT HOMOLOGY

# Using:


# 1) HEIGHT FILTRATION

# Using Lower star filtration induced by the height filtration

directions = [(1,0),(0,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,1),(-1,-1)]
n_dir = len(directions)

# Compute the 8 persistence diagrams for the image (one for direction)

#data_per = [[None]*n_dir]*n_data   <--- ERROREEEE
data_per_height = [[None for _ in range(n_dir)] for _ in range(n_data)]

for j, dir in enumerate(directions):

  height_filtration = HeightFiltration(direction=np.array(dir))
  filtrations_values = height_filtration.fit_transform(1-data_edge)#(bin_data)

  for i in range(n_data):

    data_per_height[i][j] = lower_star_persistence(data_edge[i],filtrations_values[i],homology_dimensions=[0,1])


# VECTORIZE THE PERSISTENCE DIAGRAMS


# Using the Silhouettes:

data_vec_height = [None]*n_data

n_finesse = 19

for j in range(n_dir):

    for i in range(n_data):

      v0 = (data_per_height[i][j])[0]
      v1 = (data_per_height[i][j])[1]
      v = np.concatenate((vect(v0,n_finesse),vect(v1,n_finesse)))
      #v = vect(v0, n_finesse)
      data_vec_height[i] = np.append(data_vec_height[i], v)

for i in range(n_data):     # si può fare meglioooo
  data_vec_height[i] = data_vec_height[i][1:]


# Let's see if the Silhouette are different by classes

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for i in range(n_label-1):
    ax = axes[i // 3, i % 3]

    for j in [p for p,x in enumerate(label) if x == i]:
        ax.plot(data_vec_height[j])

    ax.set_title("Silhouette of " + str(name_label[i]) + " images")

plt.tight_layout()
plt.show()


# 2) RADIAL FILTRATION


# Using Lower star filtration induced by the radial filtration

dim = np.shape(data_edge[0])[0]

centers = [(0,0),(0,dim),(dim,0),(dim,dim),(int(dim/2),int(dim/2)),(int(dim/2),0),(0,int(dim/2)),(int(dim/2),dim),(dim,int(dim/2))]
n_cen = len(centers)

# Compute the 4 persistence diagrams for the image (one for center)

data_per_radial = [[None for _ in range(n_cen)] for _ in range(n_data)]

for j, cen in enumerate(centers):

  radial_filtration = RadialFiltration(center=np.array(cen))
  filtrations_values = radial_filtration.fit_transform(1-data_edge)#(bin_data)

  for i in range(n_data):

    data_per_radial[i][j] = lower_star_persistence(data_edge[i],filtrations_values[i],homology_dimensions=[0,1])#(bin_data[i],filtrations_values[i],homology_dimensions=[0,1])


# VECTORIZE THE PERSISTENCE DIAGRAMS


# NB Since the unique holes that appear are the final ones inside the filtration,
# we don't consider the 1-homology classes, since the final holes are all detected
# by using previous method

# Using the Silhouettes:

data_vec_radial = [None]*n_data

n_finesse = 19

for j in range(n_cen):

    for i in range(n_data):

      v0 = (data_per_radial[i][j])[0]
      #v1 = (data_per_radial[i][j])[1]
      #v = np.concatenate((vect(v0,n_finesse),vect(v1,n_finesse)))
      v = vect(v0,n_finesse)
      data_vec_radial[i] = np.append(data_vec_radial[i], v)

for i in range(n_data):     # si può fare meglioooo
  data_vec_radial[i] = data_vec_radial[i][1:]


# Let's see if the Silhouette are different by classes

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for i in range(n_label-1):
    ax = axes[i // 3, i % 3]

    for j in [p for p,x in enumerate(label) if x == i]:
        ax.plot(data_vec_radial[j])

    ax.set_title("Silhouette of " + str(name_label[i]) + " images")

plt.tight_layout()
plt.show()


# 3) DENSITY FILTRATION


# NB it is lower star filtration for the density filtration of the inverted images !!!!

# Using Lower star filtration induced by the density filtration

radius = 6

data_per_density = [ None for _ in range(n_data)]

density_filtration = DensityFiltration(radius=radius)
filtrations_values = inverter(density_filtration.fit_transform(1-data_edge))#(1-bin_data)

for i in range(n_data):

    data_per_density[i] = lower_star_persistence(data_edge[i],filtrations_values[i],homology_dimensions=[0,1])#(bin_data[i],filtrations_values[i],homology_dimensions=[0,1])

# VECTORIZE THE PERSISTENCE DIAGRAMS


# Using the Silhouettes:

data_vec_density = [None]*n_data

n_finesse = 19

for i in range(n_data):

      v0 = (data_per_density[i])[0]
      v1 = (data_per_density[i])[1]
      v = np.concatenate((vect(v0,n_finesse),vect(v1,n_finesse)))
      #v = vect(v0,n_finesse)
      data_vec_density[i] = np.append(data_vec_density[i], v)

for i in range(n_data):     # si può fare meglioooo
  data_vec_density[i] = data_vec_density[i][1:]


# Let's see if the Silhouette are different by classes

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for i in range(n_label-1):
    ax = axes[i // 3, i % 3]

    for j in [p for p,x in enumerate(label) if x == i]:
        ax.plot(data_vec_density[j])

    ax.set_title("Silhouette of " + str(name_label[i]) + " images")

plt.tight_layout()
plt.show()


# 4) CONTOUR FILTRATION

# Using Lower star filtration induced by the contour filtration

data_per_contour= [ None for _ in range(n_data)]

filt_taxi = lambda ima : distance_transform_bf(ima, metric='taxicab')
contour_filtration = np.array(list(map(filt_taxi, data_edge)))

for i in range(n_data):

    data_per_contour[i] = lower_star_persistence(contour_filtration[i],contour_filtration[i],homology_dimensions=[0,1])


# VECTORIZE THE PERSISTENCE DIAGRAMS


# Using the Silhouettes:

data_vec_contour = [None]*n_data

n_finesse = 19

for i in range(n_data):

      v0 = (data_per_contour[i])[0]
      v1 = (data_per_contour[i])[1]
      v = np.concatenate((vect(v0,n_finesse),vect(v1,n_finesse)))
      #v = vect(v0,n_finesse)
      data_vec_contour[i] = np.append(data_vec_contour[i], v)

for i in range(n_data):     # si può fare meglioooo
  data_vec_contour[i] = data_vec_contour[i][1:]


# Let's see if the Silhouette are different by classes

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for i in range(n_label-1):
    ax = axes[i // 3, i % 3]

    for j in [p for p,x in enumerate(label) if x == i]:
        ax.plot(data_vec_contour[j])

    ax.set_title("Silhouette of " + str(name_label[i]) + " images")

plt.tight_layout()
plt.show()




# UNIFY ALL VECTORIZED DATA


data_vec_tot = data_vec_height.copy()

for i in range(n_data):

   data_vec_tot[i] = np.append(data_vec_tot[i], data_vec_radial[i])
   data_vec_tot[i] = np.append(data_vec_tot[i], data_vec_density[i])
   data_vec_tot[i] = np.append(data_vec_tot[i], data_vec_contour[i])


# RANDOM FOREST

# Now we can use this new data set for a RF

n_train = 400
n_trees = 1000

classifier_rf(data_vec_tot, label, n_train, n_trees)

# NB in previous approach (...) if n_train = 300 => 0.705, if = 400 => 0.74

# NAIVE APPROACH

# do flattening on the images to use them as vectors
data_flat = data.copy()
data_flat = data.reshape(n_data, 784)

#pca_fun(data_flat,label,2)

classifier_rf(data_flat, label, n_train, n_trees)
