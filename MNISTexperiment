from MNISTutils import *



# EXTRACT DATA

(train_X, train_y), (test_X, test_y) = mnist.load_data()

# we use only n images

n_data = 500
n_label = 10
data = train_X[0:n_data]
label = train_y[0:n_data]
name_label = ["0","1","2","3","4","5","6","7","8","9"]

# Visualize the data

plt.imshow(data[17], cmap='gray')
plt.colorbar()
plt.show()


# CONVERT INTO BINARY IMAGES WELL


# We apply padding + median filter + binarization to our images
padding = lambda ima : np.pad(ima, ((2,2), (2,2)), 'constant', constant_values=0)
sq=morphology.rectangle(3, 3, dtype='uint8')
median = lambda ima : filters.median(ima, sq)
binarization = lambda ima : 1*(ima>np.average(ima))
inverter = lambda ima : np.max(np.float32(ima))-ima
form_pipeline = lambda ima : (inverter(binarization(median(padding(ima)))))

data_form = np.array(list(map(form_pipeline, data)))

# Visualize the well binarizied images

plt.figure(figsize=(10, 10))

for i in range(6*6):
    plt.subplot(6, 6, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(data_form[i], cmap='gray')
    plt.xlabel(name_label[label[i]])

plt.show()


# COMPUTE PERSISTENT HOMOLOGY (height filtration)


# Using Lower star filtration induced by the height filtration

directions = [(1,0),(0,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,1),(-1,-1)]
n_dir = len(directions)

# Compute the 8 persistence diagrams for the image (one for direction)

#data_per = [[None]*n_dir]*n_data   <--- ERROREEEE
data_per_height = [[None for _ in range(n_dir)] for _ in range(n_data)]

for j, dir in enumerate(directions):

  height_filtration = HeightFiltration(direction=np.array(dir))
  filtrations_values = height_filtration.fit_transform(inverter(data_form))

  for i in range(n_data):

    data_per_height[i][j] = lower_star_persistence(data_form[i],filtrations_values[i],homology_dimensions=[0,1])



# VECTORIZE THE PERSISTENCE DIAGRAMS


# Using the Silhouettes:

data_vec_height = [None]*n_data

n_finesse = 19

for j in range(n_dir):

    for i in range(n_data):

      v0 = (data_per_height[i][j])[0]
      v1 = (data_per_height[i][j])[1]
      v = np.concatenate((vect(v0,n_finesse,0.01),vect(v1,n_finesse,0.01)))
      #v = vect(v0, n_finesse)
      data_vec_height[i] = np.append(data_vec_height[i], v)

for i in range(n_data): 
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



# RANDOM FOREST

# Now we can use this new data set for a RF

n_train = 300
n_trees = 1000

classifier_rf(data_vec_height, label, n_train, n_trees)




# COMPUTE PERSISTENT HOMOLOGY (radial filtration)


# Using Lower star filtration induced by the radial filtration

centers = [(0,0),(0,27),(27,0),(27,27)]
n_cen = len(centers)

# Compute the 4 persistence diagrams for the image (one for center)

data_per_radial = [[None for _ in range(n_cen)] for _ in range(n_data)]

for j, cen in enumerate(centers):

  radial_filtration = RadialFiltration(center=np.array(cen))
  filtrations_values = radial_filtration.fit_transform(inverter(data_form))

  for i in range(n_data):

    data_per_radial[i][j] = lower_star_persistence(data_form[i],filtrations_values[i],homology_dimensions=[0,1])



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
      v1 = (data_per_radial[i][j])[1]
      v = np.concatenate((vect(v0,n_finesse,0.01),vect(v1,n_finesse,0.01)))
      #v = vect(v0,n_finesse)
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


# RANDOM FOREST

# Now we can use this new data set for a RF

n_train = 300
n_trees = 1000

classifier_rf(data_vec_radial, label, n_train, n_trees)


# COMPUTE PERSISTENT HOMOLOGY (density filtration)


# Using Lower star filtration induced by the density filtration

radius = 6

data_per_density = [ None for _ in range(n_data)]

density_filtration = DensityFiltration(radius=radius)
filtrations_values = 1-density_filtration.fit_transform(1-data_form)

for i in range(n_data):

    data_per_density[i] = lower_star_persistence(data_form[i],filtrations_values[i],homology_dimensions=[0,1])


# VECTORIZE THE PERSISTENCE DIAGRAMS


# Using the Silhouettes:

data_vec_density = [None]*n_data

n_finesse = 19

for i in range(n_data):

      v0 = (data_per_density[i])[0]
      v1 = (data_per_density[i])[1]
      v = np.concatenate((vect(v0,n_finesse,0.01),vect(v1,n_finesse,0.01)))
      #v = vect(v0,n_finesse)
      data_vec_density[i] = np.append(data_vec_density[i], v)

for i in range(n_data):     
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


# RANDOM FOREST

# Now we can use this new data set for a RF

n_train = 300
n_trees = 1000

classifier_rf(data_vec_density, label, n_train, n_trees)


# UNIFY ALL VECTORIZED DATA


data_vec_tot = data_vec_height.copy()

for i in range(n_data):

   data_vec_tot[i] = np.append(data_vec_tot[i], data_vec_radial[i])
   data_vec_tot[i] = np.append(data_vec_tot[i], data_vec_density[i])


# RANDOM FOREST

# Now we can use this new data set for a RF

n_train = 300  #100
n_trees = 1000

classifier_rf(data_vec_tot, label, n_train, n_trees)



# NAIVE APPROACH

# do flattening on the images to use them as vectors
data_flat = data.copy()
data_flat = data.reshape(n_data, 784)

classifier_rf(data_flat, label, n_train, n_trees)
