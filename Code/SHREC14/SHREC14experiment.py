from SHREC14utils import *


# EXTRACT DATA


n_data = 300

# data is a list n_data x 2,
# with data[i][0] = np.array of the vertices of the i-th image (given in euclidean coordinates),
# with data[i][1] = np.array of the faces of the i-th image (given through indeces).

data = [[] for _ in range(n_data)]

for i in range(n_data):

  # Carica la mesh dal file .obj usando trimesh
  mesh = trimesh.load('/content/data/'+str(i)+'.obj')

  # Estrai i vertici e le facce dalla mesh
  data[i] = [mesh.vertices, mesh.faces]


# define the function of the labels

def label(index):
    if 0 <= index <= 19:
        return 'male_neutral'
    elif 20<= index <=39:
        return 'male_bodybuilder'
    elif 40<= index <=59:
        return 'male_fat'
    elif 60<= index <=79:
        return 'male_thin'
    elif 80<= index <=99:
        return 'male_average'
    elif 100<= index <=119:
        return 'female_neutral'
    elif 120<= index <=139:
        return 'female_bodybuilder'
    elif 140<= index <=159:
        return 'female_fat'
    elif 160<= index <=179:
        return 'female_thin'
    elif 180<= index <=199:
        return 'female_average'
    elif 200<= index <=219:
        return 'child_neutral'
    elif 220<= index <=239:
        return 'child_bodybuilder'
    elif 240<= index <=259:
        return 'child_fat'
    elif 260<= index <=279:
        return 'child_thin'
    elif 280<= index <=299:
        return 'child_average'
    else:
        print('What are you giving me?')

label_name = ['male_neutral','male_bodybuilder','male_fat','male_thin','male_average','female_neutral','female_bodybuilder','female_fat','female_thin','female_average','child_neutral','child_bodybuilder','child_fat','child_thin','child_average']


# COMPUTE PERSISTENT HOMOLOGY

# we compute the persistent homology for 0 and 1 classes and we vectorize the results using Silhouette for each data

data_per = [None]*n_data
max_values_filtration = [None]*n_data

n_eig = 50
t = np.array(500)  # prova con 500 e con 1000


for i in range(n_data):
  if i == 0: tic = time()
  hks = get_hks(data[i][0], data[i][1], n_eig, t)
  maxx = np.max(hks)
  max_values_filtration[i] = maxx*(7/6)  # => in order to give more importance to classes that never die
  v = lower_star_persistence(data[i][1], maxx-hks)
  data_per[i] = v
  if i == 0: print("expected time is ", n_data*(time()-tic))
  print(i)



# VECTORIZE PERSISTENT DIAGRAMS

data_vec = [None]*n_data
n_finess = 200
m = 1

for i in range(n_data):

  v0, v1 = vect1(data_per[i],n_finess,max_values_filtration[i],m)
  data_vec[i] = np.append(v0[0],v1[0])
  #data_vec[i] = v1[0]


# Let's see if the Silhouette are different by classes

fig, axes = plt.subplots(5, 3, figsize=(15, 15))

for i in range(15):
    ax = axes[i // 3, i % 3]

    for j in [p for p in range(n_data) if label(p) == label_name[i]]:
        ax.plot(data_vec[j])

    ax.set_title("Silhouette of " + str(label_name[i]) + " images")
    #ax.set_ylim(0, 0.00022)

plt.tight_layout()
plt.show()



# PCA ON VECTORIZED DATA WITHOUT OUTLIERS

pca_fun(data_vec,label_vec,2)


print(len(data_vec))
print(len(label_vec))

pca_fun(data_vec,label_vec,2)


# PCA ON VECTORIZED DATA


# See all the divions of data

label_vec = [None]*n_data

for i in range(n_data):
    label_vec[i] = label(i)

pca_fun(data_vec,label_vec,2)

# See only the general divion of data given by man/woman/child

label_vec_general = [None]*n_data

for i in range(n_data):
  if i<100:
    label_vec_general[i] = "man"
  if i<200 and i>=100:
    label_vec_general[i] = "women"
  if i<300 and i>=200:
    label_vec_general[i] = "child"

pca_fun(data_vec,label_vec_general,2)



# RANDOM FOREST

# Now we can use this new data set for a RF

n_trees = 1000
n_train = 15

if n_train>20: n_train = 20

data_vec_train = []
label_train = []

for i in range(len(label_name)):
  data_vec_train += data_vec[(i*20):(i*20+n_train)]
  label_train += label_vec[(i*20):(i*20+n_train)]

data_vec_test = []
label_test = []

for i in range(len(label_name)):
  data_vec_test += data_vec[(i*20+n_train):((i+1)*20)]
  label_test += label_vec[(i*20+n_train):((i+1)*20)]


forest=RandomForestClassifier(n_estimators=n_trees,
                              random_state=0)

forest.fit(data_vec_train, label_train)

y_pred = forest.predict(data_vec_test)

accuracy = accuracy_score(label_test, y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(label_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot();



# K-CROSS FOLD VALIDATION


# Creare un modello di Random Forest
model = RandomForestClassifier(n_estimators=1000)#, random_state=0)

# Eseguire la validazione incrociata a 10 fold e calcolare l'accuratezza
scores = cross_val_score(model, data_vec, label_vec, cv=10, scoring='accuracy')

# Stampare i risultati
print(f'Accuratezza per ogni fold: {scores}')
print(f'Accuratezza media: {scores.mean()}')
print(f'Deviazione standard dell\'accuratezza: {scores.std()}')



# RANDOM FOREST FOR GENERAL DIVISION

# Now we can use this new data set for a RF

n_trees = 1000
n_train = 15

if n_train>20: n_train = 20

data_vec_train = []
label_train = []

for i in range(len(label_name)):
  data_vec_train += data_vec[(i*20):(i*20+n_train)]
  label_train += label_vec_general[(i*20):(i*20+n_train)]

data_vec_test = []
label_test = []

for i in range(len(label_name)):
  data_vec_test += data_vec[(i*20+n_train):((i+1)*20)]
  label_test += label_vec_general[(i*20+n_train):((i+1)*20)]


forest=RandomForestClassifier(n_estimators=n_trees,
                              random_state=0)

forest.fit(data_vec_train, label_train)

y_pred = forest.predict(data_vec_test)

accuracy = accuracy_score(label_test, y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(label_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot();


# K-CROSS FOLD VALIDATION FOR GENERAL DIVISION


# Creare un modello di Random Forest
model = RandomForestClassifier(n_estimators=1000)#, random_state=0)

# Eseguire la validazione incrociata a 10 fold e calcolare l'accuratezza
scores = cross_val_score(model, data_vec, label_vec_general, cv=10, scoring='accuracy')

# Stampare i risultati
print(f'Accuratezza per ogni fold: {scores}')
print(f'Accuratezza media: {scores.mean()}')
print(f'Deviazione standard dell\'accuratezza: {scores.std()}')
