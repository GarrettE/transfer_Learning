# Transfer Leanring example:
# Use Resnet50 NN architecture trained on ImaeNet 
# to get > 98% accurracy on Kaggle Dogs versus Cat competition


# import the necessary packages
from keras.applications import ResNet50
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from library.io import HDF5DatasetWriter
from imutils import paths
import numpy as np
import progressbar
import random
import pickle
import h5py
import os

#Path to Kaggle dogs versus cats images
dataset = ".\\library\\datasets\\kaggle_dogs_vs_cats\\train"
bs = 64
jobs = 10

#Oupt trained model (HDF5) path 
#writing mdoel to disk makes it easier to deal with datasets 
#that don't fit into memeory
output = ".\\dogvcat_output2.hdf5"
db = ".\\dogvcat_output2.hdf5"
model = ".\\dogs_vs_cats.pickle"

 

# grab the list of images that we'll be describing then randomly
# shuffle them to allow for easy training and testing splits via
# array slicing during training time
print("[INFO] loading images...")
imagePaths = list(paths.list_images(dataset))
random.shuffle(imagePaths)

# extract the class labels from the image paths then encode the
# labels
labels = [p.split(os.path.sep)[-1].split(".")[0] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

# load the ResNet50 network
print("[INFO] loading network...")
model = ResNet50(weights="imagenet", include_top=False)

# initialize the HDF5 dataset writer, then store the class label
# names in the dataset
dataset = HDF5DatasetWriter((len(imagePaths), 2048),
	output, dataKey="features", bufSize=bs)
dataset.storeClassLabels(le.classes_)

# initialize the progress bar
widgets = ["Extracting Features: ", progressbar.Percentage(), " ",
	progressbar.Bar(), " ", progressbar.ETA()]
pbar = progressbar.ProgressBar(maxval=len(imagePaths),
	widgets=widgets).start()

# loop over the images in batches
for i in np.arange(0, len(imagePaths), bs):
	# extract the batch of images and labels, then initialize the
	# list of actual images that will be passed through the network
	# for feature extraction
	batchPaths = imagePaths[i:i + bs]
	batchLabels = labels[i:i + bs]
	batchImages = []

	# loop over the images and labels in the current batch
	for (j, imagePath) in enumerate(batchPaths):
		# load the input image using the Keras helper utility
		# while ensuring the image is resized to 224x224 pixels
		image = load_img(imagePath, target_size=(224, 224))
		image = img_to_array(image)

		# preprocess the image by (1) expanding the dimensions and
		# (2) subtracting the mean RGB pixel intensity from the
		# ImageNet dataset
		image = np.expand_dims(image, axis=0)
		image = imagenet_utils.preprocess_input(image)

		# add the image to the batch
		batchImages.append(image)

	# pass the images through the network and use the outputs as
	# our actual features
	batchImages = np.vstack(batchImages)
	features = model.predict(batchImages, batch_size=bs)

	# reshape the features so that each image is represented by
	# a flattened feature vector of the `MaxPooling2D` outputs
	features = features.reshape((features.shape[0], 2048))

	# add the features and labels to our HDF5 dataset
	dataset.add(features, batchLabels)
	pbar.update(i)

# close the dataset
dataset.close()
pbar.finish()




# open the HDF5 database for reading then determine the index of
# the training and testing split, provided that this data was
# already shuffled *prior* to writing it to disk
db = h5py.File(db, "r")
i = int(db["labels"].shape[0] * 0.75)

# define the set of parameters that we want to tune then start a
# grid search where we evaluate our model for each value of C
params = {"C": [0.0001, 0.001, 0.01, 0.1, 1.0]}
model = GridSearchCV(LogisticRegression(), params, cv=3,
n_jobs=jobs)
model.fit(db["features"][:i], db["labels"][:i])
print("[INFO] best hyperparameters: {}".format(model.best_params_))

# generate a classification report for the model
print("[INFO] evaluating...")
preds = model.predict(db["features"][i:])
print(classification_report(db["labels"][i:], preds,
	target_names=db["label_names"]))

# compute the raw accuracy with extra precision
acc = accuracy_score(db["labels"][i:], preds)
print("[INFO] score: {}".format(acc))

# serialize the model to disk
print("[INFO] saving model...")
f = open(model, "wb")
f.write(pickle.dumps(model.best_estimator_))
f.close()

# close the database
db.close()



