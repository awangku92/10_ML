# import the necessary packages

# ML method/ Algorithm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# A dataset splitting method used to separate our data into training and testing subsets 
from sklearn.model_selection import train_test_split

# Report utility
from sklearn.metrics import classification_report

# Iris Dataset
from sklearn.datasets import load_iris

# CMD arg parsing
import argparse

# to see the iris dataset using pandas and numpy
import pandas as pd
import numpy as np

def ML_by_models():
	# construct the argument parser and parse the arguments
	ap = argparse.ArgumentParser()

	# >>> python classify.py --model knn --> to choose KNeighborsClassifier() method to do ML
	ap.add_argument("-m", "--model", type=str, default="knn",
		help="Type of python machine learning model to use")
	args = vars(ap.parse_args())

	# define the dictionary of models our script can use, where the key
	# to the dictionary is the name of the model (supplied via command
	# line argument) and the value is the model itself
	models = {
		"knn": KNeighborsClassifier(n_neighbors=1),
		"naive_bayes": GaussianNB(),
		"logit": LogisticRegression(solver="lbfgs", multi_class="auto"),
		"svm": SVC(kernel="rbf", gamma="auto"),
		"decision_tree": DecisionTreeClassifier(),
		"random_forest": RandomForestClassifier(n_estimators=100),
		"mlp": MLPClassifier(),
		"all": []
	}

	# load the Iris dataset and perform a training and testing split,
	# using 75% of the data for training and 25% for evaluation
	print("[INFO] loading data...")
	dataset = load_iris()
	(trainX, testX, trainY, testY) = train_test_split(dataset.data, dataset.target, random_state=3, test_size=0.25)

	# to see the iris dataset
	# df = pd.DataFrame(data= np.c_[dataset['data'], dataset['target']], columns= dataset['feature_names'] + ['target'])
	# print(df)

	# train the model
	print("[INFO] Training using '{}' model".format(args["model"]))
	model = models[args["model"]]
	model.fit(trainX, trainY)
	 
	# make predictions on our data and show a classification report
	print("[INFO] evaluating...")
	predictions = model.predict(testX)
	print(classification_report(testY, predictions, target_names=dataset.target_names))

	return


if __name__ == '__main__':
	print('Start ML')

	ML_by_models()

	print('Finish ML')

