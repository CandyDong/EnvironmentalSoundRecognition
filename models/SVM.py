import numpy as np
import glob
import librosa
import librosa.display
import math
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix 


class SVMModel():
	def __init__(self, X_train, y_train, X_test, y_test, opts):
		self.train_df = train_df
		self.svc = SVC()
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test

	def run(self):
		svc = SVC()
		svc.fit(self.X_train, self.y_train)

	# predict results with the trained SVC classifier
	def predict(self, X_test):
		return svc.predict(X_test)

	def evaluate(self, X_test, y_test):
		y_pred = self.predict(self.svc, X_test)
		confusion_matrix = confusion_matrix(y_test,y_pred)
		classification_report = classification_report(y_test,y_pred)
		return confusion_matrix, classification_report


	

