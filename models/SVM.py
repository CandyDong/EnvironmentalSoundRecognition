import numpy as np
import glob
import librosa
import librosa.display
import math
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix 


class SVMModel():
	def __init__(self, opts):
		self.svc = SVC(C=28.0, gamma = 0.00001, decision_function_shape="ovr")

	def run(self, X_train, y_train):
		self.svc.fit(X_train, y_train)

	# predict results with the trained SVC classifier
	def predict(self, X_test):
		return self.svc.predict(X_test)

	def evaluate(self, X_test, y_test):
		y_pred = self.predict(X_test)
		print("y_pred:"+str(y_pred))
		confusion = confusion_matrix(y_test,y_pred)
		classi = classification_report(y_test,y_pred)
		acc = self.svc.score(X_test, y_test)
		return confusion, classi, acc


	

