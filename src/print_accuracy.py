from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
import numpy as np


data = pickle.loads(open("embeddings.pickle", "rb").read())

le = LabelEncoder()
labels = le.fit_transform(data["names"])

print(np.array(data["embeddings"]).shape)
print("Training the model...")
clf = pickle.loads(open("classifier.pickle", "rb").read())
print("Accuracy: ", clf.score(data["embeddings"], labels))