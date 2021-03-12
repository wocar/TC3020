import sys

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
assert sys.version_info >= (3, 5)
import sklearn
assert sklearn.__version__ >= "0.20"
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)
PROJECT_ROOT_DIR = "."

CHAPTER_ID = "decision_trees"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR)

os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)

    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

from graphviz import Source
from sklearn.tree import export_graphviz, DecisionTreeClassifier, export_text
from sklearn import tree
import william as w
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target,
    test_size=0.2, shuffle = True, random_state = 123)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

export_graphviz(
    clf,
    out_file=os.path.join(IMAGES_PATH, "iris_tree.dot"),
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    rounded=True,
    filled=True
)
Source.from_file(os.path.join(IMAGES_PATH, "iris_tree.dot"))
