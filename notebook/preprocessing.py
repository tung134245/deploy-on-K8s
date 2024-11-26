import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import CategoricalNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
import graphviz
# %matplotlib inline

def pre_processing(file):
    df = pd.read_csv(file)
    numerical_features = [feature for feature in df.columns if df[feature].dtypes != 'O']
    print("The number of numerical features is",len(numerical_features),"and they are : \n",numerical_features)
    categorical_features = [feature for feature in df.columns if df[feature].dtypes == 'O']
    print("The number of categorical features is",len(categorical_features),"and they are : \n",categorical_features)

    #discrete numerical features 
    discrete_feature = [feature for feature in numerical_features if df[feature].nunique()<=15 and feature != 'label']
    print("The number of discrete features is",len(discrete_feature),"and they are : \n",discrete_feature)
    continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature + ['label']]
    print("The number of continuous_feature features is",len(continuous_feature),"and they are : \n",continuous_feature)

    df = pd.get_dummies(df, columns=categorical_features,drop_first=True)
    print("This Dataframe has {} rows and {} columns after encoding".format(df.shape[0], df.shape[1]))
    print(df)
    return df

# pre_processing("/home/tung/Downloads/MLE K1/Project/data/dataset_sdn.csv")

def get_label(classes):
    y_pred = []
    for i in classes:
        if i > 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred