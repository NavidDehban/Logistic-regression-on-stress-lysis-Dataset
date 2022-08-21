import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import log_loss
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, model_selection, svm
#--------------------------------------
def jaccard(predict,real):
    n = len(real)
    c = 0 
    for i in range(n):
        if real[i] == predict[i]:
            c += 1
    return c/(2*n - c)
def confusion(predict,real):
    n = len(predict)
    conf = np.zeros((3,3))
    for i in range(n):
        if predict[i] == 'low':
            predict[i] = 0
        if predict[i] == 'mid':
            predict[i] = 1 
        if predict[i] == 'high':
            predict[i] = 2
        if real[i] == 'low':
            real[i] = 0
        if real[i] == 'mid':
            real[i] = 1 
        if real[i] == 'high':
            real[i] = 2
    for i in range(n):
        conf[real[i],predict[i]] += 1 
    return conf
#--------------------------------------
df = pd.read_csv("E:\\university\\term 6\\ML\\home works\\hw2\\ML_HW2\\Stress-Lysis.csv")
df = df[['Humidity', 'Temperature','Stress_Level']]
#--------------------------------------
stress=df['Stress_Level'].copy()
df['Stress_Level'],labels = pd.factorize(df['Stress_Level'])
stress.value_counts(),df.Stress_Level.value_counts()
#--------------------------------------
X = np.asarray(df[['Humidity', 'Temperature']])
y = np.asarray(df['Stress_Level'])
X = preprocessing.StandardScaler().fit(X).transform(X)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.8,random_state=0)
#--------------------------------------
clf = OneVsRestClassifier(SVC()).fit(X_train, y_train)
y_pre = clf.predict(X_test)
acc = clf.score(X_test,y_test)
print('accuracy:',acc)
#--------------------------------------
conf = confusion(y_pre,y_test)
print('confusion matrix:')
print(conf)
#--------------------------------------
j = jaccard(y_pre,y_test)
print('jaccard:',j)
#--------------------------------------
print('--------------------------------')
print (classification_report(y_test, y_pre))
#--------------------------------------
plt.show()

