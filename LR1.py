from pickle import TRUE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math 
import copy

def plot_density(data):
    sns.scatterplot(x = 'Temperature',y = 'Humidity',hue ='Stress_Level', data = data )
    plt.figure()
    sns.scatterplot(x = 'Temperature',y = 'Step_count',hue ='Stress_Level', data = data )
    plt.figure()
    sns.scatterplot(x = 'Step_count',y = 'Humidity',hue ='Stress_Level', data = data )
    plt.show()

def data_divider(df):
    msk = np.random.rand(len(df)) < 0.8
    train = df[msk]
    test = df[~msk]
    return [test,train]

def sigmoid(x):
    z = np.exp(-x)
    sig = 1 / (1 + z)
    return sig

def change_title(level,train):
    data = copy.deepcopy(train)
    l = data['Stress_Level'].values.tolist()
    for i in range(len(l)):
        if l[i] == level:
            l[i] = 1
        else :
            l[i] = 0
    data['Stress_Level'] = l
    return data
        
def gd(data):
    eta = 0.00000000001 
    n_iterations = 100
    #--------------------------------------
    w = np.array([[0],[0]])
    #w = np.random.randn(2,1)
    temp = data['Temperature'].values.tolist()
    humid = data['Humidity'].values.tolist()
    stress = data['Stress_Level'].values.tolist()
    #--------------------------------------
    for iteration in range(n_iterations):
        s = 0
        for i in range(len(stress)):     
            xi = np.array([[temp[i]],[humid[i]]])
            yi = stress[i]
            a = np.dot(np.transpose(w),xi)
            b = sigmoid(a[0,0])
            c = yi - b
            d = xi * c 
            s += d  
        w = w - eta * s
    #--------------------------------------
    return w    

def classify(data,w):
    temp = data['Temperature'].values.tolist()
    humid = data['Humidity'].values.tolist()
    stress = data['Stress_Level'].values.tolist()
    p = []
    #--------------------------------------
    for i in range(len(stress)):
            xi = np.array([[temp[i]],[humid[i]]])
            a = np.dot(np.transpose(w),xi)
            p.append(1/(1 + sigmoid(a[0,0])))    
    return p

def one_vs_rest(train,test):
    w = []
    l = ['low','mid','high']
    data = copy.deepcopy(train)
    #--------------------------------------
    for i in range(len(l)):
        data = change_title(l[i],train)
        w.append(gd(data))
    #--------------------------------------
    space_()
    print('weights:')
    for mat in w:
        print(mat)
    #--------------------------------------
    p_low  = classify(test , w[0])
    p_mid  = classify(test , w[1])
    p_high = classify(test , w[2])
    #--------------------------------------
    stress = test['Stress_Level'].values.tolist()
    stress_predict = []
    for i in range(len(stress)):
        p = [p_low[i],p_mid[i],p_high[i]]
        index = np.where(p == np.amax(p))[0][0]
        if index == 0:
            stress_predict.append('low')
        if index == 1:
            stress_predict.append('mid')
        if index == 2: 
            stress_predict.append('high')
    #--------------------------------------
    space_()
    count = 0
    for i in range(len(stress)):
        if stress[i] == stress_predict[i]:
            count += 1 
    print('true detected:',count)
    #--------------------------------------
    space_()
    print('confusion matrix:')
    print(confusion(stress_predict,stress))
    #--------------------------------------
    space_()
    print('jaccard:',jaccard(stress_predict,stress))
    #--------------------------------------
    space_()
    print('accuracy:',accuracy(confusion(stress_predict,stress),len(test)))
    #--------------------------------------
    space_()
    print('f1_score:',f1_score(jaccard(stress_predict,stress),accuracy(confusion(stress_predict,stress),len(test))))
       
def space_():
    print('.....................')

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
        
def jaccard(predict,real):
    n = len(real)
    c = 0 
    for i in range(n):
        if real[i] == predict[i]:
            c += 1
    return c/(2*n - c)

def accuracy(c,total):
    return (c[0,0] + c[1,1] + c[2,2])*100/total
    
def f1_score(jac,acu):
    return 2*jac*acu/(jac + acu)

def main():
    data= pd.read_csv("E:\\university\\term 6\\ML\\home works\\hw2\\ML_HW2\\Stress-Lysis.csv")
    data = pd.DataFrame(data ,columns=['Humidity','Temperature','Step_count','Stress_Level'])
    test,train = data_divider(data)
    #plot_density(data)
    one_vs_rest(train,test)

main()





