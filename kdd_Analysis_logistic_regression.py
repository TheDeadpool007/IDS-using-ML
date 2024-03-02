import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX11.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 9].values
#Conversion of catogorical data

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])

labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [2])


labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])

#onehotencoder= OneHotEncoder(categorical_features="all")
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Predicting the Test set results# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#from sklearn.metrics import accuracy
cm1 = confusion_matrix(y_test, y_pred)


#from sklearn.metrics import accuracy_score
#print ('Accuracy for Logistic Regression Classifier :', accuracy_score(y_test,  y_pred))
#print ('\n confussion matrix for Logistic Regression Classifier:\n',confusion_matrix(y_test,  y_pred))

print(classification_report(y_test, y_pred))

# True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
tp = np.sum(np.logical_and(y_test == 1, y_pred == 1))
 
# True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
tn = np.sum(np.logical_and(y_test == 0, y_pred == 0))
 
# False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
fp = np.sum(np.logical_and(y_test == 1, y_pred == 0))
 
# False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
fn = np.sum(np.logical_and(y_test == 0, y_pred== 1))

print('\nTrue Positive : %d' %(tp))
print('\nTrue Negative : %d' %(tn))
print('\nFalse Positive : %d' %(fp))
print('\nFalse Negative : %d'%(fn))
 
#print('TP: %d, FP: %d, TN: %d, FN: %d' %(tp,fp,tn,fn))


# Sensitivity, hit rate, recall, or true positive rate
tpr = tp/(tp+fn)
# Specificity or true negative rate
tnr = tn/(tn+fp) 
# Precision or positive predictive value
ppv = tp/(tp+fp)
# Negative predictive value
npv = tn/(tn+fn)
# Fall out or false positive rate
fpr = fp/(fp+tn)
# False negative rate
fnr = fn/(tp+fn)
# False discovery rate
fdr = fp/(tp+fp)

# Overall accuracy
acc = (tp+tn)/(tp+fp+fn+tn)

if tp>0:
  precision=float(tp)/(tp+fp)
  recall=float(tp)/(tp+fn)
  
print ('\n confussion matrix for Logistic Regression Classifier:\n',confusion_matrix(y_test,  y_pred))
print('\nTrue Positive : %d' %(tp))
print('\nTrue Negative : %d' %(tn))
print('\nFalse Positive : %d' %(fp))
print('\nFalse Negative : %d'%(fn))
print('\nSensitivity, hit rate, recall, or true positive rate : %f' %(tpr))
print('\nSpecificity or true negative rate : %f' %(tnr))
print('\nPrecision or positive predictive value : %f' %(ppv))
print('\nNegative predictive value : %f'%(npv))
print('\nFall out or false positive rate: %f' %(fpr))
print('\nFalse negative rate : %f' %(fnr))
print('\nFalse discovery rate : %f' %(fdr))
print('\nPrecision : %f' %(fnr))
print('\nRecall : %f' %(fdr))
print('\nOverall accuracy for Logistic Regression Classifier : %f' %(acc))
#from sklearn.model_selection import cross_val_score
#accur=cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=5)
#M=accur.mean()
#S=accur.std()
