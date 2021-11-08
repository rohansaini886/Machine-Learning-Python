# Import useful libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings("ignore")

#************************ 1. Data Loading ************************
titanic = pd.read_csv("C:\\Users\\VC\\Desktop\\Machine Learning\\Data_Sets\\Titanic_train.csv")
#print(titanic.head(891))



#************************ 2. & 3. Data Preparation & Manipulation ************************
#***** Categorical Data ***** 
titanic_categorical = titanic.select_dtypes(object)

# Delete 'Name' and 'Ticket' Columns
titanic_categorical.drop(['Name', 'Ticket'], axis = 1, inplace = True)

# Replace NaN values of Cabin with max count
titanic_categorical.Cabin.fillna(titanic_categorical.Cabin.value_counts().idxmax(), inplace = True)

# Replace NaN values of Embarked with max count
titanic_categorical.Embarked.fillna(titanic_categorical.Embarked.value_counts().idxmax(), inplace = True)

# To count number of nulls in particular column
#print(titanic_categorical.isnull().sum())
#print(titanic_categorical.describe())


#***** Numerical Data ***** 
titanic_numerical = titanic.select_dtypes(np.number)

# Delete 'PassengerId' Columns
titanic_numerical.drop(['PassengerId'], axis = 1, inplace = True)

# Replace NaN values of Age with mean value
titanic_numerical.Age.fillna(titanic_numerical.Age.mean(), inplace = True)

# To count number of nulls in particular column
#print(titanic_numerical.isnull().sum())
#print(titanic_numerical.describe())


# To replace Categorical data with Numerical Labels
lab_en = LabelEncoder();
titanic_categorical = titanic_categorical.apply(lab_en.fit_transform)

# Merge Preprocessed Categorical and Numericals datas
titanic_prepared = pd.concat([titanic_categorical, titanic_numerical], axis = 1)


#************************ 4. Model ************************
X = titanic_prepared.drop(['Survived'], axis = 1)
Y = titanic_prepared['Survived']

# Train with 80% data and test with remaining 20% data
X_train = np.array(X[0:(int)(len(X)*0.80)])
Y_train = np.array(Y[0:(int)(len(Y)*0.80)])

X_test = np.array(X[(int)(len(X)*0.80):])
Y_test = np.array(Y[(int)(len(Y)*0.80):])

print(len(X_train), len(Y_train), len(X_test), len(Y_test))

# Logistic Regression
LR = LogisticRegression()

# K-Nearest Neighbours
KNN = KNeighborsClassifier()

# Naive Bayes
NB = GaussianNB()

# Linear Support Vector Machine
LSVM = LinearSVC()

# Non-Linear Support Vector Machine
NLSVM = SVC(kernel = 'rbf')

# Decision Tree
DT = DecisionTreeClassifier()

# Random Forest
RF = RandomForestClassifier()

# Training the selected model with X_train and Y_train
LR_fit = LR.fit(X_train, Y_train)
KNN_fit = KNN.fit(X_train, Y_train)
NB_fit = NB.fit(X_train, Y_train)
LSVM_fit = LSVM.fit(X_train, Y_train)
NLSVM_fit = NLSVM.fit(X_train, Y_train)
DT_fit = DT.fit(X_train, Y_train)
RF_fit = RF.fit(X_train, Y_train)

#************************ 5. Analysis ************************
LR_pred = LR_fit.predict(X_test)
KNN_pred = KNN_fit.predict(X_test)
NB_pred = NB_fit.predict(X_test)
LSVM_pred = LSVM_fit.predict(X_test)
NLSVM_pred = NLSVM_fit.predict(X_test)
DT_pred = DT_fit.predict(X_test)
RF_pred = RF_fit.predict(X_test)

LR_accuracy = accuracy_score(LR_pred, Y_test) * 100
KNN_accuracy = accuracy_score(KNN_pred, Y_test) * 100
NB_accuracy = accuracy_score(NB_pred, Y_test) * 100
LSVM_accuracy = accuracy_score(LSVM_pred, Y_test) * 100
NLSVM_accuracy = accuracy_score(NLSVM_pred, Y_test) * 100
DT_accuracy = accuracy_score(DT_pred, Y_test) * 100
RF_accuracy = accuracy_score(RF_pred, Y_test) * 100

print("LR accuracy is " + str(round(LR_accuracy, 2)) + "%")
print("KNN accuracy is " + str(round(KNN_accuracy, 2)) + "%")
print("NB accuracy is " + str(round(NB_accuracy, 2)) + "%")
print("LSVM accuracy is " + str(round(LSVM_accuracy, 2)) + "%")
print("NLSVM accuracy is " + str(round(NLSVM_accuracy, 2)) + "%")
print("DT accuracy is " + str(round(DT_accuracy, 2)) + "%")
print("RF accuracy is " + str(round(RF_accuracy, 2)) + "%")

#**************************** Predicted Output ****************************
testData = pd.read_csv("C:\\Users\\VC\\Desktop\\Machine Learning\\Data_Sets\\Titanic_test.csv")
testData_categorical = testData.select_dtypes(object)
testData_categorical.drop(['Name', 'Ticket'], axis = 1, inplace = True)
testData_categorical.Cabin.fillna(testData_categorical.Cabin.value_counts().idxmax(), inplace = True)

testData_numerical = testData.select_dtypes(np.number)
testData_numerical.drop(['PassengerId'], axis = 1, inplace = True)
testData_numerical.Age.fillna(testData_numerical.Age.mean(), inplace = True)
testData_numerical.Fare.fillna(testData_numerical.Fare.value_counts().idxmax(), inplace = True)

testData_categorical = testData_categorical.apply(lab_en.fit_transform)
X_test = pd.concat([testData_categorical, testData_numerical], axis = 1)

LR_pred = LR_fit.predict(X_test)
RF_pred = RF_fit.predict(X_test)
testData['Survived_LR'] = LR_pred
testData['Survived_RF'] = RF_pred
testData.to_csv("C:\\Users\\VC\\Desktop\\Machine Learning\\Data_Sets\\Titanic_test_result.csv")


#**************************** Predicted Output Of Whole Data Set ****************************
testData = pd.read_csv("C:\\Users\\VC\\Desktop\\Machine Learning\\Data_Sets\\Titanic_Full.csv")
testData_categorical = testData.select_dtypes(object)
testData_categorical.drop(['Name', 'Ticket'], axis = 1, inplace = True)
testData_categorical.Cabin.fillna(testData_categorical.Cabin.value_counts().idxmax(), inplace = True)

testData_numerical = testData.select_dtypes(np.number)
testData_numerical.drop(['PassengerId'], axis = 1, inplace = True)
testData_numerical.Age.fillna(testData_numerical.Age.mean(), inplace = True)
testData_numerical.Fare.fillna(testData_numerical.Fare.value_counts().idxmax(), inplace = True)

testData_categorical = testData_categorical.apply(lab_en.fit_transform)
X_test = pd.concat([testData_categorical, testData_numerical], axis = 1)

LR_pred = LR_fit.predict(X_test)
KNN_pred = KNN_fit.predict(X_test)
NB_pred = NB_fit.predict(X_test)
#LVSM_pred = LSVM_fit.predict(X_test)
#NLVSM_pred = NLSVM_fit.predict(X_test)
DT_pred = DT_fit.predict(X_test)
RF_pred = RF_fit.predict(X_test)
testData['Survived_LR'] = LR_pred
testData['Survived_KNN'] = KNN_pred
testData['Survived_NB'] = NB_pred
#testData['Survived_LSvM'] = LSVM_pred
#testData['Survived_NLSVM'] = NLSVM_pred
testData['Survived_DT'] = DT_pred
testData['Survived_RF'] = RF_pred

testData.to_csv("C:\\Users\\VC\\Desktop\\Machine Learning\\Data_Sets\\Titanic_test_full_result.csv")
