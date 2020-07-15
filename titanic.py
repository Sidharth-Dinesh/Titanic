'''
Predict whether or not a passenger survives the Titanic.
datasets =  test - for tesing.
            train - has survival values.
            
Planning to use multiple linear regression. 
'''

# Importing the necessary libraries and functions.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


###############################################################################


# Loading the necessary files.
test = pd.read_csv('titanic/test.csv')
train = pd.read_csv('titanic/train.csv')


# Combining the test and train datasets for future analysis. Removing Survived 
# and resetting index as well.
temp = train.drop(['Survived'],axis=1)
test = temp.append(test).reset_index()
test = test.drop(['index'],axis=1)


###############################################################################
# Data is currently raw and needs to be tidied up before use.

# Very few valid Cabin values (<25% of the table). Also, Cabin is not relevant 
# to predicting Survived. Dropping Cabin.
print(test.count())
test = test.drop(['Cabin'],axis=1)
print(test)
print('\n\n')


# Embarked and Fare have some missing values(<1% of table), but not too many.
# Acceptable.Those rows will simply be removed. 
test.dropna(subset=['Fare','Embarked'],axis=0,inplace=True)
print(test)
print(test.count())
print('\n\n')


# Age needs to be improved. 
# To do so, average age of the remaining is taken. The missing ages are filled
# using previous valid age values.
test.fillna(inplace=True,method='pad')
print(test)
print(test.count())
dupe = test


###############################################################################
# Data is now tidy. However, to use logistic regression, one will need 
# numerical values for all columns. This will be done by mapping. Also, dummy
# columns will be made so that only one equation is needed for any given
# prediction.


x = pd.get_dummies(test['Sex'],columns='Male',drop_first=True)
test['Sex'] = x

x = pd.get_dummies(test['Embarked'],drop_first=True)
test = pd.concat([test,x],axis=1)

x = pd.get_dummies(test['Pclass'],drop_first=True)
test = pd.concat([test,x],axis=1)

# Also, going to drop all values not needed for predictions.
test.drop(["PassengerId","Pclass","Name","Ticket","Embarked"],axis=1,inplace=True)
print(test)

###############################################################################
# All the previous steps are going to be repeated for the training set of data.

train = train.drop(['Cabin'],axis=1)
train.dropna(subset=['Fare','Embarked'],axis=0,inplace=True)
train = train.reset_index()
train.drop(['index'],axis=1,inplace=True)
train.fillna(inplace=True,method='pad')
x = pd.get_dummies(train['Sex'],columns='Male',drop_first=True)
train['Sex'] = x
x = pd.get_dummies(train['Embarked'],drop_first=True)
train = pd.concat([train,x],axis=1)
x = pd.get_dummies(train['Pclass'],drop_first=True)
train = pd.concat([train,x],axis=1)
train.drop(["PassengerId","Pclass","Name","Ticket","Embarked"],axis=1,inplace=True)


###############################################################################
# Data is now sufficiently tidy and usable. Prediction needs to be made.
# Reminder - train is to be used for training, and has the 'Survived' values.
# test needs to be used for final predictions.


feat = train.drop('Survived',axis=1)
tar = train['Survived']

train_feat, test_feat, train_tar, test_tar = train_test_split(feat, tar)

predictor = LogisticRegression()
predictor.fit(train_feat,train_tar)

# to check metrics on accuracy
#predictions = predictor.predict(test_feat)
#from sklearn.metrics import classification_report
#print(classification_report(test_tar, predictions))


###############################################################################
# Actually making the predicitons and assigning a column in the dataset.
predictions = predictor.predict(test)
dupe['Survived'] = predictions
print(dupe)


input('Press key to continue.')