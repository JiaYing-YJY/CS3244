"""
CS3244 Machine Learning Project
Multi Layer Perceptron Classifier

Liang Zhengxin
Team 05.
"""

# Importing Required Modules
import csv
import math
import joblib
import random
import numpy as np 
import pandas as pd 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from statistics import mean
from sklearn.model_selection import cross_val_score
print("Modules Loaded",flush=True)

#_______________________________
# Defining Global Variables
filename = '/home/svu/e0540423/CS3244/dataset/fixed_data_train.csv'


#_______________________________

# Load the csv file using Pandas
df = pd.read_csv(filename, sep=',', low_memory=False) # Separate on commas
print("Dataframe Loaded",flush=True)


# Data Cleaning
del df['Unnamed: 0'] # Delete Index of CSV file
del df['client_id']  # Delete client_id as it is a string and can't be used as a feature


# Separating Training and Testing Data
# Partion the features from the target to predict
df_features = df[df.columns[df.columns != 'target']].copy() # get columns that are not 'target'; this our features
df_target = df['target'].copy() # get the column named 'target'; this is our label

# (random_state): we use a fixed random seed so we get the same results every time.
X_train, X_test, y_train, y_test = train_test_split(df_features, df_target, test_size=0.3, random_state=1) ## RANDOM STATE DETERMINED HERE

print ("Number of training instances: ", len(X_train), "\nNumber of test instances: ", len(X_test), "\nTotal number of instances: ",len(df_target),flush=True)


# Imbalance Data 
## Training Data
not_fraud = y_train[y_train == 0].index #Get index of rows that are not fraud
fraud = y_train[y_train == 1].index # #Get index of rows that are fraud
fraud = np.random.choice(fraud, len(not_fraud), replace=True) # Randomly pick not fraud rows without replacement until the len is the same as fruad
X_train = pd.concat([X_train.loc[not_fraud],X_train.loc[fraud]]) 
y_train = pd.concat([y_train.loc[not_fraud],y_train.loc[fraud]])

print("Training Data", len(X_train),len(y_train),flush=True)

## Testing Data
not_fraud = y_test[y_test == 0].index #Get index of rows that are not fraud
fraud = y_test[y_test == 1].index # #Get index of rows that are fraud
fraud = np.random.choice(fraud, len(not_fraud), replace=True) # Randomly pick not fraud rows without replacement until the len is the same as fruad
X_test = pd.concat([X_test.loc[not_fraud],X_test.loc[fraud]]) 
y_test = pd.concat([y_test.loc[not_fraud],y_test.loc[fraud]])

print("Testing Data",len(df_features),len(df_target),flush=True)

## Raw Data
not_fraud = df_target[df_target == 0].index #Get index of rows that are not fraud
fraud = df_target[df_target == 1].index # #Get index of rows that are fraud
fraud = np.random.choice(fraud, len(not_fraud), replace=True) # Randomly pick not fraud rows without replacement until the len is the same as fruad
df_features = pd.concat([df_features.loc[not_fraud],df_features.loc[fraud]]) 
df_target = pd.concat([df_target.loc[not_fraud],df_target.loc[fraud]])

print("Raw Data",len(df_features),len(df_target),flush=True)

print("Number of Features: ",len(X_train.columns),flush=True)

# MAIN FUNCTION #
def main(hidden_layer_sizes):
    # Builidng the classifier
    MLP = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=1000, verbose=0, random_state=1)
    print(f"Training Started   Hidden Layer: {hidden_layer_sizes}",flush=True)

    # Training
    MLP.fit(X_train, y_train)

    # Validation
    print("Training set score", MLP.score(X_train, y_train),flush=True)
    print("Testing set score", MLP.score(X_test, y_test),flush=True)

    # Saving the Trained Model
    # joblib.dump(MLP, "/home/svu/e0540423/CS3244/ZX-MLPClassifier-TrainedModel.pkl") 
    # print("Model Saved",flush=True)

    # Cross Validation
    # scores = cross_val_score(MLP, df_features, df_target, cv=5)
    # print(scores,flush=True)
    # print(mean(scores),flush=True)


    predicted_target = MLP.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predicted_target)
    balanced_accuracy = metrics.balanced_accuracy_score(y_test, predicted_target)
    precision = metrics.precision_score(y_test, predicted_target)
    recall = metrics.recall_score(y_test, predicted_target)
    f1 = metrics.f1_score(y_test, predicted_target)
    roc_auc = metrics.roc_auc_score(y_test, predicted_target)
    confusion_matrix = metrics.confusion_matrix(y_test, predicted_target)

    print(f"Possible Output: {set(predicted_target)}",flush=True)
    print(f"Accuracy: {accuracy}")
    print(f"Balanced Accuracy: {balanced_accuracy}",flush=True)
    print(f"Precision: {precision}",flush=True)
    print(f"Recall: {recall}",flush=True)
    print(f"F1 score: {f1}",flush=True)
    print(f"ROC AUC: {roc_auc}",flush=True)
    print(confusion_matrix,flush=True)

    return (accuracy,balanced_accuracy,precision,recall,f1,roc_auc,confusion_matrix[0,0],confusion_matrix[0,1],confusion_matrix[1,0],confusion_matrix[1,1])

with open('/home/svu/e0540423/CS3244/varyingMLP.csv','a') as file:
    csv.writer(file).writerow(['x','y','accuracy','balanced_accuracy','precision','recall','f1','roc_auc', 'cm00', 'cm01', 'cm10', 'cm11'])
    file.close()

hidden_layer_sizes = ()
for y in range(1,44):
    for x in range(1,44):
        hidden_layer_sizes = (x,y)
        print("--------------------------------------------",flush=True)
        accuracy,balanced_accuracy,precision,recall,f1,roc_auc, cm00, cm01, cm10, cm11 = main(hidden_layer_sizes)
        with open('/home/svu/e0540423/CS3244/varyingMLP.csv','a') as file:
            csv.writer(file).writerow([x,y,accuracy,balanced_accuracy,precision,recall,f1,roc_auc, cm00, cm01, cm10, cm11])
            file.close()
        print("\n\n--------------------------------------------",flush=True)