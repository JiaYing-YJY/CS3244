"""
CS3244 Machine Learning Project
Multi Layer Perceptron Classifier v2

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
from imblearn.over_sampling import RandomOverSampler 
print("Modules Loaded",flush=True)

#_______________________________
# Defining Global Variables
filename = '/Users/liangzhengxin/Library/Mobile Documents/com~apple~CloudDocs/Grand Unified Theory/Y3S1 NUS 2022/CS3244 Machine Learning/Project/Datasets/Energy/train/fixed_data_train.csv'


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
ROS = RandomOverSampler(random_state=42) # Initialise Oversampler
X_train, y_train = ROS.fit_resample(X_train, y_train) # Oversample Training Data

print("Training Data", len(X_train),len(y_train),flush=True)


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



    predicted_target = MLP.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predicted_target)
    balanced_accuracy = metrics.balanced_accuracy_score(y_test, predicted_target)
    precision = metrics.precision_score(y_test, predicted_target)
    recall = metrics.recall_score(y_test, predicted_target)
    f1 = metrics.f1_score(y_test, predicted_target)
    f2 = metrics.fbeta_score(y_test, predicted_target, beta=2.0)
    roc_auc = metrics.roc_auc_score(y_test, predicted_target)
    confusion_matrix = metrics.confusion_matrix(y_test, predicted_target)

    print(f"Possible Output: {set(predicted_target)}",flush=True)
    print(f"Accuracy: {accuracy}")
    print(f"Balanced Accuracy: {balanced_accuracy}",flush=True)
    print(f"Precision: {precision}",flush=True)
    print(f"Recall: {recall}",flush=True)
    print(f"F1 score: {f1}",flush=True)
    print(f"F2 score: {f2}",flush=True)
    print(f"ROC AUC: {roc_auc}",flush=True)
    print(confusion_matrix,flush=True)

    return (accuracy,balanced_accuracy,precision,recall,f1,f2,roc_auc,confusion_matrix[0,0],confusion_matrix[0,1],confusion_matrix[1,0],confusion_matrix[1,1])

with open('/Users/liangzhengxin/Downloads/varyingMLP-v2.csv','a') as file:
    csv.writer(file).writerow(['x','y','accuracy','balanced_accuracy','precision','recall','f1','f2','roc_auc', 'cm00', 'cm01', 'cm10', 'cm11'])
    file.close()

hidden_layer_sizes = ()
for y in range(1,44):
    for x in range(1,44):
        hidden_layer_sizes = (x,y)
        print("--------------------------------------------",flush=True)
        accuracy,balanced_accuracy,precision,recall,f1,f2,roc_auc, cm00, cm01, cm10, cm11 = main(hidden_layer_sizes)
        with open('/Users/liangzhengxin/Downloads/varyingMLP-v2.csv','a') as file:
            csv.writer(file).writerow([x,y,accuracy,balanced_accuracy,precision,recall,f1,f2,roc_auc, cm00, cm01, cm10, cm11])
            file.close()
        print("\n\n--------------------------------------------",flush=True)