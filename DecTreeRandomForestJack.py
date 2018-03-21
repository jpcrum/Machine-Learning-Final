# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 16:27:48 2017

@author: sjcrum
"""

##### 1) Import Packages #####

import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report as cr


##### 2) Importing Training and Testing Data #####

import os
os.chdir('C:/Users/sjcrum/Documents/GitHub/Machine-Learning-Final-GW-Navy')


#Original training and testing data
train_original = pd.read_csv("train.csv")
test_original = pd.read_csv("test.csv")

#Normalized training and testing data
train = pd.read_csv("train_nmlz.csv")
test = pd.read_csv("test_nmlz.csv")

#selecting feature columns for training and testing and assigning target column
train_feature_cols = train.iloc[:, 2:]
X_train = train_feature_cols
y_train = train.diagnosis

test_feature_cols = test.iloc[:, 2:]
X_test = test_feature_cols
y_test = test.diagnosis

#Mapping B to 0 and M to 1 in in diagnosis (target) column
y_train = y_train.map({'B':0, 'M':1})
y_test = y_test.map({'B':0, 'M':1})
y_train = pd.to_numeric(y_train)
y_test = pd.to_numeric(y_test)





##### 3)Create Decision Tree ######

# Assign variables as empty lists
train_accuracy = []
test_accuracy = []
maxdepth = []

#Test Decision Tree model with different maximum depths
for n in range(2,11):
    model = DecisionTreeClassifier(max_depth = n, random_state = 0)
    model.fit(X_train, y_train)
    #10-fold cross validation for the Decision Tree accuracy on the training data
    train_acc = np.mean(cross_val_score(model, X_train, y_train, cv=10))
    test_acc = model.score(X_test, y_test)
    
    #append data to above lists
    train_accuracy.append(train_acc)
    test_accuracy.append(test_acc)    
    maxdepth.append(n)
       
#Plot training and testing accuracy vs accuracy 
plt.plot(maxdepth,train_accuracy, label = "Training Accuracy")
plt.plot(maxdepth,test_accuracy, label = "Testing Accuracy")
plt.title("Decision Tree Accuracy with Varying Maximum Depths")
plt.xlabel("Maximum Depth")
plt.ylabel("Accuracy")
#Vertical line to show maximum accuracy
plt.axvline(x=6, color = "silver", linestyle='dashed')
plt.legend()
plt.show()

dtree = DecisionTreeClassifier(criterion='gini', max_depth=6, random_state = 0)


#Function that plots feature importance
def plot_features(model, model_name):
    
    model.fit(X_train, y_train)
    
    #Feature Importance
    features = pd.DataFrame({'feature':X_train.columns.values, 'importance':model.feature_importances_})
    features_sorted = features.sort_values(by = ['importance'], ascending = False)
    print(features_sorted)
        
    #Plotting feature importance
    plt.bar(features_sorted.feature, features_sorted.importance)
    plt.title("Feature Importance in {}".format(model_name))
    plt.xticks(fontsize =  8, rotation = 80)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()
    

plot_features(dtree, "Decision Tree")

#set max depth to optimal depth
dtree = DecisionTreeClassifier(criterion='gini', max_depth=6, random_state = 0)

#Function that plots classication accuracy, sensitivity, and specificity
# and plots 5-fold cross-validated accuracy 
def plot_feature_length(model, model_name):
    
    total_features = []
    accuracy = []
    sense = []
    spec = []
    cv = []
    
    for n in range(1, 31):
        
        model.fit(X_train, y_train)
        
        features = pd.DataFrame({'feature':X_train.columns.values, 'importance':model.feature_importances_})
        features_sorted = features.sort_values(by = ['importance'], ascending = False)
        
        important_features = features_sorted['feature'].head(n)
        X_train_feat = X_train.loc[:, important_features]
        X_test_feat = X_test.loc[:, important_features]
        
        #Fitting the smaller model
        model.fit(X_train_feat, y_train)
        
        #Predict target values
        y_predict = model.predict(X_test_feat)
        
        #Accuracy 
        acc_small = accuracy_score(y_test, y_predict)
        accuracy.append(acc_small)
        total_features.append(n)
        
        #Confusion Table
        conf = pd.DataFrame(confusion_matrix(y_test, y_predict),
        columns=['Predicted Benign', 'Predicted Malignant'],
        index=['True Benign', 'True Malignant'])
        
        TP = conf.iloc[1,1]
        TN = conf.iloc[0,0]
        FP = conf.iloc[0,1]
        FN = conf.iloc[1,0]
        
        #Sensitivity calculation
        sensitivity = recall_score(y_test, y_predict)
        #Specificity calculation
        specificity = TN / (TN + FP)
        
        sense.append(sensitivity)
        spec.append(specificity)
        #Cross Validation Score
        cv_acc = np.mean(cross_val_score(model, X_train_feat, y_train, cv = 5))
        cv.append(cv_acc)
    
    plt.plot(total_features, accuracy, label = "Classification Accuracy")
    plt.plot(total_features, sense, label = "Sensitivity")
    plt.plot(total_features, spec, label = "Specificity")
    plt.grid(True, linestyle='dashed', c = "lightgrey")
    plt.title("Accuracy, Sensitivity, and Specificity of {}".format(model_name))
    plt.xlabel("Number of Features")
    plt.ylabel("Metric Score")
    plt.legend()
    plt.show()
       
    plt.plot(total_features, cv)
    plt.grid(True, linestyle='dashed', c = "lightgrey")
    plt.title("Cross-Validated Accuracy of {}".format(model_name))
    plt.xlabel("Number of Features")
    plt.ylabel("Classification Accuracy")
    plt.show()

plot_feature_length(dtree, "Decision Tree")


#Function to get optimal number of features for the decision tree model
def optimal_features_scores(model, n):
    
    model.fit(X_train, y_train)
    
    #Get top n features
    features = pd.DataFrame({'feature':X_train.columns.values, 'importance':model.feature_importances_})
    features_sorted = features.sort_values(by = ['importance'], ascending = False)
    
    #Dataset with only top n features
    important_features = features_sorted['feature'].head(n)
    X_train_feat = X_train.loc[:, important_features]
    X_test_feat = X_test.loc[:, important_features]
    
    model.fit(X_train_feat, y_train)
    
    y_predict = model.predict(X_test_feat)
    
    acc = accuracy_score(y_test, y_predict)
    conf_tree = pd.DataFrame(confusion_matrix(y_test, y_predict),
        columns=['Predicted Benign', 'Predicted Malignant'],
        index=['True Benign', 'True Malignant'])
    print(conf_tree, "\n")
    print("Accuracy: ", acc)
    print("\n")
    #Other metrics to show model quality
    target_names = ['Benign', 'Malignant']
    print(cr(y_test, y_predict, target_names=target_names))


optimal_features_scores(dtree, 6)

#Making the optimal Decision Tree
dtree = DecisionTreeClassifier(criterion='gini', max_depth=6, random_state = 0)
dtree.fit(X_train, y_train)

features_dec = pd.DataFrame({'feature':X_train.columns.values, 'importance':dtree.feature_importances_})
features_sorted_dec = features_dec.sort_values(by = ['importance'], ascending = False)

important_features_dec = features_sorted_dec['feature'].head(6)
X_train_dec = X_train.loc[:, important_features_dec]
X_test_dec = X_test.loc[:, important_features_dec]

dtree_opt = DecisionTreeClassifier(criterion='gini', max_depth=6, random_state = 0)
dtree_opt.fit(X_train_dec, y_train)

print("Accuracy: ", np.mean(cross_val_score(dtree_opt, X_train_dec, y_train, cv = 5)))
optimal_features_scores(dtree_opt, 6)

#Export visualization of decision tree

from sklearn.tree import export_graphviz
export_graphviz(dtree_opt,out_file="opttree.dot",class_names=['Benign',"Malignant"],
    feature_names=X_train_dec.columns.values,impurity=False,filled=True)





##### Random Forest Model ######
    
#Assign feature labels
feat_labels = X_train.columns

#Assign variables to empty lists
train_forest_accuracy = []
test_forest_accuracy = []
maxforestdepth = []

#Testing Random Forests with varying max_depths
for n in range(1,31):
    forest = RandomForestClassifier(criterion='gini', max_depth=n, n_estimators = 1000, random_state = 0)
    #fit the random forest to the training data 
    forest.fit(X_train, y_train)
    
    #Accuracy scores of Random Forest for training and testing
    #Traded computing speed for lack of cross validation
    train_forest_acc = forest.score(X_train, y_train)
    test_forest_acc = forest.score(X_test, y_test)

    #Append to above lists
    train_forest_accuracy.append(train_forest_acc)
    test_forest_accuracy.append(test_forest_acc)    
    maxforestdepth.append(n)
       
#Plot testing and training accuracy vs Maximum Depth
plt.plot(maxforestdepth,train_forest_accuracy, label = "Training Accuracy")
plt.plot(maxforestdepth,test_forest_accuracy, label = "Testing Accuracy")
plt.title("Random Forest Accuracy with Varying Maximum Depths")
plt.xlabel("Maximum Depth")
plt.ylabel("Accuracy")
plt.axvline(x=8, color = "silver", linestyle='dashed')
plt.legend()
plt.show()

#Assign variables to empty lists
train_forest_accuracy_est = []
test_forest_accuracy_est = []
forest_est = []

#Forest size estimates
est = [100,500,800, 900, 1000, 1100, 1200, 1500, 2000]

for n in est:
    forest = RandomForestClassifier(criterion='gini', max_depth=8, n_estimators = n, random_state = 0)
    #fit the random forest to the training data 
    forest.fit(X_train, y_train)
    
    #Accuracy scores of Random Forest for training and testing
    #Traded computing speed for lack of cross validation
    train_forest_acc = forest.score(X_train, y_train)
    test_forest_acc = forest.score(X_test, y_test)

    #Append to above lists
    train_forest_accuracy_est.append(train_forest_acc)
    test_forest_accuracy_est.append(test_forest_acc)    
    forest_est.append(n)
       
#Plot testing and training accuracy vs Number of Trees
plt.plot(forest_est,train_forest_accuracy_est, label = "Training Accuracy")
plt.plot(forest_est,test_forest_accuracy_est, label = "Testing Accuracy")
plt.title("Random Forest Accuracy with Varying Number of Trees")
plt.xlabel(" Number of Trees")
plt.ylabel("Accuracy")
plt.axvline(x=1000, color = "silver", linestyle='dashed')
plt.legend()
plt.show()



#Fit the Random Forest model with all features with best max depth and number of trees
rf =  RandomForestClassifier(criterion='gini', max_depth=8, n_estimators = 1000, random_state = 0)
rf.fit(X_train, y_train)

#Plot best features
plot_features(rf, "Random Forest")


#Predict target values
y_predict_rf = rf.predict(X_test)

#User defined function above
plot_feature_length(rf, "Random Forest")

#User defined function above
optimal_features_scores(rf, 15)



#Making the optimal Random Forest
rf =  RandomForestClassifier(criterion='gini', max_depth=8, n_estimators = 1000, random_state = 0)
rf.fit(X_train, y_train)

features_rf = pd.DataFrame({'feature':X_train.columns.values, 'importance':rf.feature_importances_})
features_sorted_rf = features_rf.sort_values(by = ['importance'], ascending = False)

important_features_rf = features_sorted_rf['feature'].head(15)
X_train_small = X_train.loc[:, important_features_rf]
X_test_small = X_test.loc[:, important_features_rf]

rf_small =  RandomForestClassifier(criterion='gini', max_depth=15, n_estimators = 1000, random_state = 0)
rf_small.fit(X_train_small, y_train)

print("Accuracy: ", np.mean(cross_val_score(rf_small, X_train_dec, y_train, cv = 5)))

#User defined function above
plot_features(rf_small, "Random Forest")

#User defined function above
plot_feature_length(rf_small, "Random Forest")

#User defined function above
optimal_features_scores(rf_small, 15)


#Calculated and plot ROC Curve and AUC
def plot_roc(model, feature_X, feature_y, model_name): 
    model.fit(feature_X, feature_y)
    y_prob = model.predict_proba(feature_X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, linewidth = 5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title("ROC Curve for Cancer Prediction by {}".format(model_name))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.show()
    
    auc_score = roc_auc_score(y_test, y_prob)
    print(auc_score)
    return(auc_score)

#Calculating the AUC (Area Under the Curve) Score

plot_roc(dtree_opt, X_test, y_test, "Decision Tree")
plot_roc(rf_small, X_test_small, y_test, "Random Forest Small")








###### Plotly Graph #######

## Making a plotly graph to show category separation from only top three features

#Top three features as shown by random forest
a = train_original["concave.points_mean"]
b = train_original["perimeter_worst"]
c = train_original["texture_worst"]
 
#Importing packages and setting credentials
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
plotly.tools.set_credentials_file(username='jackcrum', api_key='CueWpz8W4OROnmTR9mko')

#Mapping colors onto original training data
size_mapping = {'B': "blue",'M': "red"}
train_original['color'] = train_original['diagnosis'].map(size_mapping)

#Create the 3D scatter plot
trace1 = go.Scatter3d(x=a, y=b, z=c, mode='markers', marker=dict(size=4, color=train_original["color"], opacity=0.8))

#Set the layout
data = [trace1]
layout = go.Layout(title='Plot Title', xaxis=dict(title='Growth Mean Radius'), yaxis=dict(title='Growth Mean Texture'))

#Plot the figure to my plotly account
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='3d-scatter-tumor')

