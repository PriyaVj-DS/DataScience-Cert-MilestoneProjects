# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 23:29:19 2020
@author: Priya Asokan

Smart Reply: Automated response suggestions for chat messages
Since the project asked for binary classification, I tried to correlate between user's text and reply size.

Ubuntu Dialogue corpus Data Set - This data set is building dialogue systems, where a human can have a natural-feeling conversation with a virtual agent.
This script prepares the data for training a predictive model to an incoming user text.
 
The following packages are used:
    1. pandas
    2. numpy
    3. sklearn
    4. matplotlib
    5. gensim - for doc2vec to convert text to a vector

Please install gensim to run my file. I used anaconda to install gensim

If it doesn't work, please recreate my environment as detailed below
---------------------------------------
Steps to recreate my environment as is:
---------------------------------------
conda env create -f PriyaAsokanCondaEnvironment.yml --name priyaasokanenv
conda activate priyaasokanenv
python PriyaAsokan-M03-DataModelFinal.py


"""
# Import the necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib
import gensim.utils
import gensim.models



print("*************Preparation of Data******************")
#1. Source citation of data - Ubuntu dialog corpus dataset
#https://www.kaggle.com/rtatman/ubuntu-dialogue-corpus

#Download the data
#Reduced the sample here.
url = "https://drive.google.com/uc?id=1hlyRt7aWHR_mUNQ3N-HJgoHD_Njgkmm-&export=download"

#2 & 3. Data Read - Number of observations- 5000 & Attributes - 6
df = pd.read_csv(url, nrows=5000)
#view the dataset
#print(df)

#view the first five rows from the dataset
df.head()

#4. View data type of each column in the dataset
df.dtypes


#Add a new column 'numberOfTokens' that counts the number of words in each row in 'text' column.
df.loc[:,'numberOfTokens'] = df.text.apply(lambda x: len(str(x).split(' ')) if str(x) != 'nan' else np.NaN)

#Create a new Data Set by adding 'Message' and 'Reply' columns to the dataset by looking at each line.
newdf = pd.DataFrame(columns = ['folder', 'dialogueID', 'date', 'from','to','text', 'reply', 'textTokens', 'replyTokens'])
for i in range(1, len(df)):
    if(df.loc[i, 'to'] == df.loc[i-1, 'from']) and (df.loc[i, 'dialogueID'] == df.loc[i-1, 'dialogueID']):
        new_row = {'folder':df.loc[i, 'folder'],  'dialogueID':df.loc[i, 'dialogueID'], 'date':df.loc[i, 'date'], 'from':df.loc[i, 'from'], 'to': df.loc[i, 'to'], 'text':df.loc[i-1, 'text'], 'reply':df.loc[i, 'text'], 'textTokens':df.loc[i-1, 'numberOfTokens'], 'replyTokens':df.loc[i, 'numberOfTokens']}
        newdf = newdf.append(new_row, ignore_index=True)
newdf.head(5)

#5. Distribution of Numerical Attributes - 'textTokens' attributes
plt.title("Distribution of Numerical column 'Text Tokens'")
plt.hist(newdf.loc[:,'textTokens'])
plt.show()

#5. Distribution of Numerical Attributes -'ReplyTokens' attributes
plt.title("Distribution of Numerical column 'Reply Tokens'")
plt.hist(newdf.loc[:,'replyTokens'])
plt.show()

#6 Distribution of categorical attribute is performed below line number - 177

#7.Comment on each attribute
#Folder - Contains the folder number for each dialogue. It is of integer data type.
#DialogueID - contains the dialogue number in the folder. It is of object data type
#Date - DataTime of the conversation. It is of object data type.
#From - From Person name
#To - To person name
#Text - Text question
#Reply - Text Answer
#TextTokens - number of words in each row of text attribute
#ReplyTokens -  number of words in each row of reply attribute

#8. Remove Missing Data is performed below in line number 293

#9. Remove Outliers from 'textTokens' column.
def remove_outliers(df, colName):
    array = df.loc[:, colName]
    #The high limit is the cutoff for good values
    LimitHi = np.mean(array) + 2* np.std(array)
    #LowLImit is the cut off for good values
    LimitLo = np.mean(array) - 2 * np.std(array)
    #create a flag for values within limits
    FlagGood = (array<=LimitHi) & (array>=LimitLo)
    #Returns array without outliers
    return df.loc[FlagGood,:]
    
#Function Call to remove outliers in textTokens and print the array without outliers
print('TextTokens without outliers')
newdf = remove_outliers(newdf, 'textTokens')

#10. Replace missing numeric data with the median of non Nan values.
#newdf.loc[:,'textTokens'] = pd.to_numeric(newdf.loc[:,'textTokens'],errors="coerce")
HasNan = np.isnan(newdf.loc[:,'textTokens'])
newdf.loc[HasNan,'textTokens'] = np.nanmedian(newdf.loc[~HasNan,"textTokens"])

HasNan = np.isnan(newdf.loc[:,'replyTokens'])
newdf.loc[HasNan,'replyTokens'] = np.nanmedian(newdf.loc[~HasNan,"replyTokens"])

#Bin numeric column - 'replyTokens' (at least 1 column).
NumberOfBins = 3
Max = np.max(newdf.loc[:,'replyTokens'])
Min = np.min(newdf.loc[:,'replyTokens'])
BinWidth = (Max - Min)/NumberOfBins
MinBin1 = float('-inf')
MaxBin1 = Min + 1 * BinWidth
MaxBin2 = Min + 2 * BinWidth
MaxBin3 = float('inf')
print([Min + i * BinWidth for i in range(1,NumberOfBins)])

print(" Bin 1 is greater than", MinBin1, "up to", MaxBin1)
print(" Bin 2 is greater than", MaxBin1, "up to", MaxBin2)
print(" Bin 3 is greater than", MaxBin2, "up to", MaxBin3)

xBinnedEqW = np.empty(len(newdf.loc[:,'replyTokens']), object)

# The conditions at the boundaries should consider the difference 
# between less than (<) and less than or equal (<=) 
# and greater than (>) and greater than or equal (>=)
# bin the data into three groups -'short', 'medium', 'long'
xBinnedEqW[(newdf.loc[:,'replyTokens'] > MinBin1) & (newdf.loc[:,'replyTokens'] <= MaxBin1)] = "Short"
xBinnedEqW[(newdf.loc[:,'replyTokens'] > MaxBin1) & (newdf.loc[:,'replyTokens'] <= MaxBin2)] = "Medium"
xBinnedEqW[(newdf.loc[:,'replyTokens'] > MaxBin2) & (newdf.loc[:,'replyTokens'] <= MaxBin3)] = "Long"
print(" replyTokens column is binned into 3 equal-width bins:", xBinnedEqW)
newdf.loc[:, 'replycategory'] = xBinnedEqW
newdf.head()

#11. Decoding - creating a new column 'replycatogoryID'
newdf.loc[newdf.loc[:,'replycategory'] == "Short", "replycatogoryID"] = 1
newdf.loc[newdf.loc[:,'replycategory'] == "Medium", "replycatogoryID"] = 2
newdf.loc[newdf.loc[:,'replycategory'] == "Long", "replycatogoryID"] = 3
#Convert the data type of the column from object to float
newdf["replycatogoryID"] = newdf["replycatogoryID"].astype(object).astype(float)
#print(newdf.dtypes)
#Replacing Nan values with median of non nan values
HasNan = np.isnan(newdf.loc[:,'replycatogoryID'])
newdf.loc[HasNan,'replycatogoryID'] = np.nanmedian(newdf.loc[~HasNan,"replycatogoryID"])

#12. Consolidating empty & 'Short'  values of categorical data column into  one 'Samll' value
newdf.loc[newdf.loc[:, 'replycategory'].isnull(), 'replycategory'] = "Small"
newdf.loc[newdf.loc[:, 'replycategory'] == "Short", 'replycategory'] = "Small"

#13 One-hot encode categorical data with at least 3 categories (at least 1 column).
newdf.loc[:, "Short"] = (newdf.loc[:, 'replycategory'] == "Small").astype(int)
newdf.loc[:, "Medium"] = (newdf.loc[:, 'replycategory'] == "Medium").astype(int)
newdf.loc[:, "Long"] = (newdf.loc[:, 'replycategory'] == "Long").astype(int)
newdf.head()

#6 Distribution of Cateogory Attributes - 'replyCategory' attributes
plt.title("Distribution of Category column 'ReplyCategory'")
plt.hist(newdf.loc[:,'replycategory'])
plt.show()

#14. Min Max - Normalize numeric column -'TextTokens', ReplyTokens,ReplyCategoryID
x = newdf.loc[:,'textTokens']
#calculate offset with min of x values
offset = min(x)
#calculate spread with max of x values
spread = max(x) - min(x)
#calculate the normalized values
newdf.loc[:,'textTokensNorm'] = (x - offset)/spread
print(" The min-max-normalized variable:", newdf.loc[:,'textTokensNorm'])
print(newdf.loc[:,'textTokensNorm'])

#Min Max - Normalize numeric column -'ReplyTokens'
x = newdf.loc[:,'replyTokens']
#calculate offset with min of x values
offset = min(x)
#calculate spread with max of x values
spread = max(x) - min(x)
#calculate the normalized values
newdf.loc[:,'replyTokensNorm'] = (x - offset)/spread
print(" The min-max-normalized variable:", newdf.loc[:,'replyTokensNorm'])
print(newdf.loc[:,'replyTokensNorm'])
# Min Max - Normalize numeric column -'Reply catogoryID'
x = newdf.loc[:,'replycatogoryID']
#calculate offset with min of x values
offset = min(x)
#calculate spread with max of x values
spread = max(x) - min(x)
#calculate the normalized values
newdf.loc[:,'replycatogoryIDNorm'] = (x - offset)/spread
print(" The min-max-normalized variable:", newdf.loc[:,'replycatogoryIDNorm'])
print(newdf.loc[:,'replycatogoryIDNorm'])

#Remove obsolete columns - Removing 'folder' column as it will not be used.
newdf = newdf.drop("folder", axis=1)

print("**************K-Means*****************")
#K-Means clustering between 'Reply tokens' and ' replycatogoryID' column. Both the column values are normalized 
# - replyTokensNorm,replycatogoryIDNorm
x = newdf[['replyTokensNorm','replycatogoryIDNorm']].values
#Choosing the right number of clusters
#calculate Within cluster sum of squares- WCSS
def get_wcss(x):
    wcss = []
    for i in range(1,11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++')
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
    return wcss
#Plot the elbow curve to get the correct number of clusters
def plot_elbow_curve(wcss):
    plt.figure(figsize = (15,10))
    plt.plot(range(1,11), wcss, color = 'blue')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within Cluster Sum of Squares')
    plt.title('Elbow Curve')
    plt.show()
    
# Get and plot the WCSS for Kmeans
wcss = get_wcss(x)
plot_elbow_curve(wcss)

# Looks like 3 can be the optimum number of clusters for the extracted dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++')
y_clusters = kmeans.fit_predict(x)

#visualizing the clusters
plt.figure(figsize = (10,10))
plt.scatter(x[y_clusters == 0, 0], x[y_clusters == 0, 1], s = 100, c = 'red', label = 'Small')
plt.scatter(x[y_clusters == 1, 0], x[y_clusters == 1, 1], s = 100, c = 'blue', label = 'Medium')
plt.scatter(x[y_clusters == 2, 0], x[y_clusters == 2, 1], s = 100, c = 'green', label = 'Long')

# Print labels on the plotter
plt.xlabel('ReplyToken')
plt.ylabel('ReplyCatgoryID')
plt.title('ReplyToken- ReplyCatogoryID Distribution of participants')

# Plot the centroids for each cluster
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'brown', label = 'Centroids')
plt.legend()

print("*****************DocToVec****************")
# Training a Doc2Vec model here so that we can encode natural language text into a vector space.
# Using our own training corpus to train the doc2vec model instead of a predefined model
# I used a doc2vec model here so that i can get better performance and understand different representations
# of same user intent. Other approaches could have been a simple bag of words model.
# Model dataset doesn't have a lot of repetitions, so performance may not be representative.
# To improve the model further, we can train on a larger dataset.
# Since the project asked for binary classification, I tried to correlate between user's text and reply size

# Hyper parameters
vectorSize = 100
epochs = 100

def read_corpus(corpus):
    for i, line in enumerate(corpus):
        tokens = gensim.utils.simple_preprocess(str(line))
        yield gensim.models.doc2vec.TaggedDocument(tokens, [i])
train_corpus = list(read_corpus(df['text']))

# train doc2vec model
model = gensim.models.doc2vec.Doc2Vec(vector_size=vectorSize, min_count=2, epochs=epochs)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

# vectorize each line of text column; using trained model.infer_vector 
textVector = np.array([model.infer_vector(gensim.utils.simple_preprocess(str(textLine))) 
                for textLine in newdf['text']])

# Append columns textVector[0..vectorSize-1] to data frame
for c in range(vectorSize):
    newdf.loc[:, 'textVector' + str(c)] = textVector[:,c]
    
#Remove Missing data
newdf = newdf.dropna()
newdf = newdf.reset_index()

print("*****************Classification Model****************")
#Binary choice Question- Is reply to a text short or not ?
#create a binary column 'ShortReply' based on the replycategory attribute value
newdf['shortReply'] = [1 if x == 'Small' else 0 for x in newdf['replycategory']]
#Training Target
y = newdf.shortReply
#Traing Feature
x = newdf[['textVector' + str(i) for i in range(vectorSize)]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.6)
x_train.head()
x_test.head()
y_train.head()
y_test.head()

# Generic function to fit and evaluate metrics
def metrics(model, modelName):
    model.fit(x_train, y_train)
    print ("predictions for test set:")
    print (model.predict(x_test))
    print ('actual class values:')
    print (y_test)
    
    modelAcc = model.score(x_test,y_test)
    print(" Accuracy of RFC: " + str(modelAcc))
    #
    ##Calculate probability
    modelPredictedValues = model.predict(x_test)
    #  
    #Confusion Matrix from your predicted values
    print ('\nConfusion Matrix and Metrics')
    #Threshold = 0.5 # Some number between 0 and 1
    #print ("Probability Threshold is chosen to be:", Threshold)
    #predictions = (probabilities > Threshold).astype(int)
    CM = confusion_matrix(y_test, modelPredictedValues)
    print ("\n\nConfusion matrix:\n", CM)
    #Get precision, recall & Accuracy
    tn, fp, fn, tp = CM.ravel()
    print ("TP, TN, FP, FN:", tp, ",", tn, ",", fp, ",", fn)
    AR = accuracy_score(y_test, modelPredictedValues)
    print ("Accuracy rate:", np.round(AR, 2))
    P = precision_score(y_test, modelPredictedValues)
    print ("Precision:", np.round(P, 2))
    R = recall_score(y_test, modelPredictedValues)
    print ("Recall:", np.round(R, 2))
    #calculate the ROC curve and it's AUC using sklearn. Present the ROC curve. Present the AUC in the ROC's plot.
    # ROC analysis
    fpr, tpr, th = roc_curve(y_test, modelPredictedValues) # False Positive Rate, True Posisive Rate, probability thresholds
    AUC = auc(fpr, tpr)
    print ("\nTP rates:", np.round(tpr, 2))
    print ("\nFP rates:", np.round(fpr, 2))
    print ("\nProbability thresholds:", np.round(th, 2))
    #Plot ROC curve with AUC value
    plt.rcParams["figure.figsize"] = [8, 8] # Square
    font = {'family' : 'normal', 'weight' : 'bold', 'size' : 18}
    matplotlib.rc('font', **font)
    plt.figure()
    plt.title(modelName + ' ROC Curve')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.plot(fpr, tpr, LW=3, label='ROC curve (AUC = %0.2f)' % AUC)
    plt.plot([0, 1], [0, 1], color='navy', LW=3, linestyle='--') # reference line for random classifier
    plt.legend(loc="lower right")
    plt.show()

print("**********************Naive Bayes classifier***********************")

# Naive Bayes classifier
print ('\n\n\n')
nbc = GaussianNB() # default parameters are fine
metrics(nbc, 'Gaussian Naive Bayes')

print("**********************Random Forest Classifier***********************")
# Random Forest classifier
estimators = 10 # number of trees parameter
mss = 2 # mininum samples split parameter
print ('\n\nRandom Forest classifier\n')
clf = RandomForestClassifier(n_estimators=estimators, min_samples_split=mss) # default parameters are fine
metrics(clf, 'Random Forest')

print("************************* KNN***********************")
# k Nearest Neighbors classifier
print ('\n\nK nearest neighbors classifier\n')
k = 5 # number of neighbors
distance_metric = 'euclidean'
knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
metrics(knn, 'KNN with 5 neighbors')







