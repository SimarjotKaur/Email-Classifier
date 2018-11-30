EMAIL-CLASSIFICATION:
--------------------
This is a text classification project which is a multi-class classification. Various classifiers are trained and tested using Python. It includes the classification of emails based on their content into three categories: Normal, Spam and Fraud.

Various classifiers including Support Vector Machine (SVM), K-Nearest Neighbour, Multinomial Naïve Bayes, Decision Tree, Logistic Regression, SVM with Stochastic Gradient Descent classifier (SGD-SVM) and Logistic Regression with Stochastic Gradient Descent classifier (SGD-LR) are trained on features extracted using TF-IDF vectorizer.
Further, ensemble classifiers including Random Forest (RF), AdaBoost, Bagging (BGC), Extra Trees and, Vote on various classifier combinations are trained in a similar manner.
Also, the effect of stemming on the model performance is observed. Additionally, classifiers are trained on the features extracted using Count Vectorizer. 

Finally, all the models are evaluated based on standard evaluation metrics: Accuracy, Precision, Recall, F-score and Confusion Matrix. It is observed that Vote on SVM, BGC and RF outperform all the models, followed by SGD-SVM, trained on TF-IDF features without stemming.

PREREQUISITES:
--------------
<ul>
   <Li> Pyhton 3.x. </li>
  <Li> Libraries: </li>
  <ul>
    <Li> Pandas </li>
    <Li> Sklearn </li>
    <Li> Nltk </li>
    <Li> Numpy </li>
    <Li> Matplotlib </li>
    <Li> String </li>
    <Li> re </li>
    <Li> Random </li>
  </Ul>
</Ul>

CODE BRIEF:
----------
The entire coding is done in Python3.5 which was executed in Spyder which is a part of Anaconda3. There are two python files ‘Extract_email.py’ and ‘Email_Classification.py’ which involves the process of Data Extraction and, Text processing and classification respectively.

I. Extract_email.py:
This file involves the process of Data Extraction. In this, 1000 fraud emails from the ‘fradulent_emails.txt’ file containing 4075 emails are extracted and, 1000 emails for each Spam and Normal category are extracted from ‘emails.csv’ file that contains 5730 emails which is a combination of both Spam and Normal emails. Finally, all the extracted emails are concatenated into one csv file. This csv file contains the final dataset that contains 3000 emails with 1000 emails for each category.

NOTE: For the proper execution of the code, update the paths for:
a. the final csv file to be created (‘final_dataset.csv’).
b. fradulent_emails.txt file (file containing fraud emails).
c. emails.csv file (file containing spam and normal emails).

II. Email_Classification.py:
This file involves the complete process of email processing and classification:
1. Data Preprocessing:
Functions are created for the removal of punctuation and stopwords. Another function is created for stemming of the content.
In order to extract the relevant features 2 vectors were used: TF-IDF vectors and Count Vectors. First, the entire process of classification is performed by the features created using TF-IDF and then the features created by Count- Vectorizer are processed and observed. Then the features are split into train and test set in the ratio 7:3 respectively.

2. Text Classification:
Various classifers are trained on the features extracted above and then, their performance is observed. Before the training of the models, Parameter Tuning is performed to identify the optimum parameters for each classifier.

3. Evaluation Metrics:
The classifiers are evaluated on the basis of:
<ul>
   <li> Accuracy</li>
   <li> Confusion Matrix</li>
   <li> Precision, Recall and F-Score</li>
</ul>

NOTE:
For the execution of the code, change the path for final_dataset.csv that was created in the previous step for the variable ‘input_dataset’.
