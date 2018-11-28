Various classifiers are trained and tested using Python. It includes the classification of emails into three categories: Normal, Spam and Fraud.

Various classifiers including Support Vector Machine (SVM), K-Nearest Neighbour, Multinomial Naïve Bayes, Decision Tree, Logistic Regression, SVM with Stochastic Gradient Descent classifier (SGD-SVM) and Logistic Regression with Stochastic Gradient Descent classifier (SGD-LR) are trained on features extracted using TF-IDF vectorizer. Further, ensemble classifiers including Random Forest (RF), AdaBoost, Bagging (BGC), Extra Trees and, Vote on various classifier combinations are trained in a similar manner. Also, the effect of stemming on the model performance is observed. Additionally, classifiers are trained on the features extracted using Count Vectorizer. 

Finally, all the models are evaluated based on standard evaluation metrics: Accuracy, Precision, Recall, F-score and Confusion Matrix. It is observed that Vote on SVM, BGC and RF outperform all the models, followed by SGD-SVM, trained on TF-IDF features without stemming.

CODE BRIEF:
----------
The entire coding is done in Python3.5 which was executed in Spyder which is a part of Anaconda3. There are two python files ‘Extract_email.py’ and ‘Email_Classification.py’ which involves the process of Data Extraction and, Text processing and classification respectively.

I. Extract_email.py
This file involves the process of Data Extraction. In this, 1000 fraud emails from the ‘fradulent_emails.txt’ file containing 4075 emails are extracted and, 1000 emails for each Spam and Normal category are extracted from ‘emails.csv’ file that contains 5730 emails which is a combination of both Spam and Normal emails. Finally, all the extracted emails are concatenated into one csv file.

NOTE: For the proper execution of the code, update the paths for:
a. the final csv file to be created (‘final_dataset.csv’).
b. fradulent_emails.txt file (file containing fraud emails).
c. emails.csv file (file containing spam and normal emails).

II. Email_Classification.py
This file involves the complete process of email processing and classification.

NOTE:
For the execution of the code, change the path for final_dataset.csv that was created in the previous step for the variable ‘input_dataset’.
