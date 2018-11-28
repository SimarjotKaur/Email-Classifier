# -*- coding: utf-8 -*-
"""
Created on Tue May  8 23:47:09 2018

@author: Simar
"""

'''SELF-IMPLEMENTED'''
#importing libraries
import re
import random

class Dataset_Creation:
    
    def __init__(self):
        print('Data Creation Started')
        print('---------------------------')
        
    def generate_csv_file(self, valid_emails):
    	print('\n')
    	print('GENERATE FINAL DATASET')
    	print('----------------------')
    	print('-- ',len(valid_emails),'Emails Found --')
    
    	file = open('F:\WebMining\SimarjotKaur_WM\Code_WM\Datasets/final_dataset.csv','w')
    	file.write('Email,Label,Length\n')	# Header Line
    	for email in valid_emails:
    		file.write('"'+email[1]+'",'+email[0]+','+str(len(email[1]))+'\n') # Added double quotes for csv to read email as one column
    	file.close()
    	print('\n')
    	print('DATSET GENERATED SUCCESSFULLY')
    	print('-----------------------------')
    
    def extract_content(self, email):
        # Most of the emails start with the following objects #
    	start_objs = ['DEAR FRIEND','DEAR SIR','DEAR PARTNER','MY DEAR','KIND ATTENTION','HELLO,','DEAR,','ATTN: SIR','HELLO FRIEND,','DEAR BELOVED','DEAREST BELOVED','DEAREST ONE','DEAR RESPECTFUL','ATTN: MD']
    	start_list = []
    	for obj in start_objs:
    		start_obj = re.search(obj+'(.*)',email,re.IGNORECASE)
    		if start_obj!=None:
    			start_list.append((start_obj.start(),[obj+start_obj.group(1)]))
    
    	if len(start_list)==0:
    		return None
    
    	start_list.sort(reverse=True)	# Sort list in descending order and select first element
    	start = start_list[0][1][0]
    
    	# Most of the emails end with the following objects #
    	end_objs = ['YOURS FAITHFULLY','YOURS SINCERELY','YOUR FAITHFULLY','YOUR SINCERELY','SINCERELY YOUR','BEST REGARD','REGARDS,','KIND REGARDS','WITH REGARDS','YOURS TRULY','THANKS AND REGARDS','THANKS AND REMAIN BLESS','REGARDS AND GOD','SINCERE REGARD']
    	end_list = []
    	for endobj in end_objs:
    		end_obj = re.search(endobj+'(.*)',email,re.IGNORECASE)
    		if end_obj!=None:
    			end_list.append((end_obj.start(),[endobj+end_obj.group(1)]))
    
    	if len(end_list)==0:
    		return None
    	end_list.sort()	# Sort list in ascending order and select first element
    	end = end_list[0][1][0]
    	
    	email_obj = re.search(start+'(.*)'+end,email,flags=re.IGNORECASE|re.DOTALL)
    	if email_obj==None:
    		return None
    
    	email_string = email_obj.group(1)
    	for html_tag in ('</div>','</p>','<br>','</span>','</table>'):
    		if re.search(html_tag,email_string,flags=re.IGNORECASE|re.DOTALL)!=None:
    			return None		# Skip any html email
    
    	return email_string
     
    def spam_normal_extraction(self,emails_dataset2):
        print('Spam_Normal_Extraction starts')
        
        spam_emails = []
        normal_emails = []

        for mail in emails_dataset2:
            mail = mail.strip('\n').split('",')	# Strip \n from end of line and split string on ", to seperate mail and its label
            mail_obj = re.search('Subject:(.*)',mail[0],flags=re.IGNORECASE)
            unique_spam = 0
            unique_normal = 0
            
            if mail_obj!=None:
                mail_string = mail_obj.group(1).replace('"',"'") # Replace double quote with single quote for csv
                unique_spam = len(set(spam_emails))
                unique_normal = len(set(normal_emails))
        
                if len(mail_string)>13000:
                    continue	# Skip any email which is unnecessarily large
        
                if mail[-1]=='1' and unique_spam<1000:
                    spam_emails.append(('SPAM',mail_string))
                elif mail[-1]=='0' and unique_normal<1000:
                    normal_emails.append(('NORMAL',mail_string))
           
            if unique_spam==1000 and unique_normal==1000:
                break
        print('-- ',len(set(spam_emails)),'Spam Emails Found --')
        print('-- ',len(set(normal_emails)),'Normal Emails Found --')
        
        return spam_emails,normal_emails

         
'''--------------------------------------------------------------------------------
************************MAIN PROGRAM STARTS FROM HERE******************************
--------------------------------------------------------------------------------'''

valid_emails = []
# Start with first dataset - Fraud Emails #
print('DATASET = FRAUD EMAILS')
print('----------------------')

#create object of the class
create_dataset= Dataset_Creation()

emails_dataset = open('F:\WebMining\SimarjotKaur_WM\Code_WM\Datasets/fradulent_emails.txt','r').read() #Read complete dataset as string
email_list = emails_dataset.split('From r')

for email in email_list:
	email_content = create_dataset.extract_content(email)
	if email_content!=None:
		single_line_email = email_content.replace('\n',' ') # Moving all email content to a single line
		if len(single_line_email)>100 and len(single_line_email)<13000:
			valid_emails.append(('FRAUD',single_line_email.replace('"',''))) # Removing all double quotes for easy csv generation
			if len(set(valid_emails))==1000:
				break

fraud_emails = set(valid_emails)
print('-- ',len(fraud_emails),'Fraud Emails Found --')

# Below is used to process second dataset containing spam and normal emails #
print('\n')
print('DATASET = SPAM AND NORMAL EMAILS')
print('-------------------------')

emails_dataset2 = open('F:\WebMining\SimarjotKaur_WM\Code_WM\Datasets/spam_normal_emails.csv','r').readlines() #Read complete dataset as string
spam_emails,normal_emails= create_dataset.spam_normal_extraction(emails_dataset2)


all_emails = list(fraud_emails) + list(set(spam_emails)) + list(set(normal_emails))

# Randomise emails #
random.shuffle(all_emails)

# Generate final dataset #
create_dataset.generate_csv_file(all_emails)
