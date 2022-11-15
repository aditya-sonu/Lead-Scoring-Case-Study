#!/usr/bin/env python
# coding: utf-8

# # Lead Scoring Case Study                                       ---By Aditya Kumar Roy              
# 
# With close to 35 predictor variables we need to select the most promising leads, i.e. the leads that are most likely to convert into paying customers for education company X Education which sells online courses to industry professionals. The company requires us to build a model wherein we need to assign a lead score to each of the leads such that the customers with higher lead score have a higher conversion chance and the customers with lower lead score have a lower conversion chance.

# # Step 1: Importing import packages and Reading Data

# In[1]:


# Importing all required packages
import numpy as np
import pandas as pd
from datetime import datetime as dt


# In[2]:


# For Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# To Scale our data
from sklearn.preprocessing import scale


# In[4]:


# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd


# In[5]:


#reading Dataset
leads = pd.read_csv("Leads.csv",  sep = ',',encoding = "ISO-8859-1")
leads.head()


# # Step 2: Inspecting the Dataframe

# In[6]:


leads.dtypes


# In[7]:


leads.shape


# # Step 3: Data Preparation

# ### Handling Duplicate Rows

# In[8]:


# removing duplicate rows
leads.drop_duplicates(subset='Lead Number')
leads.shape


# In[9]:


# Checking for total count and percentage of null values in all columns of the dataframe.
total = pd.DataFrame(leads.isnull().sum().sort_values(ascending=False), columns=['Total'])
percentage = pd.DataFrame(round(100*(leads.isnull().sum()/leads.shape[0]),2).sort_values(ascending=False)                          ,columns=['Percentage'])
pd.concat([total, percentage], axis = 1)


# ## Visualizing occurence of Null values in the columns based on rows

# In[10]:


plt.figure(figsize=(10,10))
sns.heatmap(leads.isnull(), cbar=False)

plt.tight_layout()
plt.show()


# In[11]:


# Identifying if any column exists with only null values
leads.isnull().all(axis=0).any()


# In[12]:


#Dropping all columns with only 0 values
leads.loc[:, (leads != 0).any(axis=0)]
leads.shape


# In[13]:


#Remove columns which has only one unique value

"""
Deleting the following columns as they have only one unique value and hence cannot be responsible in predicting a successful lead case

Magazine
Receive More Updates About Our Courses
Update me on Supply Chain Content
Update me on Supply Chain Content
I agree to pay the amount through cheque

"""  
leads= leads.loc[:,leads.nunique()!=1]
leads.shape


# In[14]:


#Deleting the columns 'Asymmetrique Activity Score' & 'Asymmetrique Profile Score' 
#as they will be represented by their corresponding index columns

leads = leads.drop('Asymmetrique Activity Score', axis=1)
leads = leads.drop('Asymmetrique Profile Score', axis=1)
leads.shape


# In[15]:


#Deleting the columns 'Prospect ID' as it will not have any effect in the predicting model
leads = leads.drop('Prospect ID', axis=1)
#leads = leads.drop('Lead Number', axis=1)
leads.shape


# In[16]:


#Deleting the columns 'What matters most to you in choosing a course' as it mostly has unique values and some null values.
leads = leads.drop('What matters most to you in choosing a course', axis=1)
leads.shape


# In[17]:


#Deleting the columns 'How did you hear about X Education' as it mostly has null values or 'Select' values 
#that contribute to the 'Converted' percentage.
leads = leads.drop('How did you hear about X Education', axis=1)
leads.shape


# # Removing rows where a particular column has high missing values

# In[18]:


leads['Lead Source'].isnull().sum()


# In[19]:


#removing rows where a particular column has high missing values because the column cannot be removed because of its importance
leads = leads[~pd.isnull(leads['Lead Source'])]
leads.shape


# ## Imputing with Median values because the continuous variables have outliers

# In[20]:


leads['TotalVisits'].replace(np.NaN, leads['TotalVisits'].median(), inplace =True)


# In[21]:


leads['Page Views Per Visit'].replace(np.NaN, leads['Page Views Per Visit'].median(), inplace =True)


# ### Imputing with Mode values

# In[22]:


leads['Country'].mode()


# In[23]:


leads.loc[pd.isnull(leads['Country']), ['Country']] = 'India'


# In[24]:


leads['Country'] = leads['Country'].apply(lambda x: 'India' if x=='India' else 'Outside India')
leads['Country'].value_counts()


# In[25]:


sns.barplot(y='Country', x='Converted', palette='husl', data=leads, estimator=np.sum)


# ## Assigning An Unique Category to NULL/SELECT values
# 
# Instead of deleting columns with huge null value percentage(which results in loss of data), this strategy adds more information into the dataset and results in the change of variance.

# *Creating a new category consisting on NULL/Select values for the field Lead Quality*

# ### 'Select' values in some columns :
# 
# There are some columns in dataset which have a level/value called 'Select'. This might have happened because these fields in the website might be non mandatory fields with drop downs options for the customer to choose from. Amongst the dropdown values, the default option is probably 'Select' and since these aren't mandatory fields, many customer might have have chosen to leave it as the default value 'Select'.

# In[26]:


leads['Lead Quality'].value_counts()


# In[27]:


leads['Lead Quality'].isnull().sum()


# In[28]:


sns.barplot(y='Lead Quality', x='Converted', palette='husl', data=leads, estimator=np.sum)


# ### Creating a new category consisting on NULL/Select values for the field Asymmetrique Profile Index

# In[29]:


leads['Asymmetrique Profile Index'].value_counts()


# In[30]:


leads['Asymmetrique Profile Index'].isnull().sum()


# In[31]:


leads['Asymmetrique Profile Index'].fillna("Unknown", inplace = True)
leads['Asymmetrique Profile Index'].value_counts()


# In[32]:


sns.barplot(y='Asymmetrique Profile Index', x='Converted', palette='husl', data=leads, estimator=np.sum)


# ### *Creating a new category consisting on NULL/Select values for the field   Asymmetrique Activity Index

# In[33]:


leads['Asymmetrique Activity Index'].value_counts()


# In[34]:


leads['Asymmetrique Activity Index'].isnull().sum()


# In[35]:


leads['Asymmetrique Activity Index'].fillna("Unknown", inplace = True)
leads['Asymmetrique Activity Index'].value_counts()


# In[36]:


sns.barplot(y='Asymmetrique Activity Index', x='Converted', palette='husl', data=leads, estimator=np.sum)


# ### *Creating a new category consisting on NULL/Select values for the field City*

# In[37]:


leads['City'].isnull().sum()


# In[38]:


leads['City'].fillna("Unknown", inplace = True)
leads['City'].value_counts()


# In[39]:


leads['City'].replace('Select', 'Unknown', inplace =True)
leads['City'].value_counts()


# In[40]:


sns.barplot(y='City', x='Converted', palette='husl', data=leads, estimator=np.sum)


# ### *Creating a new category consisting on NULL/Select values for the field Lead Profile*

# In[41]:


leads['Last Activity'].value_counts()


# In[42]:


leads['Last Activity'].isnull().sum()


# In[43]:


leads['Last Activity'].fillna("Unknown", inplace = True)
leads['Last Activity'].value_counts()


# In[44]:


sns.barplot(y='Last Activity', x='Converted', palette='husl', data=leads, estimator=np.sum)


# ### *Creating a new category consisting on NULL/Select values for the field Last Activity*

# In[45]:


leads['Last Activity'].value_counts()


# In[46]:


leads['Last Activity'].isnull().sum()


# In[47]:


leads['Last Activity'].fillna("Unknown", inplace = True)
leads['Last Activity'].value_counts()


# In[48]:


sns.barplot(y='Last Activity', x='Converted', palette='husl', data=leads, estimator=np.sum)


# ### *Creating a new category consisting on NULL/Select values for the field Lead Profile* 

# In[49]:


leads['Lead Profile'].value_counts()


# In[50]:


leads['Lead Profile'].isnull().sum()


# In[51]:


leads['Lead Profile'].fillna("Unknown", inplace = True)
leads['Lead Profile'].value_counts()


# In[52]:


leads['Lead Profile'].replace('Select', 'Unknown', inplace =True)
leads['Lead Profile'].value_counts()


# In[53]:


sns.barplot(y='Lead Profile', x='Converted', palette='husl', data=leads, estimator=np.sum)


# ### *Creating a new category consisting on NULL/Select values for the field What is your current occupation*

# In[54]:


leads['What is your current occupation'].value_counts()


# In[55]:


leads['What is your current occupation'].isnull().sum()


# In[56]:


leads['What is your current occupation'].fillna("Unknown", inplace = True)
leads['What is your current occupation'].value_counts()


# In[57]:


sns.barplot(y='What is your current occupation', x='Converted', palette='husl', data=leads, estimator=np.sum)


# ### *Creating a new category consisting on NULL/Select values for the field Specialization*

# In[58]:


leads['Specialization'].value_counts()


# In[59]:


leads['Specialization'].isnull().sum()


# In[60]:


leads['Specialization'].fillna("Unknown", inplace = True)
leads['Specialization'].value_counts()


# In[61]:


sns.barplot(y='Specialization', x='Converted', palette='husl', data=leads, estimator=np.sum)


# ### *Creating a new category consisting on NULL/Select values for the field Tags*

# In[62]:


leads['Tags'].value_counts()


# In[63]:


leads['Tags'].isnull().sum()


# In[64]:


leads['Tags'].fillna("Unknown", inplace = True)
leads['Tags'].value_counts()


# In[65]:


sns.barplot(y='Tags', x='Converted', palette='husl', data=leads, estimator=np.sum)


# ## Reinspecting Null Values

# In[66]:


# Checking for total count and percentage of null values in all columns of the dataframe.

total = pd.DataFrame(leads.isnull().sum().sort_values(ascending=False), columns=['Total'])
percentage = pd.DataFrame(round(100*(leads.isnull().sum()/leads.shape[0]),2).sort_values(ascending=False)                          ,columns=['Percentage'])
pd.concat([total, percentage], axis = 1).head()


# In[67]:


plt.figure(figsize=(5,5))
sns.heatmap(leads.isnull(), cbar=False)

plt.tight_layout()
plt.show()


# ## Checking for Outliers
# 

# In[68]:


# Checking outliers at 25%,50%,75%,90%,95% and 99%
leads.describe(percentiles=[.25,.5,.75,.90,.95,.99]).T


# In[69]:


numeric_variables = ['TotalVisits','Total Time Spent on Website','Page Views Per Visit']
print(numeric_variables)


# In[70]:


numeric_variables = ['TotalVisits','Total Time Spent on Website','Page Views Per Visit']

#Function to plot the distribution plot of the numeric variable list
def boxplot(var_list):
    plt.figure(figsize=(12,8))
    for var in var_list:
        plt.subplot(2,5,var_list.index(var)+1)
        #plt.boxplot(country[var])
        sns.boxplot(y=var,palette='cubehelix', data=leads)
    # Automatically adjust subplot params so that the subplotS fits in to the figure area.
    plt.tight_layout()
    # display the plot
    plt.show()
    
boxplot(numeric_variables)


# In[71]:


fig, ax = plt.subplots(figsize=(8,6))
ax.scatter(leads['TotalVisits'], leads['Total Time Spent on Website'])
ax.set_xlabel('Proportion of non-retail business acres per town')
ax.set_ylabel('Full-value property-tax rate per $10,000')
plt.show()


# In[72]:


sns.jointplot(leads['Page Views Per Visit'],leads['Total Time Spent on Website'], color="b")
plt.show()


# ### Removing outlier values based on the Interquartile distance for some of the continuous variable 

# In[73]:


Q1 = leads['TotalVisits'].quantile(0.25)
Q3 = leads['TotalVisits'].quantile(0.75)
IQR = Q3 - Q1
leads=leads.loc[(leads['TotalVisits'] >= Q1 - 1.5*IQR) & (leads['TotalVisits'] <= Q3 + 1.4*IQR)]

Q1 = leads['Page Views Per Visit'].quantile(0.25)
Q3 = leads['Page Views Per Visit'].quantile(0.75)
IQR = Q3 - Q1
leads=leads.loc[(leads['Page Views Per Visit'] >= Q1 - 1.5*IQR) & (leads['Page Views Per Visit'] <= Q3 + 1.5*IQR)]

leads.shape


# In[74]:


#Function to plot the distribution plot of the numeric variable list
def boxplot(var_list):
    plt.figure(figsize=(15,10))
    for var in var_list:
        plt.subplot(2,5,var_list.index(var)+1)
        #plt.boxplot(country[var])
        sns.boxplot(y=var,palette='BuGn_r', data=leads)
    # Automatically adjust subplot params so that the subplotS fits in to the figure area.
    plt.tight_layout()
    # display the plot
    plt.show()
    
boxplot(numeric_variables)


# In[75]:


leads.shape


# ### Converting some binary variables (Yes/No) to 0/1

# In[76]:


# List of variables to map

varlist =  ['Search','Do Not Email', 'Do Not Call', 'Newspaper Article', 'X Education Forums', 'Newspaper', 
            'Digital Advertisement','Through Recommendations','A free copy of Mastering The Interview']

# Defining the map function
def binary_map(x):
    return x.map({'Yes': 1, "No": 0})

# Applying the function to the housing list
leads[varlist] = leads[varlist].apply(binary_map)
leads.head()


# ### For categorical variables with multiple levels, creating dummy features (one-hot encoded)

# In[77]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy1 = pd.get_dummies(leads[['Country', 'Lead Source','Lead Origin','Last Notable Activity']], drop_first=True)

# Adding the results to the master dataframe
leads = pd.concat([leads, dummy1], axis=1)
leads.shape


# ### Dropping the repeated variables

# In[78]:


# We have created dummies for the below variables, so we can drop them
leads = leads.drop(['Lead Quality','Asymmetrique Profile Index','Asymmetrique Activity Index','Tags','Lead Profile',
                    'Lead Origin','What is your current occupation', 'Specialization', 'City','Last Activity', 'Country', 
                    'Lead Source','Last Notable Activity'], 1)
leads.shape


# In[79]:


leads.head()


# In[80]:


# Ensuring there are no categorical columns left in the dataframe
cols = leads.columns
num_cols = leads._get_numeric_data().columns
list(set(cols) - set(num_cols))


# In[81]:


# Creating a copy of this origial variable in case if needed later on
original_leads = leads.copy()
print(original_leads.shape)
print(leads.shape)


# # Step 4: Test-Train Split

# In[82]:


from sklearn.model_selection import train_test_split


# In[83]:


# Putting feature variable to X
X = leads.drop(['Converted','Lead Number'], axis=1)

X.head()


# In[84]:


# Putting response variable to y
y = leads['Converted']

y.head()


# In[85]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)


# # Step 5: Feature Scaling

# In[86]:


from sklearn.preprocessing import StandardScaler


# In[87]:


scaler = StandardScaler()

X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.fit_transform(X_train[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])

X_train.head()


# In[88]:


X_train.describe()


# ### Checking the Lead Conversion Rate
# 

# In[89]:


### Checking the Lead Conversion Rate
converted = (sum(leads['Converted'])/len(leads['Converted'].index))*100
converted


# We have almost 38% lead conversion rate

# # Step 6: Model Building

# In[90]:


import statsmodels.api as sm


# In[91]:


# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()


# # Step 7: Feature Selection Using RFE
# 

# In[92]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()


# In[93]:


from sklearn.feature_selection import RFE
rfe = RFE(logreg, n_features_to_select=15)             # running RFE with 15 variables as output
rfe = rfe.fit(X_train, y_train)


# In[94]:


rfe.support_


# In[95]:


list(zip(X_train.columns, rfe.support_, rfe.ranking_))


# In[96]:


col = X_train.columns[rfe.support_]
col


# In[97]:


X_train.columns[~rfe.support_]


# Assessing the model with StatsModels

# In[98]:


X_train_sm = sm.add_constant(X_train[col])
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[99]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:15]


# In[100]:


# reshaping the numpy array containing predicted values
y_train_pred = y_train_pred.values.reshape(-1)
y_train_pred[:15]


# #### Creating a dataframe with the actual churn flag and the predicted probabilities

# In[101]:


y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Conversion_Prob':y_train_pred})
y_train_pred_final['LeadID'] = y_train.index
y_train_pred_final.head()


# #### Creating new column 'predicted' with 1 if Churn_Prob > 0.5 else 0

# In[102]:


y_train_pred_final['predicted'] = y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[103]:


from sklearn import metrics


# In[104]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
print(confusion)


# In[105]:


# Predicted     not_churn    churn
# Actual
# not_churn        3270      365
# churn            579       708  


# In[106]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))


# ### Checking VIFs

# In[107]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[108]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train[col].columns
vif['VIF'] = [variance_inflation_factor(X_train[col].values, i) for i in range(X_train[col].shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ### Clearly there is not much multicollinearity present in our model among the selected features as per their VIF values.

# #### Let us now check the correlation among the features in the below heat map.

# In[109]:


# Slightly alter the figure size to make it more horizontal.
plt.figure(figsize=(20,15), dpi=80, facecolor='w', edgecolor='k', frameon='True')

cor = X_train[col].corr()
sns.heatmap(cor, annot=True, cmap="YlGnBu")

plt.tight_layout()
plt.show()


# # Step 8: Calculating Metrics beyond Accuracy

# In[110]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[111]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[112]:


# Let us calculate specificity
TN / float(TN+FP)


# In[113]:


# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))


# In[114]:


# positive predictive value 
print (TP / float(TP+FP))


# In[115]:


# Negative predictive value
print (TN / float(TN+ FN))


# # Step 9: Plotting the ROC Curve
# 
# ### An ROC curve demonstrates several things:

# 1. It shows the tradeoff between sensitivity and specificity (any increase in sensitivity will be accompanied by a decrease in    specificity).
# 
# 2. The closer the curve follows the left-hand border and then the top border of the ROC space, the more accurate the test.
# 
# 3. The closer the curve comes to the 45-degree diagonal of the ROC space, the less accurate the test.

# In[116]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return fpr,tpr, thresholds


# In[117]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob, drop_intermediate = False )


# In[118]:


draw_roc(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# ### Calculating the area under the curve(GINI)

# In[119]:


def auc_val(fpr,tpr):
    AreaUnderCurve = 0.
    for i in range(len(fpr)-1):
        AreaUnderCurve += (fpr[i+1]-fpr[i]) * (tpr[i+1]+tpr[i])
    AreaUnderCurve *= 0.5
    return AreaUnderCurve


# In[120]:


auc = auc_val(fpr,tpr)
auc


# # Step 10: Finding Optimal Cutoff Point 
# ### Optimal cutoff probability is that prob where we get balanced sensitivity and specificity

# In[121]:


# Let's create columns with different probability cutoffs 
numbers = [float(x)/10 for x in range(10)]
for i in numbers:
    y_train_pred_final[i]= y_train_pred_final.Conversion_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[122]:


# Now let's calculate accuracy sensitivity and specificity for various probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives

num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i] )
    total1=sum(sum(cm1))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
print(cutoff_df)


# ### Let's plot accuracy sensitivity and specificity for various probabilities.

# In[123]:


# Slightly alter the figure size to make it more horizontal.

#plt.figure(figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k', frameon='True')
sns.set_style("whitegrid") # white/whitegrid/dark/ticks
sns.set_context("paper") # talk/poster
cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'], figsize=(10,6))
# plot x axis limits
plt.xticks(np.arange(0, 1, step=0.05), size = 12)
plt.yticks(size = 12)
plt.show()


# #### From the curve above, 0.33 is the optimum point to take it as a cutoff probability.

# In[124]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Conversion_Prob.map( lambda x: 1 if x > 0.33 else 0)

y_train_pred_final.head()


# In[125]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[126]:


confusion1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted)
confusion1


# In[127]:


TP = confusion1[1,1] # true positive 
TN = confusion1[0,0] # true negatives
FP = confusion1[0,1] # false positives
FN = confusion1[1,0] # false negatives


# In[128]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[129]:


# Let us calculate specificity
TN / float(TN+FP)


# In[130]:


# Calculate false postive rate - predicting churn when customer does not have churned
print(FP/ float(TN+FP))


# In[131]:


# Positive predictive value 
print (TP / float(TP+FP))


# In[132]:


# Negative predictive value
print (TN / float(TN+ FN))


# #### Step 11: Precision and Recall
# ##### Precision
# ##### TP / TP + FP

# In[133]:


precision = confusion1[1,1]/(confusion1[0,1]+confusion1[1,1])
precision


# #### Recall
# ##### TP / TP + FN

# In[134]:


recall = confusion1[1,1]/(confusion1[1,0]+confusion1[1,1])
recall


# ##### Using sklearn utilities for the same

# In[135]:


from sklearn.metrics import precision_score, recall_score


# In[136]:


precision_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[137]:


recall_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# #### Precision and recall tradeoff
# 

# In[138]:


from sklearn.metrics import precision_recall_curve


# In[139]:


y_train_pred_final.Converted, y_train_pred_final.final_predicted


# In[140]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Conversion_Prob)


# In[141]:


# Slightly alter the figure size to make it more horizontal.
plt.figure(figsize=(8, 4), dpi=100, facecolor='w', edgecolor='k', frameon='True')
plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.xticks(np.arange(0, 1, step=0.05))
plt.show()


# ##### From the precision-recall graph above, we get the optical threshold value as close to .37. However our business requirement here is to have Lead Conversion Rate around 80%.
# ###### This is already achieved with our earlier threshold value of 0.33. So we will stick to this value.

# ### Calculating the F1 score
# #### F1 = 2×(Precision*Recall)/(Precision+Recall)

# In[142]:


F1 = 2*(precision*recall)/(precision+recall)
F1


# # Step 12: Making predictions on the test set
# ##### Using the scaler function from the train dataset to transfor the test dataset

# In[143]:


X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']] = scaler.transform(X_test[['TotalVisits','Total Time Spent on Website','Page Views Per Visit']])
X_test.head()


# In[144]:


X_test = X_test[col]
X_test.head()


# #### Adding the constant

# In[145]:


X_test_sm = sm.add_constant(X_test)


# ####  Making predictions on the test set

# In[146]:


y_test_pred = res.predict(X_test_sm)


# In[147]:


y_test_pred[:15]


# In[148]:


# Converting y_pred to a dataframe which is an array
y_pred_1 = pd.DataFrame(y_test_pred)


# In[149]:


# Let's see the head
y_pred_1.head()


# In[150]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[151]:


# Putting CustID to index
y_test_df['LeadID'] = y_test_df.index


# In[152]:


# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[153]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)


# In[154]:


y_pred_final.head()


# In[155]:


y_pred_final.shape


# # Step 13: Calculating Lead score for the entire dataset
# ### Lead Score = 100 * ConversionProbability
# ### This needs to be calculated for all the leads from the original dataset (train + test)

# In[159]:


# Selecting the test dataset along with the Conversion Probability and final predicted value for 'Converted'
leads_test_pred = y_pred_final.copy()
leads_test_pred.head()


# In[160]:


# Selecting the train dataset along with the Conversion Probability and final predicted value for 'Converted'
leads_train_pred = y_train_pred_final.copy()
leads_train_pred.head()


# In[161]:


# Dropping unnecessary columns from train dataset
leads_train_pred = leads_train_pred[['LeadID','Converted','Conversion_Prob','final_predicted']]
leads_train_pred.head()


# ##### Concatenating the train and the test dataset with the Conversion Probabilities 

# In[162]:


# Concatenating the 2 dataframes train and test along the rows with the append() function
lead_full_pred = leads_train_pred.append(leads_test_pred)
lead_full_pred.head()


# In[163]:


# Inspecting the shape of the final dataframe and the test and train dataframes
print(leads_train_pred.shape)
print(leads_test_pred.shape)
print(lead_full_pred.shape)


# In[164]:


# Ensuring the LeadIDs are unique for each lead in the finl dataframe
len(lead_full_pred['LeadID'].unique().tolist())


# In[168]:


# Making the LeadID column as index
# We willlater join it with the original_leads dataframe based on index
lead_full_pred = lead_full_pred.set_index('LeadID').sort_index(axis = 0, ascending = True)
lead_full_pred.head()


# In[169]:


# Slicing the Lead Number column from original_leads dataframe
original_leads = original_leads[['Lead Number']]
original_leads.head()


# #### Concatenating the 2 dataframes based on index.
# ##### This is done so that Lead Score is associated to the Lead Number of each Lead. This will help in quick identification of the lead

# In[170]:


# Concatenating the 2 dataframes based on index and displaying the top 10 rows
# This is done son that Lead Score is associated to the Lead Number of each Lead. This will help in quick identification of the lead.
leads_with_score = pd.concat([original_leads, lead_full_pred], axis=1)
leads_with_score.head(10)


# In[171]:


# Inspecting the dataframe shape
leads_with_score.shape


# In[172]:


# Inspectin if the final dataframe has any null values

total = pd.DataFrame(leads_with_score.isnull().sum().sort_values(ascending=False), columns=['Total'])
percentage = pd.DataFrame(round(100*(leads_with_score.isnull().sum()/leads_with_score.shape[0]),2).sort_values(ascending=False)                          ,columns=['Percentage'])
pd.concat([total, percentage], axis = 1)


# # Step 14: Determining Feature Importance
# #### Selecting the coefficients of the selected features from our final model excluding the intercept

# In[173]:


pd.options.display.float_format = '{:.2f}'.format
new_params = res.params[1:]
new_params


# ##### Getting a relative coeffient value for all the features wrt the feature with the highest coefficient

# In[174]:


#feature_importance = abs(new_params)
feature_importance = new_params
feature_importance = 100.0 * (feature_importance / feature_importance.max())
feature_importance


# ### Sorting the feature variables based on their relative coefficient values

# In[175]:


sorted_idx = np.argsort(feature_importance,kind='quicksort',order='list of str')
sorted_idx
##


# # Step 15: Conclusion
# ### After trying several models, we finally chose a model with the following characteristics
# 1. All variables have p-value < 0.05.
# 2. All the features have very low VIF values, meaning, there is hardly any muliticollinearity among the features. This is also evident from the heat map.
# 3. The overall accuracy of 0.9056 at a probability threshold of 0.33 on the test dataset is also very acceptable.
