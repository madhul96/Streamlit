#!/usr/bin/env python
# coding: utf-8

# # Business Objective:-
# -1.Can we predict whether a customer will churn or not based on their telecommunications usage patterns and demographic information?
# 
# -2.What are the factors that contribute the most to customer churn in the telecommunications industry, and how can we use this information to develop targeted retention strategies?
# 
# -3.How can we optimize our retention strategies to reduce customer churn and increase customer loyalty, based on the customer data we have available?
# 
# -4.Are there any significant differences in telecommunications usage patterns and customer demographics between customers who churn and those who remain loyal, and how can we leverage this information to improve our customer retention efforts?
# 
# -5.Can we develop a machine learning model to accurately predict customer churn in real-time, and use this model to proactively intervene with customers who are at high risk of leaving the company?

# # Required Labraries

# In[1]:


# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from collections import Counter
from imblearn.over_sampling import SMOTE,  ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from dataprep.eda import plot, plot_correlation, plot_missing, create_report
import plotly.express as px
from plotly.offline import plot as off
import plotly.figure_factory as ff
import plotly.io as pio
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report,confusion_matrix,accuracy_score, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score, train_test_split, RandomizedSearchCV
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#load Dataset
tele_df = pd.read_csv("churn.csv", index_col=0)


# # Understand More About The Data

# In[3]:


tele_df.head()


# In[4]:


tele_df.describe(include='all')


# In[5]:


#view the dataset below 5 rows to look the glimps of data
tele_df.tail()


# In[6]:


#getting the shape of dataset with row and columns
tele_df.shape


# In[7]:


tele_df.dtypes


# In[8]:


# day charge is obj, eve mins is obj


# In[9]:


#getting types of dataset and details about dataset
tele_df.info()


# In[10]:


print('features of data set:')
tele_df.columns


# In[11]:


day_charge =tele_df['day.charge'].astype(float)
eve_mins = tele_df['eve.mins'].astype(float)
tele_df['day.charge'] = day_charge
tele_df['eve.mins'] = eve_mins


# In[12]:


tele_df.dtypes


# In[13]:


tele_df.nunique()


# In[14]:


tele_df[tele_df.duplicated()].shape


# In[15]:


#looking for the Description of the dataset to get insights of the data
tele_df.describe(include='all')


# In[16]:


tele_df.describe(include='all').T


# In[17]:


#printing count of true and flase in'churn' feature
#0 = flase & 1 = true
tele_df.Churn.value_counts()


# In[18]:


# Identifying categorical and numerical data types
cat_features = [feat for feat in tele_df.columns if tele_df[feat].dtype == 'object']
num_features = [feat for feat in tele_df.columns if tele_df[feat].dtype != 'object']

# Identifying discrete and continuous numerical features
discrete_features = [feat for feat in num_features if len(tele_df[feat].unique()) < 25]
continuous_features = [feat for feat in num_features if feat not in discrete_features]

print(f"Discrete variable count: {len(discrete_features)}")
print(f"Continuous variable count: {len(continuous_features)}")


# # Checking missing value and duplicate values

# In[19]:


# check for count of missing value in each column.
tele_df.isna().sum()
tele_df.isnull().sum()


# missing values were found in two of the columns: day.charge, eve.mins

# In[20]:


# % Missing Value visualization

missing = pd.DataFrame((tele_df.isnull().sum())*100/tele_df.shape[0]).reset_index()
plt.figure(figsize=(20,5))
ax = sns.pointplot('index',0,data=missing)
plt.xticks(rotation =90,fontsize =20)
plt.title("Percentage of Missing values")
plt.ylabel("PERCENTAGE")
plt.show


# 
# 

# In[21]:


tele_df=tele_df.dropna()


# In[22]:


#since the percentage of null values were less we could just drop them


# In[23]:


tele_df.isnull().sum()


# In[24]:


tele_df.corr()


# In[ ]:





# * NO any duplicate value in present data set
# * there are 3333 rows and 20 columns
# * & one datatype is boolen = (churn)
# * 8 flot data type & 8 interger data type
#  
# 

# In[25]:


sns.set(style="darkgrid", palette="muted")
fig, ax = plt.subplots(figsize=(16,7))
sns.boxplot(data=tele_df)
plt.xticks(rotation=45)
plt.show()


# In[26]:


tele_df.unimputed=tele_df.copy


# In[27]:


# Identifying numerical columns with outliers
num_features = tele_df.select_dtypes(include=[np.number]).columns
outlier_cols = []
for col in num_features:
    q1 = tele_df[col].quantile(0.25)
    q3 = tele_df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5*iqr
    upper = q3 + 1.5*iqr
    if tele_df[(tele_df[col] < lower) | (tele_df[col] > upper)].shape[0] > 0:
        outlier_cols.append(col)


# In[28]:


# Imputing outliers with the median value
for col in outlier_cols:
    median_val = tele_df[col].median()
    tele_df[col] = tele_df[col].clip(lower=tele_df[col].quantile(0.05), upper=tele_df[col].quantile(0.95)).fillna(median_val)

# Save the imputed dataset to a new file
tele_df.to_csv('my_dataset_imputed.csv', index=False)

# Load the imputed dataset into a Pandas DataFrame
tele_df = pd.read_csv('my_dataset_imputed.csv')


# In[29]:


sns.set(style="darkgrid", palette="muted")
fig, ax = plt.subplots(figsize=(16,7))
sns.boxplot(data=tele_df)
plt.xticks(rotation=45)
plt.show()


# Analyzing what the dependent variable said to us(Churn)

# In[30]:


#unique value inside 'churn' column
tele_df['Churn'].unique()


# In[31]:


#count of true & false in ' churn'
tele_df.Churn.value_counts()


# In[32]:


import plotly.express as px
target_instance = tele_df["Churn"].value_counts().to_frame()
target_instance = target_instance.reset_index()
target_instance = target_instance.rename(columns={'index': 'Category'})
fig = px.pie(target_instance, values='Churn', names='Category', color_discrete_sequence=["green", "red"],
             title='Distribution of Churn')
fig.show()

Customer churned: 14.2%(705)
Loyal customer: 85.8%(4263)
# In[33]:


#let's see churn by using countplot
sns.countplot(x=tele_df.Churn)


# * After analyzing the churn column, we had little to say like almost 15% of customers have churned.
# * let's see what other features say to us and what relation we get after correlated with churn

# In[34]:


fig, axs = plt.subplots(1, 2, figsize=(12, 5))

sns.kdeplot(data=tele_df, x='day.mins', hue='Churn', ax=axs[0])
sns.kdeplot(data=tele_df, x='night.mins', hue='Churn', ax=axs[1])

axs[0].set_title('Distribution of Day Minutes for Churned and Loyal Customers')
axs[1].set_title('Distribution of Night Minutes for Churned and Loyal Customers')

plt.show()


# ** day.mins & night.mins do not have much impact on customer churn 

# # Analyzing "Account Length" column

# In[35]:


#Separating churn and non churn customers
churn_df     = tele_df[tele_df["Churn"] == bool(True)]
not_churn_df = tele_df[tele_df["Churn"] == bool(False)]


# In[36]:


sns.histplot(tele_df['account.length'])


# In[37]:


#comparison of churned account length and not churned account length 
sns.histplot(tele_df['account.length'],color = 'yellow',label="All")
sns.histplot(churn_df['account.length'],color = "red",hist=False,label="Churned")
sns.histplot(not_churn_df['account.length'],color = 'green',hist= False,label="Not churned")
plt.legend()


# * this plot show effect of account length  on churn
# * here is no sign  of customers leaving because of the length  of usage of their account
# * After analyzing various aspects of the "account length" column we didn't found any useful relation to churn. so we aren't able to build any connection to the churn as of now. let's see what other features say about the churn.

# 

# # Analyzing "International Plan" column

# In[38]:


#Show count value of 'yes','no'
tele_df['intl.plan'].value_counts()


# In[39]:


#Show the unique data of "International plan"
tele_df["intl.plan"].unique()


# In[40]:


fig, ax = plt.subplots(figsize=(3,3))
sns.countplot(data=tele_df, x='intl.plan', hue='Churn')
ax.set_title('Churn Count by Intl Plan')
ax.set_xlabel('Intl Plan')
ax.set_ylabel('Count')


# In[41]:


#Calculate the International Plan vs Churn percentage 
International_plan_data = pd.crosstab(tele_df["intl.plan"],tele_df["Churn"])
International_plan_data['Percentage Churn'] = International_plan_data.apply(lambda x : x[1]*100/(x[0]+x[1]),axis = 1)
print(International_plan_data)


# In[42]:


#To get the Donut Plot to analyze International Plan
data = tele_df['intl.plan'].value_counts()
explode = (0, 0.2)
plt.pie(data, explode = explode,autopct='%1.1f%%',shadow=True,radius = 2.0, labels = ['No','Yes'],colors=['skyblue' ,'orange'])
circle = plt.Circle( (0,0), 1, color='white')
p=plt.gcf()
p.gca().add_artist(circle)
plt.title('Donut Plot for International plan')
plt.show()


# * There are 3333 people
# * 323 have a International Plan 
# * 3010 do not have International Plan
# 

# In[43]:


#Analysing by using countplot
sns.countplot(x='intl.plan',hue="Churn",data = tele_df)


# This is a count plot which shows the churned and not churned customer respective to their international plan 
# 
# From the above data we get
# 
# There are 3010 customers who dont have a international plan.
# 
# There are 323 customers who have a international plan.
# 
# Among those who have a international plan 42.4 % people churn.
# 
# Whereas among those who dont have a international plan only 11.4 % people churn.
# 
# So basically the people who bought International plans are churning in big numbers.
# 
# Probably because of connectivity issues or high call charge.

# # Analyzing "Voice Mail Plan" column

# In[44]:


#show the unique value of the "Voice mail plan" column
tele_df["voice.plan"].unique()


# In[45]:


#Calculate the Voice Mail Plan vs Churn percentage
Voice_mail_plan_data = pd.crosstab(tele_df["voice.plan"],tele_df["Churn"])
Voice_mail_plan_data['Percentage Churn'] = Voice_mail_plan_data.apply(lambda x : x[1]*100/(x[0]+x[1]),axis = 1)
print(Voice_mail_plan_data)


# In[46]:


#To get the Donut Plot to analyze Voice mail plan
data = tele_df['voice.plan'].value_counts()
explode = (0, 0.2)
plt.pie(data, explode = explode,autopct='%1.1f%%',startangle=90,shadow=True,radius = 2.0, labels = ['NO','YES'],colors=['skyblue','red'])
circle = plt.Circle( (0,0), 1, color='white')
p=plt.gcf()
p.gca().add_artist(circle)
plt.title('Donut Plot for Voice mail plan')
plt.show()


# * There are 3333 people,
# * 922 having Voicemail plan, 
# * 2411 do not have any Voicemail plan.
# 

# In[47]:


#Analysing by using countplot
sns.countplot(x='voice.plan',hue="Churn",data = tele_df)


# * As we can see there is are no clear relation between voice mail plan and churn so we can't clearly say anything so let's move to the next voice mail feature i.e number of voice mail, let's see what it gives to us.
# * This plot shows churn corresponding with the subscription of voicemail plan Out of 922 people having Voicemail plan, 8.7% are Churn.
# 

# #  Analyzing "Customer service calls" column

# In[48]:


#Printing the data of customer service calls 
tele_df['customer.calls'].value_counts()


# In[49]:


#Calculating the Customer service calls vs Churn percentage
Customer_service_calls_data = pd.crosstab(tele_df['customer.calls'],tele_df["Churn"])
Customer_service_calls_data['Percentage_Churn'] = Customer_service_calls_data.apply(lambda x : x[1]*100/(x[0]+x[1]),axis = 1)
print(Customer_service_calls_data)


# * This table mapping number of customer calls to the churn percentage
# * It’s clear that after 4 calls at least 45% of the subscribers churn.
# * Customers with more than 4 service calls their probability of leaving is more

# In[50]:


#Analysing using countplot
sns.countplot(x='customer.calls',hue="Churn",data = tele_df)


# * It is observed from the above analysis that, mostly because of bad customer service, people tend to leave the operator.
# 
# The above data indicating that those customers who called the service center 5 times or above those customer churn percentage is higher than 60%,
# 
# And customers who have called once also have a high churn rate indicating their issue was not solved in the first attempt.
# 
# So operator should work to improve the service call.

# # Analyzing all calls minutes,all calls, all calls charge together

# In[51]:


#Print the mean value of churned and not churned customer 
print(tele_df.groupby(["Churn"])['day.calls'].mean())


# In[52]:


#Print the mean value of churned and not churned customer 
print(tele_df.groupby(["Churn"])['day.mins'].mean())


# In[53]:


tele_df['Churn'] = tele_df['Churn'].map({'yes': 1, 'no': 0})


# In[ ]:





# In[54]:


tele_df.head(15)


# In[55]:


tele_df.dtypes['Churn']


# In[56]:


##tele_df["churn"] = tele_df["churn"].astype("float64")


# In[57]:


#Print the mean value of churned and not churned customer 
print(tele_df.groupby(["Churn"])['day.charge'].mean())


# In[58]:


#show the relation using scatter plot
sns.scatterplot(x="day.mins", y="day.charge", hue="Churn", data=tele_df,palette='hls')


# In[59]:


#show the relation using box plot plot
sns.boxplot(x="day.mins", y="day.charge", hue="Churn", data=tele_df,palette='hls')


# In[60]:


#Print the mean value of churned and not churned customer 
print(tele_df.groupby(["Churn"])['eve.calls'].mean())


# In[ ]:





# In[61]:


#Print the mean value of churned and not churned customer 
print(tele_df.groupby(["Churn"])['eve.mins'].mean())


# In[62]:


#Print the mean value of churned and not churned customer 
print(tele_df.groupby(["Churn"])['eve.charge'].mean())


# In[63]:


#show the relation using scatter plot
sns.scatterplot(x="eve.mins", y="eve.charge", hue="Churn", data=tele_df,palette='hls')


# In[64]:


#show the relation using box plot plot
sns.boxplot(x="eve.mins", y="eve.charge", hue="Churn", data=tele_df,palette='hls')


# In[65]:


#Print the mean value of churned and not churned customer 
print(tele_df.groupby(["Churn"])['night.calls'].mean())


# In[66]:


#Print the mean value of churned and not churned customer 
print(tele_df.groupby(["Churn"])['night.charge'].mean())


# In[67]:


#Print the mean value of churned and not churned customer 
print(tele_df.groupby(["Churn"])['night.mins'].mean())


# In[68]:


#show the relation using scatter plot
sns.scatterplot(x="night.mins", y="night.charge", hue="Churn", data=tele_df,palette='hls')


# In[69]:


#show the relation using box plot
sns.boxplot(x="night.mins", y="night.charge", hue="Churn", data=tele_df,palette='hls')


# In[70]:


#Print the mean value of churned and not churned customer 
print(tele_df.groupby(["Churn"])['intl.mins'].mean())


# In[71]:


#Print the mean value of churned and not churned customer 
print(tele_df.groupby(["Churn"])['intl.calls'].mean())


# In[72]:


#Print the mean value of churned and not churned customer 
print(tele_df.groupby(["Churn"])['intl.charge'].mean())


# In[73]:


#show the relation using scatter plot
sns.scatterplot(x="intl.mins", y="intl.charge", hue="Churn", data=tele_df,palette='hls')


# In[74]:


#show the relation using scatter plot
sns.boxplot(x="intl.mins", y="intl.charge", hue="Churn", data=tele_df,palette='hls')


# In[75]:


#Deriving a relation between overall call charge and overall call minutes   
day_charge_perm = tele_df['day.charge'].mean()/tele_df['day.mins'].mean()
eve_charge_perm = tele_df['eve.charge'].mean()/tele_df['eve.mins'].mean()
night_charge_perm = tele_df['night.charge'].mean()/tele_df['night.mins'].mean()
int_charge_perm= tele_df['intl.charge'].mean()/tele_df['intl.mins'].mean()


# In[76]:


print([day_charge_perm,eve_charge_perm,night_charge_perm,int_charge_perm])


# In[77]:


sns.barplot(x=['Day','Evening','Night','International'],y=[day_charge_perm,eve_charge_perm,night_charge_perm,int_charge_perm])


# Below this bar plot shows the comparison between all call charges per minute
# International call charges are high as compare to others it's an obvious thing but that may be a cause for international plan customers to churn out.
# 
# After analyzing the above dataset we have noticed that total day/night/eve minutes/call/charges are not put any kind of cause for churn rate. But international call charges are high as compare to others it's an obvious thing but that may be a cause for international plan customers to churn out.

# # Graphical  Visualisation Analysis

# In[78]:


#Printing boxplot for each numerical column present in the data set
df1=tele_df.select_dtypes(exclude=['object','bool'])
for column in df1:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=df1, x=column)
plt.show()


# In[79]:


#Printing displot for each numerical column present in the data set
df1=tele_df.select_dtypes(exclude=['object','bool'])
for column in df1:
        plt.figure(figsize=(17,1))
        sns.displot(data=df1, x=column)
plt.show()


# In[80]:


#Printing strip plot for each numerical column present in the data set
df1=tele_df.select_dtypes(exclude=['object','bool'])
for column in df1:
        plt.figure(figsize=(17,1))
        sns.stripplot(data=df1, x=column)
plt.show()


# ## Univariate Plots

# In[81]:


numerical_features=[feature for feature in tele_df.columns if tele_df[feature].dtypes != 'O']
for feat in numerical_features:
    skew = tele_df[feat].skew()
    sns.distplot(tele_df[feat], kde= False, label='Skew = %.3f' %(skew), bins=30)
    plt.legend(loc='best')
    plt.show()


# In[82]:


def outlier_hunt(tele_df):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than 2 outliers. 
    """
    outlier_indices = []
    
    # iterate over features(columns)
    for col in tele_df.columns.tolist():
        # 1st quartile (25%)
        Q1 = np.percentile(tele_df[col], 25)
        
        # 3rd quartile (75%)
        Q3 = np.percentile(tele_df[col],75)
        
        # Interquartile rrange (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        
        # Determine a list of indices of outliers for feature col
        outlier_list_col = tele_df[(tele_df[col] < Q1 - outlier_step) | (tele_df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    import collections
    outlier_indices = collections.Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > 2 )
    
    return multiple_outliers   

print('The dataset contains %d observations with more than 2 outliers' %(len(outlier_hunt(tele_df[numerical_features])))) 


# In[83]:


#We are using correlation plot,correlation matrix, correletaion heatmap, pair plot


# # Feature Scaling

# In[84]:


# let's plot pair plot to visualise the attributes all at once
import seaborn as sns
sns.pairplot(data=df1, hue = 'Churn')


# In[85]:


data_ = df1.copy()
data_.drop('Churn',axis=1, inplace =True)
data_ = pd.get_dummies(data_.iloc[:,:-1])
data_.head()


# In[ ]:





# # 1)Correlation

# In[86]:


## plot the Correlation matrix
plt.figure(figsize=(17,8))
correlation=tele_df.corr()
sns.heatmap(abs(correlation), annot=True, cmap='coolwarm')


# In[87]:


df=tele_df.copy()


# In[88]:


df.corr()["Churn"].sort_values()


# #unnecessary columns
# voice_mail_plan          -0.102148
# voice_mail_messages      -0.089728
# international_calls      -0.052844
# night_calls               0.006141
# evening_calls             0.009233

# In[89]:


df1=tele_df.copy()
df1['Churn_Cat'] = tele_df['Churn'].replace({1: 'yes', 0: 'no'})
df1.head()


# In[90]:


data_ = df1.copy()
data_.drop('Churn',axis=1, inplace =True)
data_ = pd.get_dummies(data_.iloc[:,:-1])
data_.head()


# In[91]:


data_['Churn'] = df1.Churn_Cat
data_.head()


# In[92]:


le = LabelEncoder()
le.fit(data_["Churn"])
data_["Churn"]=le.transform(data_["Churn"])
data_.head()


# In[93]:


#Feature importance
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest, chi2

# split into input (X) and output (y) variables
X = data_.iloc[:, :-1]

y=  data_.Churn

# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, y)


# In[94]:


# summarize scores
scores = fit.scores_

features = fit.transform(X)


# In[95]:


score_df = pd.DataFrame(list(zip(scores, X.columns)),
               columns =['Score', 'Feature'])
score_df.sort_values(by="Score", ascending=False, inplace=True)
score_df


# In[ ]:





# In[96]:


plt.figure(figsize=(20,8))
# make barplot and sort bars
sns.barplot(x='Feature',
            y="Score", 
            data=score_df, 
            order=score_df.sort_values('Score').Feature)
# set labels
plt.xlabel("Features", size=15)
plt.ylabel("Scores", size=15)
plt.yticks(rotation = 0, fontsize = 14)
plt.xticks(rotation = 90, fontsize = 16)
plt.title("Feature Score w.r.t Churn", size=18)
plt.show()


# In[97]:


model_data = data_[['day.mins', 'voice.messages','day.charge', 'eve.mins', 'intl.plan_yes', 'customer.calls', 'night.mins', 'voice.plan_yes','eve.charge', 'intl.plan_no', 'account.length', 'Churn']]
model_data.head()


# ### Test Train Split With Imbalanced Dataset

# In[98]:


x = model_data.drop('Churn',axis=1)
y = model_data['Churn']


# In[99]:


y.unique()


# In[100]:


# Create StratifiedKFold object.
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

# enumerate the splits and summarize the distributions
for train_ix, test_ix in skf.split(x, y):
# select rows
    train_X, test_X = x.iloc[train_ix], x.loc[test_ix]
    train_y, test_y = y.iloc[train_ix], y.iloc[test_ix]
# summarize train and test composition
counter_train = Counter(train_y)
counter_test = Counter(test_y)
print('Training Data',counter_train,'Testing Data',counter_test)


# In[101]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0,stratify=y)


# In[102]:


# summarize train and test composition
counter_train = Counter(y_train)
counter_test = Counter(y_test)
print('Training Data',counter_train,'Testing Data',counter_test)


# In[103]:


print("Shape of X_train: ",x_train.shape)
print("Shape of X_test: ", x_test.shape)
print("Shape of y_train: ",y_train.shape)
print("Shape of y_test",y_test.shape)


# ### Grid search using Stratified Kfold Splits on Imbalanced Dataset

# In[104]:


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


# In[105]:


params = {
    "criterion":("gini", "entropy"), 
    "splitter":("best", "random"), 
    "max_depth":(list(range(1, 20))), 
    "min_samples_split":[2, 3, 4], 
    "min_samples_leaf":list(range(1, 20)), 
}


tree_clf = DecisionTreeClassifier(random_state=42)
tree_cv = GridSearchCV(tree_clf, params, scoring="accuracy", n_jobs=-1, verbose=1, cv=3)
tree_cv.fit(train_X, train_y)
best_params = tree_cv.best_params_
print(f"Best paramters: {best_params})")

tree_clf = DecisionTreeClassifier(**best_params)
tree_clf.fit(train_X, train_y)
print_score(tree_clf, train_X, train_y, test_X, test_y, train=True)
print_score(tree_clf, train_X, train_y, test_X, test_y, train=False)


# In[106]:


# Get score for different values of n
decision_tree = DecisionTreeClassifier()
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

results = cross_val_score(decision_tree, train_X, train_y, cv=skf)
print(results.mean())


# In[ ]:





# In[107]:


x = model_data.drop(['Churn'], axis=1)
y = model_data['Churn']

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.30, random_state=0,stratify=y)


# In[108]:


print("Shape of X_train: ",x_train.shape)
print("Shape of X_test: ", x_test.shape)
print("Shape of y_train: ",y_train.shape)
print("Shape of y_test",y_test.shape)


# ### SMOTE Oversampling technique

# In[109]:


counter = Counter(y_train)
print('Before',counter)
# oversampling the train dataset using SMOTE
smt = SMOTE()
x_train_sm, y_train_sm = smt.fit_resample(x_train, y_train)

counter = Counter(y_train_sm)
print('After',counter)


# In[110]:


counter = Counter(y_train)
print('Before',counter)
# oversampling the train dataset using ADASYN
ada = ADASYN(random_state=130)
x_train_ada, y_train_ada = ada.fit_resample(x_train, y_train)

counter = Counter(y_train_ada)
print('After',counter)


# In[111]:


##Hybridization
counter = Counter(y_train)
print('Before',counter)
# oversampling the train dataset using SMOTE + Tomek
smtom = SMOTETomek(random_state=139)
x_train_smtom, y_train_smtom = smtom.fit_resample(x_train, y_train)

counter = Counter(y_train_smtom)
print('After',counter)


# In[112]:


counter = Counter(y_train)
print('Before',counter)
#oversampling the train dataset using SMOTE + ENN
smenn = SMOTEENN()
x_train_smenn, y_train_smenn = smenn.fit_resample(x_train, y_train)

counter = Counter(y_train_smenn)
print('After',counter)


# ### Performance Analysis after Resampling

# In[113]:


sampled_data = {
    'ACTUAL':[x_train, y_train],
    'SMOTE':[x_train_sm, y_train_sm],
    'ADASYN':[x_train_ada, y_train_ada],
    'SMOTE_TOMEK':[x_train_smtom, y_train_smtom],
    'SMOTE_ENN':[x_train_smenn, y_train_smenn]
}


# In[114]:


def test_eval(clf_model, X_test, y_test, algo=None, sampling=None):
    # Test set prediction
    y_prob=clf_model.predict_proba(X_test)
    y_pred=clf_model.predict(X_test)

    print('Confusion Matrix')
    print('='*60)
    #plot_confusion_matrix(clf_model, X_test, y_test)  
    #plt.show() 
    print(confusion_matrix(y_test,y_pred),"\n")
    print('Classification Report')
    print('='*60)
    print(classification_report(y_test,y_pred),"\n")
    #print('AUC-ROC')
    #print('='*60)
    #print(roc_auc_score(y_test, y_prob[:,1], multi_class='ovo'))
    
    #x = roc_auc_score(y_test, y_prob[:,1])
    f1 = f1_score(y_test, y_pred, average='micro')
    recall = recall_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test, y_pred, average='micro')
          
    
    return algo,precision,recall,f1,sampling


# In[115]:


model_params = {
    'decision_tree' :{
        'model' :  DecisionTreeClassifier(),
        'params' : {
             'max_depth': [i for i in range(5,16,2)],
             'min_samples_split': [2, 5, 10, 15, 20, 50, 100],
             'min_samples_leaf': [1, 2, 5],
             'criterion': ['gini', 'entropy'],
             'max_features': ['log2', 'sqrt', 'auto']
        }
        
    }
    
}


# In[116]:


cv = StratifiedKFold(n_splits=5, random_state=100, shuffle=True)
output = []
for model , model_hp in model_params.items():
    for resam , data in sampled_data.items():
        clf = RandomizedSearchCV(model_hp['model'], model_hp['params'],cv = cv, scoring='roc_auc', n_jobs=-1 )
        clf.fit(data[0], data[1])
        clf_best = clf.best_estimator_
        print('x'*60)
        print(model+' with ' + resam)
        print('='*60)
        output.append(test_eval(clf_best, x_test, y_test, model, resam))

As the results can be compared, SMOTE TOMEK Hybridization technique gave the best results we are going to use it further
# In[117]:


counter = Counter(y)
print('Before',counter)
# oversampling the train dataset using SMOTE + Tomek
smtom = SMOTETomek(random_state=0)
x_train_smtom, y_train_smtom = smtom.fit_resample(x, y)

counter = Counter(y_train_smtom)
print('After',counter)


# In[118]:


x_train,x_test,y_train,y_test = train_test_split(x_train_smtom,y_train_smtom,test_size=0.3,random_state=0, stratify=y_train_smtom)

counter = Counter(y_train_smtom)
print('Before',counter)
counter = Counter(y_train)
print('After',counter)
print("Shape of X_train: ",x_train.shape)
print("Shape of X_test: ", x_test.shape)
print("Shape of y_train: ",y_train.shape)
print("Shape of y_test",y_test.shape)


# # Model Building

# In[119]:


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")

params = {
    "criterion":("gini", "entropy"), 
    "splitter":("best", "random"), 
    "max_depth":(list(range(1, 6))), 
    "min_samples_split":[2, 3, 4], 
    "min_samples_leaf":list(range(1, 6)), 
}


tree_clf = DecisionTreeClassifier(random_state=42)
tree_cv = GridSearchCV(tree_clf, params, scoring="accuracy", n_jobs=-1, verbose=1, cv=10)
tree_cv.fit(x_train, y_train)
best_params = tree_cv.best_params_
print(f"Best paramters: {best_params})")

tree_clf = DecisionTreeClassifier(**best_params)
tree_clf.fit(x_train, y_train)
print_score(tree_clf, x_train, y_train, x_test, y_test, train=True)
print_score(tree_clf, x_train, y_train, x_test, y_test, train=False)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)


base_estimator = DecisionTreeClassifier(max_depth=1)
n_estimators = 50
learning_rate = 1

ada = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, learning_rate=learning_rate)

ada.fit(X_train, y_train)

y_pred = ada.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)



# Import necessary libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Create KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the model with training data
knn.fit(X_train, y_train)

# Predict the classes of testing data
y_pred = knn.predict(X_test)

# Evaluate the accuracy of the model
print("Accuracy:", knn.score(X_test, y_test))


# Import necessary libraries
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Create Gradient Boosting classifier
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Fit the model with training data
gb.fit(X_train, y_train)

# Predict the classes of testing data
y_pred = gb.predict(X_test)

# Evaluate the accuracy of the model
print("Accuracy:", gb.score(X_test, y_test))


# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load iris dataset
iris = load_iris()

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Create Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model with training data
rf.fit(X_train, y_train)

# Predict the classes of testing data
y_pred = rf.predict(X_test)

# Evaluate the accuracy of the model
print("Accuracy:", rf.score(X_test, y_test))

# In[120]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




base_estimator = DecisionTreeClassifier(max_depth=1)
n_estimators = 50
learning_rate = 1

ada = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, learning_rate=learning_rate)

ada.fit(x_train, y_train)

y_pred = ada.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred) )


# In[121]:


# Import necessary libraries
from sklearn.neighbors import KNeighborsClassifier


# Create KNN classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Fit the model with training data
knn.fit(x_train, y_train)

# Predict the classes of testing data
y_pred = knn.predict(x_test)

# Evaluate the accuracy of the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(
      classification_report(y_test, y_pred))


# In[122]:


# Import necessary libraries
from sklearn.ensemble import GradientBoostingClassifier



# Create Gradient Boosting classifier
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Fit the model with training data
gb.fit(x_train, y_train)

# Predict the classes of testing data
y_pred = gb.predict(x_test)

# Evaluate the accuracy of the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred) )


# In[123]:


# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier


# Create Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model with training data
rf.fit(x_train, y_train)

# Predict the classes of testing data
y_pred = rf.predict(x_test)

# Evaluate the accuracy of the model
print("Accuracy:", accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred) )


# In[124]:


from sklearn.ensemble import GradientBoostingClassifier



# Create Gradient Boosting classifier
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Fit the model with training data
gb.fit(x_train, y_train)

# Predict the classes of testing data
y_pred = gb.predict(x_test)

# Evaluate the accuracy of the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred) )


# ## Model Evaluation
Since we're getting highest accuracy score in RandomForestClassifier
# In[125]:


# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier


# Create Random Forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model with training data
rf.fit(x_train, y_train)

# Predict the classes of testing data
y_pred = rf.predict(x_test)

# Evaluate the accuracy of the model
print("Accuracy:", accuracy_score(y_pred, y_test))
print(classification_report(y_test, y_pred) )


# In[126]:


n_estimators = [int(x) for x in np.linspace(start=0, stop=200, num=200)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators, 'max_features': max_features,
               'max_depth': max_depth, 'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf, 'bootstrap': bootstrap}

rf_clf = RandomForestClassifier(random_state=42)

rf_cv = RandomizedSearchCV(estimator=rf_clf, scoring='f1',param_distributions=random_grid, n_iter=100, cv=3, 
                               verbose=2, random_state=42, n_jobs=-1)

rf_cv.fit(x_train, y_train)
rf_best_params = rf_cv.best_params_
print(f"Best paramters: {rf_best_params})")

rf_clf = RandomForestClassifier(**rf_best_params)
rf_clf.fit(x_train, y_train)

print_score(rf_clf, x_train, y_train, x_test, y_test, train=True)
print_score(rf_clf, x_train, y_train, x_test, y_test, train=False)


# ### Saving the Trained Model

# In[138]:


import joblib
joblib.dump(model, "customer_churn_model.pkl")


# In[139]:


# Loading the saved model
model = joblib.load('customer_churn_model.pkl')


# In[140]:


import streamlit as st
# Creating the user interface
st.title("Customer Churn Prediction App")


# In[141]:


st.sidebar.header("Enter Customer Details")


# In[142]:


# Creating input fields for customer information
day_mins = st.sidebar.number_input("Day Mins")
voice_messages = st.sidebar.number_input("Voice Messages")
day_charge = st.sidebar.number_input("Day Charge")
eve_mins = st.sidebar.number_input("Eve Mins")
intl_plan_yes = st.sidebar.radio("International Plan (Yes/No)", ["Yes", "No"])
customer_calls = st.sidebar.number_input("Customer Calls")
night_mins = st.sidebar.number_input("Night Mins")
voice_plan_yes = st.sidebar.radio("Voice Plan (Yes/No)", ["Yes", "No"])
eve_charge = st.sidebar.number_input("Eve Charge")
intl_plan_no = 1 if intl_plan_yes == "No" else 0
account_length = st.sidebar.number_input("Account Length")


# In[143]:


# Creating a dictionary for the input data
data = {"day.mins": day_mins, "voice.messages": voice_messages, "day.charge": day_charge, "eve.mins": eve_mins,
        "intl.plan_yes": 1 if intl_plan_yes == "Yes" else 0, "customer.calls": customer_calls, "night.mins": night_mins,
        "voice.plan_yes": 1 if voice_plan_yes == "Yes" else 0, "eve.charge": eve_charge, "intl.plan_no": intl_plan_no,
        "account.length": account_length}
input_data = pd.DataFrame(data, index=[0])


# In[144]:


# Selecting only the required features from the input data
model_input = input_data[['day.mins', 'voice.messages', 'day.charge', 'eve.mins', 'intl.plan_yes', 'customer.calls',
                          'night.mins', 'voice.plan_yes', 'eve.charge', 'intl.plan_no', 'account.length']]


# In[145]:


# Predicting customer churn
if st.sidebar.button("Predict"):
    prediction = model.predict(model_input)
    if prediction[0] == 0:
        st.sidebar.success("This customer is not likely to churn.")
    else:
        st.sidebar.error("This customer is likely to churn.")


# In[ ]:





# In[ ]:




