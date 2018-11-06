
# coding: utf-8
    I will be using pandas, Clustering with k means, hence I would need numbers in each attribute as follows:
    Sex: Male-1, Female-2, Unknown-0
    Age: 0 to 20 - 1; 20 to 40 - 2; age > 50 - 3; Unknown-0
    Position: Driver-1, Rider-2, Unknown-0
    Safety: Having proper seatbelt/helmet-1, No seatbelt/Helmet-2, Unknown-0
    
    Target is the severity of the accident, this is represented by the attribute "thrown" which describes if the 
    victim was thrown out. Represented by:
    Thrown: Not Thrown Out-1, Thrown Out-2, NaN-0
    
    My columns = sex, age, position, safety
# In[7]:


import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

# Importing my CSV FILE
# In[8]:


data = pd.read_csv("C:/Users/Abhi/Downloads/data/cas - Copy.csv")


# This is to use the series by names

# In[9]:


data.columns = ["report_id","unit_number","casualty_number","casualty_type", "sex","age","position","thrown", "injury","seatbelt","helmet","hospital"]


# Dropping irrelevant attributes

# In[ ]:


data = data.drop(['report_id', 'unit_number','casualty_number', 'casualty_type','injury','hospital'],axis = 1)


# Cleaning

# In[ ]:


#This function gives a list of values that a serie has for me to check
def vallist(attr):
    values = []
    for i in attr:
        if i not in values:
            values.append(i)
    print(values)


# Cleaning age

# In[ ]:


#data.head()


# In[ ]:


data['age'] = pd.to_numeric(data['age'],errors = 'coerce')


# In[ ]:


#new dataframe for plotting purposes
data = data.fillna(0)
age=[]
for i in data.age:
    age.append(i)
#print(age)


# In[ ]:


#Continue cleaning
data.loc[data['age'] < 20,'age'] = 1
data.loc[data['age'] > 50,'age'] = 3
data.loc[data['age'].between(20,50, inclusive = True),['age']] = 2


# In[ ]:


data['age'] = data.age.map({1: 1, 2: 2,3:3 }).fillna(0).astype(int)


# Cleaning "sex"

# In[ ]:


data['sex'] = data.sex.map({'Male': 1, 'Female': 2 }).fillna(0).astype(int)


# Cleaning position

# In[ ]:


data.loc[data['position'].str.contains("Rider")==True, ['position']] = 1
data.loc[data['position'].str.contains("Passenger")==True, ['position']] = 2
data.loc[data['position'].str.contains("Driver")==True, ['position']] = 1


# In[ ]:


data['position'] = data.position.map({1: 1, 2: 2 }).fillna(0).astype(int)


# I will introduce a new column "safety" that will process information from helmet and seatbelt: 
#     Have helmet/seatbelt worn:1
#     Didn't wear them: 2
#     Blank data: 0

# In[ ]:


data.loc[data['seatbelt'].str.contains(" - Not Worn")==True,['seatbelt']] = 2
data.loc[data['seatbelt'].str.contains("Not Fitted")==True,['seatbelt']] = 2
data.loc[data['seatbelt'].str.contains(" - Worn")==True,['seatbelt']] = 1
data.loc[data['seatbelt'].str.contains("Unknown")==True,['seatbelt']] = 0


# In[ ]:


data['seatbelt'] = data.seatbelt.map({1: 1, 2: 2 }).fillna(0).astype(int)


# In[ ]:


data.loc[data['helmet'].str.contains("Not Worn")==True,['helmet']] = 2
data.loc[data['helmet'].str.contains("Worn")==True,['helmet']] = 1


# In[ ]:


data['helmet'] = data.helmet.map({1: 1, 2: 2 }).fillna(0).astype(int)


# In[ ]:


data['safety'] = data['seatbelt']+data['helmet']


# Now we do not need the seatbelt and helmet categories, so we drop that.

# In[ ]:


data = data.drop(['seatbelt','helmet'], axis=1)


# Modifying the target attribute

# In[ ]:


data.loc[data['thrown']=="Thrown Out",['thrown']] = 2
data.loc[data['thrown']=="Not Thrown Out",['thrown']] = 1


# In[ ]:


data['thrown'] = data.thrown.fillna(0).astype(int)


# Now I attempt to remove the missing values

# In[ ]:


#data = data.convert_objects(convert_numeric=True)


# In[ ]:


#data = data.replace(0,np.NaN)


# In[ ]:


# Dropping the missing values before clustering
#data = data.dropna()


# In[ ]:


#New data frame to store the target and clusters
cluster_map = pd.DataFrame()
cluster_map['Thrown'] = data.thrown


# Dropping the target before clustering

# In[ ]:


data = data.drop(["thrown"], axis=1)


# Clustering using kmeans requires me to have int datatype

# In[ ]:


data = data.astype(int)


# Creating Clusters

# In[ ]:


km = KMeans(n_clusters=2).fit(data)


# Storing the clusters

# In[ ]:


cluster_map['cluster'] = km.labels_
data['cluster'] = cluster_map.cluster
#dataframe['cluster'] = cluster_map.cluster
data['thrown'] = cluster_map.Thrown


# Analysing the results

# In[ ]:


print(age[0:10])


# In[ ]:


agep = age[0:10]
sex = data.sex
Cluster = data.cluster
centers = np.random.randn(4, 2) 

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(age,sex,c=Cluster,s=50)
#for i,j in centers:
#    ax.scatter(i,j,s=50,c='red',marker='+')
ax.set_xlabel('age')
ax.set_ylabel('sex')
plt.colorbar(scatter)
fig.show()


# This checks how many data points were correctly classified into two clusters based on the severity of the accident. It is not the best way to do this but as our target variable too, has only two possible values; not severe, severe it works.
# 

# In[ ]:


correct = 0
for i in cluster_map.cluster:
        for j in cluster_map.Thrown:
            if i+1 == j:
                correct = correct+1
                break
            else:
                break
print(correct/len(cluster_map))


# No need to run this, this is just for adding the cluster class to the original data! 

# In[ ]:


#dataframe.to_csv("C:/Users/Abhi/Downloads/data/completed.csv", encoding='utf-8')


# In[ ]:


data.head


# For weka

# In[ ]:


#data = data.drop(['cluster'], axis = 1)


# In[ ]:


#data.to_csv("C:/Users/Abhi/Downloads/data/analysiequick.csv", encoding='utf-8')


# In[ ]:


dataframe = pd.read_csv("C:/Users/Abhi/Downloads/data/cas.csv")

