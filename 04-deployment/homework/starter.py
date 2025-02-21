#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip freeze | grep scikit-learn')


# In[3]:


get_ipython().system('python -V')


# In[ ]:


get_ipython().system('pip install pyarrow')


# In[5]:


import pickle
import pandas as pd


# In[6]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[7]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[8]:


year = 2023
month = 3


# In[9]:


get_ipython().system('mkdir output')


# In[10]:


output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'


# In[11]:


df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')


# In[12]:


df.head()


# In[13]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[14]:


y_pred


# ## Q1. Std deviation

# In[15]:


y_pred.std()


# ## Q2. Preparing the output

# In[16]:


df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')


# In[17]:


df.head()


# In[18]:


df_result = pd.DataFrame()
df_result['ride_id'] = df['ride_id']
df_result['predicted_duration'] = y_pred


# In[22]:


df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


# total size: 66M
