#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd

df = pd.read_excel("data.xlsx")

df = df[df['QUANTITY'] > 0]

df = df.drop(columns=["CUST_ID", "ITEM_ID", "ITEM_NO", "TRX_DATE", "QUANTITY"])
df.info()
grouped_df = df.groupby('INVOICE_NO')['PRODUCT_TYPE'].agg(list).reset_index()
grouped_df.head()

grouped_df.to_csv("data.csv", index=False)


# In[6]:




