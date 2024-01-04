#!/usr/bin/env python
# coding: utf-8

# In[2]:


from ast import literal_eval
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import association_rules
from time import time
import pandas as pd

df = pd.read_csv("data.csv")
df['PRODUCT_TYPE'] = df['PRODUCT_TYPE'].apply(literal_eval)
data = df['PRODUCT_TYPE'].tolist()
te = TransactionEncoder()
te_ary = te.fit(data).transform(data)

df_a = pd.DataFrame(te_ary, columns=te.columns_)

#設定不同的信心度與支持度
support_values = [0.001, 0.005, 0.01]
confidence_values = [0.1, 0.2, 0.5]

results = []

#測試不同的參數組合
for support in support_values:
    for confidence in confidence_values:
        start_time = time()

        frequent_itemsets = apriori(df_a, min_support=support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)

        #檢查冗餘規則並剔除
        non_redundant_rules = []
        for idx, rule in rules.iterrows():
            antecedent = rule['antecedents']
            consequent = rule['consequents']

            is_redundant = any(
                ((antecedent < other_rule['antecedents']) and (rule['confidence'] <= other_rule['confidence']))
                for other_idx, other_rule in rules.iterrows()
                if idx != other_idx
            )
            if not is_redundant:
                non_redundant_rules.append(rule)
                
        end_time = time()
        execution_time = end_time - start_time
        
        #原始規則數量
        total_rules = len(rules)
        print("總共有"+str(total_rules)+"條規則")
        
        #剩餘規則數量
        remaining_rules = len(non_redundant_rules)

        print(f"支持度為{support}，信心度為{confidence}時，保留的規則數量為: {remaining_rules},花費時間為{execution_time}")

        
        
        results.append({
            'support': support,
            'confidence': confidence,
            'rule_count': remaining_rules,
            'execution_time': execution_time
        })

results_df = pd.DataFrame(results)


# In[3]:


recommended_products = []

for support in support_values:
    for confidence in confidence_values:
        frequent_itemsets = apriori(df_a, min_support=support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=confidence)

        #每條規則的推薦產品數量
        products_covered = [len(rule['consequents']) for idx, rule in rules.iterrows()]

        #總共的推薦產品數量
        total_products = sum(products_covered)

        recommended_products.append({
            'support': support,
            'confidence': confidence,
            'total_products': total_products,
            'total_rule_products': sum(df_a.sum(axis=0)),
            'rule_count': len(rules)
        })

products_df = pd.DataFrame(recommended_products)
print(products_df)


# In[4]:


frequent_itemsets = apriori(df_a, min_support=0.001, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)


# In[5]:


rules.to_csv('rule.csv')
loaded_rules = pd.read_csv('rule.csv')


# In[6]:


print(loaded_rules)


# In[7]:


def generate_recommendations(input_product):
    valid_products = [
        'MEMORY_EMBEDED', 'CPU / MPU', 'DISCRETE', 'PEMCO',
        'LOGIC IC', 'LINEAR IC', 'OPTICAL AND SENSOR', 'CHIPSET / ASP',
        'MEMORY_SYSTEM', 'OTHERS'
    ]
    
    if input_product not in valid_products:
        print("請輸入正確的產品名稱")
        return
    
    relevant_rules = rules[rules['antecedents'].apply(lambda x: input_product in x)]
    relevant_rules = relevant_rules.sort_values(by='confidence', ascending=False)
        
    if len(relevant_rules) > 0:
        recommendations = relevant_rules.iloc[0]['consequents']
        recommendations_list = list(recommendations)
        print(f"推薦給您的產品: {recommendations_list}")
    else:
        print("無可推薦產品")


options = [
    'MEMORY_EMBEDED', 'CPU / MPU', 'DISCRETE', 'PEMCO',
    'LOGIC IC', 'LINEAR IC', 'OPTICAL AND SENSOR', 'CHIPSET / ASP',
    'MEMORY_SYSTEM', 'OTHERS'
]
print("以下為所有產品")
for idx, option in enumerate(options, start=1):
    print( f"({idx}){option}")

user_input_product = input("輸入產品名稱: ")
generate_recommendations(user_input_product)

