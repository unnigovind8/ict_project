#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


ele_data = pd.read_csv("Kerala_GE_Project.csv")


# In[3]:


ele_data.info()


# In[4]:


ele_data.shape


# In[5]:


ele_data.describe()


# In[6]:


ele_data.isna().sum()


# In[7]:


# fillining education details

ele_data["MyNeta_education"].fillna("Not Available", inplace=True)


# In[8]:


ele_data.dropna(axis=0, subset=["Party"], inplace=True)


# In[9]:


#ele_data["Sex"].fillna(ele_data["Sex"].mode()[0], inplace=True)


# In[10]:


ele_data["Year"]


# In[11]:


# duplicates
ele_data.duplicated().sum()


# In[12]:


'''
#Outlier Analysis

plt.figure(figsize=(10,8))
plt.suptitle("Outlier analysis of different features: ", ha = "right")
plt.subplot(2,2,1)
sns.boxplot(x = ele_data["Valid_Votes"])

plt.subplot(2,2,2)
sns.boxplot(x = ele_data["Electors"])

plt.subplot(2,2,3)
sns.boxplot(x = ele_data["Margin"])

plt.subplot(2,2,4)
sns.boxplot(x = ele_data["Turnout_Percentage"])

plt.show()
'''


# In[ ]:





# In[13]:


#adding result column

ele_data["Result"] = 0

result = []

for i in ele_data["Position"]:
    if i == 1:
        result.append(1)
    else:
        result.append(0)


# In[14]:


ele_data[ele_data["Position"]==1].shape


# In[15]:


ele_data["Result"] = result

print(ele_data["Result"].value_counts())


# In[16]:


ele_data.head()


# In[17]:


#help(str.split)


# In[18]:


# Splitting candidate name,removing details other than name

ele_data["Candidate"] = ele_data["Candidate"].str.split(",", expand=True)[0]


# In[19]:


ele_data[ele_data["Candidate"].str.contains(",")]


# In[20]:


# Splitting Constituency_name column

ele_data["Constituency_Name"] = ele_data["Constituency_Name"].str.split(",", expand=True)[0]


# In[21]:


ele_data.loc[551,["Constituency_Name"]]


# In[ ]:





# In[22]:


ele_data["Constituency_Name"].value_counts()


# In[23]:


# replacing

ele_data["Constituency_Name"].replace("ERNAKULUM", "ERNAKULAM", inplace=True)

ele_data["Constituency_Name"].replace("KASARGOD","KASARAGOD", inplace=True)

ele_data["Constituency_Name"].replace("KASERGOD","KASARAGOD", inplace=True)

ele_data["Constituency_Name"].replace(["TRIVANDRUM","TRIVANDURAM"],"THIRUVANANTHAPURAM", inplace=True)

ele_data["Constituency_Name"].replace("POONANI","PONNANI", inplace=True)

ele_data["Constituency_Name"].replace(["MUVATTU PUZHA","MUVATTPUZHA"],"MUVATTUPUZHA", inplace=True)

ele_data["Constituency_Name"].replace("QULLON","QUILON", inplace=True)

ele_data["Constituency_Name"].replace("MUKANDAPURAM", "MUKUNDAPURAM", inplace=True)

ele_data["Constituency_Name"].replace("CALICUT","KOZHIKODE", inplace=True)

ele_data["Constituency_Name"].replace(["BADAGARA","VADAKARA"],"VATAKARA", inplace=True)

ele_data["Constituency_Name"].replace("CANNANORE","KANNUR", inplace = True)

ele_data["Constituency_Name"].replace(["MAVILEKARA","MAVELIKKARA"], "MAVELIKARA", inplace=True)

ele_data["Constituency_Name"].replace("OTTAPPALAM(SC)","OTTAPALAM", inplace = True)

ele_data["Constituency_Name"].replace("MALAPPURAM ","MALAPPURAM", inplace = True)

ele_data["Constituency_Name"].replace("TELLI CHERRY","TELLICHERRY", inplace = True)


# In[24]:


ele_data["Constituency_Name"].unique()


# In[25]:


ele_data.head()


# In[ ]:





# In[26]:


# no of years from the data

ax = sns.countplot(x = ele_data["Year"].sort_values())
ax.bar_label(ax.containers[0])
plt.xticks(rotation = 90)
plt.show()


# In[ ]:





# In[27]:


win_party = ele_data[ele_data["Position"] == 1]
win_party


# In[28]:


# election won by differnt parties from 1962-2021
ax = sns.barplot(y = win_party["Party"].value_counts().values, x = win_party["Party"].value_counts().index)
ax.bar_label(ax.containers[0])
plt.xticks(rotation=90)
plt.show()


# In[ ]:





# In[29]:


# no of electors in each year from all the constituents

electors = ele_data.groupby("Year")["Electors"].sum()



# In[30]:


plt.figure(figsize=(12, 6))

plt.plot(electors.index, electors.values)
plt.ticklabel_format(style='plain', axis='y')
plt.tight_layout()


# In[ ]:





# In[31]:


# comparision between 2019 Constituency electors and 2018 Constituency electors
electors_yr = ele_data[ele_data["Year"] == 2019]

yr = electors_yr.groupby("Constituency_Name")["Electors"].max()
yr.index

# 1962 Constituency electors
electors_yr = ele_data[ele_data["Year"] == 2014]

yrs = electors_yr.groupby("Constituency_Name")["Electors"].max()
yrs.index


# In[32]:


plt.figure(figsize=(12, 10))
plt.subplot(2,1,1)
plt.plot(yr.index, yr.values)
plt.ticklabel_format(style='plain', axis='y')
plt.xticks(rotation = 90)
plt.tight_layout()

plt.subplot(2,1,2)
plt.plot(yrs.index, yrs.values)
plt.ticklabel_format(style='plain', axis='y')
plt.xticks(rotation = 90)
plt.tight_layout()


# In[ ]:





# In[33]:


ele_2019 = ele_data[ele_data["Year"] == 2019]
winparty_2019 = ele_2019.loc[ele_2019.groupby('Constituency_Name')['Votes'].idxmax()][['Constituency_Name', 'Party']]

ele_2014 = ele_data[ele_data["Year"] == 2014]
winparty_2014 = ele_2014.loc[ele_2014.groupby('Constituency_Name')['Votes'].idxmax()][['Constituency_Name', 'Party']]


# In[34]:


# comparing winning party in each constitution in the years 2019 and 2014

plt.figure(figsize=(12, 10))

plt.subplot(2,1,1)
plt.bar(winparty_2019['Constituency_Name'], winparty_2019['Party'], color='salmon')
plt.title('Winning Party in Each Constituency in Year 2019')
plt.xlabel('Constituency')
plt.ylabel('Winning Party')
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
plt.tight_layout()

plt.subplot(2,1,2)
plt.bar(winparty_2014['Constituency_Name'], winparty_2014['Party'], color='salmon')
plt.title('Winning Party in Each Constituency in Year 2014')
plt.xlabel('Constituency')
plt.ylabel('Winning Party')
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
plt.tight_layout()


plt.show()


# In[ ]:





# In[35]:


# Identify the winning candidate in each constituency for each year
winning_candidates = ele_data.loc[ele_data.groupby(['Year', 'Constituency_Name'])['Votes'].idxmax()]

# Create a count plot to visualize the distribution of winning parties
plt.figure(figsize=(12, 6))
sns.countplot(x='Party', hue='Result', data=winning_candidates)
plt.title('Distribution of Winning Parties')
plt.xlabel('Party')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Winner')
plt.tight_layout()
plt.show()


# In[36]:


ele_data["Party"].replace("INDEPENDENT", "IND", inplace=True)


# In[37]:


ele_data[ele_data['Party_Type_TCPD'].isna()]


# In[38]:


# filling null values in "Party type TCPD" col

partynull = ele_data["Party_Type_TCPD"]

partynull = np.array(partynull)
partynull


# In[39]:


conditions = [
    (ele_data['Party'] == 'igp') | (ele_data['Party'] == 'SDPI'),
    (ele_data['Party'] == 'IUML'),
    (ele_data['Party'] == 'CPM') | (ele_data['Party'] == 'BJP'),
    (ele_data['Party'] == 'IND'),
    (ele_data['Party'] == 'NOTA')
]


choices = ['Local Party', 'State-based Party', 'National Party', 'Independents','NOTA']

ele_data["Party_Type_TCPD"] = np.select(conditions, choices, default=ele_data["Party_Type_TCPD"])


# In[40]:


ele_data


# In[41]:


ele_data[ele_data['Party_Type_TCPD'].isna()]


# In[42]:


ele_data[ele_data['Party']=='igp']


# In[43]:


ele_data[ele_data["Party_Type_TCPD"] == "National Party"]["Party"].nunique()


# In[44]:


# violin plot for understanding the relation ships btw " Party_type_TCPD and Result"


#  A higher value in Kruskal-Wallis Test indicates more significant difference.
#  there are systematic variations in the election results associated with different party types.

from scipy.stats import kruskal

party_national = ele_data[ele_data["Party_Type_TCPD"] == "National Party"]


plt.figure(figsize=(12,6))
sns.violinplot(data=party_national, y="Result", x = "Party")



# Perform Kruskal-Wallis test
result_by_party_type = [ele_data['Result'][ele_data['Party_Type_TCPD'] == party_type] for party_type in ele_data['Party_Type_TCPD'].unique()]
kruskal_stat, p_value = kruskal(*result_by_party_type)

# Print the test result
print(f"Kruskal-Wallis Test: Statistic = {kruskal_stat}, p-value = {p_value}")


# In[45]:


# violin plot for understanding the relation ships btw "Party and Results"

#  A higher value in Kruskal-Wallis Test indicates more significant difference.
#  there are systematic variations in the election results associated with different party types.

from scipy.stats import kruskal

party_national = ele_data[ele_data["Party"].isin(["INC","CPM","BJP"])]


plt.figure(figsize=(12,6))
sns.violinplot(data=party_national, y="Result", x = "Party")



# Perform Kruskal-Wallis test
result_by_party_type = [ele_data['Result'][ele_data['Party'] == party_type] for party_type in ele_data['Party'].unique()]
kruskal_stat, p_value = kruskal(*result_by_party_type)

# Print the test result
print(f"Kruskal-Wallis Test: Statistic = {kruskal_stat}, p-value = {p_value}")


# In[46]:


ele_data.head()


# In[47]:


ele_data["Incumbent"].fillna('NOTA',inplace=True)


# In[48]:


# violin plot for understanding the relation ships btw "Incumbent and Results"

#  A higher value in Kruskal-Wallis Test indicates more significant difference.
#  there are systematic variations in the election results associated with different party types.

from scipy.stats import kruskal

#party_national = ele_data[ele_data["Party"].isin(["INC","CPM","BJP"])]


plt.figure(figsize=(12,6))
sns.violinplot(data=ele_data, y="Result", x = "Incumbent")



# Perform Kruskal-Wallis test
result_by_party_type = [ele_data['Result'][ele_data['Incumbent'] == incumbent_type] for incumbent_type in ele_data['Incumbent'].unique()]
kruskal_stat, p_value = kruskal(*result_by_party_type)

# Print the test result
print(f"Kruskal-Wallis Test: Statistic = {kruskal_stat}, p-value = {p_value}")


# In[ ]:





# In[49]:


ele_data["Turncoat"].fillna('NOTA',inplace=True)


# In[50]:


# violin plot for understanding the relation ships btw " Turncoat and Results"

#  A higher value in Kruskal-Wallis Test indicates more significant difference.
#  there are systematic variations in the election results associated with different party types.

from scipy.stats import kruskal

#party_national = ele_data[ele_data["Party"].isin(["INC","CPM","BJP"])]


plt.figure(figsize=(12,6))
sns.violinplot(data=ele_data, y="Result", x = "Turncoat")



# Perform Kruskal-Wallis test
result_by_party_type = [ele_data['Result'][ele_data['Turncoat'] == turncoat_type] for turncoat_type in ele_data["Turncoat"].unique()]
kruskal_stat, p_value = kruskal(*result_by_party_type)

# Print the test result
print(f"Kruskal-Wallis Test: Statistic = {kruskal_stat}, p-value = {p_value}")


# In[ ]:





# In[51]:


# violin plot for understanding the relation ships btw " Contituency_name and Results"

#  A higher value in Kruskal-Wallis Test indicates more significant difference.
#  there are systematic variations in the election results associated with different party types.

from scipy.stats import kruskal

constituency_name = ele_data[ele_data["Constituency_Name"].isin(["KASARAGOD","KANNUR","WAYANAD"])]


plt.figure(figsize=(12,6))
sns.violinplot(data=constituency_name, y="Result", x = "Constituency_Name", cut=0)



# Perform Kruskal-Wallis test
result_by_party_type = [ele_data['Result'][ele_data['Constituency_Name'] == consti_type] for consti_type in ele_data["Constituency_Name"].unique()]
kruskal_stat, p_value = kruskal(*result_by_party_type)

# Print the test result
print(f"Kruskal-Wallis Test: Statistic = {kruskal_stat}, p-value = {p_value}")


# In[ ]:





# In[52]:


ele_data["Sex"].replace("male", "M", inplace=True)


# In[53]:


ele_data["Sex"].fillna("NOTA", inplace=True)


# In[54]:


# violin plot for understanding the relation ships btw "Sex and Results"

#  A higher value in Kruskal-Wallis Test indicates more significant difference.
#  there are systematic variations in the election results associated with different party types.

from scipy.stats import kruskal

#party_national = ele_data[ele_data["Party"].isin(["INC","CPM","BJP"])]


plt.figure(figsize=(12,6))
sns.violinplot(data=ele_data, y="Result", x = "Sex", cut=0)



# Perform Kruskal-Wallis test
result_by_party_type = [ele_data['Result'][ele_data['Sex'] == gender] for gender in ele_data['Sex'].unique()]
kruskal_stat, p_value = kruskal(*result_by_party_type)

# Print the test result
print(f"Kruskal-Wallis Test: Statistic = {kruskal_stat}, p-value = {p_value}")


# In[55]:


#[ele_data['Result'][ele_data['Sex'] == party_type]for party_type in ele_data['Sex'].unique()]


# In[56]:


plt.figure(figsize=(12,6))
sns.heatmap(data = ele_data.corr(),annot=True)


# In[57]:


ele_data.info()


# ### encoding

# In[58]:


ele_features = ele_data[["Year","Sex","Party","Constituency_Name","Party_Type_TCPD","Incumbent","Result"]]

#ele_features = ele_data[["Year","Sex","Party","Constituency_Name","Party_Type_TCPD","Result"]]


# In[59]:


# label encoding

from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()


# In[60]:


encode_values = ele_features.copy()


# In[61]:


encode_values["Incumbent"].replace({True:"Yes", False:"No"}, inplace=True)


# In[62]:


# will be using for encoding new candidate prediction
prediction_values = encode_values.copy()


# downloading prediction_values as csv file for flask
# prediction_values.to_csv("E:/DA/ict_dsa/project/prediction_values.csv")


# In[63]:


sex = encoder.fit_transform(encode_values["Sex"])
encode_values["Sex"] = sex


# In[64]:


encode_values["Incumbent"]


# In[65]:


encode_values["Party"] = encoder.fit_transform(encode_values["Party"])

encode_values["Constituency_Name"] = encoder.fit_transform(encode_values["Constituency_Name"])

encode_values["Party_Type_TCPD"] = encoder.fit_transform(encode_values["Party_Type_TCPD"])

encode_values["Incumbent"] = encoder.fit_transform(encode_values["Incumbent"])


# In[ ]:





# In[66]:


# downloading encoded values of prediction_values as csv file for scaling in flask
#encode_values.to_csv("E:/DA/ict_dsa/project/tobe_scaled.csv")


# ### Scaling

# In[67]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


# In[68]:


x = encode_values.drop(columns="Result")
y = encode_values["Result"]


# In[69]:


#y = y.values.reshape(-1,1)
x1 = x.copy()

x = scaler.fit_transform(x)
#y = scaler.fit_transform(y)


# In[70]:


#splitting
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=True)


# In[ ]:





# ### Modeling

# In[71]:


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier()
rf_model.fit(x_train,y_train)


# In[72]:


rf_predict = rf_model.predict(x_test)


# In[73]:


# evaluation

from sklearn.metrics import accuracy_score, classification_report

print("Accuracy: ",  accuracy_score(rf_predict,y_test))
print("\n Claaification report: \n", classification_report(rf_predict,y_test))


# In[74]:


ele_data[ele_data["Result"] == 1]


# In[75]:


ele_data[(ele_data["Incumbent"] == True) & (ele_data["Result"] == 1)]


# In[76]:


ele_data.loc[37]


# In[77]:


# prediction  1

candidate_prediction = {
    "Year": 2019,
    "Sex": "M",
    "Party": "INC",
    "Constituency_Name": "WAYANAD",
    "Party_Type_TCPD": "National Party",
    "Incumbent": "Yes"
}


candidate_predictdf = pd.DataFrame([candidate_prediction])


encoder.fit_transform(prediction_values["Sex"])
candidate_predictdf['Sex'] = encoder.transform(candidate_predictdf['Sex'])


encoder.fit_transform(prediction_values["Party"])
candidate_predictdf['Party'] = encoder.transform(candidate_predictdf['Party'])  

encoder.fit_transform(prediction_values["Constituency_Name"])
candidate_predictdf['Constituency_Name'] = encoder.transform(candidate_predictdf['Constituency_Name'])

encoder.fit_transform(prediction_values["Party_Type_TCPD"])
candidate_predictdf['Party_Type_TCPD'] = encoder.transform(candidate_predictdf['Party_Type_TCPD'])  

encoder.fit_transform(prediction_values["Incumbent"])
candidate_predictdf['Incumbent'] = encoder.transform(candidate_predictdf['Incumbent'])  



scaler.fit_transform(x1)
candidate_predict_scaled = scaler.transform(candidate_predictdf)


# In[78]:


#original scaled data
x[37]


# In[79]:


# scaled data for new candidate prediction
candidate_predict_scaled


# In[80]:


prediction = rf_model.predict(candidate_predict_scaled)

# Print the prediction
print("Predicted Result:", prediction[0])


# In[81]:


candidate_predictdf


# In[82]:


encode_values.loc[2361]


# In[83]:


# prediction 2

new_candidate_data = {
   "Year": 2019,
    "Sex": "M",
    "Party": "INC",
    "Constituency_Name": "KASARAGOD",
    "Party_Type_TCPD": "National Party",
    "Incumbent": "No"
}

# Convert input data to DataFrame
new_candidate_df = pd.DataFrame([new_candidate_data])
'''
# Encode categorical columns (using transform instead of fit_transform)
new_candidate_df['Sex'] = encoder.transform(new_candidate_df['Sex'])
new_candidate_df['Party'] = encoder.transform(new_candidate_df['Party'])
new_candidate_df['Constituency_Name'] = encoder.transform(new_candidate_df['Constituency_Name'])
new_candidate_df['Party_Type_TCPD'] = encoder.transform(new_candidate_df['Party_Type_TCPD'])
new_candidate_df['Incumbent'] = encoder.transform(new_candidate_df['Incumbent'])

# Scale numerical features
new_candidate_features_scaled = scaler.transform(new_candidate_df)  # Assuming 'scaler' is the trained StandardScaler

# Make predictions
new_candidate_result = model.predict(new_candidate_features_scaled)

# Print the prediction
print("Predicted Result for the new candidate:", new_candidate_result)
'''


# In[ ]:





# In[84]:


encode_values["Sex"].value_counts()


# In[85]:


ele_features["Sex"].value_counts()


# In[86]:


encoder.fit_transform(ele_features["Sex"])
new_candidate_df['Sex'] = encoder.transform(new_candidate_df["Sex"])


# In[87]:


encoder.fit_transform(ele_features["Party"])
new_candidate_df['Party'] = encoder.transform(new_candidate_df["Party"])


# In[88]:


encoder.fit_transform(ele_features["Party_Type_TCPD"])
new_candidate_df['Party_Type_TCPD'] = encoder.transform(new_candidate_df["Party_Type_TCPD"])


# In[89]:


encoder.fit_transform(ele_features["Constituency_Name"])
new_candidate_df['Constituency_Name'] = encoder.transform(new_candidate_df["Constituency_Name"])


# In[90]:


ele_features["Incumbent"].replace({True:"Yes", False:"No"}, inplace=True)

encoder.fit_transform(ele_features["Incumbent"])
new_candidate_df['Incumbent'] = encoder.transform(new_candidate_df["Incumbent"])


# In[91]:


# Scale numerical features
new_candidate_features_scaled = scaler.transform(new_candidate_df)  # Assuming 'scaler' is the trained StandardScaler

# Make predictions
new_candidate_result = rf_model.predict(new_candidate_features_scaled)

# Print the prediction
print("Predicted Result for the new candidate:", new_candidate_result)


# In[92]:


new_candidate_df


# In[93]:


new_candidate_features_scaled


# In[94]:


encode_values.loc[0]


# In[95]:


# remaining in colab, since it would be easier for developing flask application


# In[ ]:





# In[96]:


# pickle

import pickle

#pickle.dump(rf_model,open('rf_newmodel.pkl','wb'))


# In[97]:


pickled_model = pickle.load(open('rfmodel.pkl','rb'))

pickled_model.predict(candidate_predict_scaled)


# In[98]:


# pkl files of label encoder and scalar

#label encoder
#pickle.dump(encoder,open('l_encoder.pkl','wb'))

#scalar
#pickle.dump(scaler,open('std_scalar.pkl','wb'))


# In[ ]:





# In[99]:


#!pip install --upgrade scikit-learn


# In[ ]:





# In[100]:


#trying out sampling techniques

ele_data["Incumbent"].value_counts()


# In[ ]:




