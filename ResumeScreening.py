# Decision Tree Classification
import numpy as np
import pandas as pd
from sklearn import tree

df = pd.read_csv("DecisionTreesClassificationDataSet.csv")

# Change Hired section to numbers
change_mapping = {'Y': 1, 'N': 0}

df['Hired'] = df['Hired'].map(change_mapping)
df['Working?'] = df['Working?'].map(change_mapping)
df['Top10 University?'] = df['Top10 University?'].map(change_mapping)
df['InternThisCompany?'] = df['InternThisCompany?'].map(change_mapping)

# Change education level to numbers
change_mapping_education = {'BS' : 0, 'MS':1, 'PhD':2}
df['Education Level'] = df['Education Level'].map(change_mapping_education)
df

y = df['Hired']
x = df.drop(['Hired'], axis = 1)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)

# making prediction
# 5 years experience, working, and worked 3 different companies. Education level is BS
print(clf.predict([[5, 1, 3, 0, 0, 0]]))

# 2 years experience, worked 7 different companies, and top10 university
print(clf.predict([[2, 0, 7, 0, 1, 0]]))



