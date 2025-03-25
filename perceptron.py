import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
or_data = pd.DataFrame()
and_data = pd.DataFrame()
xor_data = pd.DataFrame()
or_data['input1']=[1,1,0,0]
or_data['input2']=[1,0,1,0]
or_data['ouput']=[1,1,1,0]
and_data['input1']=[1,1,0,0]
and_data['input2']=[1,0,1,0]
and_data['ouput']=[1,0,0,0]
xor_data['input1']=[1,1,0,0]
xor_data['input2']=[1,0,1,0]
xor_data['ouput']=[0,1,1,0]
sns.scatterplot(and_data['input1'],and_data['input2'],hue=and_data['ouput'],s=200)
or_data
sns.scatterplot(or_data['input1'],or_data['input2'],hue=or_data['ouput'],s=200)
sns.scatterplot(xor_data['input1'],xor_data['input2'],hue=xor_data['ouput'],s=200)
from sklearn.linear_model import Perceptron
clf1=Perceptron()
clf2=Perceptron()
clf3=Perceptron()
clf1.fit(and_data.iloc[:,0:2].values,and_data.iloc[:,-1].values)
clf2.fit(or_data.iloc[:,0:2].values,or_data.iloc[:,-1].values)
clf3.fit(xor_data.iloc[:,0:2].values,xor_data.iloc[:,-1].values)
clf1.intercept_
x=np.linspace(-1,1,5)
y=-x+1
plt.plot(x,y)
sns.scatterplot(and_data['input1'],and_data['input2'],hue=and_data['ouput'],s=200)
clf2.coef_
clf2.intercept_
plt.plot(x1,y1)
sns.scatterplot(or_data['input1'],or_data['input2'],hue=or_data['ouput'],s=200)
