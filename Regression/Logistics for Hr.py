import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics



data=pd.read_csv("HR_comma_sep.csv")

def plot_correlation_map( df ):
    corr = df.corr()
    s , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 10 , 220 , as_cmap = True )
    s = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 }
        )

plot_correlation_map(data)



Y=data["left"]
X=data.drop(["left","Department"],axis=1)
X['salary']=X['salary'].map({'low':1, 'medium':2, 'high':3})


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=1)


clf=RandomForestClassifier(n_estimators=150)
clf.fit(x_train,y_train)
clf.score(x_test,y_test)

model=LogisticRegression()
model.fit(x_train,y_train)
model.score(x_test,y_test)

