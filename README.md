

```python
# import some necessary libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
%matplotlib inline

# ignore annoying warning (from sklearn and seaborn)
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn 

from scipy import stats
from scipy.stats import norm, skew
```


```python
# import data
train = pd.read_csv("F:\\Python_data_set\\titanic\\train.csv")
test  = pd.read_csv("F:\\Python_data_set\\titanic\\test.csv")

print("train contains: {}".format(train.shape))
print("test contains: {}".format(test.shape))
```

    train contains: (891, 12)
    test contains: (418, 11)
    


```python
train.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>McCarthy, Mr. Timothy J</td>
      <td>male</td>
      <td>54.0</td>
      <td>0</td>
      <td>0</td>
      <td>17463</td>
      <td>51.8625</td>
      <td>E46</td>
      <td>S</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
      <td>female</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>1</td>
      <td>2</td>
      <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
      <td>female</td>
      <td>14.0</td>
      <td>1</td>
      <td>0</td>
      <td>237736</td>
      <td>30.0708</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.head(5)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



PassengerId 

Pclass => class of passenger 1> 2> 3

Name 

Sex 

Age 

SibSp => number of cousins

Parch => number of parent & child

Ticket => ticket info

Fare 

Cabin

Embarked => habour that getting aboard

# Data Exploration

Distinguish numeric and categorical features

Visualizing distribution and correlation


```python
features = pd.concat([train, test])
features.drop("Survived",axis=1,inplace=True)

numeric_feats = features.dtypes[features.dtypes!="object"].index
categorical_feats = features.dtypes[features.dtypes=="object"].index

print("all data contains: {}".format(features.shape))
print("numeric_feats are: {}".format(numeric_feats))
print("categorical_feats are: {}".format(categorical_feats))
```

    all data contains: (1309, 11)
    numeric_feats are: Index(['Age', 'Fare', 'Parch', 'PassengerId', 'Pclass', 'SibSp'], dtype='object')
    categorical_feats are: Index(['Cabin', 'Embarked', 'Name', 'Sex', 'Ticket'], dtype='object')
    

## categorical features distribution

Temporally fill NA with Missing and droped in case of differnet columns between Survived_0 and Survived_1

Viewing whether there are some categorical features linear to the surviving ratio num(1)/num(0)


```python
feats = ["Embarked","Sex"]
feats2 = ["Cabin","Name","Ticket"]
train_plt = train

plt.figure(figsize=(18,18))
for i,c in enumerate(feats):
    train_plt[c] = train[c].fillna("Missing")
    train_plt[c] =train_plt[c][train_plt[c]!="Missing"]
    Survived_0 = train[c][train_plt["Survived"]==0].value_counts()
    Survived_1 = train[c][train_plt["Survived"]==1].value_counts()
    
    ind = [i+1 for i in range(len(Survived_0))]      # bar only support x that has same data type as y
    plt.subplot(len(feats),2,2*i+1)
    plt.bar(ind,Survived_1.values,label='Survived')
    plt.bar(ind,Survived_0.values,bottom=Survived_1.values,label='Not Survived')
    plt.xticks(ind,Survived_0.index)
    plt.ylabel("frequency")
    plt.legend(loc="upper right")
    plt.title(feats[i])
    
    plt.subplot(len(feats),2,2*i+2)
    plt.bar(ind, Survived_1.values/Survived_0.values)
    plt.xticks(ind,Survived_0.index)
    plt.ylabel("Surviving Ratio")
    plt.title(feats[i])
    
#plt.figure(figsize=(18,18))
#for i,c in enumerate(feats2):
#    train[c] = train[c].fillna("Missing")
#    plt.subplot(np.ceil(len(feats2)/2),2,i+1)
#    sns.countplot(train[c],hue=train["Survived"])
#    plt.xlabel(train[c])
#    plt.ylabel("Frequency")
#    plt.title(feats2[i])
```


![png](output_10_0.png)


## numeric features distribution

Viewing whether there are some bizzarre relationship between numeric features and the label


```python
plt.figure(figsize=(30,30))
for i,c in enumerate(numeric_feats):
    plt.subplot(len(numeric_feats),2,i*2+1)
    sns.violinplot(train_plt["Survived"],train_plt[c],hue=train["Sex"],split=True)
    plt.ylabel("frequency")
    plt.title(numeric_feats[i])
    
for i,c in enumerate(numeric_feats):
    train_plt[c] = train[c].fillna(train[c].mean())
    plt.subplot(len(numeric_feats),2,i*2+2)
    sns.distplot(train[c])
    plt.ylabel("frequency")
    plt.title(numeric_feats[i])
```


![png](output_13_0.png)


## data correlation 


```python
plt.figure(figsize=(12,10))
g = sns.heatmap(train.corr(),vmax =0.9,square=False,annot=True,fmt=".2f")
```


![png](output_15_0.png)


# Data Preprocessing

## extract title from name


```python
features["Title"] = features.Name.str.extract("(\w+)\.")
features["Title"].value_counts()
```




    Mr          757
    Miss        260
    Mrs         197
    Master       61
    Dr            8
    Rev           8
    Col           4
    Mlle          2
    Major         2
    Ms            2
    Sir           1
    Dona          1
    Mme           1
    Capt          1
    Countess      1
    Don           1
    Jonkheer      1
    Lady          1
    Name: Title, dtype: int64




```python
pd.crosstab(features["Title"],features["Sex"]).T
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Title</th>
      <th>Capt</th>
      <th>Col</th>
      <th>Countess</th>
      <th>Don</th>
      <th>Dona</th>
      <th>Dr</th>
      <th>Jonkheer</th>
      <th>Lady</th>
      <th>Major</th>
      <th>Master</th>
      <th>Miss</th>
      <th>Mlle</th>
      <th>Mme</th>
      <th>Mr</th>
      <th>Mrs</th>
      <th>Ms</th>
      <th>Rev</th>
      <th>Sir</th>
    </tr>
    <tr>
      <th>Sex</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>female</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>260</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>197</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>male</th>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>757</td>
      <td>0</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
features["Title"] = features["Title"].replace(list(features["Title"].value_counts()[4:].index),"Rare")
features["Title"].value_counts()
```




    Mr        757
    Miss      260
    Mrs       197
    Master     61
    Rare       34
    Name: Title, dtype: int64



## missing data


```python
print(features.shape)
features.isnull().sum()[features.isnull().sum()>0].sort_values(ascending=False)
```

    (1309, 12)
    




    Cabin       1014
    Age          263
    Embarked       2
    Fare           1
    dtype: int64



Cabin: has lost too much data, we probabilly need to delete it 

Fare : strong connection with Pclass, use mediean of same Pclass's Fare


```python
features["Fare"] = features.groupby("Pclass")["Fare"].transform(
    lambda x: x.fillna(x.median()))
```

Embarked : using mode


```python
features["Embarked"].fillna(features["Embarked"].mode()[0],inplace=True)
```

Age is hard to tell, we can buld a model for predicting it by using other features


```python
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV 

def pred_age(data,model,param):
    
    train = data[data["Age"].notnull()]
    test  = data[data["Age"].isnull()]
    
    Xtrain = train.drop("Age",axis=1)
    Ytrain = train["Age"]
    test   = test.drop("Age",axis=1)
    
    n_folds=5
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(Xtrain)
    models = GridSearchCV(model,param,scoring="neg_mean_squared_error",cv=kf)
    models.fit(Xtrain,Ytrain)
        
    data.loc[data["Age"].isnull(),"Age"] = models.best_estimator_.predict(test)
      
    return data,models.best_params_,models.best_score_
```

check whether there is no missing value except cabin & age


```python
features.isnull().sum()[features.isnull().sum()>0].sort_values(ascending=False)
```




    Cabin    1014
    Age       263
    dtype: int64



# Feature Engineering

## delete feature
delete useless features


```python
features.drop(["PassengerId","Cabin","Ticket","Name"],axis=1,inplace=True)
```

## numeric2categorical
from observing numeric feautres disstribution


```python
features.Parch = features.Parch.astype(str)
features.SibSp = features.SibSp.astype(str)
features.Pclass = features.Pclass.astype(str)
```

## categorical2numeric
from observing categorical feautres disstribution


```python
## label encoding
#from sklearn.preprocessing import LabelEncoder

#cols = None

## process columns, apply LabelEncoder to categorical features
#for c in cols:
#    lbl = LabelEncoder() 
#    lbl.fit(list(features[c].values)) 
#    features[c] = lbl.transform(list(features[c].values))

## shape        
#print('Shape features: {}'.format(features.shape))
```

## skew features
normal distribution for features with high skewness by using boxcox1p


```python
# 5 making high skewness features to satisfy normal distribution by using boxcox
from scipy.stats import skew
from scipy.special import boxcox1p
numeric_feats = features.dtypes[features.dtypes != "object"].index

skewed_feats = features[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[abs(skewed_feats) > 0.75].index

lam=0.15
features[skewed_feats] = boxcox1p(features[skewed_feats],lam)
```

## get dummy
OneHotEncoder


```python
# 6 OneHotEncoder
category_feats = features.dtypes[features.dtypes == "object"].index
features = pd.get_dummies(features,columns=category_feats)    
print(features.shape)
```

    (1309, 30)
    

## age modeling 


```python
from sklearn.linear_model import Lasso
lasso = Lasso()
param = {"alpha": np.logspace(-5,0,100).round(5)}

features,param, score = pred_age(features,lasso,param)
print("rmse = {:.2f}, with param = {}".format(np.sqrt(-score),param))
```

    rmse = 10.92, with param = {'alpha': 0.0059899999999999997}
    

## new train and test 


```python
train_features = features[:train.shape[0]]
test_features  = features[train.shape[0]:]
train_labels   = train["Survived"].values
```

## normalization

RobustScaler: robust to the outliers

StandardScaler: std


```python
from sklearn.preprocessing import RobustScaler,StandardScaler
RS = StandardScaler()
RS.fit(train_features)
train_features_scale = RS.transform(train_features)
test_features_scale = RS.transform(test_features)
```

# Modelling

## import lib


```python
from sklearn.linear_model import SGDClassifier,LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
```

## cv coding


```python
from sklearn.model_selection import cross_val_score

#Validation function
def accuracy_cv(model):
    sss = StratifiedShuffleSplit(n_splits=5,test_size = 0.2,random_state=1)
    accuarcy_cv= cross_val_score(model, train_features_scale, train_labels, scoring="accuracy",cv = sss)
    return(accuarcy_cv)
```

## gridSearch coding


```python
def param_select(model,param):
    sss = StratifiedShuffleSplit(n_splits=5,test_size = 0.2, random_state=1)
    models = GridSearchCV(model,param,scoring="neg_mean_squared_error",cv=sss)
    models.fit(train_features_scale,train_labels)
    return models.best_estimator_,models.best_params_
```

## base models

### LogisticRegression


```python
param = {"C": np.logspace(-5,0,100).round(5),"max_iter":np.linspace(100,500,5)}
LR,best_param = param_select(LogisticRegression(),param)
print(best_param)
```

    {'C': 0.35111999999999999, 'max_iter': 100.0}
    


```python
LR = LogisticRegression(C =0.3511,max_iter=100, random_state=3)
```

### SVC


```python
param = {"C": np.logspace(-5,0,100).round(5)}
svc,best_param = param_select(SVC(random_state=3),param)
print(best_param)
```

    {'C': 0.24771000000000001}
    


```python
svc = SVC(C =0.24771, random_state=3)
```

### SGDClassifier 


```python
param = {"alpha": np.logspace(-5,0,100).round(5),"l1_ratio":np.linspace(0.1,0.9,18)}
sgdc,best_param = param_select(SGDClassifier(loss='hinge', penalty='elasticnet'),param)
print(best_param)
```

    {'alpha': 0.0053400000000000001, 'l1_ratio': 0.2411764705882353}
    


```python
sgdc = SGDClassifier(loss='hinge', penalty='elasticnet', alpha=0.00534, l1_ratio=0.24118)
```

### RandomForestClassifier 


```python
param = {"n_estimators":range(10,200,100), "min_samples_split":range(2,10,5),
         "min_samples_leaf":range(1,5,5), "max_depth":range(1,6,5)}
rfc,best_param = param_select(RandomForestClassifier(random_state=4),param)
print(best_param)
```

    {'max_depth': 1, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 110}
    


```python
rfc = RandomForestClassifier(n_estimators=110, min_samples_split=2, min_samples_leaf=1, max_depth=1, random_state=4)
```

### XGBoost


```python
param = {"colsample_bytree": np.linspace(0.2,1,20), "gamma": np.logspace(-2,1,5),
         "learning_rate": np.logspace(-2,0,5), "max_depth": range(3,11,8),
         "n_estimators": range(10,200,25), "subsample": np.linspace(0.2,1,20)}
XGB,best_param = param_select(xgb.XGBClassifier(random_state=5),param)
print(best_param)
```

    {'colsample_bytree': 1.0, 'gamma': 1.7782794100389228, 'learning_rate': 0.31622776601683794, 'max_depth': 3, 'n_estimators': 60, 'subsample': 0.57894736842105265}
    


```python
XGB = xgb.XGBClassifier(colsample_bytree=1, gamma=1.77828, 
                        learning_rate=0.03163, max_depth=3, 
                        n_estimators=60, reg_alpha=0.0, reg_lambda=0.0,
                        subsample=0.57895, random_state=5)
```

### LightGBM


```python
param = {"n_estimators":range(10,200,100), "min_samples_split":range(2,10,5),
         "min_samples_leaf":range(1,5,5), "max_depth":range(1,5,5)}
LGB,best_param = param_select(lgb.LGBMClassifier(),param)
print(best_param)
```

    {'max_depth': 1, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 110}
    


```python
LGB = lgb.LGBMClassifier(n_estimators= 110, min_samples_split= 2,
                         min_samples_leaf= 1, max_depth= 1
```

## base models scores


```python
score = accuracy_cv(LR)
print("\nLogisitciRegression score: {:.4f} +/- {:.4f}\n".format(score.mean(), score.std()))
```

    
    LogisitciRegression score: 0.8201 +/- 0.0139
    
    


```python
score = accuracy_cv(svc)
print("\nSVC score: {:.4f} +/- {:.4f}\n".format(score.mean(), score.std()))
```

    
    SVC score: 0.8246 +/- 0.0076
    
    


```python
score = accuracy_cv(sgdc)
print("\nSGDClassifier score: {:.4f} +/- {:.4f}\n".format(score.mean(), score.std()))
```

    
    SGDClassifier score: 0.8078 +/- 0.0195
    
    


```python
score = accuracy_cv(rfc)
print("\nRandomForestClassifier score: {:.4f} +/- {:.4f}\n".format(score.mean(), score.std()))
```

    
    RandomForestClassifier score: 0.7989 +/- 0.0150
    
    


```python
score = accuracy_cv(XGB)
print("\nXGBoost score: {:.4f} +/- {:.4f}\n".format(score.mean(), score.std()))
```

    
    XGBoost score: 0.8302 +/- 0.0144
    
    


```python
score = accuracy_cv(LGB)
print("\nlightGBM score: {:.4f} +/- {:.4f}\n".format(score.mean(), score.std()))
```

    
    lightGBM score: 0.8067 +/- 0.0148
    
    

## Stacking models

### Stucked base models class


```python
class StuckingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self,BaseModels,MetaModel):
        self.BaseModels = BaseModels
        self.MetaModel  = MetaModel
        
    def fit(self,x_data,y_data):
        self.BaseModels_ = [list() for x in self.BaseModels]
        self.MetaModel_  = clone(self.MetaModel)
        kf = KFold(n_splits= 5, shuffle= True, random_state= 10)
        meta_fold = np.zeros((x_data.shape[0],len(self.BaseModels)))
        
        for i,model in enumerate(self.BaseModels):
            for train_index, test_index in kf.split(x_data,y_data):
                instance = clone(model)
                self.BaseModels_[i].append(instance)
                instance.fit(x_data[train_index],y_data[train_index])
                pred = instance.predict(x_data[test_index])
                meta_fold[test_index, i] = pred
                
        self.MetaModel_.fit(meta_fold,y_data) 
        return self
    
    def predict(self,x_data):
        meta_features = np.column_stack([
            np.column_stack([model.predict(x_data) for model in BaseModels]).mean(axis=1)
            for BaseModels in self.BaseModels_])
        return self.MetaModel_.predict(meta_features)
```

### Stucked base models score


```python
StuckedModels =StuckingModels(BaseModels=(svc,sgdc,rfc,XGB,LGB),MetaModel=LR)
score = accuracy_cv(StuckedModels)
print(" Stucked base models score:{:.4f} +/- {:.4f}".format(score.mean(),score.std()))
# Stucked base models score:0.8380 +/- 0.0050
```

     Stucked base models score:0.8458 +/- 0.0097
    

# Prediction


```python
StuckedModels.fit(train_features_scale,train_labels)
pred = StuckedModels.predict(test_features_scale)
```


```python
filepath = "F:\\Python_data_set\\titanic\\"
submission = pd.DataFrame({"PassengerId":test.PassengerId,"Survived":pred})
submission.to_csv(filepath + "submission.csv",index=False)
```


```python

```

