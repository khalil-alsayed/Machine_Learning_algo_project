# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 02:05:50 2021

@author: khalil et zahira
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
data = pd.read_excel("C:/Users/khali/Downloads/default(1).xls") 
data1=data.rename(columns={'X1':'m.credit','X2':'genre','X3':'education','X4':'sit.familiale','X5':'age','X6':'h.p.9','X7':'h.p.8','X8':'h.p.7','X9':'h.p.6','X10':'h.p.5','X11':'h.p.4','X12':'r.f.9','X13':'r.f.8','X14':'r.f.7','X15':'r.f.6','X16':'r.f.5','X17':'r.f.4','X18':'m.p.9','X19':'m.p.8','X20':'m.p.7','X21':'m.p.6','X22':'m.p.5','X23':'m.p.4'})
data2 = data1.drop('Y', axis = 1)
target = data1.Y

# In[1]: Base Test et base apprentissage

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data2, target, test_size=0.3)

# In[3]:  kpp voisins
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
parameters = {'n_neighbors': list(range(1,20,2)),"weights":['uniform','distance']}
# La fonction GridSearchCV automatise la recherche d’un optimum parmi les hyperparamètre, elle utilise notamment la validation croisée.
knn = KNeighborsClassifier()
gridknn = GridSearchCV(knn, parameters, scoring="roc_auc",cv=5)
gridknn.fit(X_train, y_train)
gridknn.best_params_
conf_knn = confusion_matrix(y_test, gridknn.predict(X_test))
cf_knn = pd.DataFrame(conf_knn, columns=['prédit ' + _ for _ in ["white", "red"]])
cf_knn.index = ['vrai ' + _ for _ in ["white", "red"]]
(1-((conf_knn[0,0] +  conf_knn[1,1])/y_test.shape[0]))*100
probas_kpp = gridknn.predict_proba(X_test)
fpr0, tpr0, thresholds1 = roc_curve(y_test, probas_kpp[:, 0], pos_label = gridknn.classes_[0] ,  drop_intermediate=False)
auc_kpp = auc(fpr0, tpr0) 

# In[4]: decision_tree

from sklearn.tree import DecisionTreeClassifier, plot_tree
param={"criterion":["gini","entropy"],"max_depth":list(range(1,10)),"min_samples_split":range(2,10),"min_samples_leaf":range(1,5)}
tree= GridSearchCV(DecisionTreeClassifier(),param,scoring="roc_auc",cv=5) 

tree.fit(X_train, y_train)
tree.best_estimator_
"on trouve que le max_depth=4"
clf = DecisionTreeClassifier(criterion="entropy",max_depth=5,min_samples_leaf=4,min_samples_split=5)# A REMPLIR
clf.fit(X_train, y_train)
clf.predict(X_test)
plt.figure(figsize=(20,20))
plot_tree(clf)
plt.show()
clf.score(X_test,y_test)
conf_cart = confusion_matrix(y_test, clf.predict(X_test))
cf_cart = pd.DataFrame(conf_cart, columns=['prédit ' + _ for _ in ["white", "red"]])
cf_cart.index = ['vrai ' + _ for _ in ["white", "red"]]
cf_cart
(1-((conf_cart[0,0] +  conf_cart[1,1])/y_test.shape[0]))*100




probas1 =clf.predict_proba(X_test)
fpr1, tpr1, thresholds0 = roc_curve(y_test,probas1[:, 0], pos_label=clf.classes_[0], drop_intermediate=False)# A REMPLIR
fpr1.shape

auc_tree = auc(fpr1, tpr1) 

# In[5]: foret_aleatoire

from sklearn.ensemble import RandomForestClassifier

param_rf={"max_depth":list(range(9,20)),"n_estimators":[500,1000,2000,3000]}
modele_grid=GridSearchCV(RandomForestClassifier(),param_rf,scoring="roc_auc",cv=5)
modele_grid.fit(X_train, y_train)
modele_grid.best_params_

forest = RandomForestClassifier(n_estimators=1000,max_depth=9)
forest.fit(X_train,y_train)
(1-forest.score(X_test, y_test))*100
probas2 = forest.predict_proba(X_test)

fpr2, tpr2, thresholds2 = roc_curve(y_test, probas2[:, 0], pos_label=forest.classes_[0], drop_intermediate=False)


auc_for = auc(fpr2, tpr2) 

# In[6]: methode MMG

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
gridMMG = dict()
gridMMG['solver'] = ['svd', 'lsqr', 'eigen']
modele_gridMMG=GridSearchCV(LinearDiscriminantAnalysis(),gridMMG,scoring="roc_auc",cv=5)
modele_gridMMG.fit(X_train, y_train)
modele_gridMMG.best_params_


lda = LinearDiscriminantAnalysis(solver="svd")
lda.fit(X_train, y_train)
(1-lda.score(X_test,y_test))*100
probas3 = lda.predict_proba(X_test)
fpr3, tpr3, thresholds3 = roc_curve(y_test, probas3[:, 0], pos_label=lda.classes_[0], drop_intermediate=False)
auc_lda = auc(fpr3, tpr3) 

# In[7] Regression logistique

from sklearn.linear_model import LogisticRegression
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['none','l1','l2','elasticnet']
c_values = [100, 10, 1.0, 0.1, 0.01]

gridlogit = dict(solver=solvers,penalty=penalty,C=c_values)
modele_gridlogit=GridSearchCV(LogisticRegression(),gridlogit,scoring="roc_auc",cv=5)
modele_gridlogit.fit(X_train, y_train)
modele_gridlogit.best_params_


SK_logit = LogisticRegression(max_iter =1000, penalty='l1', fit_intercept=True,solver='liblinear',C=0.01)

SK_logit.fit(X_train, y_train)

conf = confusion_matrix(y_test, SK_logit.predict(X_test))
conf
(1-((conf[0,0]+conf[1,1])/y_test.shape[0]))*100
(1-SK_logit.score(X_test,y_test))*100
probas4 = SK_logit.predict_proba(X_test)
fpr4, tpr4, thresholds4 = roc_curve(y_test, probas4[:, 0], pos_label=SK_logit.classes_[0], drop_intermediate=False)
fpr0.shape

auc_SK = auc(fpr4, tpr4) 

# In[8]: Gradient boosting machine (GBM):
from sklearn.ensemble import GradientBoostingClassifier
n_estimators = [10, 100,200]
learning_rate = [0.001, 0.01, 0.1]
subsample = [0.5, 0.7, 1.0]
max_depth = [3, 7, 9]   
paramboost = dict(learning_rate=learning_rate, n_estimators=n_estimators, subsample=subsample, max_depth=max_depth)
grid_boost = GridSearchCV(GradientBoostingClassifier(),paramboost, cv=5, scoring="roc_auc")
grid_boost.fit(X_train, y_train)
grid_boost.best_params_



boost = GradientBoostingClassifier(n_estimators=200,learning_rate=0.1,max_depth=3,subsample=0.7)

boost.fit(X_train, y_train)




conf = confusion_matrix(y_test, boost.predict(X_test))
conf

(1-boost.score(X_test,y_test))*100
probas5 = boost.predict_proba(X_test)
fpr5, tpr5, thresholds5 = roc_curve(y_test, probas5[:, 0], pos_label=boost.classes_[0], drop_intermediate=False)
fpr0.shape

auc_boost = auc(fpr5, tpr5) 

# In[9]: courbe ROC 
fig, ax = plt.subplots(1, 1, figsize=(4,4))
ax.plot([0, 1], [0, 1], 'k--')
ax.plot(fpr0, tpr0, label= 'kpp auc=%1.5f' % auc_kpp)
ax.plot(fpr1, tpr1, label= 'CART auc=%1.5f' % auc_tree)

ax.plot(fpr2, tpr2, label= 'Forest auc=%1.5f' % auc_for)

ax.plot(fpr3, tpr3, label= 'MMG auc=%1.5f' % auc_lda)

ax.plot(fpr4, tpr4, label= 'logit auc=%1.5f' % auc_SK)
ax.set_title('Courbe ROC')

ax.plot(fpr5, tpr5, label= 'boost auc=%1.5f' % auc_boost)
ax.set_title('Courbe ROC')

ax.set_xlabel("FPR")
ax.set_ylabel("TPR");
ax.legend();
