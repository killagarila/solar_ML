# %% [markdown]
# # Group M - Solar Potential ML
# ## Members:
# ### 21030062 - Eimantas Miliauskis
# ### - Harvey
# ### - Yasine

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay, classification_report
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
import seaborn as sns

from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, precision_score, classification_report

# %%
readfile = pd.read_excel("solar-potential.xlsx", usecols=["Insolation", "Generation","Area"], nrows=10000)
print(f"# num values:\n{readfile.isnull().sum()}\n")
readfile = readfile[readfile.Generation <= 7000] 

X = readfile[["Insolation", "Area"]].copy()
y = readfile["Generation"].copy()
print(X.head(), end="\n\n")
print(y.head())
train_x, test_x, train_y, test_y = train_test_split(X,y, test_size=0.19)
min_max_scalar = preprocessing.MinMaxScaler()
X = min_max_scalar.fit_transform(X)
# y = min_max_scalar.fit_transform(y)

# %%
# readfile.hist()
# plt.show()

# # %%
# sns.boxplot(readfile["Generation"])

# # %%
# print(np.where(readfile["Generation"]>1000000))

# # %%
# fig, ax = plt.subplots(figsize = (18,10))

# ax.scatter(readfile["Generation"], readfile["Area"])

# plt.show()

# # %%
# rfc = RandomForestClassifier(n_estimators=50, max_features="sqrt", max_samples=None)

# # Fit RFC and predict using the testing set
# rfc.fit(train_x, train_y)
# pred2 = rfc.predict(test_x)

# # Performance Report of rfc
# print(f"Accuracy Score of Random Forest Classifier: {accuracy_score(pred2,test_y)*100}%")
# cm = confusion_matrix(test_y, pred2, labels = rfc.classes_)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rfc.classes_)
# disp.plot()
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
# rfc_disp = RocCurveDisplay.from_estimator(rfc, test_x, test_y, ax=ax1)
# prec, recall, _ = precision_recall_curve(test_y, pred2, pos_label=rfc.classes_[1])
# pr_display = PrecisionRecallDisplay(precision=prec, recall=recall)

# print(classification_report(test_y, pred2, target_names=["suitable", "well suitable", "excellent suitable"]))


# %%
# regr = make_pipeline(StandardScaler(), SVR(C=1.0, degree=3, epsilon=20))
# regr.fit(train_x, train_y)
# pred3 = regr.predict(test_x)
# print(regr.score(train_x,train_y))


# %%
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logmodel = LinearRegression()

pipe = make_pipeline(StandardScaler(), logmodel)
pipe.fit(train_x,train_y)  # apply the pipeline on training data
print(pipe.score(test_x,test_y))



# %%
x_surf, y_surf = np.meshgrid(np.linspace(readfile.Insolation.min(), readfile.Insolation.max(), 10), np.linspace(readfile.Area.min(), readfile.Area.max(), 10))
onlyX = pd.DataFrame({'Insolation':x_surf.ravel(), 'Area':y_surf.ravel()})
fittedY=pipe.predict(onlyX)
fittedY=np.array(fittedY)

print(f"Insolation max:{readfile.Insolation.max()}/min:{readfile.Insolation.min()}")
print(f"Area max:{readfile.Area.max()}/min:{readfile.Area.min()}")
print(f"Generation max:{readfile.Generation.max()}/min:{readfile.Generation.min()}")

# readfile["Generation"].plot.line()

from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure(figsize=(20,10))
### Set figure size
ax = fig.add_subplot(111, projection='3d')
ax.scatter(test_x['Insolation'],test_x['Area'],test_y,c='red', marker='o', alpha=0.5)
ax.plot_surface(x_surf,y_surf,fittedY.reshape(x_surf.shape), color='b', alpha=0.3)
ax.set_xlabel('Insolation')
ax.set_ylabel('Area')
ax.set_zlabel('Generation')
plt.show()


