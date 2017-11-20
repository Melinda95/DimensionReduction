import numpy as np
#M=np.array([[1,2],[2,4]])
#print(np.linalg.matrix_rank(M,tol=None))
import pandas as pd
digits_train=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',header=None)
digits_test=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',header=None)

X_digits=digits_train[np.arange(64)]
y_digits=digits_train[64]

from sklearn.decomposition import PCA
estimator=PCA(n_components=2)
X_pca=estimator.fit_transform(X_digits)

from matplotlib import pyplot as plt
def plot_pca_scatter():
    colors=['black','blue','purple','yellow','white','red','lime','cyan','orange','gray']

    for i in range(len(colors)):
        px=X_pca[:,0][y_digits.as_matrix()==i]
        py=X_pca[:,1][y_digits.as_matrix()==i]
        plt.scatter(px,py,c=colors[i])

    plt.legend(np.arange(0,10).astype(str))#标签
    plt.xlabel('First Principle Component')
    plt.ylabel('Second Principle Component')
    plt.show()

plot_pca_scatter()

X_train=digits_train[np.arange(64)]
y_train=digits_train[64]
X_test=digits_test[np.arange(64)]
y_test=digits_test[64]

from sklearn.svm import LinearSVC
lsvc=LinearSVC()
lsvc.fit(X_train,y_train)
y_predict = lsvc.predict(X_test)

from sklearn.decomposition import PCA
estimator=PCA(n_components=20)
pca_X_train=estimator.fit_transform(X_train)
pca_X_test=estimator.transform(X_test)

pcalsvc=LinearSVC()
pcalsvc.fit(pca_X_train,y_train)
pca_y_predict = pcalsvc.predict(pca_X_test)

from sklearn.metrics import classification_report
print('Accuracy of 64:',lsvc.score(X_test,y_test))
print(classification_report(y_test,y_predict,target_names=np.arange(10).astype(str)))

print('Accuracy of 20:',pcalsvc.score(pca_X_test,y_test))
print(classification_report(y_test,pca_y_predict,target_names=np.arange(10).astype(str)))