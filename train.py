import pickle
from six.moves import cPickle as pickle
from sklearn import linear_model
import numpy as np

def convert(train_dataset):
    nsamples, nx, ny = train_dataset.shape
    d2_train_dataset = train_dataset.reshape((nsamples,nx*ny))
    return d2_train_dataset

save=pickle.load(open('notMNIST.pickle','rb'))
#print('Training dataset size:',save["train_dataset"])
x=convert(save["train_dataset"])
y=save["train_labels"]
xval=convert(save["valid_dataset"])
yval=save["valid_labels"]
xtest=convert(save["test_dataset"])
ytest=save["test_labels"]
row,col=xtest.shape
i=1.0
regmax=20
j=0
success=np.zeros((regmax*2,1))
while i<regmax:
    print('\nIteration number:',i)
    logreg = linear_model.LogisticRegression(C=i)
    logreg.fit(x[0:59500,:],y[0:59500])
    ypredict=logreg.predict(xval)
    success[j]=np.sum(ypredict==yval)
    d=np.argmax(success)
    if d==j:
        maxindex=i
    i+=0.5
    j+=1
    
print(maxindex)
logreg = linear_model.LogisticRegression(C=maxindex)
logreg.fit(x[0:59500,:],y[0:59500])
ypredict=logreg.predict(xtest)
correct=np.sum(ypredict==ytest)
accuracy=correct*100/row
print(accuracy)
