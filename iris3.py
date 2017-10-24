from scipy.spatial import distance             # to find dist

def euclid(a,b):
	return distance.euclidean(a,b)


class myknn():
	def fit(self,x_train,y_train):           #defining   fit   method to train
		self.x_train=x_train             #x train & y train inside the fun 
		self.y_train=y_train



	def predict(self,x_test):              #       defining  predict fun
		predictions=[]            # to store a list of predictions
		for row in x_test:
			labels=self.closest(row)
			predictions.append(labels)
		return predictions
	def closest(self,row):  #defining closest fun
		best_dist=euclid(row,self.x_train[0])
		best_index=0
		for i in range(1,len(self.x_train)):
			dist=euclid(row,self.x_train[i])
			if dist < best_dist :
				best_dist=dist
				best_index=i
		return self.y_train[best_index]


from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

iris=load_iris()

features=iris.data
labels=iris.target

from sklearn.cross_validation import train_test_split         #split into train and test
knn = myknn()
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size=.3)

knn.fit(x_train,y_train)

p=knn.predict(x_test)
from sklearn.metrics import accuracy_score
print("accuracy=",accuracy_score(y_test,p)) 


