from  sklearn.tree import DecisionTreeClassifier
features=[[140,0],[130,0],[150,1],[170,1]]
#0=apple     1=orange
labels=[0,0,1,1]

clf=DecisionTreeClassifier()

clf.fit(features,labels) #train

p=clf.predict([100,1])       #prediction sample
print("prediction=",p)
