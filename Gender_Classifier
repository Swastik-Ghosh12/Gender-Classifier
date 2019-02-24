import sklearn
from sklearn import tree
from sklearn import neighbors
from sklearn import svm




# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
prediction = clf.predict([[185, 82, 44.5]])

clf2=neighbors.KNeighborsClassifier()
clf2=clf2.fit(X,Y)
prediction2 = clf2.predict([[185, 82, 44.5]])

clf3=svm.SVC()
clf3.fit(X,Y)
prediction3 = clf3.predict([[185, 82, 44.5]])


print(prediction)
print(prediction2)
print(prediction3)
