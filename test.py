
import matplotlib.pyplot as plt
import pandas as pd
import pydotplus
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pickle as pk
# filename = '../save/train_data.pk'
# f = open(filename, 'rb')
# data = pk.load(f)
# f.close()

# x = []
# y1 = []
# y2 = []
# for i in range(len(data)):
#     x.append(i)
#     y1.append(data[i][0])
#     y2.append(data[i][1])

# filename = '../save/reward.pk'
# f = open(filename, 'rb')
# data = pk.load(f)
# f.close()
# y = []
# for i in range(len(data)):
#     y.append(data[i]/100000)


# plt.plot(x,y1,x,y2, x, y) 
# plt.axis([0, 45, 0, 1])   
# plt.show()

# class DTPolicy:
#     def __init__(self, max_depth):
#         self.max_depth = max_depth
    
#     def fit(self, obss, acts):
#         self.tree = DecisionTreeClassifier(max_depth=self.max_depth)
#         self.tree.fit(obss, acts)

#     def train(self, obss, acts, train_frac):
#         obss_train, acts_train, obss_test, acts_test = split_train_test(obss, acts, train_frac)
#         # print('train_obss')
#         # print(obss_train[0])
#         # print(type(obss_train))
#         # print(obss_train.shape)
#         # input()
#         self.fit(obss_train, acts_train)
#         training_accuracy = accuracy(self, obss_train, acts_train)
#         test_accuracy = accuracy(self, obss_test, acts_test)
#         log('Train accuracy: {}'.format(training_accuracy), INFO)
#         log('Test accuracy: {}'.format(test_accuracy), INFO)
#         log('Number of nodes: {}'.format(self.tree.tree_.node_count), INFO)
#         # input('hello!')
#         return training_accuracy, test_accuracy

#     def predict(self, obss):
#         return self.tree.predict(obss)

#     def clone(self):
#         clone = DTPolicy(self.max_depth)
#         clone.tree = self.tree
#         return clone

with open('./tmp/tree.pk', 'rb') as f:
	clf = pk.load(f)

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
		filled=True, rounded=True,
		special_characters=True)#,feature_names = feature_cols)
		#special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree.png')
Image(graph.create_png())
