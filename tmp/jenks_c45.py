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

class DTPolicy:
    def __init__(self, max_depth):
        self.max_depth = max_depth
    
    def fit(self, obss, acts):
        self.tree = DecisionTreeClassifier(max_depth=self.max_depth)
        self.tree.fit(obss, acts)

    def train(self, obss, acts, train_frac):
        obss_train, acts_train, obss_test, acts_test = split_train_test(obss, acts, train_frac)
        # print('train_obss')
        # print(obss_train[0])
        # print(type(obss_train))
        # print(obss_train.shape)
        # input()
        self.fit(obss_train, acts_train)
        training_accuracy = accuracy(self, obss_train, acts_train)
        test_accuracy = accuracy(self, obss_test, acts_test)
        log('Train accuracy: {}'.format(training_accuracy), INFO)
        log('Test accuracy: {}'.format(test_accuracy), INFO)
        log('Number of nodes: {}'.format(self.tree.tree_.node_count), INFO)
        # input('hello!')
        return training_accuracy, test_accuracy

    def predict(self, obss):
        return self.tree.predict(obss)

    def clone(self):
        clone = DTPolicy(self.max_depth)
        clone.tree = self.tree
        return clone



col_names = ['RTT', 'CWND', 'RWND', 'retransmission', 'fastretransmission', 'timeout', 'bytes_acked', 'bytes_received', 'samplesum', 'samplecounter', 'packets_out', 'duration', 'inflight_bytes', 'CCA']
feature_cols = ['RTT', 'CWND', 'RWND', 'retransmission', 'fastretransmission', 'timeout', 'bytes_acked', 'bytes_received', 'samplesum', 'samplecounter', 'packets_out', 'duration', 'inflight_bytes']
# network = pd.read_csv(app + "_jenks_DT_" + str(int_num) + ".csv", header=None, names=col_names)

# x = network[feature_cols]
# y = network.CCA

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
# clf = DecisionTreeClassifier()
# clf = clf.fit(x_train,y_train)
# y_pred = clf.predict(x_test)

# print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
with open('./dt_policy.pk', 'rb') as f:
	clf = pk.load(f)

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,
		filled=True, rounded=True,
		special_characters=True)#,feature_names = feature_cols)
		#special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree.png')
Image(graph.create_png())
