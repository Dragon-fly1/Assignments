import numpy as np
import pandas as pd

class node:
    def __init__(self,feature_val=None,threshold=None,info_gain=None,left_child=None,right_child=None,value=None):
        
        #if it is a decision node
        self.feature_val = feature_val
        self.threshold = threshold
        self.info_gain = info_gain
        self.left_child = left_child
        self.right_child = right_child

        #if it is leaf node
        self.value = value

class DecisionTreeClassifier:
    def __init__(self,min_sample_split,max_depth):
        self.root = None
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
    
    def build_tree(self,X,y,cur_depth=0):
        num_samples,num_features = np.shape(X)
        if num_samples>=self.min_sample_split and cur_depth<self.max_depth:
            best_split = self.get_best_split(X,y)
            for x in best_split:
                print(x,end=":\n")
                print(best_split[x])
            if best_split["info_gain"]>0:
                left_subtree = self.build_tree(best_split["left_data_X"],best_split["left_data_y"],cur_depth+1)
                right_subtree = self.build_tree(best_split["right_data_X"],best_split["right_data_y"],cur_depth+1)
                return node(best_split["feature_val"],best_split["threshold"],best_split["info_gain"],left_subtree,right_subtree)
            
        leaf_node_value = self.get_leaf_node_value(y)
        return node(value=leaf_node_value)
    
    def get_best_split(self,X,y):
        best_split = {}
        max_ig = -1
        for feature in X:
            feature_values = np.unique(X[feature])
            for threshold in feature_values:
                left_X,left_y,right_X,right_y = self.split(X,y,feature,threshold)
                if len(left_X)>0 and len(right_X)>0:
                    cur_ig = self.ig(y,left_y,right_y)
                    if cur_ig>max_ig:
                        best_split["feature_val"] = feature
                        best_split["threshold"] = threshold
                        best_split["left_data_X"] = left_X
                        best_split["left_data_y"] = left_y
                        best_split["right_data_X"] = right_X
                        best_split["right_data_y"] = right_y
                        best_split["info_gain"] = cur_ig
                        max_ig = cur_ig
        return best_split

    def split(self,X,y,feature,threshold):
        left_X,left_y = X[X[feature]<=threshold],y[X[feature]<=threshold]
        right_X,right_y = X[X[feature]>threshold],y[X[feature]>threshold]
        return left_X,left_y,right_X,right_y
    
    def ig(self,y,left_y,right_y,type="entropy"):
        left_weight = len(left_y)/len(y)
        right_weight = len(right_y)/len(y)

        if type=="gini":
            gain = self.gini_index(y) - (left_weight*self.gini_index(left_y) + right_weight*self.gini_index(right_y))
        else:
            gain = self.entropy(y) - (left_weight*self.entropy(left_y) + right_weight*self.entropy(right_y))
        return gain
    
    def get_leaf_node_value(self,y):
        return max(list(y),key=list(y).count)

    def fit(self,X,y):
        self.root = self.build_tree(X,y)
    
    def predict(self,X):
        def make_prediction(x,tree):
            if tree.value!=None: return tree.value     #if its a leaf node
            
            if x[tree.feature_val]<=tree.feature_val:
                return make_prediction(x,tree.left_child)
            else:
                return make_prediction(x,tree.right_child)

        preditions = [make_prediction(x, self.root) for x in X]
        return preditions
    
    def entropy(self, y):
        ''' function to compute entropy '''
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
            #print("here",entropy)
        return entropy
    
    def gini_index(self, y):
        ''' function to compute gini index '''
        
        class_labels = np.unique(y)
        gini = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini

    def print_tree(self,tree=None,indent='  '):
        if tree==None:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_"+str(tree.feature_val), "<=", tree.threshold, "?", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left_child, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right_child, indent + indent)


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = load_iris()
X = pd.DataFrame(data.data)
y = pd.Series(data.target)
# X = pd.read_csv("data.csv")
# X=X[["Feature1","Feature2","Feature3"]]
# y = pd.read_csv("data.csv")
# y=y["Label"]
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

tree = DecisionTreeClassifier(2,5)
# Fit your custom decision tree
tree.fit(X, y)

# # Predict on test set
#y_pred = tree.predict(X_test)
tree.print_tree()