from sklearn import neighbors, datasets, tree, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
# import lightgbm
from sklearn.ensemble import BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import numpy as np

class Comparison:

    def __init__(self, train_data, train_label, test_data, test_label):
        self.train_data = train_data
        self.train_label = train_label
        self.test_data = test_data
        self.test_label = test_label

    def comparison(self):

        SVM = svm.SVC(C=1, kernel='rbf')  # 支持向量机，SVM
        KNN = neighbors.KNeighborsClassifier(3)  # K最近邻,KNN
        MNB = MultinomialNB()  # 朴素贝叶斯，MNB
        DT = DecisionTreeClassifier()  # 决策树,DT
        RF = RandomForestClassifier()  # 随机森林，RFC
        LGR = LogisticRegression()  # 逻辑回归，LGR
        MLP = MLPClassifier()  # 多层感知机，MLP
        AB = AdaBoostClassifier()  # AdaBoost
        # LGBM = lightgbm.LGBMClassifier()  # lightgbm
        Bagging = BaggingClassifier()  # Bagging
        LDA = LinearDiscriminantAnalysis()  # LinearDiscriminantAnalysis
        # SVDD

        #classify_type_list = [SVM, KNN, MNB, DT, RF, LGR, MLP, AB, LGBM, Bagging, LDA]
        classify_type_list = [KNN, SVM]
        temp_accuracy_list = []
        temp_f1_score_List = []
        temp_cost_time = []
        for item in classify_type_list:
            item.fit(self.train_data, self.train_label.ravel())
            start_time = time.time()
            # accuracy = item.score(self.test_data, self.test_label.ravel())
            predict_label = item.predict(self.test_data)
            end_time = time.time() -start_time
            # accuracy
            acc_score = accuracy_score(self.test_label.ravel(), predict_label)
            # AUC

            temp_f1_score = f1_score(self.test_label, predict_label.reshape((self.test_label.shape[0], 1)), labels=list(set(self.test_label.ravel())), average="macro")

            temp_accuracy_list.append(acc_score * 100)
            temp_f1_score_List.append(temp_f1_score * 100)
            temp_cost_time.append(end_time)
        return temp_accuracy_list, temp_f1_score_List, temp_cost_time