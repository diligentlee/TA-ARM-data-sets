# -*-coding:utf-8 -*-
__author__ = 'diligentLee'
import csv
import os
import pymssql
import random
import re
import numpy as np

import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

dataBaseTrain = []  # list用来存储整个分类数据，并最终存储为csv格式
dataBaseTest = []
'''读取数据库，获取样本和分类并将文件写入到csv文件中'''
server = "."
user = "sa"
password = "123456"
database = "InTe"

with pymssql.connect(server, user, password, database) as conn:
    with conn.cursor(as_dict=True) as cursor:
        cursor.execute(
            'select * from Examination where KnowledgePointID is not null')
        for row in cursor:
            dataLine = []  # //存储每条文本以及所属分类
            question = ''  # 要参与分类的样本，选择题把各选项文本并入题目作为分类样本，填空题把答案并入题目作为分类样本
            question += row['QuestionTitle']
            if row['QuestionType'] == 1 or row['QuestionType'] == 3:
                # print(row['QuestionTitle'], row['OptionA'], row['OptionB'], row['OptionC'], row['OptionD'], end='')
                question = question + " " + row['OptionA'] + ' ' + row['OptionB'] + ' ' + row['OptionC'] + '' + row[
                    'OptionD']
            elif row['QuestionType'] == 4:
                # print(row['CorrectOption'])
                question = question + ' ' + row['CorrectOption']
            dataLine.append(question)
            # if row['KnowledgePointID'] != '1.5':
            #     dataLine.append(row['KnowledgePointID'])
            # else:
            #     dataLine.append('1.4')
            dataLine.append(re.match(r"(\d+?)\.\d+?", row['KnowledgePointID']).group(1))
            matchGe = re.search(r'智博二零一七十二月份(\d)$', row['Memo'])
            matchShi = re.search(r'智博二零一七十二月份([1-8]\d)$', row['Memo'])
            match90 = re.search(r'智博二零一七十二月份(90)$', row['Memo'])
            if match90 or matchGe or matchShi:
                if match90:
                    dataLine.append(match90.group(1))
                elif matchGe:
                    dataLine.append(matchGe.group(1))
                else:
                    dataLine.append(matchShi.group(1))
            else:
                dataLine.append(False)
            if dataLine[2]:
                dataBaseTest.append(dataLine)
            else:
                dataBaseTrain.append(dataLine)
# 这里应该给训练集做一个数据清洗，把重复数据去掉
dataTrainFilter = []
for item in dataBaseTrain:
    if item not in dataTrainFilter:
        dataTrainFilter.append(item)
print('数据清洗前训练集数量为：', len(dataBaseTrain), '    测试集数量为：', len(dataBaseTest))
print('数据清洗后训练集数量为：', len(dataTrainFilter), '    测试集数量为：', len(dataBaseTest))

dataBaseAll = dataTrainFilter + dataBaseTest  # 所有的数据集，包含了训练集和测试集
random.shuffle(dataBaseAll)
print('总的数据集数量为：', len(dataBaseAll))

with open(os.path.join('..\\filesKnn', 'example.csv'), 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    for row in dataBaseAll:
        writer.writerow(row)
print('已将数据集文本保存至 filesKnn\\example.csv')

'''从csv文件中把数据读出来，以后就可以不用执行上面读取数据库的步骤了'''
with open(os.path.join('..\\filesKnn', 'example.csv'), encoding='utf-8') as f:
    reader = csv.reader(f)
    dataBaseAll = list(reader)
print('从文件filesKnn\\example.csv中将数据读取出，数量为：', len(dataBaseAll))

'''开始对数据中的每一行的第一列使用jieba进行分词，这里我们保存在一个新的list里面'''
# 定义一个新的list，命名为jiebaList
jiebaList = []
for line in dataBaseAll:
    # cutByJieba = jieba.cut_for_search(line[0])  # 我们尝试使用搜索引擎模式，这样可以在长词的基础上进行再次分类，这里可以对三种分词类型做不同的尝试，查看哪种类型的分类效果更好
    cutByJieba = jieba.cut(line[0], cut_all=False)
    cutStr = ' '.join(cutByJieba)
    jiebaList.append([cutStr, line[1], line[2]])

'''读取停用词表，去除停用词'''
# 创建停用词list
stopwordslist = []
with open(os.path.join('..\\filesKnn', 'stopwordsfile.txt'), 'r', encoding='utf-8') as f:
    for line in f:
        stopwordslist.append(line.strip())
print('读取停用词，停用词数量：', len(stopwordslist))

# 遍历，并去除停用词
afterstopwords = []
for line in jiebaList:
    remainword = []
    for word in line[0].split():
        if word not in stopwordslist and not re.search(r'^\d*\.?\d*$', word):
            remainword.append(word)
    afterstopwords.append([' '.join(remainword), line[1], line[2]])
print('已去除停用词')

'''把过滤后的词语保存到一个csv文件中，这样就可以使用文本分类方法进行训练和分类了'''
with open(os.path.join('..\\filesKnn', 'afterfilterTrain.csv'), 'w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    for row in afterstopwords:
        writer.writerow(row)

corpus = []  # 存储所有的数据集
category_list = []  # 存储类别信息
isTest = []  # 标识是否为测试集
for wordline in afterstopwords:
    corpus.append(wordline[0])
    category_list.append(wordline[1])
    isTest.append(wordline[2])

vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
tfidf = transformer.fit_transform(
    vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
weight = tfidf.toarray()  # 这样就转化为了矩阵，可以使用分类器进行分类了
print('TF-IDF计算完毕')

X_train = []
y_train = []

X_test = []
y_test = []
question_num = []
# 开始区分训练集和测试集
for i in range(len(isTest)):
    if isTest[i] != 'False':
        X_test.append(weight[i])
        y_test.append(category_list[i])
        question_num.append(isTest[i])
    else:
        X_train.append(weight[i])
        y_train.append(category_list[i])
print('已区分训练集和测试集')

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
print(len(X_train[0]))

model = KNeighborsClassifier()
# model = LogisticRegression()
model.fit(X_train, y_train)
print(model.get_params())

predictList = model.predict(X_test)
for i in range(len(predictList)):
    if predictList[i] != y_test[i]:
        print('第 ', question_num[i], ' 道题错了。\t正确的知识点应该是：', y_test[i], '而模型分类到：', predictList[i])

print('未调参的方法准确率为：', model.score(X_test, y_test))

predictProba = model.predict_proba(X_test)
for i in range(len(predictList)):
    if predictList[i]!=y_test[i]:
        print("被分错的题目是第 ",question_num[i], " 题，被分到各个类别的概率是：", predictProba[i])
    else:
        print(max(predictProba[i]))
