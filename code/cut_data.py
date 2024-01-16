#!/user/bin/python
import random

train_rate = 0.8

R = []
with open('../data/ub.txt', 'r') as infile:
    for line in infile.readlines():
        user, item, rating = line.strip().split('\t')
        R.append([user, item, rating])

random.shuffle(R)
train_num = int(len(R) * train_rate)

with open('../data/ub_' + str(train_rate) + '.train', 'w') as trainfile,\
     open('../data/ub_' + str(train_rate) + '.test', 'w') as testfile:
     for r in R[:train_num]:
         trainfile.write('\t'.join(r) + '\n')
     for r in R[train_num:]:
         testfile.write('\t'.join(r) + '\n')



'''
上述代码的作用是从一个存储用户评分数据的文件中分割出训练集和测试集。代码的主要步骤如下：

1. **设置训练集比例**：变量 `train_rate` 被设置为 0.8，意味着 80% 的数据将用作训练集，其余 20% 用作测试集。

2. **读取数据**：使用 `open` 函数和 `readlines` 方法从文件 `'../data/ub.txt'` 中读取数据。这个文件应该包含用户评分数据，其中每行包含一个用户ID、一个项目ID和相应的评分，这三个值之间用制表符分隔。

3. **数据预处理**：对读取的每一行数据进行处理，将其分割为用户ID、项目ID和评分，然后将这些数据添加到列表 `R` 中。

4. **打乱数据**：使用 `random.shuffle` 方法随机打乱列表 `R` 中的数据，以确保训练集和测试集的随机性。

5. **分割数据**：根据 `train_rate` 计算训练集的数量，然后将 `R` 列表分成两部分，一部分用于训练集，另一部分用于测试集。

6. **写入文件**：将分割好的训练集和测试集分别写入两个新的文件中。文件名包含了 `train_rate` 的值，以区分不同的训练集和测试集。训练集数据写入 `'../data/ub_' + str(train_rate) + '.train'` 文件，测试集数据写入 `'../data/ub_' + str(train_rate) + '.test'` 文件。

总的来说，这段代码的目的是为机器学习或数据分析任务准备训练集和测试集，确保数据的随机性并有效地分割数据集。
'''