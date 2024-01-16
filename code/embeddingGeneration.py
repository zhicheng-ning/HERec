import os

train_rate = 0.8
dim = 128
walk_len = 5
win_size = 3
num_walk = 10

metapaths = ['ubu', 'ubcabu', 'ubcibu', 'bub', 'bcab', 'bcib']

for metapath in metapaths:
	metapath = metapath + '_' + str(train_rate)
	input_file = '../data/metapath/' + metapath + '.txt'
	output_file = '../data/embeddings/' + metapath + '.embedding'

	cmd = 'deepwalk --format edgelist --input ' + input_file + ' --output ' + output_file + \
	      ' --walk-length ' + str(walk_len) + ' --window-size ' + str(win_size) + ' --number-walks '\
	       + str(num_walk) + ' --representation-size ' + str(dim)

	print(cmd)
	os.system(cmd)

'''
这段Python代码的主要作用是使用DeepWalk算法生成一系列元路径文件的嵌入。DeepWalk是一种流行的图嵌入方法，通常用于学习图结构数据（如社交网络、推荐系统中的用户-物品关系图）的低维表示。代码通过迭代不同的元路径文件，为每个文件生成嵌入，并将输出保存到指定的目录。

下面是代码的具体功能解释：

1. **初始化变量**:
   - `train_rate`, `dim`, `walk_len`, `win_size`, `num_walk` 是DeepWalk算法的参数。
   - `train_rate` 表示训练集的比例。
   - `dim` 表示嵌入的维度。
   - `walk_len` 表示随机游走的长度。
   - `win_size` 表示DeepWalk算法中窗口的大小。
   - `num_walk` 表示每个节点的随机游走次数。

2. **遍历元路径**:
   - `metapaths` 列表包含了不同类型的元路径（如`'ubu'`, `'ubcabu'`, `'ubcibu'` 等）。
   - 对于每种元路径，代码构建了输入文件和输出文件的路径。输入文件包含图的边列表，输出文件用于保存生成的嵌入。

3. **构建并执行DeepWalk命令**:
   - 对于每个元路径，代码构建了一个DeepWalk命令，包括所有必要的参数和文件路径。
   - 使用 `os.system(cmd)` 执行这个命令，这将启动DeepWalk过程，处理输入文件并生成嵌入。

4. **输出和保存嵌入**:
   - DeepWalk的结果（即节点的低维表示）被保存到指定的输出文件中。

### 代码的实际应用

这段代码在推荐系统、社交网络分析或任何需要对图结构数据进行特征学习的领域都非常有用。通过为不同类型的元路径生成嵌入，可以捕捉到图中节点的复杂关系，这对于后续的数据分析、机器学习建模或推荐算法都是有益的。
'''