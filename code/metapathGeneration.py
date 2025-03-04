#!/usr/bin/python
import sys
import numpy as np
import random
import cupy as cp

class metapathGeneration:
    def __init__(self, unum, bnum, conum, canum, cinum):
        self.unum = unum + 1
        self.bnum = bnum + 1
        self.conum = conum + 1
        self.canum = canum + 1
        self.cinum = cinum + 1
        ub = self.load_ub('../data/ub_0.8.train')
        '''
        论文中提到的六种元路径类型：
        UBU：用户-商家-用户（对同一个商家有过评价的用户们）
        UBCaBU：用户-商家-类别-商家-用户（对同一类商家有过评价的用户们）
        UBCiBU：用户-商家-城市-商家-用户（对同一个城市的商家有过评价的用户们）
        BUB：商家-用户-商家（受过同一个用户评价的商家们）
        BCiB：商家-城市-商家（位于同一个城市的商家们）
        BCaB：商家-类别-商家（属于同一类的商家们）
        '''
        self.get_UBU(ub, '../data/metapath/ubu_0.8.txt')
        self.get_UBCaBU(ub, '../data/bca.txt', '../data/metapath/ubcabu_0.8.txt')
        self.get_UBCiBU(ub, '../data/bci.txt', '../data/metapath/ubcibu_0.8.txt')
        self.get_BUB(ub, '../data/metapath/bub_0.8.txt')
        self.get_BCiB('../data/bci.txt', '../data/metapath/bcib_0.8.txt')
        self.get_BCaB('../data/bca.txt', '../data/metapath/bcab_0.8.txt')

    def load_ub(self, ubfile):
        ub = np.zeros((self.unum, self.bnum))
        with open(ubfile, 'r') as infile:
            for line in infile.readlines():
                user, item, rating = line.strip().split('\t')
                ub[int(user)][int(item)] = 1 
        return ub
    
    def get_UCoU(self, ucofile, targetfile):
        print('UCoU...')
        uco = np.zeros((self.unum, self.conum))
        with open(ucofile, 'r') as infile:
            for line in infile.readlines():
                u, co, _ = line.strip().split('\t')
                uco[int(u)][int(co)] = 1

        uu = uco.dot(uco.T)
        print(uu.shape)
        print('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(uu.shape[0]):
                for j in range(uu.shape[1]):
                    if uu[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(uu[i][j])) + '\n')
                        total += 1
        print('total = ', total)
    
    def get_UU(self, uufile, targetfile):
        print('UU...')
        uu = np.zeros((self.unum, self.unum))
        with open(uufile, 'r') as infile:
            for line in infile.readlines():
                u1, u2, _ = line.strip().split('\t')
                uu[int(u1)][int(u2)] = 1
        r_uu = uu.dot(uu.T)

        print(r_uu.shape)
        print('writing to file...')
        total = 0 
        with open(targetfile, 'w') as outfile:
            for i in range(r_uu.shape[0]):
                for j in range(r_uu.shape[1]):
                    if r_uu[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(r_uu[i][j])) + '\n')
                        total += 1
        print('total = ', total)
                                                                                                                                     

    def get_UBU(self, ub, targetfile):
        print('UMU...')
        # 将NumPy数组转换为CuPy数组
        ub_gpu = cp.array(ub)
        # 执行GPU加速的矩阵乘法
        uu_gpu = ub_gpu.dot(ub_gpu.T)
        # 如果需要，将结果转回NumPy数组
        uu = cp.asnumpy(uu_gpu)
        # uu = ub.dot(ub.T)
        print(uu.shape)
        print('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(uu.shape[0]):
                for j in range(uu.shape[1]):
                    if uu[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(uu[i][j])) + '\n')
                        total += 1
        print('total = ', total)
    
    def get_BUB(self, ub, targetfile):
        print('MUM...')
        ub_gpu = cp.array(ub)
        mm_gpu = ub_gpu.T.dot(ub_gpu)
        mm = cp.asnumpy(mm_gpu)
        # mm = ub.T.dot(ub)
        print(mm.shape)
        print('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(mm.shape[0]):
                for j in range(mm.shape[1]):
                    if mm[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(mm[i][j])) + '\n')
                        total += 1
        print('total = ', total)
    
    def get_BCiB(self, bcifile, targetfile):
        print('BCiB..')

        bci = np.zeros((self.bnum, self.cinum))
        with open(bcifile) as infile:
            for line in infile.readlines():
                m, d, _ = line.strip().split('\t')
                bci[int(m)][int(d)] = 1

        bci_gpu = cp.array(bci)
        mm_gpu = bci_gpu.dot(bci_gpu.T)
        mm = cp.asnumpy(mm_gpu)
        # mm = bci.dot(bci.T)
        print('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(mm.shape[0])[1:]:
                for j in range(mm.shape[1])[1:]:
                    if mm[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(mm[i][j])) + '\n')
                        total += 1
        print('total = ', total)

    def get_BCaB(self, bcafile, targetfile):
        print('BCaB..')

        bca = np.zeros((self.bnum, self.canum))
        with open(bcafile) as infile:
            for line in infile.readlines():
                m, a,__ = line.strip().split('\t')
                bca[int(m)][int(a)] = 1
        bca_gpu = cp.array(bca)
        mm_gpu = bca_gpu.dot(bca_gpu.T)
        mm = cp.asnumpy(mm_gpu)
        # mm = bca.dot(bca.T)
        print('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(mm.shape[0])[1:]:
                for j in range(mm.shape[1])[1:]:
                    if mm[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(mm[i][j])) + '\n')
                        total += 1
        print('total = ', total)
    
    def get_MTM(self, mtfile, targetfile):
        print('MTM..')

        mt = np.zeros((self.mnum, self.tnum))
        with open(mtfile) as infile:
            for line in infile.readlines():
                m, a,__ = line.strip().split('\t')
                mt[int(m)][int(a)] = 1

        mm = mt.dot(mt.T)
        print('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(mm.shape[0])[1:]:
                for j in range(mm.shape[1])[1:]:
                    if mm[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(mm[i][j])) + '\n')
                        total += 1
        print('total = ', total)
    
    def get_UBCaBU(self, ub, bcafile, targetfile):
        print('UBCaBU...')

        bca = np.zeros((self.bnum, self.canum))
        with open(bcafile, 'r') as infile:
            for line in infile.readlines():
                m, d, _ = line.strip().split('\t')
                bca[int(m)][int(d)] = 1
        ub_gpu = cp.array(ub)
        bca_gpu = cp.array(bca)
        uu_gpu = ub_gpu.dot(bca_gpu).dot(bca_gpu.T).dot(ub_gpu.T)
        uu = cp.asnumpy(uu_gpu)
        # uu = ub.dot(bca).dot(bca.T).dot(ub.T)
        print('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(uu.shape[0]):
                for j in range(uu.shape[1]):
                    if uu[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(uu[i][j])) + '\n')
                        total += 1
        print('total = ', total)
    
    def get_UBCiBU(self, ub, bcifile, targetfile):
        print('UBCiBU...')

        bci = np.zeros((self.bnum, self.cinum))
        with open(bcifile, 'r') as infile:
            for line in infile.readlines():
                m, a, _ = line.strip().split('\t')
                bci[int(m)][int(a)] = 1

        ub_gpu = cp.array(ub)
        bci_gpu = cp.array(bci)
        uu_gpu = ub_gpu.dot(bci_gpu).dot(bci_gpu.T).dot(ub_gpu.T)
        uu = cp.asnumpy(uu_gpu)
        # uu = ub.dot(bci).dot(bci.T).dot(ub.T)
        print('writing to file...')
        total = 0
        with open(targetfile, 'w') as outfile:
            for i in range(uu.shape[0]):
                for j in range(uu.shape[1]):
                    if uu[i][j] != 0 and i != j:
                        outfile.write(str(i) + '\t' + str(j) + '\t' + str(int(uu[i][j])) + '\n')
                        total += 1
        print('total = ', total)

if __name__ == '__main__':
    #see __init__() 
    metapathGeneration(unum=16239, bnum=14284, conum=11, canum=511, cinum=47)

'''
这段Python代码定义了一个名为 `metapathGeneration` 的类，其主要功能是生成和处理不同类型的元路径（metapath）数据，这通常在复杂网络分析、推荐系统或相关领域中使用。代码利用了CuPy库，这是一个基于NumPy API但利用GPU进行计算的库，以加速矩阵运算。以下是对代码主要部分的解释：

### 类 `metapathGeneration`

- **初始化方法 `__init__`**:
  - 输入参数：`unum`, `bnum`, `conum`, `canum`, `cinum` 分别表示不同类型实体的数量（用户数、商家数等）。
  - 在初始化时，类会加载用户-商家评分数据，并生成多种类型的元路径，例如用户-商家-用户（UBU）、用户-商家-类别-商家-用户（UBCaBU）等。

- **方法 `load_ub`**:
  - 从给定的文件中加载用户-商家关系，并创建一个用户-商家矩阵。

- **各种 `get_` 方法**:
  - 这些方法用于生成特定类型的元路径（如 `get_UBU`, `get_BUB`, `get_BCiB` 等）并将它们写入到文件中。它们通过矩阵乘法来实现这一点，例如 `UBU` 通过 `ub.dot(ub.T)` 计算，其中 `ub` 是用户-商家矩阵。
  - 与原版代码不同的是，这里使用了CuPy来执行GPU加速的矩阵乘法，提高了运算效率。

### 应用场景

这段代码在推荐系统、社交网络分析或任何需要通过链接实体来挖掘信息的领域中非常有用。通过不同类型的元路径，可以探索实体之间的复杂关系，并可能用于改进推荐算法或网络分析。

### 技术细节

- 使用了NumPy库来处理初始的数据加载和矩阵初始化。
- 利用CuPy进行GPU加速的矩阵运算，以提高处理大型数据集时的效率。
- 代码中的打印语句用于跟踪操作进度和结果。

### 注意事项

- 运行此代码需要具备GPU硬件支持以及正确安装配置了CuPy和CUDA环境。
- 代码注释部分提供了关于元路径类型的说明，有助于理解代码的上下文和目的。
- 这个脚本特别适合处理大规模数据集，特别是当涉及到密集的矩阵运算时。
'''


'''
在这段代码中，生成的元路径是通过矩阵运算得到的，并且被写入到文本文件中。元路径的格式和它们表示的含义通常与实体之间的关系有关。在这个特定的 `metapathGeneration` 类中，考虑到Yelp数据集的背景，元路径可能代表用户和商家之间的复杂关系。下面是对每种元路径格式及其含义的解释：

### 元路径的格式

生成的元路径数据通常以文本文件形式存储，每行代表一个元路径实例。每行数据通常包括若干字段，字段之间由制表符（`\t`）分隔。这些字段通常表示实体的ID或元路径中各个实体之间的关系强度。

例如，在 `get_UBU` 方法中，元路径的每一行可能包括三个字段：两个用户ID和它们之间的关系强度。这个关系强度是通过计算两个用户分别与同一商家有关联的次数得出的
（其他元路径同理，基本是两个用户在特定元路径下的关联次数或者两个商家在特定元路径下的关联次数）。

### 元路径的含义

1. **UBU (用户-商家-用户)**: 这种元路径连接评价过同一家商家的用户。如果两个用户经常评价同样的商家，他们之间的UBU连接就会更强。

2. **UBCaBU (用户-商家-类别-商家-用户)**: 这种元路径关联了对同一类型商家进行评价的用户。例如，喜欢同一类型餐馆的用户在这个元路径中会被连接起来。

3. **UBCiBU (用户-商家-城市-商家-用户)**: 这种元路径关联了评价过同一城市商家的用户。

4. **BUB (商家-用户-商家)**: 这种元路径连接了被同一用户评价过的商家。如果两家商家都被相同的用户评价过，则这两家商家之间的BUB连接就更强。

5. **BCiB (商家-城市-商家)**: 连接同一城市中的商家。

6. **BCaB (商家-类别-商家)**: 连接同一类别的商家。

### 应用

这些元路径在推荐系统和社交网络分析中非常有用。例如，在推荐系统中，可以使用UBU元路径来找到具有类似口味的用户，或者使用BUB元路径来发现可能相似的商家。通过分析这些复杂的关系网络，可以更好地理解用户行为，提升推荐的质量

和精确度。在社交网络分析中，这些元路径可以用来识别影响力用户、社区结构或用户群体之间的关系。通过这样的分析，可以揭示隐藏在大量数据中的模式和趋势，对市场营销策略、产品开发或用户体验改进等方面提供支持。
'''