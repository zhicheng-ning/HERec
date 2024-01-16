https://github.com/librahu/HIN-Datasets-for-Recommendation-and-Network-Embedding

## 数据集 Yelp
(Containing rating information)
### Entity Statistics
| Entity         |#Entity        |
| :-------------:|:-------------:|
| User           | 16,239        |
| Business       | 14,284        |
| Compliment     | 11            |
| Category       | 511            | 
| City           | 47           |

### Relation Statistics
| Relation            |#Relation      |
| :------------------:|:-------------:|
| User - Business     | 198,397       |
| User - User         | 158,590       |
| User - Compliment   | 76,875        |
| Business - City     | 14,267        |
| Business - Category | 40,009        |

上述的的 relation 分别对应五个文件：
- ub.txt（User - Business）
- uu.txt（User - User）
- uco.txt（User - Compliment）
- bci.txt（Business - City）
- bca.txt（Business - Category）



在Yelp数据集中，各种实体（如Compliment、User、Business、Category、City）扮演着特定的角色，每个实体都有其独特的属性和用途。以下是这些实体的一般描述：

1. **Compliment**:
   - 描述：在Yelp的上下文中，"Compliment"是用户之间互相给予的正面反馈。例如，一个用户可能会给另一个用户的评价发送“赞扬”以表扬其有帮助或信息丰富。
   - 属性：通常包括赞扬的类型（如有用、有趣、鼓舞人心）、发送者（赞扬的用户）、接收者（被赞扬的用户）、赞扬的日期和时间等。

2. **User**:
   - 描述：代表使用Yelp平台的个人用户。用户可以进行诸如发布评价、上传照片、给商家打分、互相发送"compliments"等活动。
   - 属性：可能包括用户ID、用户名、用户位置、加入日期、评价数量、好友数量、用户照片、个人简介等。

3. **Business**:
   - 描述：代表在Yelp上列出的商家，如餐厅、零售店、服务提供商等。
   - 属性：商家名称、地址、电话号码、营业时间、类别、星级评分、评价数量、设施（如免费WiFi、停车服务等）、照片等。

4. **Category**:
   - 描述：用于分类Yelp上的商家。每个商家可以归类到一个或多个类别，如“意大利餐厅”、“美甲服务”、“购物”等。
   - 属性：类别名称。在数据结构中可能还包括与其他类别的关系或层级结构。

5. **City**:
   - 描述：表示商家所在的城市或地理位置。
   - 属性：城市名称、可能还包括地理坐标、所属州或国家、人口等信息。

在Yelp的数据集中，这些实体通常以表格形式存在，每个实体的不同属性作为字段出现。这些实体之间可能存在各种关系，如用户评价特定商家，商家属于特定的类别和城市等。通过分析这些实体及其关系，可以揭示用户偏好、商业

趋势、区域特色等多种有价值的信息。例如，通过分析用户评价和“赞扬”，可以了解哪些商家受到欢迎，哪些用户是社区中的活跃参与者；通过分析商家类别和城市数据，可以了解不同地区的商业分布和特色。这些信息对于市场研究、商业决策和消费者行为分析等方面都非常重要。

Cate

## GPU 加速的矩阵乘法

old:
```python
        uu = ub.dot(ub.T)
```

new:
```python
        # 将NumPy数组转换为CuPy数组
        ub_gpu = cp.array(ub)
        # 执行GPU加速的矩阵乘法
        uu_gpu = ub_gpu.dot(ub_gpu.T)
        # 如果需要，将结果转回NumPy数组
        uu = cp.asnumpy(uu_gpu)
```  
