# tianchi_2018_guangdong_image
2018年天池图像比赛 广东工业智造大数据创新大赛


# 阿里天池2018广东工业智造图像比赛



---



该笔记是我们小队，关于阿里天池2018广东工业智造图像比赛的解决方案。首次参加天池比赛，这里记录一下自己的处理过程。模型是VGG19 + group normalization 这里是[比赛链接](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.0.0.79d733afHH1TQu&raceId=231682)。
> * 分析题意，确定目标
> * 数据预处理：包括图片读取，数据增强，划分训练集、测试集
> * 选择模型训练
> * 调参

### 1.分析题意，确定目标

将大赛给的四百多张铝型材图片，根据所损坏的程度分成12类。确定为图形分类问题。

| 瑕疵名        | 提交结果   |  
| --------   | -----:  | 
| 正常     | norm |  
| 不导电        |   defect1   |  
| 擦花        |    defect2    |  
| 横条压凹        |    defect3    | 
| 桔皮        |    defect4    | 
| 漏底        |    defect5    | 
| 碰伤        |    defect6    | 
| 起坑        |    defect7    | 
| 凸粉        |    defect8    | 
| 涂层开裂        |    defect9    | 
| 脏点        |    defect10    | 
| 其他        |    defect11    | 


```python
label_warp = {'正常': 0,
              '不导电': 1,
              '擦花': 2,
              '横条压凹': 3,
              '桔皮': 4,
              '漏底': 5,
              '碰伤': 6,
              '起坑': 7,
              '凸粉': 8,
              '涂层开裂': 9,
              '脏点': 10,
              '其他': 11,
              }

```


### 2.数据预处理：包括图片读取，数据增强，划分训练集、测试集


#### （1）图片读取


测试集
```python
def read_img(path):
    map_path, map_relative = [path +x for x in os.listdir(path) if os.path.isfile(path + x) ], [y for y in os.listdir(path)]
    return map_path, map_relative
    
path_root_test = 'Q:/Github/GuanDong-AL-train/data/guangdong_round1_test_a_20180916/'

# test data
map_path_test, map_relative_test = read_img(path_root_test)

test_file = pd.DataFrame({'img_path': map_path_test})
test_file.to_csv('data/test.csv', index=False)


```


训练集

```python
#train data
img_path, label = [], []
for first_dir in map_relative_train:
    first_path = path_root_train + first_dir + '/'
    if '无瑕疵样本' in first_path:
        map_path_train_norm, map_relative_train_norm = read_img(first_path)
        for temp in map_path_train_norm:
            img_path.append(temp)
            label.append('正常')
    else:
        map_path_train_other, map_relative_train_other = read_img(first_path)
        for second_dir in map_relative_train_other:
            second_path = first_path + second_dir + '/'
            if '其他' in second_path:
                map_path_train_other_11, map_relative_train_other_11 = read_img(second_path)
                for third_dir in map_relative_train_other_11:
                    if '.DS_Store' == third_dir:
                        continue
                    third_path = second_path + third_dir + '/'
                    map_path_train_other_11_son_img_path, map_relative_train_other_11_son = read_img(third_path)
                    for temp in map_path_train_other_11_son_img_path:
                        if temp == 'Q:/Github/GuanDong-AL-train/data/guangdong_round1_train2_20180916/瑕疵样本/其他/粘接/.DS_Store':
                            continue
                        img_path.append(temp)
                        label.append('其他')            
            else:
                map_path_train_other_son_imgs_path, map_relative_train_other_son = read_img(second_path)
                for temp in map_path_train_other_son_imgs_path:
                    img_path.append(temp)
                    label.append(second_dir)
                    

label_file = pd.DataFrame({'img_path': img_path, 'label': label})
label_file['label'] = label_file['label'].map(label_warp)
label_file.to_csv('data/label.csv', index=False)

```


#### （2）数据增强

```python
#对train 数据增强
import random
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
angles = [0,90,180,270]
angles = list(angles)
angles_num = len(angles)
for i in img_path:
    index = random.randint(0, angles_num - 1)
    im = Image.open(i)
    im.rotate(angles[index])
    im.save(i)
    print(i +" ----- is ok")

```

#### （3）划分训练集、测试集

```python
 #读取的图片，顺序打乱，划分测试和训练
def read_new_img(img_path,label):
    imgs=[]
    labels = label
    for idx in img_path:
        img = io.imread(idx)
        img = transform.resize(img, (image_size, image_size))
        imgs.append(img)
        print(idx + '------------is ok ')
    x_data, x_label = np.array(imgs), np.array(def_one_hot(np.array(labels)))
    data = []
    sigle_data = []
    for i in range(len(x_data)):
        sigle_data.append(x_data[i])
        sigle_data.append(x_label[i])
        data.append(sigle_data)
        sigle_data = []
        # 打乱顺序
    data = np.array(data)
    num_example = data.shape[0]
    np.random.shuffle(data)
    # 将所有数据分为训练集和验证集
    ratio = 0.8
    img_train = []
    label_train = []
    img_test = []
    label_test = []
    for i in range(int(ratio*num_example)):
        img_train.append(data[i][0])
        label_train.append(data[i][1])
    img_train = np.array(img_train)
    label_train = np.array(label_train)
    for i in range(int(ratio*num_example),num_example):
        img_test.append(data[i][0])
        label_test.append(data[i][1])
    img_test = np.array(img_test)
    label_test = np.array(label_test)
    print(img_train)
    return img_train,img_test,label_train,label_test


```


### 3.选择模型训练

我们队选择了VGG19模型 唯一不同的是，加入了2018年刚出的group normalization.下面是论文地址：

[group normalization](https://arxiv.org/abs/1803.08494)

![cmd-markdown-logo](http://attachbak.dataguru.cn/attachments/portal/201804/03/101604hs5or3eelrldd5d5.jpg)


![cmd-markdown-logo](http://attachbak.dataguru.cn/attachments/portal/201804/03/101604sr3tllljtja6t5zu.jpg)


以下是代码实现：group normalization

```python

def norm(x, norm_type, is_train,i, G=32, esp=1e-5):
    with tf.variable_scope('{}_norm_{}'.format(norm_type,i)):
        if norm_type == 'none':
            output = x
        elif norm_type == 'batch':
            output = tf.contrib.layers.batch_norm(
                x, center=True, scale=True, decay=0.999,
                is_training=is_train, updates_collections=None
            )
        elif norm_type == 'group':
            # normalize
            # tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
            x = tf.transpose(x, [0, 3, 1, 2])
            N, C, H, W = x.get_shape().as_list()
            G = min(G, C)
            x = tf.reshape(x, [-1, G, C // G, H, W])   # <------------------------------这里源码错了 需要改成这样 https://github.com/shaohua0116/Group-Normalization-Tensorflow/issues/1
            mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
            x = (x - mean) / tf.sqrt(var + esp)
            # per channel gamma and beta
            gamma = tf.get_variable('gamma', [C],
                                    initializer=tf.constant_initializer(1.0))
            beta = tf.get_variable('beta', [C],
                                   initializer=tf.constant_initializer(0.0))
            gamma = tf.reshape(gamma, [1, C, 1, 1])
            beta = tf.reshape(beta, [1, C, 1, 1])

            output = tf.reshape(x, [-1, C, H, W]) * gamma + beta   # 《------------------------ 这里源码错了 需要改成这样 https://github.com/shaohua0116/Group-Normalization-Tensorflow/issues/1
            # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
            output = tf.transpose(output, [0, 2, 3, 1])
        else:
            raise NotImplementedError
    return output


```

VGG19网络结构

```python

 # build_network

    W_conv1_1 = tf.get_variable('conv1_1', shape=[3, 3, 3, 64], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv1_1 = bias_variable([64])
    output = tf.nn.relu(norm(conv2d(x, W_conv1_1) + b_conv1_1,norm_type='group',is_train = True,i='1_1'))
    print(output)
    
    W_conv1_2 = tf.get_variable('conv1_2', shape=[3, 3, 64, 64], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv1_2 = bias_variable([64])
    output = tf.nn.relu(norm(conv2d(output, W_conv1_2) + b_conv1_2,norm_type='group',is_train = True,i='1_2'))
    output = max_pool(output, 2, 2, "pool1")
    print(output)

    W_conv2_1 = tf.get_variable('conv2_1', shape=[3, 3, 64, 128], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv2_1 = bias_variable([128])
    output = tf.nn.relu(norm(conv2d(output, W_conv2_1) + b_conv2_1,norm_type='group',is_train = True,i='2_1'))
    print(output)

    W_conv2_2 = tf.get_variable('conv2_2', shape=[3, 3, 128, 128], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv2_2 = bias_variable([128])
    output = tf.nn.relu(norm(conv2d(output, W_conv2_2) + b_conv2_2,norm_type='group',is_train = True,i='2_2'))
    output = max_pool(output, 2, 2, "pool2")
    print(output)

    W_conv3_1 = tf.get_variable('conv3_1', shape=[3, 3, 128, 256], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_1 = bias_variable([256])
    output = tf.nn.relu( norm(conv2d(output,W_conv3_1) + b_conv3_1,norm_type='group',is_train = True,i='3_1'))
    print(output)

    W_conv3_2 = tf.get_variable('conv3_2', shape=[3, 3, 256, 256], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_2 = bias_variable([256])
    output = tf.nn.relu(norm(conv2d(output, W_conv3_2) + b_conv3_2,norm_type='group',is_train = True,i='3_2'))
    print(output)

    W_conv3_3 = tf.get_variable('conv3_3', shape=[3, 3, 256, 256], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_3 = bias_variable([256])
    output = tf.nn.relu( norm(conv2d(output, W_conv3_3) + b_conv3_3,norm_type='group',is_train = True,i='3_3'))
    print(output)

    W_conv3_4 = tf.get_variable('conv3_4', shape=[3, 3, 256, 256], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv3_4 = bias_variable([256])
    output = tf.nn.relu(norm(conv2d(output, W_conv3_4) + b_conv3_4,norm_type='group',is_train = True,i='3_4'))
    output = max_pool(output, 2, 2, "pool3")
    print(output)

    W_conv4_1 = tf.get_variable('conv4_1', shape=[3, 3, 256, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_1 = bias_variable([512])
    output = tf.nn.relu(norm(conv2d(output, W_conv4_1) + b_conv4_1,norm_type='group',is_train = True,i='4_1'))
    print(output)

    W_conv4_2 = tf.get_variable('conv4_2', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_2 = bias_variable([512])
    output = tf.nn.relu(norm(conv2d(output, W_conv4_2) + b_conv4_2,norm_type='group',is_train = True,i='4_2'))
    print(output)

    W_conv4_3 = tf.get_variable('conv4_3', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_3 = bias_variable([512])
    output = tf.nn.relu(norm(conv2d(output, W_conv4_3) + b_conv4_3,norm_type='group',is_train = True,i='4_3'))
    print(output)

    W_conv4_4 = tf.get_variable('conv4_4', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv4_4 = bias_variable([512])
    output = tf.nn.relu(norm(conv2d(output, W_conv4_4) + b_conv4_4,norm_type='group',is_train = True,i='4_4'))
    output = max_pool(output, 2, 2)
    print(output)

    W_conv5_1 = tf.get_variable('conv5_1', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_1 = bias_variable([512])
    output = tf.nn.relu(norm(conv2d(output, W_conv5_1) + b_conv5_1,norm_type='group',is_train = True,i='5_1'))
    print(output)

    W_conv5_2 = tf.get_variable('conv5_2', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_2 = bias_variable([512])
    output = tf.nn.relu(norm(conv2d(output, W_conv5_2) + b_conv5_2,norm_type='group',is_train = True,i='5_2'))
    print(output)

    W_conv5_3 = tf.get_variable('conv5_3', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_3 = bias_variable([512])
    output = tf.nn.relu(norm(conv2d(output, W_conv5_3) + b_conv5_3,norm_type='group',is_train = True,i='5_3'))
    print(output)

    W_conv5_4 = tf.get_variable('conv5_4', shape=[3, 3, 512, 512], initializer=tf.contrib.keras.initializers.he_normal())
    b_conv5_4 = bias_variable([512])
    output = tf.nn.relu(norm(conv2d(output, W_conv5_4) + b_conv5_4,norm_type='group',is_train = True,i='5_4'))
    print(output)

    # output = tf.contrib.layers.flatten(output)
    output = tf.reshape(output, [-1, 2*2*512])
    print(output)

    W_fc1 = tf.get_variable('fc1', shape=[2048, 4096], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc1 = bias_variable([4096])
    output = tf.nn.relu(norm(tf.matmul(output, W_fc1) + b_fc1,norm_type='batch',is_train = True,i='fc1') )
    output = tf.nn.dropout(output, keep_prob)
    print(output)

    W_fc2 = tf.get_variable('fc7', shape=[4096, 4096], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc2 = bias_variable([4096])
    output = tf.nn.relu(norm(tf.matmul(output, W_fc2) + b_fc2,norm_type='batch',is_train = True,i='fc2'))
    output = tf.nn.dropout(output, keep_prob)
    print(output)

    W_fc3 = tf.get_variable('fc3', shape=[4096, 12], initializer=tf.contrib.keras.initializers.he_normal())
    b_fc3 = bias_variable([12])
    output = tf.nn.relu(norm(tf.matmul(output, W_fc3) + b_fc3,norm_type='batch',is_train = True,i='fc3'))
    print(output)
    # output  = tf.reshape(output,[-1,10])

    # loss function: cross_entropy
    # train_step: training operation
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
    l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    train_step = tf.train.MomentumOptimizer(learning_rate, momentum_rate, use_nesterov=True).\
        minimize(cross_entropy + l2 * weight_decay)

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


```


### 4.调参

主要涉及的参数：

```python

#全局one-hot编码空间
label_binarizer = ""

#初始的定义的数据
class_num = 12
image_size = 32
img_channels = 3
iterations = 72
batch_size = 24
total_epoch = 100
weight_decay = 0.0003
dropout_rate = 0.5
momentum_rate = 0.9
log_save_path = './vgg_19_logs'
model_save_path = './model/'
batch_size_test = 4

```

目前比赛还在进行中，调参有待提高



### 5.目前成绩：
当日排名175 综合排名602（截至10.4日一共2950支队伍，本科生第一次参加，第一次提交，轻喷）


## 关于我

Github:https://github.com/Qinxianshen

CSDN: https://blog.csdn.net/Qin_xian_shen

个人博客: http://saijiadexiaoqin.cn/

Gitchat:https://gitbook.cn/gitchat/author/59ef0b02a276fd1a69094634

哔哩哔哩：https://space.bilibili.com/126021651/#/

微信公众号：松爱家的小秦

更多LIVE：

[如何利用 Selenium 爬取评论数据？](https://gitbook.cn/gitchat/activity/59ef0fbf54011222e227c720)

[Neo4j 图数据库在社交网络等领域的应用](https://gitbook.cn/gitchat/activity/5a310961259a166307ceadb4)

[如何快速编写小程序商城和安卓 APP 商城](https://gitbook.cn/gitchat/activity/5b628776ff984e633d987f7d)


![微信赞赏](http://pc2bqmnuo.bkt.clouddn.com/249781965284692510.jpg)

![支付宝赞赏](http://pc2bqmnuo.bkt.clouddn.com/667424079218363348.jpg)
