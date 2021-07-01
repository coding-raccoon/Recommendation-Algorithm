# Recommendation-Algorithm

​		这是一个关于推荐算法的仓库，主要是我这个菜鸡根据网上资料和各路大佬的学习记录备份。

​		主要包括论文、笔记、代码、推荐案例

​		- 论文选择都是推荐领域的经典论文，包括传统方法（非深度学习）、深度匹配模型、Embedding技术、基于CTR 预估的排序模型、序列推荐模型、多任务推荐模型。（持续更新...）

​		论文笔记参考了论文原文以及网上大佬写的相关技术博客，是对论文的简要介绍；（持续更新...）

​		代码部分参考网上大佬开源的实现，对应用广泛的模型进行了复现，并在开源的数据集上进行测试；(持续更新...)

​		推荐案例主要是一些推荐相关的比赛和小项目，其中包含了个人的尝试和探索.（待完成...）

### Paper

​		持续更新...

​		读过的推荐算法领域的经典的论文, 暂时按照类别分成了一下几个部分：

- **Traditional Algorithm：**

  |               Model               |                            Paper                             |
  | :-------------------------------: | :----------------------------------------------------------: |
  |       Matrix Factorization        | [IEEE 2009] Matrix Factorization Techniques for Recommender Systems |
  |                BPR                | [UAI 2009] BPR: Bayesian Personalized Ranking from Implicit Feedback |
  |       Factorization Machine       |              [ICDM 2010] Factorization Machines              |
  | Field-aware Factorization Machine | [ReSys 2016] Field-aware Factorization Machines for CTR Prediction |

- **Deep Match：**

  |    Model    |                            Paper                             |
  | :---------: | :----------------------------------------------------------: |
  |    DSSM     | [CIKM 2013] Deep Structured Semantic Models for Web Search using Clickthrough Data |
  | Youtube DNN | [RecSys 2016] Deep Neural Networks for YouTube Recommendations |
  |  Neural CF  |          [WWW 2017] Neural Collaborative Filtering           |
  |   SASRec    |     [ICDM 2018] Self-Attentive Sequential Recommendation     |
  |    STAMP    | [KDD 2018] Short-Term Attention/Memory Priority Model for Session based Recommendation |
  |   AttRec    | [AAAAI 2019] Next Item Recommendation with Self-Attentive Metric Learning |
  |     SDM     | [CIKM 2019] SDM: Sequential Deep Matching Model for Online Large-scale Recommender System |
  |    MIND     | [CIKM 2019] Multi-interest network with dynamic routing for recommendation at Tmall |

- **Embedding：**

  |   Model   |                            Paper                             |
  | :-------: | :----------------------------------------------------------: |
  | Word2Vec  | [2013] Efficient Estimation of Word Representations in Vector Space |
  | Item2Vec  |       Neural Item Embedding for Collabrotive Filtering       |
  | DeepWalk  | [KDD 2014] DeepWalk: Online Learning of Social Representations |
  |   LINE    |  [WWW 2015] LINE: Large-scale Information Network Embedding  |
  | Node2Vec  | [KDD 2016] node2vec: Scalable Feature Learning for Networks  |
  |   SDNE    |         [KDD 2016] Structural Deep Network Embedding         |
  | Struc2Vec | [KDD 2017] struc2vec: Learning Node Representations from Structural Identity |

- **Click Through Rate：**

  |                 Model                  |                            Paper                             |
  | :------------------------------------: | :----------------------------------------------------------: |
  | Factorization-supported Neural Network | [ECIR 2016] Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction |
  |      Product-based Neural Network      | [ICDM 2016] Product-based Neural Networks for User Response Prediction |
  |              Wide & Deep               |   [DLRS 2016] Wide & Deep Learning for Recommender Systems   |
  |             Deep Crossing              | [KDD 2016] Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features |
  |                 DeepFM                 | [IJCAI 2017] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction |
  |          Deep & Cross Network          |  [ADKDD 2017] Deep & Cross Network for Ad Click Predictions  |
  |   Attentional Factorization Machine    | [IJCAI 2017] Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks |
  |      Neural Factorization Machine      | [SIGIR 2017] Neural Factorization Machines for Sparse Predictive Analytics |
  |                xDeepFM                 | [KDD 2018] xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems |

- **Sequential Recommendation：**

  | Model |                            Paper                             |
  | :---: | :----------------------------------------------------------: |
  |  DIN  | [KDD 2018] Deep Interest Network for Click-Through Rate Prediction |
  | DIEN  | [AAAI 2019] Deep Interest Evolution Network for Click-Through Rate Prediction |
  |  BST  | [DLP-KDD 2019] Behavior sequence transformer for e-commerce recommendation in Alibaba |
  | DSIN  | [IJCAI 2019] Deep Session Interest Network for Click-Through Rate Prediction |
  |       | Practice on Long Sequential User Behavior Modeling for Click-Through Rate Prediction |

- **Multi-Task Recommendation：**

  | Model |                            Paper                             |
  | :---: | :----------------------------------------------------------: |
  | 综述  |  An Overview of Multi-Task Learning in Deep Neural Networks  |
  | MMoe  | [KDD 2018] Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts |
  |       | [RecSys 2019] Recommending What Video to Watch Next: A Multitask Ranking System |
  | ESMM  | [KDD 2018] Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate |
  |       | A pareto-efficient algorithm for multiple objective optimization in e-commerce recommendation |



### Notes

​		持续更新...

​		上述主要论文的阅读笔记，包含关于内容主要内容的简介，以及一些思考题和常见的面试考察点；

​		除了论文原文，主要参考学习了：

1. 《王喆的机器学习笔记》: https://zhuanlan.zhihu.com/p/51117616

2. Datawhale 推荐小组：https://github.com/datawhalechina/fun-rec

   

### Codes

​		持续更新...

​		关于一些经典的推荐算法的实现，每个算法都在一个公开的数据集上进行测试；

​		主要参考学习了：

  1. 潜心大佬的实现： https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0

  2. 浅梦大佬的DeepCTR：https://github.com/shenweichen/DeepCTR	

  3. 浅梦大佬的DeepMatch：https://github.com/shenweichen/DeepMatch

  4. 浅梦大佬的GraphEmbedding：https://github.com/shenweichen/GraphEmbedding

     ​		建议直接看大佬代码，我只是看完后尝试自己写了一下，主要学习他们的代码组织方式和模型细节的实现方式；其中 DeepCTR 简洁易用，在比赛和项目中都可以直接应用，方便快速比较各种算法的性能。

     

### Samples

​		待更新......

​		一些实际的推荐案例，包括参加过的比赛，以及一些小项目；

​		每一个比赛和小项目都会把数据集链接放上，都会给出一些自己尝试和探索出的解决方案和结果，也欢迎各位提出更好的解决方案进行交流学习。

- **案例一：阿里天池新闻推荐算法（新人入门赛）**

  - 比赛链接：https://tianchi.aliyun.com/competition/entrance/531842/information

    

- **案例二：2021微信大数据-视频推荐大赛（初赛）**

  - 比赛链接：https://algo.weixin.qq.com/

    

- **案例三：2021科大讯飞基于用户画像的商品推荐挑战赛**

  - 比赛链接：http://challenge.xfyun.cn/topic/info?type=user-portrait

    

