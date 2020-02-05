 **1.NFFM** 

参考：NFFM是去年腾讯赛冠军杨毅分享的模型，结构如下：
    
![输入图片说明](https://images.gitee.com/uploads/images/2018/1121/170241_bf011388_1805564.png "内网通截图20181121165642.png")
NFFM参考了渣大的分享，根据讯飞初赛数据结构自己重新写了一遍NFFM的建模，建了模型后，为了对应dnn模型，没有用之前lgb的统计类特征，重新写了一堆id类特征，不过最后识别率还没有lgb好，权当练习看图搭网络了。后来看到讯飞赛的冠军也说试了ffm+dnn的方式，结果没有树模型好。看样子点击率预测用dnn的方式也不一定吊打树模型。也可能dnn所需要的特征没做好，完全原始特征不做特征工程效果还是不行。另一方面dnn的一般特征工程都是id交叉编码和交叉统计，维度很大，然后全扔进去，我扔了一部分就显卡内存不够了，batch调的很小也不行，硬件略差，可能大内存全部扔进去训练效果会不一样吧，以后有足够的硬件条件再试试dnn。其实用来和树模型融合应该蛮好的，没有接着往下面做了。


 **2.data_analysis** 
数据分析的函数和一些可视化的图片，对数据的分布和范围等的一些观察。类似如下结果图片：
![输入图片说明](https://images.gitee.com/uploads/images/2018/1122/110520_b0f97757_1805564.png "null_feature_ratio_numeric.png")
![输入图片说明](https://images.gitee.com/uploads/images/2018/1122/110531_b334d7c9_1805564.png "day_time_click_numeric.png")
    ![输入图片说明](https://images.gitee.com/uploads/images/2018/1122/110544_a66d960d_1805564.png "num_null_numeric.png")

