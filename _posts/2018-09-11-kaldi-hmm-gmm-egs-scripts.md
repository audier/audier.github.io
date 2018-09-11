---
layout:     post   				    # 使用的布局（不需要改）
title:      kaldi HMM-GMM全部训练脚本分解   # 标题 
subtitle:   kaldi hmm-gmm model trainning			#副标题
date:       2018-09-11 				# 时间
author:     Sun Hongwen						# 作者
header-img: img/home-bg-pic.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - 语音识别
    - kaldi
---


## train_mono.sh
单音素训练脚本：
```
//初始化，[topo feats] -> [0.mdl tree]
gmm-init-mono 
//生成训练图，[0.mdl text l.fst] -> [train.fst]
compile-train-graph
//对标签进行初始化对齐[train.fst feats 0.mdl tree] -> [1.ali]
align-equal-compiled
//统计估计模型所需统计量，[feats 1.ali] -> [1.acc]
gmm-acc-stats-ali
//参数重估，估计新的模型 [1.acc] -> [1.mdl]
gmm-est

//迭代训练
for i < iteration
	//重新统计所需统计量，[$i.ali] -> [$i.acc]
	gmm-acc-stats-ali
	//估计新的模型，[$i.acc] -> [$i.mdl]
	gmm-est
	//重新对齐，[train.fst $i.mdl] ->[$i+1.ali] 
	gmm-align-compiled
//输出最后的模型
final.mdl = $i.mdl
```

## train_deltas.sh
三音素训练脚本：
```
//特征处理 [feats] -> [feats]
apply-cmvn | add-deltas
//由生成的单音素模型的对齐结果对三音素参数统计，用于生成决策树[final.ali feats] -> [treeacc]
acc-tree-stats
//三音素绑定，[treeacc] -> [tree]
cluster-phone
compile-questions
build-tree //该步骤完成决策树三音素聚类
//三音素模型初始化，[treeacc tree topo] -> [1.occ 1.mdl] -> [1.mdl]
gmm-init-model | gmm-mixup
//将单音素对其文件中的元素替换为决策树的叶子，[final.mdl 1.mdl final.ali] -> [ali.new]
convert-ali 
//生成训练图，[1.mdl text l.fst] -> [train.fst]
compile-train-graph

//迭代训练
for i < iteration
	//重新对齐，[train.fst $i.mdl] ->[$i+1.ali] 
	gmm-align-compiled
	//重新统计所需统计量，[$i.ali] -> [$i.acc]
	gmm-acc-stats-ali
	//估计新的模型，[$i.acc] -> [$i.mdl]
	gmm-est //该步骤增加混合高斯分量的数目
//输出最后的模型
final.mdl = $i.mdl
```

## train_lda_mllt.sh
lda-mllt训练脚本，非说话人自适应，mllt的作用是减少协方差矩阵对角化的损失：
```
//生成先验概率，统计计算lda所需统计量，[splice-feats final.ali] -> [lda.acc]
ali-to-post
weight-silence-post
acc-lda
//估计lda矩阵，[lda.acc] -> [lda.mat]
est-lda
//通过对转换后的特征重新统计，用于生成决策树[final.ali feats.*lda.mat] -> [treeacc]
acc-tree-stats
//三音素绑定，[treeacc] -> [tree]
cluster-phone
compile-questions
build-tree //该步骤完成决策树三音素聚类
//三音素模型初始化，[treeacc tree topo] -> [1.occ 1.mdl]
gmm-init-model
//将三音素决策树的叶子替换为转换后模型决策树的叶子，[final.mdl 1.mdl final.ali] -> [ali.new]
convert-ali 
//生成训练图，[1.mdl text l.fst] -> [train.fst]
compile-train-graph

//迭代训练
for i < iteration
	//重新对齐，[train.fst $i.mdl] ->[$i+1.ali] 
	gmm-align-compiled
		//同lda，估计mllt的矩阵
		ali-to-post | weight-silence-post | gmm-acc-mllt
		est-mllt
		//对gmm模型进行变换，[mllt.mat mdl] -> [new.mdl]
		gmm-transform-means
		//组合变换矩阵，[lda.mat mllt.mat] -> [lda.mat]
		compose-transforms
	//重新统计所需统计量，[$i.ali] -> [$i.acc]
	gmm-acc-stats-ali
	//估计新的模型，[$i.acc] -> [$i.mdl]
	gmm-est //该步骤增加混合高斯分量的数目
//输出最后的模型
final.mdl = $i.mdl
```


## train_sat.sh
说话人自适应模型，fmllr训练脚本：
```
//生成先验概率，统计计算fmllr所需统计量，[splice-feats spk2utt] -> [trans]
ali-to-post
weight-silence-post
gmm-est-fmllr
//通过对转换后的特征重新统计，用于生成决策树[final.ali feats.*lda.mat] -> [treeacc]
acc-tree-stats
//三音素绑定，[treeacc] -> [tree]
cluster-phone
compile-questions
build-tree //该步骤完成决策树三音素聚类
//三音素模型初始化，[treeacc tree topo] -> [1.occ 1.mdl]
gmm-init-model
//将三音素决策树的叶子替换为转换后模型决策树的叶子，[final.mdl 1.mdl final.ali] -> [ali.new]
convert-ali 
//生成训练图，[1.mdl text l.fst] -> [train.fst]
compile-train-graph

//迭代训练
for i < iteration
	//重新对齐，[train.fst $i.mdl] ->[$i+1.ali] 
	gmm-align-compiled
		//同lda，估计fmllr的矩阵 -> [fmllr.trans]
		ali-to-post | weight-silence-post | gmm-est-fmllr
		//组合变换矩阵，[trans.mat fmllr.trans] -> [trans.mat]
		compose-transforms
	//重新统计所需统计量，[$i.ali] -> [$i.acc]
	gmm-acc-stats-ali
	//估计新的模型，[$i.acc] -> [$i.mdl]
	gmm-est //该步骤增加混合高斯分量的数目
//输出最后的模型
final.mdl = $i.mdl
```

