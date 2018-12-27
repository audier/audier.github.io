---
layout:     post   				# 使用的布局（不需要改）
title:      个人深度学习项目整理		    # 标题 
subtitle:   personal learing plan	#副标题
date:       2018-10-25 				# 时间
author:     Sun Hongwen				# 作者
header-img: img/home-bg-pic.jpg 		#这篇文章标题背景图片
catalog: true 					# 是否归档
tags:						#标签
    - 学习计划
---

# 个人深度学习项目整理
最近准备整理一些使用深度模型实现的项目，作为最近工作学习的一个整理，因为之前学习太不系统，有些杂乱，希望这个项目系列完成之后能够对自己的学习有一个总结，有些新的收获。
**该系列项目尽量都使用TensorFlow、keras分别进行实现，其内容包括：**
- 深度模型入门
- 语音识别应用
- 自然语言处理
- 图像识别
- GAN
- ...
整个系列项目及文档将在12月之前完成，先立个flag，希望自己好好整理，能够产出一些真正有用的东西，为他人所用。
PS: 之前的一个语音识别的项目已经都没有怎么去维护整理了，一方面是最近确实是又被安排去做其他的事情了，没有连续下来，另一方面是发现自己对于这些框架的基础确实薄弱，限制了自己的创造力。这次希望整理这些东西，能够扎实自己的基础吧。

## 深度模型入门项目
1.	**TensorFlow实现mnist分类**
	- DNN示例
	- CNN示例
	- RNN示例
2.	**keras实现mnist分类**
	- DNN示例
	- CNN示例
	- RNN示例

## 自然语言处理
1. **文章自动生成**
	- lstm : tensorflow
	- lstm : keras
2. **翻译系统**
	- seq2seq (tensorflow)
	- seq2seq + attention (tensorflow)
3. **对话系统**
	- seq2seq +attention (keras)
4. **输入法系统**
	- CBHG (tensorflow)
	- CBHG (keras)

## 语音识别 [speech recognition](https://blog.csdn.net/chinatelecom08/article/details/82557715)
1. CTC + RNN
2. CTC + CNN 
3. seq2seq +attention (keras)

## 图像识别
1. 目标检测
2. 风格迁移
3. 文本生成

## GAN
1. mnist图像生成
	- TensorFlow
	- keras


这是给自己立的一个flag，当这些都完成后，希望能够更好的理解这些深度框架，以及一些细节理论。
当然，也有很大的可能完不成这些任务，图像相关的任务自己也只做过mnist，其他的也不是很了解。但是不管怎么说，还是希望能够把列出来的这些任务都能做一遍，更好的理解深度模型在这些任务中是如何发挥作用的。加油。