---
layout:     post   				    # 使用的布局（不需要改）
title:      python HMM-GMM孤立词识别   # 标题 
subtitle:   python hmm-gmm model trainning			#副标题
date:       2018-09-30 				# 时间
author:     Sun Hongwen						# 作者
header-img: img/home-bg-pic.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - 语音识别
    - python
---


@[toc](基于HMM-GMM的孤立词识别)
# 简介


- 本文实现了一个基于HMM-GMM的独立词识别模型，数据集有限，训练数据为独立词，为10类。训练样本100个，测试样本10个，测试集上能够达到90%的识别率。
- 直接[下载项目](https://github.com/audier/my_hmm_gmm_speech_recognition)到本地，运行`.py`文件就可以得到下面的结果，成功执行：

```py
训练集：
识别得到结果：
 ['1', '10', '2', '3', '4', '5', '6', '7', '8', '10', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '5', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '10', '1', '10', '2', '4', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '10', '1', '10', '2', '3', '4', '5', '6', '7', '8', '10', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9']
原始标签类别：
 ['1', '10', '2', '3', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9']
识别率: 0.94
测试集：
识别得到结果：
 ['1', '10', '2', '3', '4', '5', '6', '7', '8', '3']
原始标签类别：
 ['1', '10', '2', '3', '4', '5', '6', '7', '8', '9']
 识别率: 0.9
```
# 基础准备
原理部分需要了解hmm-gmm在语音识别中的工作原理是什么，**特别要理解一个hmm-gmm模型对应一个孤立词这个概念，弄清楚非常重要。**
不过你即使不算很明白其中的含义，也可以成功的执行项目，可以在原理和代码中反复思考传统模型的实现原理。
这部分网上讲的很多，这里不再赘述。

# python建模
## 数据预处理
首先，进行数据预处理，输入为训练集路径或者测试集路径`wavpath`：

```py
# -----------------------------------------------------------------------------------------------------
'''
&usage:		准备所需数据
'''
# -----------------------------------------------------------------------------------------------------
# 生成wavdict，key=wavid，value=wavfile
def gen_wavlist(wavpath):
	wavdict = {}
	labeldict = {}
	for (dirpath, dirnames, filenames) in os.walk(wavpath):
		for filename in filenames:
			if filename.endswith('.wav'):
				filepath = os.sep.join([dirpath, filename])
				fileid = filename.strip('.wav')
				wavdict[fileid] = filepath
				# 获取文件的类别
				label = fileid.split('_')[1]
				labeldict[fileid] = label
	return wavdict, labeldict
```
查看wavdict和labeldict的数据内容，以test为例，其中音频的类别为文件名后面的数字：
![数据格式](https://img-blog.csdn.net/20180930091335772?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2NoaW5hdGVsZWNvbTA4/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
```py
wavdict, labeldict = gen_wavlist('test_data')
print(wavdict, labeldict)
```
输出文件路径和标注类别：

```
wavdict:
{'1_1': 'test_data\\1_1.wav', '1_10': 'test_data\\1_10.wav', '1_2': 'test_data\\1_2.wav', '1_3': 'test_data\\1_3.wav', '1_4': 'test_data\\1_4.wav', '1_5': 'test_data\\1_5.wav', '1_6': 'test_data\\1_6.wav', '1_7': 'test_data\\1_7.wav', '1_8': 'test_data\\1_8.wav', '1_9': 'test_data\\1_9.wav'}
labeldict:
{'1_1': '1', '1_10': '10', '1_2': '2', '1_3': '3', '1_4': '4', '1_5': '5', '1_6': '6', '1_7': '7', '1_8': '8', '1_9': '9'}
```
## 特征提取
我们直接调包`python_speech_features `实现mfcc，偷懒做法，这样看起来代码比较简洁，如果需要深入了解算法可以自己coding实现。

```py
# pip install python_speech_features ，不行的话百度一下
from python_speech_features import mfcc
# 特征提取，feat = compute_mfcc(wadict[wavid])
def compute_mfcc(file):
	fs, audio = wavfile.read(file)
	# 这里我故意fs/2,有些类似减小step，不建议这样做，投机取巧做法
	mfcc_feat = mfcc(audio, samplerate=(fs/2), numcep=26)
	return mfcc_feat
```
## 搭建孤立词模型
我们利用`hmmlearn`工具包搭建hmm-gmm，可以提前了解一下hmmlearn的[使用方法](https://hmmlearn.readthedocs.io/en/latest/)。
- 首先，需要初始化10个独立的hmm-gmm模型，分别对应十个独立词，主要是初始化一个hmm-gmm模型的集合 `self.models`：
```py
class Model():
	def __init__(self, CATEGORY=None, n_comp=3, n_mix = 3, cov_type='diag', n_iter=1000):
		super(Model, self).__init__()
		self.CATEGORY = CATEGORY
		self.category = len(CATEGORY)
		self.n_comp = n_comp
		self.n_mix = n_mix
		self.cov_type = cov_type
		self.n_iter = n_iter
		# 关键步骤，初始化models，返回特定参数的模型的列表
		self.models = []
		for k in range(self.category):
			model = hmm.GMMHMM(n_components=self.n_comp, n_mix = self.n_mix, 
								covariance_type=self.cov_type, n_iter=self.n_iter)
			self.models.append(model)
```
各个参数的意义：
```
	CATEGORY:	所有标签的列表
	n_comp:		每个孤立词中的状态数
	n_mix:		每个状态包含的混合高斯数量
	cov_type:	协方差矩阵的类型
	n_iter:		训练迭代次数
```
- 然后，用同一种类的数据训练特定的模型。

```py
	# 模型训练
	def train(self, wavdict=None, labeldict=None):
		for k in range(10):
			subdata = []
			model = self.models[k]
			for x in wavdict:
				if labeldict[x] == self.CATEGORY[k]:
					mfcc_feat = compute_mfcc(wavdict[x])
					model.fit(mfcc_feat)
```

-  最后，对待测试的数据分别用十个模型打分，选出得分最高的为识别结果。

```py
	# 使用特定的测试集合进行测试
	def test(self, wavdict=None, labeldict=None):
		result = []
		for k in range(self.category):
			subre = []
			label = []
			model = self.models[k]
			for x in wavdict:
				mfcc_feat = compute_mfcc(wavdict[x])
				# 生成每个数据在当前模型下的得分情况
				re = model.score(mfcc_feat)
				subre.append(re)
				label.append(labeldict[x])
			# 汇总得分情况
			result.append(subre)
		# 选取得分最高的种类
		result = np.vstack(result).argmax(axis=0)
		# 返回种类的类别标签
		result = [self.CATEGORY[label] for label in result]
		print('识别得到结果：\n',result)
		print('原始标签类别：\n',label)
		# 检查识别率，为：正确识别的个数/总数
		totalnum = len(label)
		correctnum = 0
		for i in range(totalnum):
		 	if result[i] == label[i]:
		 	 	correctnum += 1 
		print('识别率:', correctnum/totalnum)
```
- 你也可以保存和载入模型
```py
	# 利用external joblib保存生成的hmm模型
	def save(self, path="models.pkl"):
		joblib.dump(self.models, path)
		
	# 利用external joblib载入保存的hmm模型
	def load(self, path="models.pkl"):
		self.models = joblib.load(path)
```
## 模型的训练和测试
利用上面搭建的模块对模型进行训练和测试，完整项目包括数据在[hmm-gmm声学模型](https://github.com/audier/my_hmm_gmm_speech_recognition)，下载下来可以直接运行，如果对你有帮助的话，有账户求star啊。

```py
# 准备训练所需数据
CATEGORY = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
wavdict, labeldict = gen_wavlist('training_data')
testdict, testlabel = gen_wavlist('test_data')
print(testdict, testlabel)
# 进行训练和测试
models = Model(CATEGORY=CATEGORY)
models.train(wavdict=wavdict, labeldict=labeldict)
models.save()
models.load()
models.test(wavdict=wavdict, labeldict=labeldict)
models.test(wavdict=testdict, labeldict=testlabel)
```
识别结果：
```py
训练集：
识别得到结果：
 ['1', '10', '2', '3', '4', '5', '6', '7', '8', '10', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '5', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '10', '1', '10', '2', '4', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '10', '1', '10', '2', '3', '4', '5', '6', '7', '8', '10', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9']
原始标签类别：
 ['1', '10', '2', '3', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9', '1', '10', '2', '3', '4', '5', '6', '7', '8', '9']
识别率: 0.94
测试集：
识别得到结果：
 ['1', '10', '2', '3', '4', '5', '6', '7', '8', '3']
原始标签类别：
 ['1', '10', '2', '3', '4', '5', '6', '7', '8', '9']
 识别率: 0.9
```

# hmmlearn安装报错
该工具包安装很可能报错。
可以尝试去[https://www.lfd.uci.edu/~gohlke/pythonlibs/#hmmlearn](https://www.lfd.uci.edu/~gohlke/pythonlibs/#hmmlearn)下载对应你电脑版本的文件，我的是64位，python36。
然后cd到你下载的文件的目录下，执行：
```cmd
pip install hmmlearn‑0.2.1‑cp36‑cp36m‑win_amd64.whl
```
就可以安装成功了。

# 一些想法
- 由于基于python的hmm-gmm语音识别模型较少，网上竟然没找到好使的代码，无奈下只能自己查资料写了一个小样。
- 目前该项目只是一个demo，数据也较少，后续希望能够增加一些连续语音识别模型，搞成一个传统模型的语音识别系统。
- 有兴趣的老哥可以查看我的另一个基于深度学习的中文识别系统[ch_speech_recognition](https://github.com/audier/my_ch_speech_recognition)。

> 整理不易，转载请注明出处[https://blog.csdn.net/chinatelecom08](https://blog.csdn.net/chinatelecom08)。
> 该项目github地址：[https://github.com/audier/my_hmm_gmm_speech_recognition](https://github.com/audier/my_hmm_gmm_speech_recognition)
