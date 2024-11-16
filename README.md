Awesome Model Watermarking And Fingerprint
========================
**This repository compiles a list of papers related to intellectual property (IP) protection for deep learning models.**
  
## <span id="back">Contents</span>
- [Survey](#Survey)
- [Model Watermark](#Watermark)
- [Model Fingerprint](#Fingerprint)

# <span id="Survey">Survey</span> [^](#back)
### 基于水印技术的深度神经网络模型知识产权保护  
**[[Paper]](https://crad.ict.ac.cn/cn/article/pdf/preview/10.7544/issn1000-1239.202440413.pdf)**  | *计算机研究与发展 2024*

从用于模型版权声明的鲁棒模型水印和用于模型完整性验证的脆弱模型水印 2 个维度出发，着重评述基于水印技术的 DNN模型知识产权保护方法，探讨不同方法的特点、优势及局限性。

### 深度神经网络模型水印研究进展
**[[Paper]](https://jns.usst.edu.cn/shlgdxxbzk/article/pdf/20240301?st=article_issue)**  | *上海理工大学学报 2024*

梳理了目前为了保护模型知识产权而提出的各类水印方案，按照提取水印时所具备的不同条件，将其分为白盒水印、黑盒水印和无盒水印 3 类方法，并对各类方法按照水印嵌入机制或适用模型对象的不同进行细分，深入分析了各类方法的主要原理、实现手段和发展趋势。

### Deep Watermarking for Deep Intellectual Property Protection: A Comprehensive Survey
**[[Paper]](https://papers.ssrn.com/sol3/Delivery.cfm/f9d1a497-01d6-4174-90bd-c81d0a38946d-MECA.pdf?abstractid=4697020&mirid=1)**  | *2024*

对主流的深度学习模型 IP 保护方法进行了全面综述，涵盖了深度IP保护的多个方面：问题定义、主要威胁和挑战、深度水印方法的优缺点、评价指标和性能讨论。


# <span id="Watermark">Watermark</span> [^](#back)
### Embedding watermarks into deep neural networks
**[[Paper]](https://arxiv.org/pdf/1701.04082)**  | **[[code]](https://github.com/yu4u/dnn-watermark)**  |  **[[link]](https://zhuanlan.zhihu.com/p/689918179)**  | *ICMR 2017*

模型水印的开山之作，在模型训练的过程中添加水印，随机从DNN中选取一层或多层参数，将其与一个密钥矩阵相乘，经过sigmoid()函数映射为一串二进制编码，通过最小化二进制编码和自定义水印之间的差异，实现水印的嵌入。

### DeepMarks: A Digital Fingerprinting Framework for Deep Neural Networks
**[[Paper]](https://arxiv.org/pdf/1804.03648v1)**  | *ICMR 2019*

基于内部权重的方法将水印嵌入在模型权重中。使用抗共谋码（如平衡不完全区块设计），在每个模型副本中嵌入不同的用户指纹，以抵御共谋攻击。

### Watermarking Neural Network with Compensation Mechanism
**[[Paper]](https://link.springer.com/content/pdf/10.1007/978-3-030-55393-7_33.pdf)**  | *KSEM 2020*

伪随机地选择要嵌入的水印的权重位置，对选定的权重进行正交变换，通过二值化方法将水印嵌入到获得的系数中，并对水印系数进行逆正交变换以获得水印权重。 最后，提出了一种具有补偿机制的模型微调方案，在不破坏模型中水印的情况下消除二值化带来的轻微精度下降。

### White-box watermarking scheme for fully-connected layers in fine-tuning model
**[[Paper]](https://dl.acm.org/doi/pdf/10.1145/3437880.3460402)**  | *IH&MMSec
 2021*

通过微调模型全连接层，对采样权值的频率分量进行水印嵌入操作，使水印信号能够扩散到采样点上。使用 QIM方法来估计嵌入过程引起的变化量，减少对 DNN模型精度的影响。

### RIGA: Covert and Robust White-BoxWatermarking of Deep Neural Networks
**[[Paper]](https://arxiv.org/pdf/1910.14268.pdf)** | **[[code]](https://github.com/Jiachen-T-Wang/riga)**  | *WWW 2021*

水印嵌入通过在原始损失函数上添加两个正则化项（嵌入水印信息 和 评估水印权重分布与非水印权重分布的相似性）实现。使用了单独的提取器网络，将目标权重映射到水印（可以是比特串或LOGO图像），非目标权重映射到随机信息；采用对抗学习的思想，引入一个检测器网络以区分不含水印的权重和含有水印的权重。

### Watermarking Deep Neural Networks with Greedy Residuals
**[[Paper]](http://proceedings.mlr.press/v139/liu21x/liu21x.pdf)** | **[[code]](https://github.com/eil/greedy-residuals)** | **[[link]](https://zhuanlan.zhihu.com/p/563045721)** | *ICML 2021*

该方法要嵌入的水印是一个由 RSA算法对原始的版权声明进行私钥加密，再对解码结果取符号得到的，通过 hinge 损失将嵌入到从模型重要参数构造出的残差向量的符号中。该方法不需要触发集或密钥矩阵这样的显式所有权指示符。

### Reversible Watermarking in Deep Convolutional Neural Networks for Integrity Authentication
**[[Paper]](http://proceedings.mlr.press/v139/liu21x/liu21x.pdf)** | *MM 2020*

选择重要性低的核权重矩阵组成一个序列矩阵，通过截取序列矩阵元素的有效数字并调整其范围，生成可用于水印嵌入的矩阵；利用直方图移动法将水印信息嵌入到权重矩阵中，并通过修改权重值的某些位来确保水印的存在；在提取阶段，通过逆操作提取水印信息并恢复原始权重矩阵，从而还原模型。  

# <span id="Fingerprint">Fingerprint</span> [^](#back)
### Fingerprinting Deep Neural Networks – A DeepFool Approach  
**[[Paper]](https://dr.ntu.edu.sg/bitstream/10356/147023/2/2021021379.pdf)**  | *ISCAS 2021*

利用 DeepFool 算法生成一组具有最小扰动的对抗样本，这些样本接近分类边界，并且目标模型对它们的分类结果具有明确性。生成的指纹由这些对抗样本及其标签组成，用于记录目标模型的独特行为。在验证阶段，将指纹输入可疑模型，计算其预测结果与目标模型标签的一致概率，以此来判断该模型是否为目标模型的副本。

### MetaV: A Meta-Verifier Approach to Task-Agnostic Model Fingerprinting
**[[Paper]](https://arxiv.org/pdf/2201.07391v3)** | *SIGKDD  2022*

方法首先生成一组正模型（混淆技术在目标模型上生成）和负模型（独立训练），通过训练一个自适应指纹和元验证器组合，使得元验证器能够基于模型对自适应指纹的响应准确区分正负模型，从而检测目标模型是否被盗用。

### GNNFingers: A Fingerprinting Framework for Verifying Ownerships of Graph Neural Networks
**[[Paper]](https://openreview.net/pdf?id=RNl51vzvDE)** | **[[link]](https://cn-sec.com/archives/2616607.html)** | *WWW  2024*

针对不同的图任务，设计了相应的指纹模板，通过对抗攻击方式更新图的结构和节点属性，生成具有独特性的指纹样本，以此代表模型的特征。使用FC作为判别器，将所有图指纹样本经过模型产生的输出拼接成特征向量，输入判别器，输出二元标签用于区分是否为目标模型。

