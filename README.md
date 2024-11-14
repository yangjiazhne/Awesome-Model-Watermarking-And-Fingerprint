Awesome Model Watermarking And Fingerprint
========================
**This repository compiles a list of papers related to intellectual property (IP) protection for deep learning models.**
  
## <span id="back">Contents</span>
- [Survey](#Survey)
- [Model Watermarking](#Watermarking)
- [Model Fingerprint](#Fingerprint)

# <span id="Survey">Survey</span> [^](#back)
### 基于水印技术的深度神经网络模型知识产权保护  
**[[Paper]](https://crad.ict.ac.cn/cn/article/pdf/preview/10.7544/issn1000-1239.202440413.pdf)**  | *计算机研究与发展 2024*

从用于模型版权声明的鲁棒模型水印和用于模型完整性验证的脆弱模型水印 2 个维度出发，着重评述基于水印技术的 DNN模型知识产权保护方法，探讨不同方法的特点、优势及局限性。

### 深度神经网络模型水印研究进展
**[[Paper]](https://jns.usst.edu.cn/shlgdxxbzk/article/pdf/20240301?st=article_issue)**  | *上海理工大学学报 2024*

梳理了目前为了保护模型知识产权而提出的各类水印方案，按照提取水印时所具备的不同条件，将其分为白盒水印、黑盒水印和无盒水印 3 类方法，并对各类方法按照水印嵌入机制或适用模型对象的不同进行细分，深入分析了各类方法的主要原理、实现手段和发展趋势。

# <span id="Watermarking">Watermarking</span> [^](#back)


# <span id="Fingerprint">Fingerprint</span> [^](#back)
### Fingerprinting Deep Neural Networks – A DeepFool Approach  
**[[Paper]](https://dr.ntu.edu.sg/bitstream/10356/147023/2/2021021379.pdf)**  | *ISCAS 2021*

利用 DeepFool 算法生成一组具有最小扰动的对抗样本，这些样本接近分类边界，并且目标模型对它们的分类结果具有明确性。生成的指纹由这些对抗样本及其标签组成，用于记录目标模型的独特行为。在验证阶段，将指纹输入可疑模型，计算其预测结果与目标模型标签的一致概率，以此来判断该模型是否为目标模型的副本。

### GNNFingers: A Fingerprinting Framework for Verifying Ownerships of Graph Neural Networks
**[[Paper]](https://openreview.net/pdf?id=RNl51vzvDE)** | **[[link]](https://cn-sec.com/archives/2616607.html)** | *WWW  2024*

针对不同的图任务，设计了相应的指纹模板，通过对抗攻击方式更新图的结构和节点属性，生成具有独特性的指纹样本，以此代表模型的特征。使用FC作为判别器，将所有图指纹样本经过模型产生的输出拼接成特征向量，输入判别器，输出二元标签用于区分是否为目标模型。

