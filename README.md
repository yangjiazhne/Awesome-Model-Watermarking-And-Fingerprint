Awesome Model Watermarking And Fingerprint
========================
**This repository compiles a list of papers related to intellectual property (IP) protection for deep learning models.**
  
**持续更新中....  过段时间会更新较新的相关文章，对内容进一步分类！

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

## White-box

### Embedding watermarks into deep neural networks
**[[Paper]](https://arxiv.org/pdf/1701.04082)**  | **[[Code]](https://github.com/yu4u/dnn-watermark)**  |  **[[Link]](https://zhuanlan.zhihu.com/p/689918179)**  | *ICMR 2017*

模型水印的开山之作，在模型训练的过程中添加水印，随机从DNN中选取一层或多层参数，将其与一个密钥矩阵相乘，经过sigmoid()函数映射为一串二进制编码，通过最小化二进制编码和自定义水印之间的差异，实现水印的嵌入。

### DeepMarks: A Digital Fingerprinting Framework for Deep Neural Networks
**[[Paper]](https://arxiv.org/pdf/1804.03648v1)**  | *ICMR 2019*

基于内部权重的方法将水印嵌入在模型权重中。使用抗共谋码（如平衡不完全区块设计），在每个模型副本中嵌入不同的用户指纹，以抵御共谋攻击。

### Watermarking Neural Network with Compensation Mechanism
**[[Paper]](https://Link.springer.com/content/pdf/10.1007/978-3-030-55393-7_33.pdf)**  | *KSEM 2020*

伪随机地选择要嵌入的水印的权重位置，对选定的权重进行正交变换，通过二值化方法将水印嵌入到获得的系数中，并对水印系数进行逆正交变换以获得水印权重。 最后，提出了一种具有补偿机制的模型微调方案，在不破坏模型中水印的情况下消除二值化带来的轻微精度下降。

### White-box watermarking scheme for fully-connected layers in fine-tuning model
**[[Paper]](https://dl.acm.org/doi/pdf/10.1145/3437880.3460402)**  | *IH&MMSec
 2021*

通过微调模型全连接层，对采样权值的频率分量进行水印嵌入操作，使水印信号能够扩散到采样点上。使用 QIM方法来估计嵌入过程引起的变化量，减少对 DNN模型精度的影响。

### RIGA: Covert and Robust White-BoxWatermarking of Deep Neural Networks
**[[Paper]](https://arxiv.org/pdf/1910.14268.pdf)** | **[[Code]](https://github.com/Jiachen-T-Wang/riga)**  | *WWW 2021*

水印嵌入通过在原始损失函数上添加两个正则化项（嵌入水印信息 和 评估水印权重分布与非水印权重分布的相似性）实现。使用了单独的提取器网络，将目标权重映射到水印（可以是比特串或LOGO图像），非目标权重映射到随机信息；采用对抗学习的思想，引入一个检测器网络以区分不含水印的权重和含有水印的权重。

### Watermarking Deep Neural Networks with Greedy Residuals
**[[Paper]](http://proceedings.mlr.press/v139/liu21x/liu21x.pdf)** | **[[Code]](https://github.com/eil/greedy-residuals)** | **[[Link]](https://zhuanlan.zhihu.com/p/563045721)** | *ICML 2021*

该方法要嵌入的水印是一个由 RSA算法对原始的版权声明进行私钥加密，再对解码结果取符号得到的，通过 hinge 损失将嵌入到从模型重要参数构造出的残差向量的符号中。该方法不需要触发集或密钥矩阵这样的显式所有权指示符。

### Reversible Watermarking in Deep Convolutional Neural Networks for Integrity Authentication
**[[Paper]](http://proceedings.mlr.press/v139/liu21x/liu21x.pdf)** | *MM 2020*

选择重要性低的核权重矩阵组成一个序列矩阵，通过截取序列矩阵元素的有效数字并调整其范围，生成可用于水印嵌入的矩阵；利用直方图移动法将水印信息嵌入到权重矩阵中，并通过修改权重值的某些位来确保水印的存在；在提取阶段，通过逆操作提取水印信息并恢复原始权重矩阵，从而还原模型。 

### DeepSigns: an end-to-end watermarking framework for ownership protection of deep neural networks
**[[Paper]](https://arxiv.org/pdf/1804.00750)** | **[[Code]](https://github.com/Bitadr/DeepSigns)** | *ASPLOS 2019*

将水印嵌入到网络中间层的激活图中，其中水印正则化器由三项内容构成，嵌入的水印表示为每个类别选择样本的中间层激活的高斯均值，并在损失中最小化同类激活的方差，最大化不同类间的激活差异。

### MOVE: Effective and Harmless Ownership Verification via Embedded External Features
**[[Paper]](https://arxiv.org/pdf/2208.02820)** | **[[Code]](https://github.com/THUYimingLi/MOVE)** | **[[Link]](https://zhuanlan.zhihu.com/p/427111937)** | *arXiv 2022*

利用风格迁移嵌入外部特征，针对白盒和黑盒场景分别构建用于区分嵌入与非嵌入特征的元分类器，其中白盒场景以梯度符号向量为输入，黑盒场景以预测差异特征为输入。随后，随机采样多个变换后的图像，通过元分类器获得分类结果，基于假设检验进行所有权验证。

### Radioactive data: tracing through training
**[[Paper]](https://arxiv.org/pdf/2002.00937)** | **[[Code]](https://github.com/facebookresearch/radioactive_data)** | **[[Link]](https://zhuanlan.zhihu.com/p/682110841)** | *ICML 2020*

在数据的潜在空间中添加标记（一个方向向量），使用标记将特征移向某个方向，再把标记从特征空间反向传回像素；这些标记在整个训练过程中保持可检测，最后算分类层权重和这个标记向量的余弦相似度，通过统计检验p-value来验证是否使用了标记数据。

### Structural Watermarking to Deep Neural Networks via Network Channel Pruning
**[[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9648376)** | *WIFS 2021*

通过量化索引调制（QIM），在模型剪枝期间将水印嵌入到修剪率中，在嵌入过程将水印分割为若干个比特段，每个剪枝率由比特段量化得到，根据密钥选择通道进行修剪。在验证阶段，根据确定的通道剪枝率重构水印。

### DeepIPR: Deep neural network intellectual property protection with passports
**[[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9648376)** | **[[Code]](https://github.com/kamwoh/DeepIPR)** | **[[Link]](https://zhuanlan.zhihu.com/p/447766318)** | *TPAMI 2021*

在模型的卷积层之后加入一个护照层（类似于归一化层），该层的缩放和平移因子由护照计算得到，当别人进行歧义攻击的时候，模型的效果会大打折扣。

### A robustness-assured white-box watermark in neural networkspassports
**[[Paper]](https://ieeexplore.ieee.org/document/10038500)** | **[[Code]](https://github.com/lvpeizhuo/HufuNet)** | *TDSC 2023*

在水印生成阶段，训练一个具有少量参数的神经网络（称为 HufuNet），以在测试时产生高精度。 训练样本和测试样本集是公共参考，必须在所有权验证阶段使用。 HufuNet 经过训练和测试后，被分成两部分，左部分作为水印嵌入到 DNN 模型中用于所有权保护(训练工程中冻结参数)，右部分由模型所有者保留作为所有权验证的密钥。

### Free fine-tuning: A plug-and-play watermarking scheme for deep neural networks
**[[Paper]](https://arxiv.org/pdf/2210.07809)** | **[[Code]](https://github.com/AntigoneRandy/PTYNet)** | *MM 2023*

设计了一种新的专用模型PTYNet（Resnet18），专门用于嵌入水印（用于确定输入是否包含特定模式，对于正常输入保持沉默，对于生成的背景进行特定响应），在接收验证样本时激活所有权验证。将PTYNet 和目标模型的输出通过加权组合生成最终的预测概率分布。  

### Delving in the loss landscape to embed robust watermarks into neural networks
**[[Paper]](https://ailb-web.ing.unimore.it/icpr/media/posters/10989.pdf)** | **[[Code]](https://github.com/EIDOSlab/Delving-in-the-loss-landscape-to-embed-robust-watermarks-into-neural-networks)** | *ICPR 2021*

利用模型的冗余和适应能力，锁定一部分参数来携带水印序列。通过在损失函数中添加项，找到损失函数的最小值，确保水印嵌入的参数位于landscape的谷底。

### Protecting Deep Learning Model Copyrights with Adversarial Example-Free Reuse Detection
**[[Paper]](https://arxiv.org/pdf/2407.03883)** | *arXiv 2024*

基于神经元功能分析的重用检测器，它只需要正常的测试样本即可通过测量模型在新提出的模型表征（即神经元功能（NF））上的差异来检测重用关系。文章使用欧氏距离和余弦距离来衡量神经元功能的相似性。可以适用于黑盒/白盒环境。

### Steganographic Passport: An Owner and User Verifiable Credential for Deep Model IP Protection Without Retraining
**[[Paper]](https://arxiv.org/pdf/2404.02889)** | *CVPR 2024*

提出了一种新颖的隐写护照，将用户许可证验证与所有权验证解耦。所有权验证，通过将模型权重与所有者端的哈希输出对齐来保持模型权重的唯一护照，允许模型所有者证明分布式模型的所有权和真实性。许可权验证，可以使用用户侧护照提取任何许可用户的隐藏身份 (ID) 图像。实现在不需要重新训练模型的情况下，未注册用户提供可验证的使用许可。

## Black-box

### Turning your weakness into a strength: Watermarking deep neural networks by backdooring
**[[Paper]](https://openreview.net/pdf?id=RNl51vzvDE)** | **[[Code]](https://github.com/adiyoss/WatermarkNN)**  | **[[Link]](https://medium.com/@carstenbaum/the-ubiquity-of-machine-learning-and-its-challenges-to-intellectual-property-dc38e7d66b05)** | *USENIX  2018*

第一篇后门水印，利用神经网络的过参数化易受后门攻击这一弱点，将其转变为神经网络版权保护的优势。利用触发集训练模型实现水印嵌入，在验证阶段，将触发集中的水印图像输入到含水印的模型中，若模型输出水印标签，则说明该模型含有水印。

### Protecting Intellectual Property of Deep Neural Networks with Watermarking
**[[Paper]](https://openreview.net/pdf?id=RNl51vzvDE)** | **[[Link]](https://blog.csdn.net/qq_36332660/article/details/133688057)** | *ASIACCS  2018*

利用输入与输出的映射关系嵌入水印，即给定特定的水印图像，模型若能够输出预设的水印标签，则认为该模型含有水印。其中，水印图像和水印标签构成触发集。将水印标签预设为airplane。提出三种图像变换方法，在原始图像上添加有意义的内容（TEST字样），将原始图像替换为训练集外的图像，在原始图像上添加预设的噪声。

### Identification for deep neural network: Simply adjusting few weights!
**[[Paper]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9835648)** | *ICDE  2022*

基于熵的不确定性采样，选取接近决策边界的关键样本，利用它们的高可修改性和对模型功能性影响小的特点，在深度神经网络参数中嵌入水印。具体来说，采用掩码梯度下降方法，选择对关键样本梯度强、对未选样本和自然样本梯度弱的参数进行微调，以最小改动实现水印嵌入。

### Entangled Watermarks as a Defense against Model Extraction
**[[Paper]](https://arxiv.org/pdf/2002.12200)** | **[[Code]](https://github.com/cleverhans-lab/entangled-watermark)**  |  **[[Link]](https://blog.csdn.net/weixin_44063375/article/details/127465654)** | *USENIX  2021*

EWE 在任务数据中插入添加了触发器的水印样本，并将这些样本标注为语义上不可能的目标类别，通过联合优化交叉熵损失和 SNNL 损失，鼓励模型以相同的神经元组学习任务数据和水印数据，从而增强水印的鲁棒性和不可移除性。

### How to Prove Your Model Belongs to You: A Blind-Watermark based Framework to Protect Intellectual Property of DNN
**[[Paper]](https://arxiv.org/pdf/1903.01743)** | **[[Code]](https://github.com/zhenglisec/Blind-Watermark-for-DNN)**  |  **[[Link]](https://blog.csdn.net/qq_36332660/article/details/133805704)** | *ACSAC  2019*

基于动态水印信号生成的黑盒水印方法，通过轻量级自编码器（Auto Encoder, AE）和判别器的双重约束，将特定 logo 与原始图像结合生成水印图像，同时确保水印图像的质量和有效性。AE 以原始图像和 logo 图像为输入，输出与原始图像在像素空间和结构特性上高度相似的水印图像，而判别器进一步约束水印图像与原始图像在分布上的不可分性，确保水印的视觉和统计不可感知性。

### Deep neural network watermarking against model extraction attack
**[[Paper]](https://dl.acm.org/doi/10.1145/3581783.3612515)** | **[[Code]](https://github.com/jxtalent/SSW-DNN-Watermark)** | *MM  2023*

基于影子模型的触发集嵌入方法 （ symmetric shadow model based watermarking，SSW），使用一个正影子模型模拟攻击者通过模型提取攻击得到的替代模型，再用另一个负影子模型模拟不含水印的干净模型。在嵌入水印的同时，主动优化触发集中的样本，使得它们更容易被原始模型和正影子模型预测为目标标签，而负影子模型中的预测结果与目标标签不一致。

### Protecting ip of deep neural networks with watermarking: A new label helps
**[[Paper]](https://link.springer.com/content/pdf/10.1007/978-3-030-47436-2_35.pdf)** | *PAKDD  2020*
生成一个包含添加微小扰动的图像数据集，这些样本被赋予一个新的类别标签，用作水印样本。将生成的关键样本加入原始数据集，并训练模型使其能够识别这些水印样本所代表的新增类别。通过向可疑模型查询关键样本，检测是否能正确分类到新增类别，验证模型是否嵌入水印。

### Secure Neural Network Watermarking Protocol against Forging Attack
**[[Paper]](https://dl.acm.org/doi/abs/10.1186/s13640-020-00527-1)** | *EURASIP Journal on Image and Video Processing 2020*

引入单向哈希函数，使得用于证明所有权的触发集样本必须通过连续的哈希逐个形成，并且它们的标签也按照样本的哈希值指定。通过这样设定，没有网络训练权限的攻击者就不可能构建伪造水印。

### Margin-based Neural Network Watermarking
**[[Paper]](https://openreview.net/pdf?id=bwBeCyuFZh)** | **[[Code]](https://github.com/matbambbang/margin-based-watermarking)** | *ICML  2023*

提出了一种基于边缘的DNN水印方法，该方法通过在训练期间使用投影梯度上升法来最大化水印样本的边缘，使得攻击者在不损害模型准确性的情况下无法改变预测标签。

### Certified Neural Network Watermarks with Randomized Smoothing
**[[Paper]](https://arxiv.org/pdf/2207.07972)** | **[[Code]](https://github.com/arpitbansal297/certified_watermarks)** | *ICML 2022*

不直接将信息嵌入到模型参数中，但由于验证水印时仍然需要访问模型参数，因此这种水印方法仍然属于白盒水印。通过引入触发集准确性函数的平滑版本来进行所有权验证，而验证过程中需要对模型参数的访问。

### Towards Robust Model Watermark via Reducing Parametric Vulnerability
**[[Paper]](https://openaccess.thecvf.com/content/ICCV2023/papers/Gan_Towards_Robust_Model_Watermark_via_Reducing_Parametric_Vulnerability_ICCV_2023_paper.pdf)** | **[[Code]](https://github.com/GuanhaoGan/robust-model-watermarking)** | *ICCV 2023*

水印模型的参数空间附近存在许多水印去除模型，提出了一种最小最大公式来找到这些水印去除模型并恢复它们的水印行为。通过最大化扰动来找到具有最低水印成功率（WSR）的最坏情况模型，并最小化扰动来恢复水印行为。

### Explanation as a Watermark: Towards Harmless and Multi-bit Model Ownership Verification via Watermarking Feature Attribution
**[[Paper]](https://arxiv.org/pdf/2405.04825)** | **[[Code]](https://github.com/shaoshuo-ss/EaaW)** | *NDSS  2025*

后门水印的缺点——有害性(向带水印的发布模型引入了恶意可控的错误分类行为)，模糊性(恶意用户可以通过寻找其他错误分类的样本来轻松通过验证，从而导致所有权模糊)。这两个限制都源于现有水印方法都是利用错误分类进行预测，该方法在不改变原始预测的情况下，将“多位”水印嵌入到特定触发样本的特征归因解释中。

### MEA-Defender: A Robust Watermark against Model Extraction Attack
**[[Paper]](https://arxiv.org/pdf/2401.15239)** | **[[Code]](https://github.com/lvpeizhuo/MEA-Defender)** | *S&P 2024*

通过在输入域中组合来自两个源类的两个样本来获得水印，并设计一个水印损失函数，使水印的输出域位于主任务样本的输出域内。设计了一个独特的后门，称为共生后门，并将其作为水印嵌入到待保护的DNN模型中。 共生后门保证了水印样本的输入域（即输入样本的数据分布）和输出特征域（即分类层输出特征的数据分布）应处于主水印样本的分布中。 

### Non-transferable learning：A new approach for model ownership verification and applicability authorization
**[[Paper]](https://openreview.net/pdf?id=tYRrOdSnVUy)** | **[[Code]](https://github.com/conditionWang/NTL)** | *ICLR 2022 Oral*

限制模型的泛化能力，使其更加专注于特定的领域，从而验证模型的所有权并授权其在特定数据上的适用性。
通过让模型专注于学习与特定领域相关的特征，从而使模型在目标领域上具有低性能。这种方法与领域泛化和适应的思路不同，后者通常致力于提高模型在不同领域间的适用性。NTL则反其道而行，使模型只适用于特定领域，同时在其他领域上表现较差。

### Probabilistically Robust Watermarking of Neural Networks
**[[Paper]](https://arxiv.org/pdf/2401.08261)** | *IJCAI  2024*

不需要额外的模型训练，可以应用于任何模型架构。 方法的关键思想是计算触发集，该触发集可以在源模型和代理模型集之间以高概率转移。

### Multi-bit, Black-box Watermarking of Deep Neural Networks in Embedded Applications
**[[Paper]](https://openaccess.thecvf.com/content/CVPR2024W/EVW/papers/Leroux_Multi-bit_Black-box_Watermarking_of_Deep_Neural_Networks_in_Embedded_Applications_CVPRW_2024_paper.pdf)** | *CVPR 2024*

引入了一种新颖的多位水印方法，能够有效地生成大量模型实例。 这些实例中的每一个都保持功能等效性，但在使用预定义的触发输入进行提示时表现出独特的行为。

### Not Just Change the Labels, Learn the Features: Watermarking Deep Neural Networks with Multi-View Data
**[[Paper]](https://arxiv.org/pdf/2403.10663)** | **[[Code]](https://github.com/liyuxuan-github/MAT)** | *ECCV 2024*

从特征学习的角度引入了触发集水印方法的全新视角。通过选择具有多重特征的数据（即多视角数据），可以有效地防御功能窃取攻击。基于这一视角，提出了一种基于多视角数据的新型水印技术，称为MAT，用于有效地嵌入水印到DNN中。该方法包括构建一个包含多视角数据的触发集，并结合一种简单的基于特征的正则化方法来训练源模型。

# <span id="Fingerprint">Fingerprint</span> [^](#back)
### Sensitive-Sample Fingerprinting of Deep Neural Networks
**[[Paper]](https://openaccess.thecvf.com/content_CVPR_2019/html/He_Sensitive-Sample_Fingerprinting_of_Deep_Neural_Networks_CVPR_2019_paper.html)** | **[[Code]](https://github.com/zechenghe/Sensitive_Sample_Fingerprinting)** | *CVPR 2019*

提出了一种基于敏感样本的深度神经网络篡改检测方法。首先通过生成敏感样本，识别出模型在权重扰动下的变化，进而检测是否存在篡改。然而，单一的敏感样本可能无法覆盖所有神经元，导致部分权重变化无法被检测。为了解决这一问题，提出了最大激活神经元覆盖（MANC）算法，利用贪心策略从多个敏感样本中选择最能激活神经元的样本集合，以确保尽可能多的神经元被激活，从而生成准确的DNN模型“指纹”，用于模型验证和篡改检测。


### Fingerprinting Deep Neural Networks – A DeepFool Approach  
**[[Paper]](https://dr.ntu.edu.sg/bitstream/10356/147023/2/2021021379.pdf)**  | *ISCAS 2021*

利用 DeepFool 算法生成一组具有最小扰动的对抗样本，这些样本接近分类边界，并且目标模型对它们的分类结果具有明确性。生成的指纹由这些对抗样本及其标签组成，用于记录目标模型的独特行为。在验证阶段，将指纹输入可疑模型，计算其预测结果与目标模型标签的一致概率，以此来判断该模型是否为目标模型的副本。

### MetaV: A Meta-Verifier Approach to Task-Agnostic Model Fingerprinting
**[[Paper]](https://arxiv.org/pdf/2201.07391v3)** | *SIGKDD  2022*

方法首先生成一组正模型（混淆技术在目标模型上生成）和负模型（独立训练），通过训练一个自适应指纹和元验证器组合，使得元验证器能够基于模型对自适应指纹的响应准确区分正负模型，从而检测目标模型是否被盗用。

### GNNFingers: A Fingerprinting Framework for Verifying Ownerships of Graph Neural Networks
**[[Paper]](https://openreview.net/pdf?id=RNl51vzvDE)** | **[[Link]](https://cn-sec.com/archives/2616607.html)** | *WWW  2024*

针对不同的图任务，设计了相应的指纹模板，通过对抗攻击方式更新图的结构和节点属性，生成具有独特性的指纹样本，以此代表模型的特征。使用FC作为判别器，将所有图指纹样本经过模型产生的输出拼接成特征向量，输入判别器，输出二元标签用于区分是否为目标模型。


### Are you stealing my model? sample correlation for fingerprinting deep neural networks
**[[Paper]](https://arxiv.org/pdf/2210.15427)** | **[[Code]](https://github.com/guanjiyang/SAC)** | *NeurIPS  2022*

指纹生成阶段利用错误分类样本或数据增强样本，计算源模型的样本相关性矩阵作为指纹；提取阶段，通过计算嫌疑模型与源模型在样本相关性上的距离，判断是否为窃取模型。该方法无需对模型进行额外训练，高效且鲁棒，能够有效应对多种模型窃取攻击场景。

### Fingerprinting deep neural networks globally via universal adversarial perturbations
**[[Paper]](https://arxiv.org/pdf/2202.08602)** | **[[Code]](https://github.com/faiimea/UAP)** | *CVPR  2022*

利用模型决策边界的全局几何特性捕捉窃取模型与受保护模型的相似性，同时区分独立模型。指纹生成阶段，通过对受保护模型构造能扰乱几乎所有输入数据的全局扰动向量，并结合聚类选取具有代表性的输入样本，生成指纹以描述模型的决策边界特性；提取阶段，通过对嫌疑模型施加相同的扰动并比较指纹的相似性，判断嫌疑模型是否为窃取模型。


### Deep Neural Network Fingerprinting by Conferrable Adversarial Examples
**[[Paper]](https://arxiv.org/pdf/1912.00888)** | **[[Code]](https://github.com/ayberkuckun/DNN-Fingerprinting?tab=readme-ov-file)** | *ICLR  2021*

基于可传递对抗样本，利用源模型与替代模型在对抗性弱点上的共同特性。通过优化一种特殊的目标函数，最大化样本在替代模型上的传递性，同时最小化其在参考模型上的传递性，生成可以区分替代模型和参考模型的对抗样本。

### PublicCheck: Public Integrity Veriﬁcation for Services of Run-time Deep Models
**[[Paper]](https://ieeexplore.ieee.org/document/10179380)** | *S&P  2023*

验证模型的完整性。为了捕获和识别运行时模型的固有预测行为，PublicCheck 生成平滑转换和增强的封装样本，这些样本包含在模型的决策边界周围，同时确保验证查询与正常查询无法区分。

### Intersecting-Boundary-Sensitive Fingerprinting for Tampering Detection of DNN Models
**[[Paper]](https://dl.acm.org/doi/10.5555/3692070.3694306)** | **[[Code]](https://github.com/CGCL-codes/IBSF)** | *ICML  2024*

提出了交叉边界敏感指纹识别 (IBSF)，仅使用 top-1 标签对 DNN 模型进行黑盒完整性验证。篡改模型会改变其决策边界，IBSF 通过最大化所选类别子集的部分香农熵，将指纹样本定位在子集中类别相交的决策边界附近，从而从正常样本中制作指纹样本。

### Model Barrier: A Compact Un-Transferable Isolation Domain for Model Intellectual Property Protection
**[[Paper]](https://arxiv.org/pdf/2303.11078)** | *CVPR  2023*

出了一种称为紧凑不可转移隔离（CUTI）域的新方法，以防止深度模型从授权域非法转移到未授权域。

### MAP: MAsk-Pruning for Source-Free Model Intellectual Property Protection
**[[Paper]](https://arxiv.org/abs/2403.04149)** | **[[Code]](https://github.com/ispc-lab/MAP)** | *CVPR  2024*

在训练良好的模型中，存在与特定目标域相关的参数。通过学习一个目标特定的二进制掩码来剪枝这些参数，从而在最小化对授权数据性能影响的同时，防止未授权数据的使用。


