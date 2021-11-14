# <div align=center>代码实现LSTM模型学习心得报告</div>
<div align=center>人工智能1902班&emsp;&emsp;陶锦超&emsp;&emsp;20195218</div>

							
<br> <br/>

## <div align=center>LSTM的出现</div>
&ensp;&ensp;&ensp;&ensp;**LSTM**(Long Short-term Memory)长短时记忆模型是基于**RNN**((Recurrent Neural Network)循环神经网络提出的模型，所以在谈论LSTM时绕不开基于时序信息的深度学习模型——RNN。

&ensp;&ensp;&ensp;&ensp;RNN是专门用于处理序列任务，其每个循环单元的输出都依赖于之前**所有**的循环单元。 因此是适合NLP等需要处理时序信息的任务的模型。但是RNN的缺点也显而易见，基于人类视角下的解释，RNN试图利用隐藏状态$H_t$来记录之前的所有信息。一方面，随着时序的增长，$H_t$所记录的信息越来越来越淡，产生缺失；另一方面，$H_t$毫无重点的记录所有信息，并不能把握住信息的主要特征。实际上RNN的预测效果确实随着序列的增长而显著变差。

&ensp;&ensp;&ensp;&ensp;于是乎，基于RNN实现的LSTM模型应运而生。LSTM模型是RNN的一种改进，主要体现在循环单元的设计上。相比于RNN仅传递前一个时刻的隐藏层状态$H_{t-1}$，LSTM额外传递了一个新变量$C_{t-1}$。它被称作为记忆单元，用来显性地记录**需要记录**的历史内容。同时，LSTM中引入了“门”的概念，通过门单元来**动态地选择遗忘和记忆多少**之前的信息。 LSTM中分别有“遗忘门”、“输入门”、“输出门”。这些门单元都是通过$Sigmoid$激活函数来实现的。

&ensp;&ensp;&ensp;&ensp;事实上，以上对于LSTM的解释都是从人类理解角度去解释的，试想LSTM模型像人类学习一样，从开始学习到渐渐遗忘，再到重新学习刺激记忆，如此往来把握住所学习知识的重点。但是对于机器来说，无论是“遗忘门”、“输入门”还是“输出门”，它都是一视同仁的学习参数，至于其间对谁的“另眼相看”则正是理论试图解释深度学习奥秘的钥匙。
## <div align=center>LSTM的组成</div>
#### 输入门、遗忘门和输出门：
&ensp;&ensp;&ensp;&ensp;作为门控单元，将**当前时间步的输入**和**前一个时间步的隐藏状态**作为数据送入长短时记忆网络门中。它们由三个具有$Sigmoid$激活函数的全连接层处理，以计算输入门、遗忘门和输出门的值。因此，这三个门的值都在$(0,1)$的范围内。
​<div align=center>![输入门、遗忘门和输出门](https://img-blog.csdnimg.cn/6582bee0652b4db99a2c65e2977c7ba3.png)</div>

数学描述如下：
假设有 $h$个隐藏单元，批量大小为$n$，输入数为$d$。如此，输入为$X_t∈R^{n×d}$，前一时间步的隐藏状态为$H_{t−1}∈R^{n×h}$。相应地，时间步 $t$ 的门被定义如下：输入门是$I_t∈R^{n×h}$，遗忘门是$F_t∈R^{n×h}$，输出门是 $O_t∈R^{n×h}$。它们的计算方法如下：
 - $I_t=σ(X_tW_{xi}+H_{t−1}W_{hi}+b_i)$
 - $F_t=σ(X_tW_{xf}+H_{t−1}W_{hf}+b_f)$
 - $O_t=σ(X_tW_{xo}+H_{t−1}W_{ho}+b_o)$


### 候选记忆单元:
&ensp;&ensp;&ensp;&ensp;候选记忆单元:$\widetilde{C}∈R^{n×h}$的计算与上面描述的三个门的计算类似，但是使用$tanh$函数作为激活函数，函数的值范围为$(-1,1)$。计算方法如下：
 - $\widetilde{C}=tanh(X_tW_{xc}+H_{t−1}W_{hc}+b_c)$
​<div align=center>![候选记忆单元](https://img-blog.csdnimg.cn/097e626f8c504814ac8df2cddd8234df.png)</div>


### 记忆单元、 隐藏状态的更新:
&ensp;&ensp;&ensp;&ensp;记忆单元：$C_{t−1}∈R^{n×h}$的更新用到控制输入和遗忘（或跳过）的输入门$I_t$和遗忘门$F_t$。其中输入门$I_t$来控制采用多少来自$\widetilde{C}$的新数据，而遗忘门$F_t$控制保留了多少旧记忆单元$\widetilde{C}_{t-1}$的内容。使用按元素做乘法的方法（哈达玛积），得到以下更新公式：
- $C_t=F_t⊙C_{t−1}+I_t⊙\widetilde{C}$

&ensp;&ensp;&ensp;&ensp;隐藏状态：$H_{t−1}∈R^{n×h}$的更新用到剩下的输出门。在更新中使用了$tanh$激活函数以确保了$H_t$的值始终在区间$(−1,1)$内。

 - $H_t=O_t⊙tanh(C_t).$
​<div align=center>![长短时记忆网络](https://img-blog.csdnimg.cn/4f3a10f497d4441291fd067cd5e11396.png)</div>

## <div align=center>LSTM代码实现（pytorch）</div>
### 调用`nn.LSTM`函数：

```python
class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        self.LSTM = nn.LSTM(input_size=emb_size, hidden_size=n_hidden)
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, X):
        X = self.C(X)
        X = X.transpose(0, 1) # X : [n_step, batch_size, embeding size]
        outputs, (_, _) = self.LSTM(X)
        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        outputs = outputs[-1] # [batch_size, num_directions(=1) * n_hidden]
        model = self.W(outputs) + self.b # model : [batch_size, n_class]
        return model
```
&ensp;&ensp;&ensp;&ensp;其中`TestLSTM`第一层使用`nn.Embedding`层，它的作用是随模型训练得到适合的**词向量**，`n_class` 、`emb_size`分别是**词典的大小尺寸**和**嵌入向量的维度**。（这里为了代码编写方便，这里的`n_class` 、`emb_size`、`n_hidden`都是主函数下的全局变量）


&ensp;&ensp;&ensp;&ensp;第二层就调用了`nn.LSTM`层，其中`input_size`、`hidden_size`分别代表**输入**$X_t$、**隐藏状态**$H_t$的尺寸。


&ensp;&ensp;&ensp;&ensp;最后定义输出层参数`self.w`、`self.b`，其中`self.w`交给线性层`nn.Linear`初始化，`self.b`手动初始化。


&ensp;&ensp;&ensp;&ensp;在前向传播`forward`函数中，模型进行透明处理，输入是维度为`(batch_size ,n_step)`的每一个**batch**，输出就是维度为`(batch_size ,n_class)`的预测结果。函数内部实现中要注意两点：一是对于`nn.Embeddding`输出维度为`(batch_size ,n_step,embedding_size)`的`Tensor`需要将其第`0`维和第`1`维进行转置，以便后续对每一个**时间步**（`n_step`）上不同的**batch**（`batch_size`）进行训练；二是对`nn.LSTM`层输出选最后一个$H_t$作为最终进入输出层的输入。


&ensp;&ensp;&ensp;&ensp;在直接调用`nn.LSTM`下来实现LSTM的情形下，我们忽略了具体实现细节，但是了解了不同层之间的输入输出维度之间的关系，这在宏观视角上告诉了我们LSTM具体的功能是什么，**它处理什么**，**又生成什么**。它处理的是在**每一个时间步上**的对应的一些**batch**，它生成的是**每个时间步上**的输出$H_t$的**List**。(实际上`nn.LSTM`返回的是一个`torch.Tensor`)而要理清楚其中关系，最重要的是要明白层次之间输入输出的**维度**，也就是`Tensor`的尺寸，所以在实际调试时，经常用一些`print`函数来显示`Tensor`的`shape`。
### 自己动手实现：
```python
class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        n_inputs = emb_size
        #参数初始化函数
        def init_parameter():
            return (nn.Linear(n_inputs, n_hidden, bias=False, device=device),
                    nn.Linear(n_hidden, n_hidden, bias=False, device=device),
                    nn.Parameter(torch.zeros(n_hidden, device=device)))

        self.W_xi, self.W_hi, self.b_i = init_parameter()   #输入门参数
        self.W_xf, self.W_hf, self.b_f = init_parameter()   #遗忘门参数
        self.W_xo, self.W_ho, self.b_o = init_parameter()   #输出门参数
        self.W_xc, self.W_hc, self.b_c = init_parameter()   #候选记忆单元参数

        #输出层参数
        self.W = nn.Linear(n_hidden, n_class, bias=False, device=device)
        self.b = nn.Parameter(torch.zeros(n_class, device=device))

    def LSTM(self, inputs, state=None):
        if state is None:
        	# [batch_size, n_hidden]
            hidden_state = torch.zeros(inputs.shape[1], n_hidden, device=device)
            # [batch_size, n_hidden]
            cell_state = torch.zeros(inputs.shape[1], n_hidden, device=device)
            state = (hidden_state, cell_state)
        else:
            state = self.preserve_state(state)
        (H, C) = state
        outputs = []
        for X in inputs:
            I = torch.sigmoid(self.W_xi(X) + self.W_hi(H) + self.b_i)
            F = torch.sigmoid(self.W_xf(X) + self.W_hf(H) + self.b_f)
            O = torch.sigmoid(self.W_xo(X) + self.W_ho(H) + self.b_o)
            C_tilda = torch.tanh(self.W_xc(X) + self.W_hc(H) + self.b_c)
            C = F * C + I * C_tilda
            H = O * torch.tanh(C)
            outputs.append(H)
        return torch.stack(outputs), self.preserve_state((H, C))

    def preserve_state(self, state):
        return state

    def forward(self, X):
        X = self.C(X)
        X = X.transpose(0, 1)  # X : [n_step, batch_size, embeding size]
        outputs, (_, _) = self.LSTM(X)

        # outputs : [n_step, batch_size, num_directions(=1) * n_hidden]
        outputs = outputs[-1]  # [batch_size, num_directions(=1) * n_hidden]
        model = self.W(outputs) + self.b  # model : [batch_size, n_class]
        return model
```
&ensp;&ensp;&ensp;&ensp;自己动手实现`TestLSTM`时，第一层依然用到了`nn.Embedding`层，这里就不做解释（一些主函数全局变量也同调用实现）。事实上我们只是将`nn.LSTM`自己手动实现在`TestLSTM`中，所以我们会发现在前向传播函数`forward`中，并没有进行改动。在`TestLSTM`初始化函数 `__init__`中除了输出层参数的初始化不变外，我们增加了LSTM层具体实现所需要学习的参数，它们分别是**输入门**、**遗忘门**、**输出门**、**候选记忆单元**的相关参数:
 - $I_t=σ(X_tW_{xi}+H_{t−1}W_{hi}+b_i)$
 - $F_t=σ(X_tW_{xf}+H_{t−1}W_{hf}+b_f)$
 - $O_t=σ(X_tW_{xo}+H_{t−1}W_{ho}+b_o)$
 - $\widetilde{C}=tanh(X_tW_{xc}+H_{t−1}W_{hc}+b_c)$


&ensp;&ensp;&ensp;&ensp;包括其中的$W_{xi}$、$W_{hi}$、$b_i$、$W_{xf}$、$W_{hf}$、$b_f$、$W_{xo}$、$W_{ho}$、$b_o$、$W_{xc}$、$W_{hc}$、$b_c$
对应代码中的 `self.W_xi`、`self.W_hi`、 `self.b_i`、`self.W_xf`、`self.W_hf`、 `self.b_f`、 `self.W_xo`、`self.W_ho`、 `self.b_o` 、`self.W_xc`、`self.W_hc`、 `self.b_c`
为了代码简便，手写了初始化参数的函数`init_parameter()`，同**输出层**初始化参数方法进行初始化，不过要注意把初始化的`Tensor`的放在**gpu**上，即`device=device` 。这在调用`nn.LSTM`时是不需要考虑的细节，但在自己动手实现时，后续调试会因为`Tensor`的位置（默认位置是**cpu**）在两个`device`上而报`RuntimeError`的错误。

&ensp;&ensp;&ensp;&ensp;到了具体实现`LSTM(self, inputs, state=None)`函数，我们需要把握住函数整体的**输入输出**，不能看出，对于每一个**cell**来说,输入是**上一个记忆单元**$C_{t−1}$，**上一个隐藏状态**$H_{t−1}$和**输入**$X_t$；输出是**当前记忆单元**$C_{t}$，**当前隐藏状态**$H_{t}$和当前**输出**$H_{t}$。函数将**上一个记忆单元**$C_{t−1}$，**上一个隐藏状态**$H_{t−1}$打包成一个**元组**：`state`，而在处理返回的`state`时，我们使用一个 `preserve_state`函数对状态进行一个**保存**，千万不要忘了对状态`state`的一个初始化（默认参数`state=None`）。（利用一个`if`判断）
&ensp;&ensp;&ensp;&ensp;真正处理运算时，因为前向传播函数`forward`处理了输入的**维度转置**（调用部分提及到，此时`inputs`的维度是`(n_step,batch_size ,embedding_size)`），所以在一个`for`循环中，直接对**每一个时间步**的**batch**进行运算，运算公式如上。最后的`return`部分，因为我们对输出采取**list**的`append`方式，所以要采用`torch.stack()`函数将`list`转变为`Tensor`返回（防止后续的运算报错）。


&ensp;&ensp;&ensp;&ensp;如此，一个自己动手实现的LSTM就完成了。
### 效果比较：
#####  实际训练时的相关参数：
​<div align=center>![实际训练时的相关参数](https://img-blog.csdnimg.cn/20c8190a4291481399d9ee67a2cd8992.png)</div>
##### 调用`nn.LSTM`函数：
###### 模型：
​<div align=center>![调用nn.LSTM的模型](https://img-blog.csdnimg.cn/d567842a3a1b4041b29b8a3f3c469831.png)</div>
###### 一次训练结果：
​<div align=center>![调用nn.LSTM的训练结果](https://img-blog.csdnimg.cn/5f0c253349ec4c3a844ed84e74199e0d.png)</div>
##### 自己动手实现：
###### 模型：
​<div align=center>![自己动手实现模型](https://img-blog.csdnimg.cn/b0ccc44aaa5b4a0ebbbbbb6acb24cedb.png)</div>

###### 一次训练结果：
​<div align=center>![自己动手实现训练结果](https://img-blog.csdnimg.cn/af5b71e3d0e247df8586c65032177d75.png)</div>

&ensp;&ensp;&ensp;&ensp;真正在运行时候，自己动手实现的LSTM和 调用`nn.LSTM`函数在训练效果上,训练集和测试集**训练误差**和**困惑值**上相差不大，但在训练速度上，自己动手实现的明显**慢于**调用，而在观察**gpu**的占用时可以发现调用实现时**gpu**占用**达到90%以上**，但自己动手实现的占用**只在50~60%之间**。查阅相关说明后有了一些猜测，调用对**多个小矩阵乘法**拼接处理成**少的大矩阵乘法**，提高了gpu的占用，大大加快了训练速度；调用的函数部分可能**事先已经编译**成了可执行文件，直接调用减少了python的编译时间。
## <center>双层LSTM的尝试（pytorch）
&ensp;&ensp;&ensp;&ensp;当我自己动手实现了LSTM的时候，我已经自信的认为我学会了LSTM，然而在尝试实现双层LSTM时，我发现我好像还是没有理解LSTM。


&ensp;&ensp;&ensp;&ensp;这是最简单的多层感知机MLP的层次结构：
​<div align=center>![mlp](https://img-blog.csdnimg.cn/ad02b50127774120ad6805d9198f1e55.png)</div>

&ensp;&ensp;&ensp;&ensp;这是复杂一点卷积神经网络CNN的层次结构：
​<div align=center>![CNN](https://img-blog.csdnimg.cn/cb398b606c514222b772491a39e91810.png)</div>

&ensp;&ensp;&ensp;&ensp;它们之间虽然不同，但可以很清楚的看到一种**一步一步**的层次结构。但是基于RNN来的LSTM，它和RNN一样是处理时序信息的模型，正是因为增加了**时序信息**的一个维度，让LSTM或者RNN难想了一点。直到我在网络上看到了RNN模型的这张示意图：
​<div align=center>![RNN](https://img-blog.csdnimg.cn/6c7a0a7236d049b9b920b8ae71d59e41.png)</div>

&ensp;&ensp;&ensp;&ensp;那么在此基础上去理解双层LSTM甚至多层LSTM**似乎**就容易了点：
​<div align=center>![多层LSTM](https://img-blog.csdnimg.cn/efdfefe119d148fd9693456734421245.png)</div>

&ensp;&ensp;&ensp;&ensp;我简单的理解为将上一层的的**output**作为下一层的输入（注意调整相关维度），利用已实现的LSTM进行代码改写：

定义`LSTM_ONE`类：
```python
class LSTM_ONE(nn.Module):
    def __init__(self,input_size, hidden_size, *args, **kwargs):
        super(LSTM_ONE, self).__init__()
        self.input_size, self.hidden_size = input_size, hidden_size
        #参数初始化函数
        def init_parameter():
            return (nn.Linear(self.input_size, self.hidden_size, bias=False, device=device),
                    nn.Linear(self.hidden_size, self.hidden_size, bias=False, device=device),
                    nn.Parameter(torch.zeros(self.hidden_size, device=device)))
        self.W_xi, self.W_hi, self.b_i = init_parameter()   #输入门参数
        self.W_xf, self.W_hf, self.b_f = init_parameter()   #遗忘门参数
        self.W_xo, self.W_ho, self.b_o = init_parameter()   #输出门参数
        self.W_xc, self.W_hc, self.b_c = init_parameter()   #候选记忆单元参数

    def preserve_state(self, state):
        return state

    def forward(self, inputs, state=None):
        if state is None:
        	# [batch_size, n_hidden]
            hidden_state = torch.zeros(inputs.shape[1], self.hidden_size, device=device)
            # [batch_size, n_hidden] 
            cell_state = torch.zeros(inputs.shape[1], self.hidden_size, device=device)  
            state = (hidden_state, cell_state)
        else:
            state = self.preserve_state(state)
        (H, C) = state
        outputs = []
        for X in inputs:
            I = torch.sigmoid(self.W_xi(X) + self.W_hi(H) + self.b_i)
            F = torch.sigmoid(self.W_xf(X) + self.W_hf(H) + self.b_f)
            O = torch.sigmoid(self.W_xo(X) + self.W_ho(H) + self.b_o)
            C_tilda = torch.tanh(self.W_xc(X) + self.W_hc(H) + self.b_c)
            C = F * C + I * C_tilda
            H = O * torch.tanh(C)
            outputs.append(H)
        return torch.stack(outputs), self.preserve_state((H, C))
```
定义`TextLSTM`类：

```python
class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM, self).__init__()
        self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        n_inputs = emb_size
        self.LSTM_1 = LSTM_ONE(emb_size, n_hidden)
        self.LSTM_2 = LSTM_ONE(n_hidden, n_hidden)
        #输出层参数
        self.W = nn.Linear(n_hidden, n_class, bias=False, device=device)
        self.b = nn.Parameter(torch.zeros(n_class, device=device))
    def forward(self, X):
        X = self.C(X)
        X = X.transpose(0, 1)  # X : [n_step, batch_size, embeding size]
        outputs_1, (_, _) = self.LSTM_1(X)
        outputs_2, (_, _) = self.LSTM_2(outputs_1)
        # outputs2 : [n_step, batch_size, num_directions(=1) * n_hidden]
        outputs = outputs_2[-1]  # [batch_size, num_directions(=1) * n_hidden]
        model = self.W(outputs) + self.b  # model : [batch_size, n_class]
        return model
```
&ensp;&ensp;&ensp;&ensp;但是如此构建的双层LSTM的训练效果非但没有提高，还略有下降，加上更慢的运行速度，不由得对“**我理解的双层LSTM**”产生了怀疑。但是由于时间关系，并没有继续研究下去，对双层LSTM的尝试便就此作罢了。
###### 一次训练结果：
​<div align=center>![双层LSTM训练结果](https://img-blog.csdnimg.cn/0db4e4fdba5548c28b043e83ae9db4d1.png)</div>
## <div align=center>LSTM学习心得</div>
 &ensp;&ensp;&ensp;&ensp;LSTM相比RNN而言，所需要训练的参数将近是RNN的4倍，所需要的内存也因为记忆单元的引入而增加，但确实大量实验表明：使用LSTM构造的循环神经网络比普通的RNN循环神经网络性能要好，而且还可以避免**RNN梯度消失**和**梯度爆炸**的问题。随着硬件的越做越强，深度学习的模型合理的逐渐丰富扩大，利用更多的数据，更大的内存，更强的计算能力来达到一个更优秀的效果，这是深度学习发展过程中非常明显的体会。而随着深度学习发展一路走来，我们也很容易在LSTM模型中隐隐约约看到其他模型的影子，除了RNN，我在门控单元对当前输入、记忆单元、隐藏状态的部分丢失（遗忘）和部分选择（记忆）中仿佛看到了**Dropout**的影子。为了实现机器对人类学习的模仿，各种各样的模型理论发出，而深度学习作为这几年火热的方向，在获得的效果上越来越强大明显，但在理论可解释性上一直存在着一个紧闭的黑匣子，而我相信随着不同模型的提出，深度学习能力的不断提高，终有一天会找到打开黑匣子的钥匙。

 &ensp;&ensp;&ensp;&ensp;作为新时代人工智能的学生，我们应该积极学习人工智能整个体系内的知识，无论是**应用层、算法层**还是**系统层、芯片层**。把知识串联，构建自己的知识体系很有益于自己对新知识的快速学习和对过往知识的时常回顾。通过“自然语言处理”这门课，让我在后疫情时代大三上这段时间里再一次的感受的汲取知识的快乐。肖桐老师幽默风趣的课堂授课让我在课堂上学习到了深度学习在NLP上的发展和应用，马安香老师和几位助教学长不辞辛劳的实验教学让我在困惑于理论知识具体实现时及时的给我解释，让我在自己动手的下把理论知识上的公式图片转变成了手指下的代码片段，清晰明朗的课程安排网页整合了很多学习的资源，让我在课后学习非常便利。（正是因为课程安排网页对资源的整合让我发现了李沐老师的入门深度学习课程，让我在每一个空暇时间里都有着非常充实的安排）虽然“自然语言处理”的授课结束了，但对于我而言，仍然有着很多困惑等待着我通过更深入的学习解答。感谢肖桐老师、马安香老师、穆永誉学长、吕传昊学长、刘新宇学长的授课和对我学习上的帮助。

<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({ tex2jax: {inlineMath: [['$', '$']]}, messageStyle: "none" });
</script>