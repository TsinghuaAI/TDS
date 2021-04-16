# TDS

[版本更新记录](release-note.md)|[English Document](../README.md)

## 介绍

*   Tsinghua/Temporary DeepSpeed (TDS) 是Microsoft DeepSpeed的插件，修复了DeepSpeed PipelineEngine的实现问题。 

*  尽管DeepSpeed提供了支持流水线并行训练的接口。它的代码中仍然存在一些错误和hack实现，尤其是在不同流水阶段之间发送张量的代码。因此，我们在TDS中重新实现了DeepSpeed的PipelineEngine，并采用适配器模式简单封装了其他的DeepSpeed接口。

## 如何使用TDS

1. 使用的第一步需要安装DeepSpeed。如何安装DeepSpeed可以参考
[DeepSpeed Installation](https://github.com/microsoft/DeepSpeed#installation).

1. 将文件夹"tds"复制到您的项目中，并且将您项目中所有的"import deepspeed"改为"import tds as deepspeed"。

2. 如果您要使用流水并行来加速训练，您需要额外添加一些代码让你的模型知道一些前向与后向计算中的设置细节。这些设置包括所有的张量类型（包括输入数据和隐层状态），这些张量是否需要保存梯度，以及是否需要在GPU上划分这些张量进行存储以节省显存。我们以训练GPT-2为例来解释使用TDS的一些微小变化，详细代码可从以下网址找到[GPT-2](https://github.com/microsoft/DeepSpeedExamples/blob/master/Megatron-LM-v1.1.5-3D_parallelism/pretrain_gpt2.py).

    *  使用 DeepSpeed 原有代码
    
    ```python
    def model_provider():
        """Build the model for GPT-2."""
        args = get_args()
        print_rank_0('building GPT2 model ...')
        if args.pipe_parallel_size == 0:
            model = GPT2Model(num_tokentypes=0, parallel_output=True)
        else:
            model = GPT2ModelPipe(num_tokentypes=0, parallel_output=True, topology=mpu.get_topology())
            model._megatron_batch_fn = get_batch_pipe
        return model
    ```


    *   使用 TDS 插件之后

    ```python
    def model_provider():
    """Build the model for GPT-2."""
    args = get_args()
    print_rank_0('building GPT2 model ...')
    if args.pipe_parallel_size == 0:
        model = GPT2Model(num_tokentypes=0, parallel_output=True)
    else:
        model = GPT2ModelPipe(num_tokentypes=0, parallel_output=True, topology=mpu.get_topology())
        model._megatron_batch_fn = get_batch_pipe
        # The first input tensor is input embeddings and hidden states, it requires to save its gradients. The second input tensor is attention mask. 
        model._input_grad = [True, False]
        # The first input tensor is input embeddings and hidden states, its type is float. The second input tensor is attention mask, its type is boolean.
        model._input_type = ['float', 'bool']
        # Input embeddings and hidden states can be partitioned across GPUs to save memory.
        model._input_pipe_partitioned = [True, False]
    return model
    ```        


1. 其他各种模型实现与并行加速的操作可以直接参照[DeepSpeed](https://github.com/microsoft/DeepSpeed) and [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples)。



## 样例

我们在[CPM-Pretrain](https://github.com/TsinghuaAI/CPM-Pretrain)中详细给出了如何使用TDS训练GPT-2和T5。

## 引用

如果您使用了我们的代码，请您引用下面的文章。

```[latex]
@article{cpm-v1,
  title={CPM: A Large-scale Generative Chinese Pre-trained Language Model},
  author={Zhang, Zhengyan and Han, Xu, and Zhou, Hao, and Ke, Pei, and Gu, Yuxian and Ye, Deming and Qin, Yujia and Su, Yusheng and Ji, Haozhe and Guan, Jian and Qi, Fanchao and Wang, Xiaozhi and Zheng, Yanan and Zeng, Guoyang and Cao, Huanqi and Chen, Shengqi and Li, Daixuan and Sun, Zhenbo and Liu, Zhiyuan and Huang, Minlie and Han, Wentao and Tang, Jie and Li, Juanzi and Sun, Maosong},
  year={2020}
}
```
