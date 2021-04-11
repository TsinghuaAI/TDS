# TDS

*   Tsinghua/Temporary DeepSpeed (TDS) is a  plug-in of Microsoft DeepSpeed to fix the bug of the DeepSpeed PipelineEngine. 

*   Although DeepSpeed provides interfaces to support pipeline-parallel training. There are still some bugs and hack implementation in its code, especially the code to send tensors between different stages.  We thus reimplement the PipelineEngine of DeepSpeed in TDS.


# How to use TDS

1. The first step is to install DeepSpeed. How to install DeepSpeed can refer to [DeepSpeed Installation](https://github.com/microsoft/DeepSpeed#installation).

2. Copy the folder "tds" into your project, and use "import tds as deepspeed" instead of "import deepspeed" in your code.

3. If you want to use pipeline-parallel training, you must add the code to let your model know some essential settings for its forward and backward operations. These settings consist of tensor (including both input data and hidden states) types, whether these tensors need to save gradients, and whether these tensors need to be partitioned across GPUs to save memory. We take training GPT-2 as an example, the detailed code can be found from [GPT-2](https://github.com/microsoft/DeepSpeedExamples/blob/master/Megatron-LM-v1.1.5-3D_parallelism/pretrain_gpt2.py).

    *   The code of using DeepSpeed
    
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


    *   The code of using TDS

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


4. All other operations can directly follow [DeepSpeed](https://github.com/microsoft/DeepSpeed) and [DeepSpeedExamples](https://github.com/microsoft/DeepSpeedExamples).



# Examples

More examples like using TDS for GPT-2 and T5 can refer to [CPM-Pretrain](https://github.com/TsinghuaAI/CPM-Pretrain).
