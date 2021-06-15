import tensorflow as tf
from ATTENTION.ATTENTION import self_attention,encoder_decoder_attention
from ADD_NORM.ADD_NORM import layer_norm,add
from FEED_FORWARD.FEED_FORWARD import feed_forward

class decoder:
    def __init__(self,input_nums,hidden_nums,output_nums,mask=False,multi_head=1):
        '''
        function:
            初始化decoder,由self-attention,encoder-decoder-attention,feedforward三部分组成
        parameter:
            input_nums:一般是词向量大小
            hidden_nums：q_w,k_w,v_w的大小
            output_nums：一般是词向量大小
            mask：判断是否需要遮掩数据.decoder部分在训练时，这里是True,其他情况是False
            multi_head:多头注意力机制
        '''
        self.self_atten=self_attention(input_nums,hidden_nums,output_nums,mask,multi_head)
        self.enco_deco_atten=encoder_decoder_attention(input_nums,hidden_nums,output_nums,mask,multi_head)
        self.fdfd=feed_forward(hidden_nums,output_nums)
        
    def __call__(self,x,k,v):
        #x(batch_size,seq_size,embedding_size),k(batch_size,seq_size,hidden_size),v(batch_size,seq_size,hidden_size)
        
        #self-attention
        z_output,_,_=self.self_atten(x)
        z_output=add(z_output,x)#残差连接
        z_output_self_atten=layer_norm(z_output)#Layer Normalization，对一个样本，不同的神经元neuron间做归一化。BN是从神经元的角度出发的，LN是从样本角度出发的。BN(不同样本的相同维度)，LN(不同维度的相同样本)
        
        #encoder-decoder-attention
        z_output,_,_=self.enco_deco_atten(z_output,k,v)
        z_output=add(z_output_self_atten,z_output)#残差连接
        z_output_enco_deco_atten=layer_norm(z_output)#Layer Normalization

        #feed-forward
        z_output=self.fdfd(z_output_enco_deco_atten)
        z_output=add(z_output_enco_deco_atten,z_output)#残差连接
        z_output=layer_norm(z_output)#Layer Normalization

        return z_output

    def get_params(self):
        params=[]

        params.extend(self.self_atten.get_params())
        params.extend(self.enco_deco_atten.get_params())
        params.extend(self.fdfd.get_params())
        
        return params
    def init(self):
        self.self_atten.init()
        self.enco_deco_atten.init()
    def t_plus_one(self):
        self.self_atten.t_plus_one()
        self.enco_deco_atten.t_plus_one()
