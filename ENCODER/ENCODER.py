#transformer的encoder
import tensorflow as tf
from ATTENTION.ATTENTION import self_attention
from ADD_NORM.ADD_NORM import layer_norm,add
from FEED_FORWARD.FEED_FORWARD import feed_forward
class encoder:
    def __init__(self,input_nums,hidden_nums,output_nums,mask=False,multi_head=1):
        '''
        function:
            初始化decoder,由self-attention,feed forward二部分组成
        parameter:
            input_nums:一般是词向量大小
            hidden_nums：q_w,k_w,v_w的大小
            output_nums：一般是词向量大小
            mask：判断是否需要遮掩数据.decoder部分在训练时，这里是True,其他情况是False
            multi_head:多头注意力机制
        '''
        self.self_atten=self_attention(input_nums,hidden_nums,output_nums,mask,multi_head)
        self.fdfd=feed_forward(hidden_nums,output_nums)

    def __call__(self,x):

        #self-attention
        z_output,K,V=self.self_atten(x)
        z_output=add(z_output,x)#残差连接
        z_output_self_attentiont=layer_norm(z_output)#LN

        #feed forward
        z_output=self.fdfd(z_output_self_attentiont)
        z_output=add(z_output_self_attentiont,z_output)#残差连接
        z_output_fdfd=layer_norm(z_output)#LN
        
        return z_output_fdfd,K,V
    
    def get_params(self):
        params=[]

        params.extend(self.self_atten.get_params())
        params.extend(self.fdfd.get_params())

        return params

    def get_params_last_layer(self):
        
        params=[]
        params.extend(self.self_atten.get_k_v_params())
        
        return params
    
 
