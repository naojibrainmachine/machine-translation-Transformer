import tensorflow as tf

class feed_forward:
    '''
    class:
        这是接收self-attention的输出的feed forward
        对输入的向量采用全连接的方式进行输出
    '''
    def __init__(self,hidden_nums,output_nums):
        self.w1=tf.Variable(tf.random.truncated_normal([output_nums,hidden_nums],stddev=tf.math.sqrt(2.0/(hidden_nums+output_nums))))
        self.b1=tf.Variable(tf.random.truncated_normal([hidden_nums]))

        self.w2=tf.Variable(tf.random.truncated_normal([hidden_nums,output_nums],stddev=tf.math.sqrt(2.0/(hidden_nums+output_nums))))
        self.b2=tf.Variable(tf.zeros([output_nums]))

    def __call__(self,z_output_self_attentiont):
        #z_output_self_attentiont（batch_size,seq_size,embbedding_size）
        z_output=tf.matmul(z_output_self_attentiont,self.w1)+self.b1
        z_output=tf.nn.relu(z_output)
        z_output=tf.matmul(z_output,self.w2)+self.b2
        return z_output

    def get_params(self):
        params=[]

        params.append(self.w1)
        params.append(self.b1)
        params.append(self.w2)
        params.append(self.b2)

        return params
