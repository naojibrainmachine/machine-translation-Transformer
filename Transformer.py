import tensorflow as tf
import numpy as np
import zipfile
import os
import math
import random
from DECODER.DECODER import decoder
from ENCODER.ENCODER import encoder
from EMBEDDING.EMBEDDING import embedding



class learning_rate_decay(tf.keras.optimizers.schedules.LearningRateSchedule):
    '''
    warmup 策略
    '''
    def __init__(self, initial_learning_rate, d_model, warmup_steps):
        super().__init__()
        self.initial_learning_rate = tf.cast(initial_learning_rate, dtype=tf.float32)
        self.d_model = tf.cast(d_model, dtype=tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, dtype=tf.float32)
        #self.lr=1
        self.curr_learning_rate=self.initial_learning_rate
        
    def __call__(self, step):
        self.curr_learning_rate=(self.d_model**(-0.5))*(min(step**(-0.5),step*(self.warmup_steps**(-1.5))))
        #print(step)
        return self.curr_learning_rate
    def get_lr(self):
        return self.curr_learning_rate
   
class transformer:
    def __init__(self,input_nums,hidden_nums,output_nums,multi_head,layer_nums,vocab_size_current,vocab_size_target,lr=1e-3):
        '''
        func:
            初始化transformer
        param:
            input_nums:embeddig的大小（batch_size,seq_size,embedding_size）
            hidden_nums:q_matrix_weight,k_matrix_weight,v_matrix_weight,(batch_size,seq_size,hidden_size)
            multi_head:一般为8。多头注意力机制
            layer_nums：encoder和decoder的层数
            vocab_size:encoder部分的词库大小
            vocab_size_target:decoder部分的词库大小
            lr:学习率
            
        '''
        self.warmup_steps=4000
        self.output_nums=output_nums
        
        self.layer_nums=layer_nums

        self.embedd_current=embedding(vocab_size_current,input_nums)
        self.embedd_target=embedding(vocab_size_target,input_nums)

        self.pe=self.embedd_current.positional_encoding(50,512)#位置编码
        
        self.enco_layers=[encoder(input_nums,hidden_nums,output_nums,mask=False,multi_head=multi_head) for _ in range(layer_nums)]
        self.deco_layers=[decoder(input_nums,hidden_nums,output_nums,mask=True,multi_head=multi_head) for _ in range(layer_nums)]

        self.w=tf.Variable(tf.random.truncated_normal([output_nums,vocab_size_target],stddev=tf.math.sqrt(2.0/(output_nums+vocab_size_target))),name="transformer w")
        self.b=tf.Variable(tf.zeros([vocab_size_target]),name="transformer b")

        self.lr_schedules=learning_rate_decay(initial_learning_rate=lr,d_model=output_nums,warmup_steps=self.warmup_steps)
        
        #self.opt=tf.keras.optimizers.Adam(learning_rate=self.lr_schedules, beta_1=0.9, beta_2=0.98, epsilon=1e-09)#使用warmup策略
        self.opt=tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.98, epsilon=1e-09)

        #adam优化器参数
        self.v=[]#第一矩向量
        self.m=[]#第二矩向量
        self.lr=lr
        self.beta_1=0.9
        self.beta_2=0.98
        self.epsilon=1e-09
        self.t = 0
        
    def __call__(self,x,y):
        #x,y是one_hot格式
        
        embedd_x=self.embedd_current.embedding_lookup(x)
        embedd_x=embedd_x+self.pe[:,0:embedd_x.shape[1],0:embedd_x.shape[-1]]
        
        embedd_y=self.embedd_target.embedding_lookup(y)
        embedd_y=embedd_y+self.pe[:,0:embedd_y.shape[1],0:embedd_y.shape[-1]]
        
        z_output_x=embedd_x#+self.embedd_target.positional_encoding(embedd_x.shape[1],embedd_x.shape[-1])
        z_output_y=embedd_y#+self.embedd_target.positional_encoding(embedd_y.shape[1],embedd_y.shape[-1])
        k=tf.zeros([1,1])
        v=tf.zeros([1,1])
        for i in range(self.layer_nums):
            #encoder
            z_output_x,k,v=self.enco_layers[i](z_output_x)
            
            
        for j in range(self.layer_nums):
            #decoder
            z_output_y=self.deco_layers[j](z_output_y,k,v)
            
                
        z_output_y=tf.matmul(z_output_y,self.w)+self.b
        z_output_y=tf.nn.softmax(z_output_y,2)
        
        return z_output_y
    
    def get_params(self):
        '''
        func:
            返回模型变量
        '''
        params=[]

        #跳过encoder最后一层跟Q和feed forward相关的variable
        for i in range(self.layer_nums):
            if(i<(self.layer_nums-1)):
                for inner_cell in self.enco_layers[i].get_params():
                    params.append(inner_cell)
            else:
                for inner_cell in self.enco_layers[i].get_params_last_layer():
                    params.append(inner_cell)
                       
        
        params.extend([inner_cell for cell in self.deco_layers for inner_cell in cell.get_params()])

        params.extend(self.embedd_current.get_params())

        params.extend(self.embedd_target.get_params())

        params.append(self.w)

        params.append(self.b)
        
        return params

    def get_all_params(self):
        
        params=[]
        
        params.extend([inner_cell for cell in self.enco_layers for inner_cell in cell.get_params()])

        params.extend([inner_cell for cell in self.deco_layers for inner_cell in cell.get_params()])

        params.extend(self.embedd_current.get_params())

        params.extend(self.embedd_target.get_params())

        params.append(self.w)

        params.append(self.b)
        
        return params
    
    def loss(self,output,Y,mask):
        '''
        function:
            transformer的损失函数
        parameter:
            output：模型的输出
            Y：目标语言
        '''
        return -1*tf.reduce_mean(tf.multiply(tf.math.log(output+ 1e-10),Y))

    
    def update_params(self,grads,params):
        
        print("学习率%f"%self.lr_schedules.get_lr())
        m=[]
        '''
        #print(self.m,"前")
        self.t=self.t+1
        
        for i in range(len(params)):
            if(self.t==1 and isContinue==False):
                self.m.append((1-self.beta_1)*grads[i])
                self.v.append((1-self.beta_1)*grads[i]*grads[i])
            elif self.t==1 and isContinue==True:
                for j in range(len(params)):
                    with zipfile.ZipFile("data\\m_trained.zip",'r') as zipobj:
                        filepath=+"p"+str(i)+".txt"
                        zipobj.extract(filepath,"data")
                        txtP="data"+"\\"+filepath
                        if os.path.exists(txtP):
                            self.m.append((np.loadtxt(txtP,dtype=np.float32)).reshape(grads[i].shape))
                            os.remove(txtP)
                    zipobj.close()
                    with zipfile.ZipFile("data\\v_trained.zip",'r') as zipobj:
                        filepath=+"p"+str(i)+".txt"
                        zipobj.extract(filepath,"data")
                        txtP="data"+"\\"+filepath
                        if os.path.exists(txtP):
                            self.v.append((np.loadtxt(txtP,dtype=np.float32)).reshape(grads[i].shape))
                            os.remove(txtP)
                    zipobj.close()
            elif self.t>1:
                #print(len(grads))
                #print(grads[i],"self.m[i]self.m[i]self.m[i]self.m[i]")
                self.m[i]=self.beta_1*self.m[i]+(1-self.beta_1)*grads[i]
                self.v[i]=self.beta_1*self.v[i]+(1-self.beta_1)*grads[i]*grads[i]
            self.m[i]=self.m[i]/(1-self.beta_1**self.t)
            self.v[i]=self.v[i]/(1-self.beta_2**self.t)
            params[i].assign(params[i]-self.lr*(self.m[i]/(tf.math.sqrt(self.v[i])+self.epsilon)))
        #print(self.m,"后")
        save_weight("data",self.m,name='\\m_trained.zip')
        save_weight("data",self.v,name='\\v_trained.zip')
        '''
        self.opt.apply_gradients(grads_and_vars=zip(grads,params))
    

    
def return_accuracy(temp_predict,temp_batch_label,batch_size):
    '''
    计算准确率
    '''
    
    temp_data= tf.unstack(temp_predict, axis=0)
    temp_label= tf.unstack(temp_batch_label, axis=0)
    
    temp_label2=[label for label in temp_label if tf.reduce_sum(label) -1.0==0.0]
    index=[]
    for i in range(temp_batch_label.shape[0]):
        if (tf.reduce_sum(temp_batch_label[i][:]).numpy()==0.0):
            del temp_label[i-len(index)]
            del temp_data[i-len(index)]
            index.append(i)
            
    
    temp_label=tf.stack(temp_label,0)
    temp_data=tf.stack(temp_data,0)
    
    nums=temp_data.shape[0]
    
    rowMaxSoft=np.argmax(temp_data, axis=1)+1
    rowMax=np.argmax(temp_label, axis=1)+1
    rowMaxSoft=rowMaxSoft.reshape([1,-1])
    rowMax=rowMax.reshape([1,-1])
    '''
    rowMaxSoft=np.argmax(tf.reshape(temp_predict,[-1,temp_predict.shape[-1]]), axis=1)+1
    rowMax=np.argmax(tf.reshape(temp_batch_label,[-1,temp_batch_label.shape[-1]]), axis=1)+1
    rowMaxSoft=rowMaxSoft.reshape([1,-1])
    rowMax=rowMax.reshape([1,-1])
    '''
    nonO=rowMaxSoft-rowMax
    exist = (nonO != 0) * 1.0
    factor = np.ones([nonO.shape[1],1])
    res = np.dot(exist, factor)
    accuracy=(float(nums)-res[0][0])/float(nums)
    
    return accuracy

def save_weight(path,params,name='weight_trained.zip'):
    '''
    func:
        返回模型变量
    param:
        path:路径，只到文件夹部分
        params:模型参数
        
    '''
    path=path+"\\"+name
    if os.path.exists(path):
        os.remove(path)
    with zipfile.ZipFile(path,'w') as zipobj:
        for k in range(len(params)):
            filepath="p"+str(k)+".txt"
            np.savetxt(filepath,(params[k].numpy()).reshape(1,-1))
            zipobj.write(filepath)
            os.remove(filepath) 
    zipobj.close()
    
def load_weight(path,params,name='weight_trained.zip'):
    '''
    func:
        返回模型变量
    param:
        path:路径，只到文件夹部分
        params:模型参数
        
    '''
    tmpPath=path
    path=path+"\\"+name
    with zipfile.ZipFile(path,'r') as zipobj:
        for k in range(len(params)):
            filepath="p"+str(k)+".txt"
            zipobj.extract(filepath,tmpPath)
            txtP=tmpPath+"\\"+filepath
            print(txtP,"txtP")
            if os.path.exists(txtP):
                params[k].assign((np.loadtxt(txtP,dtype=np.float32)).reshape(params[k].shape))
                os.remove(txtP)
        '''
            for k in range(len(params)):
                filepath="p"+str(k)+".txt"
                with zipobj.open(filepath,'r') as f:
                    #print(np.loadtxt(f,dtype=np.float32),filepath)
                    print(params[k],"前")
                    params[k].assign((np.loadtxt(f,dtype=np.float32)).reshape(params[k].shape))
                    print(params[k],"后")
        '''
    zipobj.close()
    
