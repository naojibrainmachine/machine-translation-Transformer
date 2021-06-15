import tensorflow as tf
import numpy as np
from Transformer import transformer,save_weight,load_weight,return_accuracy
from LOADDATA import get_corpus_indices,data_format,get_data,build_vocab


def train(model,params,train_vocab_Ch,train_vocab_En,vocab_size_ch,vocab_size_en,chars_to_idx_Ch,chars_to_idx_En,target_language,batch_size,clip_norm):
    '''
    func:
        训练函数
    param：
        model：模型
        params：模型参数
        train_vocab_Ch：中文训练集
        train_vocab_En：英文对应中文的翻译
        vocab_size_ch：中文词库大小
        vocab_size_en：英文词库大小
        chars_to_idx_Ch：中文词库到索引映射
        chars_to_idx_En：英文词库到索引映射
        target_language：要翻译成的语言，如"English"
        batch_size：批次大小
        clip_norm：梯度裁剪阈值
    '''
    acc=[]
    iter_data=get_data(train_vocab_Ch,train_vocab_En,chars_to_idx_Ch,chars_to_idx_En,batch_size=batch_size,target_language=target_language)
    outputs=[]
    Ys=[]
    for x,y,label in iter_data:

        X,Y=data_format(x,y)#格式化数据
        
        label,_=data_format(label,label,Y[0].shape[-1])#格式化标签，这个标签比Y少一个开始符

        X=tf.concat(X,0)
        Y=tf.concat(Y,0)

        label=tf.concat(label,0)

        X=tf.one_hot(X,vocab_size_ch)
        Y=tf.one_hot(Y,vocab_size_en)

        with tf.GradientTape() as tape:
            tape.watch(params)
            
            output=model(X,Y)
            
            if target_language=="English":
                label=tf.one_hot(label,vocab_size_en)
            elif target_language=="Chinese":
                label=tf.one_hot(label,vocab_size_ch)
            else:
                raise Exception("main中的target_language必须为Chinese或English")
            
            loss=model.loss(output,label)
            
        grads=tape.gradient(loss,params)
        
        #grads,globalNorm=tf.clip_by_global_norm(grads, clip_norm)#梯度裁剪
        
        model.update_params(grads,params)
        print("loss:",np.array(loss).item())
        
        output,label=tf.reshape(output,[output.shape[0]*output.shape[1],output.shape[-1]]),tf.reshape(label,[label.shape[0]*label.shape[1],label.shape[-1]])
        ac=return_accuracy(output,label,label.shape[0])
        print("训练准确率：%f"%ac)
        acc.append(ac)
        
        
    filepath="acc.txt"
    flie=open(filepath,"a+")
    
    flie.write(str(tf.math.reduce_mean(acc).numpy())+"\n")
    flie.close()
    

if __name__ == "__main__":
    
    batch_size=8#批次大小

    input_nums=8#模型输入大小，既embedding的大小

    num_hiddens=16#模型隐藏状态大小，既把输入变换成q,k,v的矩阵参数大小

    num_outputs=8#模型输出大小，既embedding的大小

    layer_nums=2#encoder和decoder的层数，2代表encoder两层，decoder两层

    multi_head=8#多头注意力机制

    clip_norm=1.0
    
    target_language="English"
    
    train_vocab_Ch,train_vocab_En,test_vocab_Ch,test_vocab_En,idx_to_chars_Ch,chars_to_idx_Ch,idx_to_chars_En,chars_to_idx_En,vocab_size_ch,vocab_size_en=build_vocab('data//simplified Chinese to English.txt')
   
    model=transformer(input_nums=input_nums,hidden_nums=num_hiddens,output_nums=num_outputs,multi_head=multi_head,layer_nums=layer_nums,vocab_size_current=vocab_size_ch,vocab_size_target=vocab_size_en,lr=1e-4)#transformer初始化
   
    params=model.get_params()

    all_aprams=model.get_all_params()
    
    epochs=3000

    isContinue=True

    if isContinue==True:
        load_weight("data",params)

    for i in range(epochs):
        train(model,params,train_vocab_Ch,train_vocab_En,vocab_size_ch,vocab_size_en,chars_to_idx_Ch,chars_to_idx_En,target_language,batch_size,clip_norm)
        save_weight("data",params)
        save_weight("d",all_aprams)

        
