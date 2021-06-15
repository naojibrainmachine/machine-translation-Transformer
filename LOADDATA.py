import os
import fool
import nltk
import math
import pandas as pd
import numpy as np
import random
import tensorflow as tf

random.seed(1)
np.random.seed(1)

def get_corpus_indices(data,chars_to_idx,language="Chinese"):
    """
    转化成词库索引
    
    """
    label_chars=[]
    corpus_indices=[]
    label_indices=[]
    if language=="Chinese" or language=="中文":
        #print(language,language)
        for d in data:
            
            d=d.replace('\n','').replace('\r','').replace(' ','').replace('\u3000','').replace('\u200b','').replace('\U00021861','')
            corpus_chars=fool.cut(d)
            corpus_chars=corpus_chars[0]
            label_chars=corpus_chars+['<结束>']
            #corpus_chars=['<开始>']+corpus_chars+['<结束>']
            corpus_chars=['<开始>']+corpus_chars
            corpus_indices.append([chars_to_idx[char] for char in corpus_chars])#语料索引，既读入的文本，并通过chars_to_idx转化成索引

            label_indices.append([chars_to_idx[char] for char in label_chars])
    elif language=="English" or language=="英文":
        #print(language,language)
        for d in data:
            corpus_chars=nltk.word_tokenize(d)
            corpus_chars=[char.lower() for char in corpus_chars]
            label_chars=corpus_chars+['<end>']
            #corpus_chars=['<start>']+corpus_chars+['<end>']
            #print(len(chars_to_idx.keys()),"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            corpus_chars=['<start>']+corpus_chars
            corpus_indices.append([chars_to_idx[char] for char in corpus_chars])#语料索引，既读入的文本，并通过chars_to_idx转化成索引

            label_indices.append([chars_to_idx[char] for char in label_chars])
            
    else:
        corpus_indices.extend("-1")
        raise Exception("language 必须为Chinese(中文)或者English(英文)")
    return corpus_indices,label_indices

def data_format(data,labels,max_size=0):
    '''
    数据格式化，把整个批次的数据转化成最大数据长度的数据相同的数据长度（以-1进行填充）
    '''
    
    max_size=max_size
    new_data=[]
    
    #获取最大数据长度
    for x in data:
        if(max_size<len(x)):
            max_size=len(x)

    #格式化数据
    for x_t in data:
        if(abs(len(x_t)-max_size)!=0):
            for i in range(abs(len(x_t)-max_size)):
                x_t.extend([-1])
        new_data.append(tf.reshape(x_t,[1,-1]))

    new_labels = []

    max_size=0
    #获取最大数据长度
    for label in labels:
        if(max_size<len(label)):
            max_size=len(label)
    
    #格式化标签
    for label in labels:
        if(abs(len(label)-max_size)!=0):
            for i in range(abs(len(label)-max_size)):
                label.extend([-1])
        new_labels.append(tf.reshape(label,[1,-1]))
    
    return new_data,new_labels

def get_data(data,labels,chars_to_idx,label_chars_to_idx,batch_size,target_language="English"):
    '''
    一个批次一个批次的yield数据
    data:需要批次化的一组数据
    labels:data对应的情感类型
    chars_to_idx;词汇到索引的映射
    label_chars_to_idx;标签到索引的映射
    batch_size;批次大小
    '''
    num_example=math.ceil(len(data)/batch_size)
    
    example_indices=list(range(num_example))
    random.shuffle(example_indices)
    for i in example_indices:
        start=i*batch_size
        if start >(len(data)-1):
            start=(len(data)-1)
            
        
        end=i*batch_size+batch_size
        if end >(len(data)-1):
            end=(len(data)-1)+1
        
        X=data[start:end]
        Y=labels[start:end]
        #print(label_chars_to_idx,"label_chars_to_idx")
        if target_language=="English":
            X,_=get_corpus_indices(X,chars_to_idx)
            Y,label=get_corpus_indices(Y,label_chars_to_idx,language=target_language)
            yield X,Y,label #只是索引化的文本，且长度不一
        elif target_language=="Chinese":
            X,label=get_corpus_indices(X,chars_to_idx)
            Y,_=get_corpus_indices(Y,label_chars_to_idx,language=target_language)
            yield X,Y,label #只是索引化的文本，且长度不一
        

def build_vocab(path):
    """
    构建词库
    path：数据集路径
    """
    df = pd.read_csv(path,sep = '\t')

    #打乱索引
    rand=np.random.permutation(len(df))
    
    #获取数据总条数
    num_sum=len(df['Chinese'])

    #获取所有数据，为构建词库做准备
    vocab_Chinese = list(df['Chinese'])
    vocab_English = list(df['English'])
    #print(vocab_English)
    
    #获取训练数据，所有数据的90%为训练数据
    train_vocab_ch, train_vocab_en = list(df['Chinese'].iloc[rand])[0:int(num_sum*0.9)], list(df['English'].iloc[rand])[0:int(num_sum*0.9)]

    #获取测试数据，所有数据的10%为测试数据
    test_vocab_ch,test_vovab_en=list(df['Chinese'].iloc[rand])[int(num_sum*0.9):num_sum], list(df['English'].iloc[rand])[int(num_sum*0.9):num_sum]
    
    idx_to_chars_en=[]#索引到词汇的映射
    chars_to_idx_en={}#词汇到索引的映射

    idx_to_chars_ch=[]#索引到词汇的映射
    chars_to_idx_ch={}#词汇到索引的映射

    #print(nltk.word_tokenize("I love you"),"safafsdfsdgfd")
    
    #构建词库，用foolnltk进行分词
    if os.path.exists("data\\idx_to_chars_ch.csv")==False and os.path.exists("data\\idx_to_chars_en.csv")==False:
        print("lai")
        for i in range(num_sum):
            corpus_ch=vocab_Chinese[i].replace('\n','').replace('\r','').replace(' ','').replace('\u3000','').replace('\u200b','').replace('\U00021861','')
            corpus_chars_ch=fool.cut(corpus_ch)
            corpus_chars_ch=corpus_chars_ch[0]
            idx_to_chars_ch.extend(corpus_chars_ch)

            corpus_en=vocab_English[i]
            corpus_chars_en=nltk.word_tokenize(corpus_en)
            corpus_chars_en=[char.lower() for char in corpus_chars_en]
            idx_to_chars_en.extend(corpus_chars_en)

        idx_to_chars_ch.extend(['<开始>'])

        idx_to_chars_ch.extend(['<结束>'])

        idx_to_chars_en.extend(['<start>'])

        idx_to_chars_en.extend(['<end>'])

        #print(idx_to_chars_en)
        idx_to_chars_ch=list(set(idx_to_chars_ch))#索引到词汇的映射
        #print(idx_to_chars_ch)
        idx_to_chars_en=list(set(idx_to_chars_en))#索引到词汇的映射
        #print(idx_to_chars_en)

        df_cn = pd.DataFrame(idx_to_chars_ch, columns=['vocabulary'])
        df_cn.to_csv("data\\idx_to_chars_ch.csv",index=0)

        df_en = pd.DataFrame(idx_to_chars_en, columns=['vocabulary'])
        df_en.to_csv("data\\idx_to_chars_en.csv",index=0)
    else:
        
        vocab_cn=pd.read_csv("data\\idx_to_chars_ch.csv")
        idx_to_chars_ch=list(vocab_cn['vocabulary'])
        
        vocab_en=pd.read_csv("data\\idx_to_chars_en.csv")
        idx_to_chars_en=list(vocab_en['vocabulary'])
    

    
    chars_to_idx_en=dict([(char,i) for i,char in enumerate(idx_to_chars_en)])#词汇到索引的映射
    
    chars_to_idx_ch=dict([(char,i) for i,char in enumerate(idx_to_chars_ch)])#词汇到索引的映射

    #print(chars_to_idx_en)
    #print(chars_to_idx_ch)
    
    vocab_size_ch=len(idx_to_chars_ch)#词库大小

    vocab_size_en=len(idx_to_chars_en)#词库大小
   
    return train_vocab_ch,train_vocab_en,test_vocab_ch,test_vovab_en,idx_to_chars_ch,chars_to_idx_ch,idx_to_chars_en,chars_to_idx_en,vocab_size_ch,vocab_size_en

#build_vocab('data//simplified Chinese to English.csv')
#vocabulary,labels ,chars_to_idx,idx_to_chars,vocab_size,label_idx_to_chars,label_chars_to_idx,label_size=build_vocab('data//data_single.csv')
#get_data(data=vocabulary,labels=labels,chars_to_idx=chars_to_idx,label_chars_to_idx=label_chars_to_idx,batch_size=3)
#train_vocab_Ch,train_vocab_En,test_vocab_Ch,test_vovab_En,idx_to_chars_Ch,chars_to_idx_Ch,idx_to_chars_En,chars_to_idx_En,vocab_size_ch,vocab_size_en=build_vocab('data//simplified Chinese to English.txt')
