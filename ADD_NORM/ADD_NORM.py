import tensorflow as tf

def layer_norm(z):
    ax=[]
    ax2=[z.shape[0]]
    for i in range(len(z.shape)):
        if i is not 0:
            ax.append(i)
            ax2.append(1)
    mean=tf.reduce_mean(z,ax)
    mean=tf.reshape(mean,ax2)
    variance=tf.reduce_mean(tf.math.square(z-mean),ax)
    variance=tf.reshape(variance,ax2)
    output=(z-mean)/(tf.math.sqrt(variance+1e-5))
    return output

def add(x,y):
    return x+y
