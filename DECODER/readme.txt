值得注意的是：encoder-decoder attention 的q,k,v,向量只有q是decoder的，k和v都是最后一个encoder的
mask：是为了让未看见的词的score对应位置变为负无穷(mask矩阵应该是一个主对角线为1，主对角线下三角全为1，
主对角线上三角全为负无穷，再加到原score上)，然后进入softmax，负无穷在softmax后会趋近于0