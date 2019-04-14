# coding: UTF-8
'''''''''''''''''''''''''''''''''''''''''''''''''''''
 This program is to train or generate peom.
 & This program is a developed version that based on
 the code by JP Huang.
'''''''''''''''''''''''''''''''''''''''''''''''''''''
from header import *
import numpy as np


def readpoems(filename):

    poems = []
    file = open(filename, "r", encoding="gbk")
    for line in file:  # 每行储存一首诗
        # 代码仿照写处1：将line开头及结尾处的空格删除，然后分成题目title与诗体peom
        title,poem=line.strip().split(':')
        ## strip()将line开头及结尾处的空格等其它非文字符号删除
        ## split是分割函数，将字符串分割成“字符”，保存在一个列表中。

        poem = poem.replace(' ','') #去除诗中的空格
        ## replace替换函数。
        if '_' in poem or '《' in poem or '[' in poem or '(' in poem or '（' in poem:
            continue
        ## 去除poem的其它字符

        if len(poem) < 10 or len(poem) > 128:
            continue
        ## 确保poem长度适当

        poem = '[' + poem + ']' #add start and end signs
        poems.append(poem)
        # 将poem加到列表 poems当中

    print("Number of Poems： %d"%len(poems))
    #counting words
    allWords = {}
    # 定义一个空的字典(dictionary)
    # key:为汉字
    # 字典的值value:为该汉字出现的频次
    # 代码仿照写处2
    for poem in poems: # 枚举诗歌列表中所有诗
        for word in poem: # 枚举每首诗中所有的字

            if word not in allWords:
                allWords[word] = 1  # 假如该汉字还没出现过
            else:
                allWords[word] += 1
    #'''
    # 删除低频字
    erase = []
    for key in allWords: # 枚举字典allwords中的所有字key
        if allWords[key] < 2:  #假如其频次小于2，加入待删列表
            erase.append(key)
    for key in erase:
        del allWords[key]
    #'''

    wordPairs = sorted(allWords.items(), key = lambda x: -x[1])
    # 所有出现的汉字按出现频次，由多到少排序。
    # 函数items，用于取出字典的key和值。
    # key = lamda表示按key的值进行排序，按x的第1维进行排序

    words, a= zip(*wordPairs)  # *wordpairs 将wordpairs解压（unzip)
                               # zip()将输入变量变成turple
    words += (" ", )           # 将空格加在最后。
    wordToID = dict(zip(words, range(len(words)))) #word to ID
                                                   # 把每个汉字赋予一个数。

    wordTOIDFun = lambda A: wordToID.get(A, len(words))  # 转化成函数，
                                                        # 输入汉字，输出它对应的数。
    # 代码粘贴处3
    poemsVector = [([wordTOIDFun(word) for word in poem]) for poem in poems] # poem to vector


    ## 将所有的诗转化成向量，其中一首诗中，数字用turple储存，在外面用列表储存。
    ## 类似这样。[(1 3 2 4), (2 3 4 1), (1 5 6 2)]
    #print(poemsVector)
    #padding length to batchMaxLength
    batchNum  = (len(poemsVector) - 1) // batchSize  # // 除法将余数去掉，如10//3 = 3



    X = []
    Y = []
    #create batch
    for i in range(batchNum):
        batch = poemsVector[i * batchSize: (i + 1) * batchSize]
        # batch 储存了 batchsize诗的向量

        maxLength = max([len(vector) for vector in batch])
        # 得到一个batch其中一首最长的诗的长度

        temp = np.full((batchSize, maxLength), wordTOIDFun(" "), np.int32)
        #将temp初始化成batchsize * maxlength的矩阵，其中矩阵元素初始值皆为空格对应的ID。

        for j in range(batchSize):
            temp[j, :len(batch[j])] = batch[j]
        #将temp 储存了一批诗对应的矩阵
        # 代码粘贴处4
        X.append(temp)  # 把这个矩阵放入列表X中，
        temp2 = np.copy(temp)
        temp2[:, :-1] = temp[:, 1:]#将诗向前挪一个字的位置对应的向量放入Y中。
        Y.append(temp2)
        #提示：比如：输入X =“白日依山近…”，输出Y=“日依山近, …”。



        # 将字往向前移一个字。比如
        # temp = [(白日依山进),()]->temp2 [(日依山进进), ()]
        # X -> Y相当于只挪动一个字。


    return X, Y, len(words) + 1, wordToID, words
