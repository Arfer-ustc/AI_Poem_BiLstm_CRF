# coding: UTF-8
'''''''''''''''''''''''''''''''''''''''''''''''''''''

'''''''''''''''''''''''''''''''''''''''''''''''''''''
import tensorflow as tf
import numpy as np
from header import *

def buildModel(wordNum, gtX, hidden_units = 128, layers = 2):
    """build rnn"""

    with tf.variable_scope("embedding"): #embedding
           embedding = tf.get_variable("embedding", [wordNum, hidden_units], dtype = tf.float32)
           # embedding 为 n * m维的矩阵，其中n为涉及的汉字的数目，m为隐藏LSTM单元的数目
           # 将汉字用hidden_units维向量重新表征，训练RNN的其中一个目的就是为了找到embedding这个矩阵
           ## 代码仿写处 1
           #将输入buildmodel的变量gtX（整数表征），通过内嵌表embedding，转变成向量表征变量inputbatch。
           inputbatch = tf.nn.embedding_lookup(embedding, gtX)
           # 将16首诗转化为向量表达。
           # embedding_lookup(params, ids)其实就是按照ids顺序返回params中的第ids行。比如说，
           #ids=[1,3,2],就是返回params中第1,3,2行。返回结果为由params的1,3,2行组成的tensor。
           # gtx: batchsize * hidden_units, embedding (wordnum*hidden)
           # inputbatch: batchsize * maxLength * hidden
    ## 代码粘贴处2
    #BasicLSTMCell：搭建LSTM长短记忆基本模块
    basicCell = tf.contrib.rnn.BasicLSTMCell(hidden_units, state_is_tuple = True)   


    ## 代码粘贴处3
    #将LSTM单元垒起来
    stackCell = tf.contrib.rnn.MultiRNNCell([basicCell] * layers)

    ## 代码粘贴处4
    #构建动态RNN。
    initState = stackCell.zero_state(np.shape(gtX)[0], tf.float32)
    outputs, finalState = tf.nn.dynamic_rnn(stackCell, inputbatch, initial_state = initState)



    outputs = tf.reshape(outputs, [-1, hidden_units])

    with tf.variable_scope("softmax"):
        w = tf.get_variable("w", [hidden_units, wordNum])
        b = tf.get_variable("b", [wordNum])
        logits = tf.matmul(outputs, w) + b

    probs = tf.nn.softmax(logits)
    return logits, probs, stackCell, initState, finalState




def train(X, Y, wordNum, reload=True):
    """train model"""
    gtX = tf.placeholder(tf.int32, shape=[batchSize, None])  # input
    gtY = tf.placeholder(tf.int32, shape=[batchSize, None])  # output

    logits, probs, a, b, c = buildModel(wordNum, gtX)
    targets = tf.reshape(gtY, [-1]) # gtY [16,122]
    # logits,probs [16,122,5669(wordNum)]
    #loss
    #代码粘贴处5
    #成本函数
    loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets],
                          [tf.ones_like(targets, dtype=tf.float32)], wordNum)


    # ones_like(a)产生一个形状与a一样，但元素皆为1的变量。在这里使其权重皆为1
    # ...sequence_loss_by_example(logits(16*wordnum),targ(16),weights(16)):以交叉熵为例，
#  第一步将targ转化 16*wordnum维度，第二步-targ*logits

    cost = tf.reduce_mean(loss) # 定义损耗函数

    # 代码粘贴处6
    tvars = tf.trainable_variables() #获得可训练的变量
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)  #计算梯度
    ## clip_by_global_norm 使得整体梯度的范数平方和不超5，超过部分将变成5

    learningRate = learningRateBase
    # 代码粘贴处7
    #选择优化方法
    optimizer = tf.train.AdamOptimizer(learningRate)

    # 代码粘贴处8
    #构建训练器
    trainOP = optimizer.apply_gradients(zip(grads, tvars))


    globalStep = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if reload:
            checkPoint = tf.train.get_checkpoint_state(checkpointsPath)
            # if have checkPoint, restore checkPoint
            if checkPoint and checkPoint.model_checkpoint_path:
                saver.restore(sess, checkPoint.model_checkpoint_path)
                print("restored %s" % checkPoint.model_checkpoint_path)
            else:
                print("no checkpoint found!")

        for epoch in range(epochNum):
            if globalStep % learningRateDecreaseStep == 0: #learning rate decrease by epoch
                learningRate = learningRateBase * (0.95 ** epoch)
            epochSteps = len(X) # equal to batch
            for step, (x, y) in enumerate(zip(X, Y)):

                globalStep = epoch * epochSteps + step
                a, loss = sess.run([trainOP, cost], feed_dict = {gtX:x, gtY:y})
                print("epoch: %d steps:%d/%d loss:%3f" % (epoch,step,epochSteps,loss))
                if globalStep%1000==0:
                    print("save model")
                    saver.save(sess,checkpointsPath + "/poem",global_step=epoch)

def probsToWord(weights, words):
    """probs to word"""
    #代码粘贴处9
    t = np.cumsum(weights) #prefix sum
    s = np.sum(weights)    # 总概率
    coff = np.random.rand(1) 
    index = int(np.searchsorted(t, coff * s)) # 由于t是累积分布，因此5667个数由小到大，此函数返回 coff*s这个数排在第几


    #index= int(np.argmax(weights))
    return words[index]

def test(wordNum, wordToID, words):
    """generate poem"""
    gtX = tf.placeholder(tf.int32, shape=[1, None])  # input
    logits, probs, stackCell, initState, finalState = buildModel(wordNum, gtX)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        checkPoint = tf.train.get_checkpoint_state(checkpointsPath)
        # if have checkPoint, restore checkPoint
        if checkPoint and checkPoint.model_checkpoint_path:
            saver.restore(sess, checkPoint.model_checkpoint_path)
            print("restored %s" % checkPoint.model_checkpoint_path)
        else:
            print("no checkpoint found!")
            exit(0)

        poems = []
        for i in range(generateNum):
            state = sess.run(stackCell.zero_state(1, tf.float32))
            x = np.array([[wordToID['[']]]) # init start sign
            probs1, state = sess.run([probs, finalState], feed_dict={gtX: x, initState: state})
            word = probsToWord(probs1, words)
            poem = ''
            while word != ']' and word != ' ':
                poem += word
                if word == '。':
                    poem += '\n'
                x = np.array([[wordToID[word]]])
                #print(word)
                probs2, state = sess.run([probs, finalState], feed_dict={gtX: x, initState: state})
                word = probsToWord(probs2, words)
            print(poem)
            poems.append(poem)
        return poems

def testHead(wordNum, wordToID, words, characters):
    gtX = tf.placeholder(tf.int32, shape=[1, None])  # input
    logits, probs, stackCell, initState, finalState = buildModel(wordNum, gtX)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        checkPoint = tf.train.get_checkpoint_state(checkpointsPath)
        # if have checkPoint, restore checkPoint
        if checkPoint and checkPoint.model_checkpoint_path:
            saver.restore(sess, checkPoint.model_checkpoint_path)
            print("restored %s" % checkPoint.model_checkpoint_path)
        else:
            print("no checkpoint found!")
            exit(0)
        flag = 1
        endSign = {-1: "，", 1: "。"}
        poem = ''
        state = sess.run(stackCell.zero_state(1, tf.float32))
        x = np.array([[wordToID['[']]])

        probs1, state = sess.run([probs, finalState], feed_dict={gtX: x, initState: state})
        for c in characters:            
                word = c
                flag = -flag
                while word != ']' and word != '，' and word != '。' and word != ' ':
                    # 代码粘贴处10
                    poem += word
                    x = np.array([[wordToID[word]]])
                    probs2, state = sess.run([probs, finalState], feed_dict={gtX: x, initState: state})
                    
                    word = probsToWord(probs2, words)

                poem += endSign[flag]
                # keep the context, state must be updated
                if endSign[flag] == '。':
                    probs2, state = sess.run([probs, finalState],
                                             feed_dict={gtX: np.array([[wordToID["。"]]]), initState: state})
                    poem += '\n'
                else:
                    probs2, state = sess.run([probs, finalState],
                                             feed_dict={gtX: np.array([[wordToID["，"]]]), initState: state})

        print(characters)
        print(poem)
        return poem
