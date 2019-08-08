import  tensorflow as tf
import pandas as pd 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

num_epoch = 10
dropout = 0.5
batch_size = 256
number_of_train_examples = 42000
steps = int (number_of_train_examples/batch_size)
number_of_test_examples = 28000
test_steps = int(number_of_test_examples /batch_size)
 # load and  preprocess training examples
def read_training_data(start ,end):    
    df = pd.read_csv('train.csv',skiprows = start , nrows = end - start)
    train_x_y = df.to_numpy()
    train_x = train_x_y[:,1:]
    train_y = train_x_y[:,0].reshape([train_x_y.shape[0] ,1])
    train_y = np.eye(10)[train_y] 
    train_y = train_y.reshape( [train_y.shape[0] ,train_y.shape[2]])
    train_x = train_x.reshape([train_x.shape[0],28,28,1])
    train_x = train_x /255.0
    return train_x , train_y

# load and preprocess test examples
def read_test_data(start , end):
    df = pd.read_csv('test.csv',skiprows = start , nrows = end - start)
    test_x = df.to_numpy()
    test_x = test_x.reshape([test_x.shape[0],28,28,1])
    test_x = test_x /255.0
    return test_x

# for i in range(0,100) :
#     x_sample = train_x[i,:,:,:].reshape((28,28)).astype('uint8')
#     im = Image.fromarray(x_sample)
#     im.save("images/"+ str(i) + ".jpeg")

X = tf.placeholder(dtype = tf.float32, shape=[None,28,28,1])
Y = tf.placeholder(dtype = tf.float32, shape=[None,10])
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, w, b, strides=1):
    out = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
    out = out+ b
    return tf.nn.relu(out)

def max_pooling(x, k):
     return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1] , padding='SAME')

def model(x,keep_prob,Weights,Bias):
    
    # first convolution layer
    conv1  = conv2d(x ,Weights["w_conv1"] ,Bias["b_conv1"], 1)
    conv1  = max_pooling(conv1,2)
    
    # second convolution layer
    conv2  = conv2d(conv1 ,Weights["w_conv2"] ,Bias["b_conv2"], 1)
    conv2  = max_pooling(conv2,2)
    
    # first  fully connected layer
    fully1 = tf.reshape(conv2,shape  = [-1 ,7*7*64 ] )  
    fully1 = tf.nn.relu( tf.add( tf.matmul(fully1 , Weights['w_fc1']) ,Bias['b_fc1']))
    fully1 = tf.nn.dropout(fully1 , keep_prob)    
    # second  fully connected layer
    fully2 = tf.nn.relu( tf.add( tf.matmul(fully1 , Weights['w_fc2']) ,Bias['b_fc2']))
    fully2 = tf.nn.dropout(fully2 , keep_prob)    
    
   #softmax layer
    logits = tf.add(tf.matmul(fully2  , Weights['w_soft']) ,Bias['b_soft'])
    return logits

Weights = {
        "w_conv1" :tf.Variable (tf.truncated_normal(shape=[5,5,1,32] , stddev = 0.1 )),
        "w_conv2" :tf.Variable (tf.truncated_normal(shape=[5,5,32,64] , stddev = 0.1 )),
        "w_fc1" :tf.Variable ( tf.truncated_normal(shape=[7*7*64,1024] , stddev = 0.1)),
        "w_fc2" :tf.Variable ( tf.truncated_normal(shape=[1024,512] , stddev = 0.1)),
        "w_soft" :tf.Variable ( tf.truncated_normal(shape=[512,10] , stddev = 0.1 )),
       }
    
bias = {
        "b_conv1" :tf.Variable ( tf.constant(0.1,shape=[32] )),
        "b_conv2" :tf.Variable ( tf.constant(0.1,shape=[64] )),
        "b_fc1" :tf.Variable ( tf.constant( 0.1,shape=[1024] )),
        "b_fc2" :tf.Variable ( tf.constant( 0.1,shape=[512] )),
        "b_soft" :tf.Variable ( tf.constant(0.1,shape=[10] ))
        }
    

out  = model(X,keep_prob,Weights,bias)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y , logits = out ))

optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
train = optimizer.minimize(cost)

correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()
save_path = 'checkpoints/best_validation'

with tf.Session() as sess:
    sess.run(init)
    #training
    for j in range(0,num_epoch):
        total_epoch_acc = 0
        print("epoch" + str(j))
        for i in range(0,steps+1):
            start = i*batch_size
            end = (i+1)*batch_size
            batch_x ,batch_y =  read_training_data(start ,end)
            sess.run(train,feed_dict = {X :batch_x ,Y:batch_y , keep_prob : dropout})
            loss , acc =sess.run([cost,accuracy] ,feed_dict = {X :batch_x ,Y:batch_y , keep_prob :1.0})
            total_epoch_acc += acc
            print("Step " + str(i) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
        print("total_epoch_acc " + str(total_epoch_acc / (steps+1)) )
        saver.save(sess, save_path)
    #testing
#     for i in range(0,test_steps+1):
#         start = i*batch_size
#         end = (i+1)*batch_size
#         batch_x  =  read_test_data(start ,end)
#         test_output =sess.run([out] ,feed_dict = {X :batch_x , keep_prob :1.0})
#         print(test_output)
from pandas import DataFrame
session =tf.Session()
saver.restore(session ,save_path)
test_array = np.zeros((28000,1))


for i in range(0,test_steps+1):
    start = i*batch_size
    end = (i+1)*batch_size
    batch_x  =  read_test_data(start ,end)
    test_output =session.run([out] ,feed_dict = {X :batch_x , keep_prob :1.0})
    test_soft = tf.convert_to_tensor(test_output[0])
    test_soft =tf.argmax(tf.nn.softmax(test_soft) , axis = 1)
    res = session.run(test_soft)
    test_array[start:end,:] =  res.reshape([res.shape[0] , 1])
    print(test_array[start,:])
    data = batch_x[0,:,:,:] * 255
    data = (data.reshape((28,28))).astype('uint8')
    img = Image.fromarray(data)
    img.save( 'images/'+str(i)+'.jpeg')


image_id = np.arange(1,28001)
image_id.reshape((28000,1))
csv_file_array = np.zeros((28000,2) , dtype=int)
csv_file_array[:,0] = image_id
csv_file_array[:,1] = test_array.reshape((28000))
df = DataFrame(csv_file_array, columns= ['ImageId', 'Label'])
export_csv = df.to_csv (r'sample_submission.csv', index = False, header=True)