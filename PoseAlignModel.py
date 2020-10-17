import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
from tensorflow.keras.layers import Dense
from tensorflow.compat.v1 import placeholder, Session, global_variables_initializer
from tensorflow.keras.layers import Flatten
from tensorflow import math

class PoseAlignModel(object):
    
    def __init__(self, input_shape, layers):
      
        #inputs are two sets of poses
        n_in = input_shape[1] * input_shape[2]
        self.njoints = input_shape[1]
        self.input_1 = placeholder(shape=[None, input_shape[1], input_shape[2]], dtype=tf.float32, name="input_1")
        self.input_2 = placeholder(shape=[None, input_shape[1], input_shape[2]], dtype=tf.float32, name="input_2")

        #input layer
        self.flatten_1 = Flatten()
        self.flatten_2 = Flatten()
        self.in_layer_1 = Dense(n_in, activation='relu')
        self.in_layer_2 = Dense(n_in, activation='relu')
        
        self.in1_f = self.flatten_1(self.input_1)
        self.in2_f = self.flatten_2(self.input_2)
        self.in_c_ = self.in_layer_1(self.in1_f) + self.in_layer_2(self.in2_f)
        
        u = self.in_c_
        #next layers
        self.layers = []
        for lsize in layers:
            hl = Dense(lsize, activation='relu')
            self.layers.append(hl)
            u = hl(u)
        
        #output
        self.out_layer = Dense(6, activation='linear')
        self.coeffs = self.out_layer(u)
        #output coefficients: theta1, theta2, a1, a2, b1, b2
        
        #loss
        #self.aligned_x = tf.identity(self.input_1)[:,::2] * self.coeffs[:,0]
        self.in1_x = self.input_1[:,:,0:1]
        self.in1_y = self.input_1[:,:,1:2]
        self.bsize = placeholder(shape=(), dtype=tf.float32, name="batch_size")
        
        cos0 = tf.reshape(tf.repeat(math.cos(self.coeffs[:,0:1]), self.njoints), (self.bsize, self.njoints, 1))
        cos1 = tf.reshape(tf.repeat(math.cos(self.coeffs[:,1:2]), self.njoints), (self.bsize, self.njoints, 1))
        sin0 = tf.reshape(tf.repeat(math.sin(self.coeffs[:,0:1]), self.njoints), (self.bsize, self.njoints, 1))
        sin1 = tf.reshape(tf.repeat(math.sin(self.coeffs[:,1:2]), self.njoints), (self.bsize, self.njoints, 1))
        
        self.rotated_x = self.in1_x*cos1 + self.in1_y*sin0*sin1
        self.rotated_y = self.in1_y*cos0
        
        c_a1 = tf.reshape(tf.repeat(self.coeffs[:,2:3], self.njoints), (self.bsize, self.njoints, 1))
        c_a2 = tf.reshape(tf.repeat(self.coeffs[:,3:4], self.njoints), (self.bsize, self.njoints, 1))
        c_b1 = tf.reshape(tf.repeat(self.coeffs[:,4:5], self.njoints), (self.bsize, self.njoints, 1))
        c_b2 = tf.reshape(tf.repeat(self.coeffs[:,5:6], self.njoints), (self.bsize, self.njoints, 1))
        
        self.aligned_x = self.rotated_x*c_a1 + c_b1
        self.aligned_y = self.rotated_y*c_a2 + c_b2
        self.in1_aligned = tf.concat([self.aligned_x,self.aligned_y],2)
        self.loss_ = tf.reduce_sum([(self.in1_aligned[:,1] - self.input_2[:,1])**2,
                                    (self.in1_aligned[:,2] - self.input_2[:,2])**2,
                                    (self.in1_aligned[:,5] - self.input_2[:,5])**2,
                                    (self.in1_aligned[:,8] - self.input_2[:,8])**2,
                                    (self.in1_aligned[:,11] - self.input_2[:,11])**2
                                   ])
#         #session
        self.sess = Session()
        self.sess.run(global_variables_initializer())

    #training
    def train(self,p1,p2,n_epochs,learning_rate):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train = optimizer.minimize(self.loss_)
        
        for i in range(n_epochs):
            print(self.sess.run([train, self.loss_], 
                                feed_dict={self.input_1: p1,
                                           self.input_2: p2,
                                           self.bsize: p1.shape[0]
                                          }))
    
    #align poses
    def align(self,p1,p2):
        aligned = self.sess.run(self.in1_aligned, 
             feed_dict={self.input_1: p1,
                        self.input_2: p2,
                        self.bsize: p1.shape[0]})
        return aligned
    
    #rotation parameters
    def predict(self, p1, p2):
        coeffs = self.sess.run(self.coeffs, 
             feed_dict={self.input_1: p1,
                        self.input_2: p2,
                        self.bsize: p1.shape[0]})
        return coeffs
    
    #test method
    def debug(self, p1, p2):
        return self.sess.run([self.in1_x,
                              self.in1_y,
                              self.rotated_x,
                              self.rotated_y,
                              self.aligned_x,
                              self.aligned_y,
                              self.in1_aligned,
                              self.in1_aligned - self.input_2,
                             ],
                             feed_dict={self.input_1: p1,
                             self.input_2: p2}
                            )