import tensorflow as tf
import numpy as np

class QNet():
    def __init__(self,sess,in_size,out_size,model_file):
        self.sess = sess #tensor flow session
        self.model_file = model_file # filename to save the model
        self.build_model(in_size,out_size) #build network
        self.restore() #load saved model

    def build_model(self,in_size,out_size,hidden_layers=[128,128,128]):
        self.X = tf.placeholder(tf.float32,shape=[in_size,None])
        self.Y = tf.placeholder(tf.float32,shape=[out_size,None])
        def weight_variable(shape):
            initial = tf.random_normal(shape, stddev=0.1)
            return tf.Variable(initial)
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)
        ll = in_size
        Q = self.X
        for l in hidden_layers:
            W = weight_variable((l,ll))
            b = bias_variable((l,1))
            Q = tf.matmul(W,Q) + b
            Q = tf.nn.relu(Q)
            ll = l
        W = weight_variable((out_size,ll))
        b = bias_variable((out_size,1))
        self.Q = tf.matmul(W,Q) + b #output layer has no activation
        self.loss = tf.losses.mean_squared_error(labels=self.Y,predictions=self.Q)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

    def restore(self):
        try:
            tf.train.Saver().restore(self.sess,self.model_file)
        except:
            self.sess.run(tf.global_variables_initializer())

    def train(self,x_train,y_train,n=1,save=True):
        for i in range(n):
            self.sess.run(self.optimizer,feed_dict={self.X:x_train,self.Y:y_train})
        if save:
            tf.train.Saver().save(self.sess,self.model_file)

    def predict(self,x_predict):
        return self.sess.run(self.Q,feed_dict={self.X:x_predict})

class ActorCriticNet():
    def __init__(self,sess,in_size,out_size,model_file):
        self.sess = sess
        self.model_file = model_file
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.variables = self.build_model(in_size,out_size)
        self.saver = tf.train.Saver()
        self.restore()

    def build_model(self,in_size,out_size,hidden_layers=[128,128,128]):
        self.s = tf.placeholder(tf.float32,shape=[in_size,None])
        self.R = tf.placeholder(tf.float32,shape=[1,None])
        self.a = tf.placeholder(tf.int32,shape=[None])
        self.actor_A = tf.placeholder(tf.float32,[1,None])
        def weight_variable(shape):
            initial = tf.random_normal(shape, stddev=0.1)
            return tf.Variable(initial)
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)
        ll = in_size
        Q = self.s

        for l in hidden_layers:
            W = weight_variable((l,ll))
            b = bias_variable((l,1))
            Q = tf.matmul(W,Q) + b
            Q = tf.nn.relu(Q)
            ll = l
        W = weight_variable((out_size,ll))
        b = bias_variable((out_size,1))
        pi = tf.matmul(W,Q) + b
        self.log_pi = tf.nn.log_softmax(pi,axis=0) #policy
        W = weight_variable((1,ll))
        b = bias_variable((1,1))
        self.V = tf.matmul(W,Q) + b
        self.A = self.R - self.V
        self.critic_loss = tf.reduce_mean(tf.square(self.A))
        self.critic_optimizer = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss)
        mask = tf.cast(tf.one_hot(self.a,out_size,axis=0),tf.float32)
        actor_A = tf.tile(self.actor_A,(out_size,1))

        self.actor_objective = tf.reduce_mean(self.log_pi*mask*actor_A)*tf.constant(out_size,dtype=tf.float32)
        self.actor_optimizer = tf.train.AdamOptimizer(self.actor_lr).minimize(-self.actor_objective)
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    def train(self,s,R,a,n=1,save=True):
        for i in range(n):
            _,A = self.sess.run([self.critic_optimizer,self.A],feed_dict={self.s:s,self.R:R})
            _ = self.sess.run(self.actor_optimizer,feed_dict={self.s:s,self.a:a,self.R:R,self.actor_A:A})
        if save:
            self.saver.save(self.sess,self.model_file)

    def predict(self,s):
        log_pi = self.sess.run(self.log_pi,feed_dict={self.s:s})
        return np.exp(log_pi).reshape(-1)

    def value(self,s):
        #print('vs=',s)
        #print(self.sess.run(self.V,feed_dict={self.s:s}))
        return self.sess.run(self.V,feed_dict={self.s:s})
    def restore(self):
        try:
            self.saver.restore(self.sess,self.model_file)
        except:
            self.sess.run(tf.global_variables_initializer())
    def save(self):
        self.saver.save(self.sess,self.model_file)


class A3CNet():
    def __init__(self,sess,in_size,out_size,model_file):
        self.sess = sess
        self.model_file = model_file
        self.actor_lr = 0.0001
        self.critic_lr = 0.001
        self.variables = self.build_model(in_size,out_size)
        self.saver = tf.train.Saver()
        #self.restore()
        self.sess.run(tf.global_variables_initializer())
        
        self.mini_batch_size = 64

    def build_model(self,in_size,out_size,hidden_layers=[64,64,64]):
        self.s = tf.placeholder(tf.float32,shape=[in_size,None])
        self.R = tf.placeholder(tf.float32,shape=[1,None])
        self.a = tf.placeholder(tf.int32,shape=[None])
        self.actor_A = tf.placeholder(tf.float32,[1,None])
        def weight_variable(shape):
            initial = tf.random_normal(shape, stddev=0.1)
            return tf.Variable(initial)
        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)
        ll = in_size
        Q = self.s

        for l in hidden_layers:
            W = weight_variable((l,ll))
            b = bias_variable((l,1))
            Q = tf.matmul(W,Q) + b
            Q = tf.nn.relu(Q)
            ll = l
        W = weight_variable((out_size,ll))
        b = bias_variable((out_size,1))
        pi = tf.matmul(W,Q) + b
        self.log_pi = tf.nn.log_softmax(pi,axis=0) #policy
        W = weight_variable((1,ll))
        b = bias_variable((1,1))
        self.V = tf.matmul(W,Q) + b
        self.A = self.R - self.V
        self.critic_loss = tf.reduce_mean(tf.square(self.A))
        self.critic_optimizer = tf.train.AdamOptimizer(self.critic_lr).minimize(self.critic_loss)
        #self.critic_optimizer = tf.train.AdamOptimizer(self.critic_lr)

        mask = tf.cast(tf.one_hot(self.a,out_size,axis=0),tf.float32)
        actor_A = tf.tile(self.actor_A,(out_size,1))

        self.actor_objective = tf.reduce_mean(self.log_pi*mask*actor_A)*tf.constant(out_size,dtype=tf.float32)
        self.actor_optimizer = tf.train.AdamOptimizer(self.actor_lr).minimize(-self.actor_objective)
        #s#elf.actor_optimizer = tf.train.AdamOptimizer(self.actor_lr)
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    def get_variables(self):
        return self.sess.run(self.variables)

    def train(self,s,R,a,n=1,save=False):
        L = len(a)
        index = list(range(L))
        np.random.shuffle(index)
        s = s[:,index]
        R = R[:,index]
        a = a[index]
        for i in range(n):
            j0 = 0
            j1 = min(j0+self.mini_batch_size,L)
            si = s[:,j0:j1]
            Ri = R[:,j0:j1]
            ai = a[j0:j1]
            _,A = self.sess.run([self.critic_optimizer,self.A],feed_dict={self.s:si,self.R:Ri})
            _ = self.sess.run(self.actor_optimizer,feed_dict={self.s:si,self.a:ai,self.R:Ri,self.actor_A:A})
        if save:
            self.saver.save(self.sess,self.model_file)

    def compute_grads(self,s,R,a,n=1):
        original_vars = self.get_variables()
        #print('original_vars:',original_vars)
        #print(len(original_vars))
        self.train(s,R,a,n,save=False)
        new_vars = self.get_variables()
        #print('new_vars:',new_vars)
        return [v1-v0 for v1,v0 in zip(new_vars,original_vars)]
        
    def predict(self,s):
        log_pi = self.sess.run(self.log_pi,feed_dict={self.s:s})
        return np.exp(log_pi).reshape(-1)

    def value(self,s):
        return self.sess.run(self.V,feed_dict={self.s:s})
    def restore(self):
        try:
            self.saver.restore(self.sess,self.model_file)
        except:
            self.sess.run(tf.global_variables_initializer())
            self.saver.save(self.sess,self.model_file)
    def save(self):
        self.saver.save(self.sess,self.model_file)

if __name__ == '__main__':
    print('for test use only')
    with tf.Session() as sess:
        AC = A3CNet(sess,4,2,'./checkpoints/test_Actor_critic')
        s = np.array([[0,0,0,1],[0,0,0,0]]).T
        R = np.array([[1,2]])
        a = np.array([0,1])
        grads = AC.compute_grads(s,R,a)
        print('grads:',grads)
        print('--------')
