import numpy as np
import gym
import tensorflow as tf
from networks import ActorCriticNet

class Game():
    def __init__(self,sess,game,model_file):
        self.sess = sess
        self.env = gym.make(game)
        self.model_file = './checkpoints/actor_critic_'+model_file+'_'+game
        self.in_size = self.env.reset().shape[0]
        self.out_size = self.env.action_space.n
        self.net = ActorCriticNet(sess,self.in_size,self.out_size,self.model_file)
        self.action_space = list(range(self.env.action_space.n))
    def get_a(self,s):
        s = np.array([s]).T
        pi = self.net.predict(s)
        #print('pi=',pi)
        return np.random.choice(self.action_space,p=pi)
    def value(self,s):
            s = np.array([s]).T
            #print('v=',self.net.value(s))

            return self.net.value(s)[0,0]

    def play(self,n=1000,episodes=1):
        s = self.env.reset()
        for ep in range(episodes):
            for t in range(n):
                self.env.render()
                a = self.get_a(s)
                s,r,done,info = self.env.step(a)
                if done:
                    print("episode {}, played {} steps".format(ep,t+1))
                    s = self.env.reset()
                    break

    def performance(self,episodes=10,n=1000):
        s = self.env.reset()
        steps = []
        for ep in range(episodes):
            for t in range(n):
                a = self.get_a(s)
                s,r,done,info = self.env.step(a)
                if done:
                    steps.append(t+1)
                    s = self.env.reset()
                    break
        return np.mean(steps),np.std(steps)

    def record_performance(self,episode,show=True):
        performance_file = self.model_file+'_performance'
        with open(performance_file,'a') as f:
            mean,std = self.performance()
            f.write(','.join([str(episode),str(mean),str(std)])+'\n')
            if show:
                print('performance:',episode,mean,std)

    def AC_train(self,episodes=1000,max_step=499,gamma=0.99,start_epsilon=0,min_epsilon=0):
        epsilon = start_epsilon
        for ep in range(episodes):
            epsilon *= 0.99
            epsilon = max(min_epsilon,epsilon)
            ss = []
            Rs = []
            aas = []
            s = self.env.reset()
            #print('s=',s)
    #        print('self.value(s)=',self.value(s))
            for t in range(max_step):
                if np.random.uniform() < epsilon:
                    a = self.env.action_space.sample()
                else:
                    a = self.get_a(s)
                #print('s=',s,'a=',a)
                #V = self.value(s)
                ss.append(s)
                aas.append(a)
                s_next,r,done,info = self.env.step(a)
                if not done:
                    R = r + gamma*self.value(s_next)
                    Rs.append(R)
                    #self.net.train(np.array([s]).T,np.array([[R]]),np.array([a]))
                    s = s_next
                else:
                    Rs.append(0)
                    #self.net.train(np.array([s]).T,np.array([[0]]),np.array([a]))
                    break
            print(epsilon,t)
            ss = np.array(ss).T
            Rs = np.array([Rs])
            aas = np.array(aas)
            #print('ss=',ss)
            #print('Rs=',Rs)
            #print('aas=',aas)
            #print(epsilon,ss.shape[1])
            self.net.train(ss,Rs,aas,n=10,save=(ep%10==0))
            if ep%10 == 0:
                self.record_performance(ep)


with tf.Session() as sess:
    game = Game(sess,'CartPole-v1',model_file='actor_critic_performance')
    game.AC_train(episodes=1000,min_epsilon=0.1)
    game.play(episodes=20)

