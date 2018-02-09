import numpy as np
import gym
import tensorflow as tf
from networks import QNet

class Game():
    def __init__(self,sess,game,model_file):
        self.env = gym.make(game)
        self.model_file = './checkpoints/'+model_file+'_'+game
        self.in_size = self.env.reset().shape[0] + 1
        self.out_size = 1
        self.net = QNet(sess,self.in_size,self.out_size,self.model_file)
    def get_Qsa(self,s,a):
        data = np.hstack((s,a)).reshape(self.in_size,1)
        return self.net.predict(data)[0,0]
    def get_max_a(self,s,sa):
        n = len(sa)
        sa = np.array(sa).reshape(-1,1)
        s = np.repeat(np.array(s).reshape(1,-1),n,axis=0)
        data = np.hstack((s,sa)).T
        predictions = self.net.predict(data).reshape(-1)
        return np.argmax(predictions),np.max(predictions)

    def play(self,n=1000,episodes=1):
        s = self.env.reset()
        for ep in range(episodes):
            for t in range(n):
                self.env.render()
                a,Qsa = self.get_max_a(s,list(range(self.env.action_space.n)))
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
                a,Qsa = self.get_max_a(s,list(range(self.env.action_space.n)))
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



    def DQN(self,episodes=1000,alpha=0.5,gamma=0.99,max_step=501,
                    start_epsilon=1,min_epsilon=0.1,fail_penalty=-100):
        epsilon = start_epsilon
        for ep in range(episodes):
            epsilon *= 0.99
            epsilon = max(0.1,epsilon)
            s = self.env.reset()
            sar = []
            for t in range(max_step):
                if np.random.uniform() < epsilon:
                    a = self.env.action_space.sample()
                    #Qsa = self.get_Qsa(s,a)
                else:
                    a,_ = self.get_max_a(s,list(range(self.env.action_space.n)))
                s_next,r,done,info = self.env.step(a)
                sar.append([s,a,r])
                s = s_next
                if done:
                    if len(sar) < 500:
                        Q_value = fail_penalty
                    else:
                        Q_value = 0
                    break
            data = []
            while sar:
                s,a,r = sar.pop()
                Q_value = r + gamma*Q_value
                Qsa = self.get_Qsa(s,a)
                Qsa = (1-alpha)*Qsa + alpha*Q_value
                data.append(np.hstack((s,a,Qsa)))
                a1,Q_value = self.get_max_a(s,list(range(self.env.action_space.n)))
            print(epsilon,len(data))
            data = np.array(data)
            #print(data)
            self.net.train(data[:,:5].T,data[:,5:].T,save=(ep%10==0))
            if ep%10 == 0:
                self.record_performance(ep)
            if ep % 100 == 0:
                print('episode:',ep)

with tf.Session() as sess:
    game = Game(sess,'CartPole-v1',model_file='DQN-performance')
    #game.DQN(episodes=1000,start_epsilon=1,fail_penalty=0)
    game.play(episodes=20)
    #print(game.record_performance(1))

