import tensorflow as tf 
from multiprocessing import Lock
import sys
from networks import A3CNet
import numpy as np
import gym
import time


class Worker():
        def __init__(self,sess,worker_id,game,model_file,global_variables_q,stopq):
            self.sess = sess
            self.id = worker_id
            self.env = gym.make(game)
            self.model_file = model_file 
            self.evaluation_file = self.model_file+'_evaluation'
            self.global_variables_q = global_variables_q #Share global variables with Queue()
            self.stopq = stopq #Global stop sign
            self.in_size = self.env.reset().shape[0]
            self.out_size = self.env.action_space.n
            self.net = A3CNet(self.sess,self.in_size,self.out_size,self.model_file) #Local network
            self.action_space = list(range(self.env.action_space.n))
            self.lock = Lock()
            self.update_size = 128
            self.pull_global_variables()

        def stop(self):
            self.stopq.put(1)

        def should_stop(self):
            return not self.stopq.empty()

        def pull_global_variables(self):
            gn = self.global_variables_q.get()
            global_variables = gn.get_variables(self.lock)
            self.global_variables_q.put(gn)
            self.sess.run([a.assign(b) for a,b in zip(self.net.variables,global_variables)])
        
        def compute_local_grads(self,s,R,a,n=1):
            return self.net.compute_grads(s,R,a,n)

        def update_global_variables_grads(self,grads):
            gn = self.global_variables_q.get()
            gn.apply_grads(self.lock,grads)
            self.global_variables_q.put(gn)

        def update_and_pull_global_variables(self,s,R,a,n=1):
            grads = self.compute_local_grads(s,R,a,n)
            self.update_global_variables_grads(grads)
            self.pull_global_variables()

        def get_a(self,s):
                s = np.array([s]).T
                pi = self.net.predict(s)
                return np.random.choice(self.action_space,p=pi)

        def value(self,s):
                s = np.array([s]).T
                return self.net.value(s)[0,0]

        def performance_and_save(self,episodes=10,max_step=1000):
            '''
            evaluate the performance of the model
            episodes: number of episodes to evluate
            max_step: maximum steps per episode
            '''
            s = self.env.reset()
            steps = []
            self.pull_global_variables() #use global variables
            self.net.save()
            print('Network saved')
            sys.stdout.flush()
            for ep in range(episodes):
                for t in range(max_step):
                    a = self.get_a(s)
                    s,r,done,info = self.env.step(a)
                    if done:
                        steps.append(t+1)
                        s = self.env.reset()
                        break
            return np.mean(steps),np.std(steps)

        def evaluate(self,interval=10,max_time=3600):
            '''
            evaluate the model at different times
            interval: seconds between two evaluations of performance
            max_time: (seconds) max time for all Workers
            model will be saved during evaluation
            '''
            start_time = time.time()
            print('start_valuation:')
            t = time.time() - start_time
            while t < max_time:
                mean,std = self.performance_and_save()
                with open(self.evaluation_file,'a') as f:
                    f.write(','.join([str(t),str(mean),str(std)])+'\n')
                print('performance:',int(t),'s',mean,std)
                sys.stdout.flush()
                time.sleep(interval)
                t = time.time() - start_time
            self.stop() #send stop signal to other processes
            return

        def work(self,episodes=2000,max_step=499,gamma=0.99,start_epsilon=0.1,min_epsilon=0.1):
            print('worker',self.id,'starts exploring')
            epsilon = start_epsilon
            for ep in range(episodes):
                if self.should_stop():
                    return
                epsilon *= 0.99
                epsilon = max(min_epsilon,epsilon)
                ss = []
                Rs = []
                aas = []
                s = self.env.reset()
                for t in range(max_step):
                    if np.random.uniform() < epsilon:
                        a = self.env.action_space.sample()
                    else:
                        a = self.get_a(s)
                    ss.append(s)
                    aas.append(a)
                    s_next,r,done,info = self.env.step(a)
                    R = r + gamma*self.value(s_next) if not done else 0
                    Rs.append(R)
                    s = s_next
                    if done or len(aas) >= self.update_size:
                        ss = np.array(ss).T
                        Rs = np.array([Rs])
                        aas = np.array(aas)
                        self.update_and_pull_global_variables(ss,Rs,aas,n=10)
                        ss,Rs,aas = [],[],[]
                        if done:
                            break
                if ep%50==0:
                     print('worker',self.id,ep,t+1)
                     sys.stdout.flush()

