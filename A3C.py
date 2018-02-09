import numpy as np
import gym
import tensorflow as tf
import time
import sys
from multiprocessing import Process,Queue,Lock
import multiprocessing

class Para():
    '''
    class to hold global variables
    '''
    def __init__(self,variables):
        self.variables = variables
        self.n = len(self.variables)
    def apply_grads(self,lock,grads):
        with lock:
            for i in range(self.n):
                self.variables[i] += grads[i]
            return self.variables
    def get_variables(self,lock):
        with lock:
            return self.variables


def work(worker_id,game,model_file,global_variables_q,stopq,evaluation=False):
    #used in a subprocess
    import tensorflow as tf 
    from multiprocessing import Lock
    import sys
    from networks import A3CNet
    from workers import Worker

    with tf.Session() as sess:
        actor = Worker(sess,worker_id,game,model_file,global_variables_q,stopq)
        if evaluation:
            actor.evaluate() #one evaluator to evalute the performance
        else:
            actor.work()      
        
def A3C(game,num_workers=2):
    from networks import A3CNet
    env = gym.make(game)
    in_size = env.reset().shape[0]
    out_size = env.action_space.n
    model_file = './checkpoints/A3C_workers_'+str(num_workers)+'_'+game
    ## create global_net to get gobal variables
    ###########
    with tf.Session() as sess:
        global_net = A3CNet(sess,in_size,out_size,model_file)
        variables = global_net.get_variables()
        print('global net creation')
        del global_net
    global_variables = Para(variables)
    global_variables_q = Queue() #Queue() can be shared between sub-processes
    global_variables_q.put(global_variables)
    stopq = Queue()
    actors = []
    for i in range(num_workers):
        t = Process(target=work,args=(i,game,model_file,global_variables_q,stopq))
        t.start()
        actors.append(t)

    t = Process(target=work,args=(num_workers,game,model_file,global_variables_q,stopq,True))
    t.start()
    actors.append(t)
    for actor in actors:
        actor.join() # main process will wait for all subprocesses

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    # This works but the default 'fork' deosn't work, don't know why
    if len(sys.argv) >= 2:
        num_workers = int(sys.argv[1])
    else:
        num_workers = 2
    A3C('CartPole-v1',num_workers)
