import model
from model import DQN_Agent
import random
import os, sys
import getopt
import numpy as np
import pyopencl as cl
import time
import torch
os.environ["PYOPENCL_CTX"] = '1'
from stable_baselines3.common.env_util import make_vec_env 
import matplotlib.pyplot as plt
from pyexcel.cookbook import merge_all_to_a_book
import glob
import xlrd
import gym
import tqdm
from prettytable import PrettyTable


def parse_arg(argv):
    inputfile = ''
    env=""
    xclbin_kernel=""
    mode_train=1 #default:train
    plot_flag=0 #default:no plot
    video_flag=0 #default:no video
    try:
        opts, args = getopt.getopt(argv,"hi:e:b:m:p:v:",["ifile=","env=","bitxm=","mode="])
    except getopt.GetoptError:
        print ('test.py -i <inputfile> -env <gym benchmark name> -bitstream <xxx.xclbin> -mode <eval/train> -plot <0/1> -video <0/1>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print ('test.py -i <inputfile> -env <gym benchmark name> -bitstream <xxx.xclbin> -mode <eval/train> -plot <0/1> -video <0/1>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-e", "--env"):
            env = arg
        elif opt in ("-b", "--bitxm"):
            xclbin_kernel = arg
        elif opt in ("-m", "--mode"):
            mode = arg
        elif opt in ("-p", "--plot"):
            plot_flag = int(arg)
        elif opt in ("-v", "--video"):
            video_flag = int(arg)
    # print (inputfile,env,xclbin_kernel,mode)
    if (mode=="eval"):
        mode_train=0
    return (inputfile,env,xclbin_kernel,mode_train,plot_flag,video_flag)

def read_in_params(f):
    # with open('host_params.in', 'r') as file:
    with open(f, 'r') as file:
        data = file.read().replace('\n',' ').split(' ')
        # data = data.split(' ')
    n_param = data[1]
    t_param = data[3]
    m_param = data[5]
    k_param = data[7]
    return [n_param,t_param,m_param,k_param]



class RL_data:
    def __init__(self):
        self.observation = []
        self.reward = []
        self.action = []
        self.doneVec = []

def setup_device():
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags

    dev =cl.get_platforms()[1].get_devices()
    binary = open(xclbin_kernel, "rb").read()

    prg = cl.Program(ctx,dev,[binary])
    prg.build()
    print(dev)
    print("Device is programmed, testing...")

    krnl_vadd = cl.Kernel(prg, "top")

    return [ctx,queue,mf,krnl_vadd]

def clrf(parallel_env,env):
    print("\n########################")
    print("...Key Execution Parameters...")
    print("----------------------------")
    print("Number of parallel environments(actors): ", parallel_env)
    
    action = np.full((parallel_env), 1) #dummy action vector
    action_output = env.action_space.sample()
    env.reset()
    observation, reward, done, info = env.step(action)
    observation_flat = observation.flatten()
    # reward_flat = reward.flatten()

    print("----------------------------")
    print("Environment Space")
    print("Observation space: ", env.observation_space)
    print("Action space: ", env.action_space)


    print("----------------------------")
    print("Environment IO Data sizes")
    print("Observation element type: ", type(observation_flat[0]))
    print("Action element type: ", type(action_output))
    print("----------------------------")

if __name__ == "__main__":
    tup = parse_arg(sys.argv[1:])
    
    ################## User input ###########################
    # use host_params.in to define the parameters
    [n_param,t_param,m_param,k_param] = read_in_params(tup[0])
    environment = tup[1]
    # environment = 'CartPole-v0'
    xclbin_kernel =tup[2]
    # xclbin_kernel = "top_allprofile.xclbin"
    #########################################################  

    #assume this is what these params are
    parallel_env = int(n_param) #16
    iteration_max = int(t_param) #100
    # iteration_max = 100
    episode_num = 10
    n_batch=int(m_param) #16
    train_ep=int(k_param) #1

    env = make_vec_env(environment, n_envs=parallel_env)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    print("input_dim,output_dim:",input_dim,output_dim)
    observation = env.reset()
    observation_flat = observation.flatten()

    [ctx,queue,mf,krnl_vadd] = setup_device()

    clrf( parallel_env,env)


    ###### create output buffer, change this if your output is different than the type of observation 
        # output = np.full((parallel_env), np.uint8(1))
    streamin = np.array(parallel_env*input_dim*[1]).astype(np.float32)
    streamout = np.array(parallel_env*output_dim*[1]).astype(np.float32)
    throwaway_time1 = time.time()

    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, streamout.nbytes)

    test_pf = True
    action = np.full((parallel_env), 1)
    total_gym_time = 0
    total_vitis_time = 0
    total_openCL_time = 0

    test_data = RL_data()

    list_rewards = [[] for i in range(parallel_env)]
    obs_list=[]
    list_episodes = []
    # list_iteration = []
    iteration = 0

    def get_action1(Qout, action_space_len, epsilon):
        # We do not require gradient at this point, because this function will be used either
        # during experience collection or during inference
        
        Q,A = torch.max(torch.from_numpy(Qout), axis=0)
        A = A if torch.rand(1,).item() > epsilon else torch.randint(0,action_space_len,(1,))
        return A

    exp_replay_size = 256
    agent = DQN_Agent(seed = 1423, layer_sizes = [input_dim, 64, output_dim], lr = 1e-3, sync_freq = 5, exp_replay_size = exp_replay_size)

    # initiliaze experiance replay      
    index = 0
    for i in range(exp_replay_size):
    	obs=[random.uniform(-1.0,1.0) for _ in range(4)]
    	obs_next=[random.uniform(-1.0,1.0) for _ in range(4)]
    	agent.collect_experience([obs, random.randint(0, 1), 1.0, obs_next])


    # pcie_hd=0
    # pcie_dh=0
    kernel_time_total=0
    envtime=0
    train_time=0
    others_time=0
    # main training loop
    for x in tqdm.tqdm(range(episode_num)):
        print("########## Episode number: ", x, " ##########")
        test_data.observation = env.reset()
        # print(test_data.observation.shape)
        test_data.doneVec = np.full((parallel_env), False)
        rew=np.full(shape =(parallel_env), fill_value =100,dtype =float)
        train_c=0
        for count in range(iteration_max): #how many iterations to complete, will likely finish when 'done' is true from env.step
            observation = test_data.observation.flatten('F')
            # print("observation_dim: ", test_data.observation.shape,observation.shape)

            #####################################
            #call kernel
            t0 = time.time()
            obs_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=observation)
            krnl_vadd(queue, (1,), (1,), obs_buf, res_g)
            res_np = np.empty_like(streamout)
            cl.enqueue_copy(queue, res_np, res_g)
            t3 = time.time()
            if (count==0):
            	kernel_time_total += t3-t0

            test_data.action = np.reshape(res_np, (parallel_env, output_dim))

            # vitis_time = kernel_time - openCL_time
            ########################################
            #get action finished

            #####################################
            # create observation and reward from Gym
            # obs shape: (batch, single_shape)
            # act shape: (batch, single_shape)
            test_data_new = RL_data()
            # action = env.action_space.sample()
            for i in range(parallel_env):
                action[i] = get_action1(test_data.action, env.action_space.n, epsilon=1)
            t4 = time.time()
            test_data_new.observation, test_data_new.reward, done, info = env.step(action)
            t5 = time.time()
            if (count==0):
            	envtime+=t5-t4
            t8=time.time()
            obs_list.append(test_data_new.observation[0])
            for i in range(parallel_env):
            	agent.collect_experience([test_data.observation[i], action[i], test_data_new.reward[i], test_data_new.observation[i]])
            if (count==0):
            	others_time+=time.time()-t8
            train_c+=1
            if (tup[3]==1):
                if(train_c > 4):
                    train_c = 0
                    t6=time.time()
                    for _ in range(train_ep):
                        loss = agent.train(batch_size=n_batch)
                    t7=time.time()
                    train_time=t7-t6

            t9=time.time()
            test_data.observation = test_data_new.observation

            test_data_new.reward = test_data_new.reward.astype(np.float32) #observation_flat[0])
            if (count==0):
            	others_time+=time.time()-t9

            for i in range(parallel_env):
                if done[i] or test_data.doneVec[i]:
                    if not test_data.doneVec[i]:
                        print("Episode: ",x+1," Env ", i, " finished after {} timesteps".format(count))
                        rew[i]=count
                    test_data.doneVec[i] = True
            
            if(test_data.doneVec.all()):
                break

        for ag in range(parallel_env):
        	list_rewards[ag].append(rew[ag])
        # list_iteration.append(iteration)
        list_episodes.append(x)    

    print("########## Test completed ##########")

    env.close()

    ################## Agents Reward Plot ###########################
    if (tup[4]==1):
        fig, axs = plt.subplots(4, 4)
        axs[0, 0].plot(list_episodes, list_rewards[0])
        axs[0, 0].set_title('Agent 1')
        axs[0, 1].plot(list_episodes, list_rewards[1], 'tab:orange')
        axs[0, 1].set_title('Agent 2')
        axs[0, 2].plot(list_episodes, list_rewards[2], 'tab:green')
        axs[0, 2].set_title('Agent 3')
        axs[0, 3].plot(list_episodes, list_rewards[3], 'tab:red')
        axs[0, 3].set_title('Agent 4')
        axs[1, 0].plot(list_episodes, list_rewards[4])
        axs[1, 0].set_title('Agent 5')
        axs[1, 1].plot(list_episodes, list_rewards[5], 'tab:orange')
        axs[1, 1].set_title('Agent 6')
        axs[1, 2].plot(list_episodes, list_rewards[6], 'tab:green')
        axs[1, 2].set_title('Agent 7')
        axs[1, 3].plot(list_episodes, list_rewards[7], 'tab:red')
        axs[1, 3].set_title('Agent 8')
        axs[2, 0].plot(list_episodes, list_rewards[8])
        axs[2, 0].set_title('Agent 9')
        axs[2, 1].plot(list_episodes, list_rewards[9], 'tab:orange')
        axs[2, 1].set_title('Agent 10')
        axs[2, 2].plot(list_episodes, list_rewards[10], 'tab:green')
        axs[2, 2].set_title('Agent 11')
        axs[2, 3].plot(list_episodes, list_rewards[11], 'tab:red')
        axs[2, 3].set_title('Agent 12')
        axs[3, 0].plot(list_episodes, list_rewards[12])
        axs[3, 0].set_title('Agent 13')
        axs[3, 1].plot(list_episodes, list_rewards[13], 'tab:orange')
        axs[3, 1].set_title('Agent 14')
        axs[3, 2].plot(list_episodes, list_rewards[14], 'tab:green')
        axs[3, 2].set_title('Agent 15')
        axs[3, 3].plot(list_episodes, list_rewards[15], 'tab:red')
        axs[3, 3].set_title('Agent 16')

        for ax in axs.flat:
            ax.set(xlabel='Episodes', ylabel='Reward')
        for ax in axs.flat:
            ax.label_outer()

        plt.savefig("./allagents_plot.png")

    if (tup[5]==1):
        img = obs_list # some array of images
        frames = [] # for storing the generated images
        fig = plt.figure()
        for i in tqdm.tqdm(range(len(obs_list)//100)):
            frames.append([plt.imshow(img[i], cmap=cm.Greys_r,animated=True)])

        ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                        repeat_delay=1000)
        ani.save('./movie_8_7.mp4')

    kernel_time_avg=kernel_time_total/episode_num
    envtime=envtime/episode_num
    others_time=others_time/episode_num

    # str(float("{0:.2f}".format(pcie_hd*1000)))+" ms"
    # ################## Profiling Result ###########################
    print("=============Average Execution Time Breakdwon per Iteration============")
    t = PrettyTable(['Inferences','Env Step','Training','Others(Sampling,etc)'])
    t.add_row([str(float("{0:.2f}".format(kernel_time_avg*1000)))+" ms",str(float("{0:.2f}".format(envtime*1000)))+" ms",str(float("{0:.2f}".format(train_time*1000)))+" ms",
    	str(float("{0:.2f}".format(others_time*1000)))+" ms"])
    print(t)

