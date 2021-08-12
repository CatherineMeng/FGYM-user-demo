
import os, sys
import getopt
import numpy as np
import pyopencl as cl
import time
os.environ["PYOPENCL_CTX"] = '1'
from stable_baselines3.common.env_util import make_vec_env 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
import torch
from model import Policy
import gym
import tqdm
import random
from prettytable import PrettyTable

# command: -i <input_file> -env <env name> -bitstream <xxx.xclbin> -mode <eval/train> -plot <0/1> -video <0/1>
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

    krnl_policy = cl.Kernel(prg, "top")

    return [ctx,queue,mf,krnl_policy]

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
    # episode_num = 3
    # iteration_max = 10000
    # parallel_env = 4
    # environment = 'Pong-v4'
    # environment = "PongNoFrameskip-v4"
    # env = gym.make("PongNoFrameskip-v4")
    env = gym.make(tup[1])
    # xclbin_kernel = "mlp_DDR_pong_1.xclbin"
    xclbin_kernel =tup[2]

    #########################################################

    [n_param,t_param,m_param,k_param] = read_in_params(tup[0])

    #assume this is what these params are
    parallel_env = int(n_param)
    iteration_max = int(t_param)
    # iteration_max = 19000
    # iteration_max = 3000
    # episode_num = int(k_param)
    episode_num = int(3)
    n_batch=int(m_param)
    train_ep=int(k_param)

    # env = make_vec_env(environment, n_envs=parallel_env)
    input_dim = 12000
    output_dim = 2

    [ctx,queue,mf,krnl_policy] = setup_device()

    clrf( parallel_env,env)

    ###### create output buffer, change this if your output is different than the type of observation 
    streamin = np.array(parallel_env*input_dim*[1]).astype(np.float32)
    streamout = np.array(parallel_env*output_dim*[1]).astype(np.float32)
    throwaway_time1 = time.time()

    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, streamout.nbytes)

    #############################################
    test_pf = True
    action = np.full((parallel_env), 1)
    total_gym_time = 0
    total_vitis_time = 0
    total_openCL_time = 0

    test_data = RL_data()

    list_rewards = []
    list_episodes = []
    rew_list=[]
    obs_list=[]
    iteration = 0

    policy = Policy()
    policy.load_state_dict(torch.load('params_6_15.ckpt'))
    policy.eval()

    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)

    w1nparray=policy.layers[0].weight.detach().numpy()[:,:]
    w1nparray_part=w1nparray.reshape((4, 128, 12000))
    w2nparray=policy.layers[2].weight.detach().numpy() 

    obs = env.reset()
    # policy=policy.to(device)

    test_data.doneVec = np.full((parallel_env), False)
    rew=np.full(shape =(parallel_env), fill_value =0,dtype =float)

    # prev_obs = None
    # d_obs=policy.pre_process(obs, prev_obs)

    # pre-process observation into the required data layout by the fpga kernel
    # pre-process weights into the required data layout by fpga kernel
    w1_1d_all = np.array([])
    for i in range(w1nparray_part.shape[0]):
        w1_1d_all=np.concatenate([w1_1d_all, w1nparray_part[i].flatten('F')])
    w1_1d_all=w1_1d_all.astype(np.float32)
    w2_1d_all=w2nparray.flatten('F')
    w2_1d_all=w2_1d_all.astype(np.float32)


    b1_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=w1_1d_all)
    b2_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=w2_1d_all)
    res_g = cl.Buffer(ctx, mf.WRITE_ONLY, streamout.nbytes)

    def get_action1(a,a_prob):
        # We do not require gradient at this point, because this function will be used either
        # during experience collection or during inference
        if a is None:
            logits= torch.tensor([a,a_prob])
            c = torch.distributions.Categorical(logits=logits)
            action = int(c.sample().cpu().numpy()[0])
            action_prob = float(c.probs[0, action].detach().cpu().numpy())
            return action, action_prob

    # pcie_hd=0
    # pcie_dh=0
    kernel_time_total=0
    envtime=0
    train_time=0
    others_time=0

    d_obs_history, action_history, action_prob_history, reward_history = [], [], [], []
    for x in range(episode_num):
        # print("########## Episode number: ", x, " ##########")
        list_episodes.append(x) 
        test_data.observation = env.reset()
        prev_obs = None

        # print(test_data.observation.shape)
        test_data.doneVec = np.full((parallel_env), False)
        # rew=np.full(shape =(parallel_env), fill_value =0,dtype =float)
        rew=0
        for count in tqdm.tqdm(range(iteration_max)): #how many iterations to complete, will likely finish when 'done' is true from env.step
            random.seed(a=None, version=2)
            # observation = test_data.observation.flatten('F')
            # print("observation_dim: ", test_data.observation.shape,observation.shape)
            t6=time.time()
            d_obs=policy.pre_process(test_data.observation, prev_obs)
            # pre-process observation into the required data layout by the fpga kernel
            d_obs_fpga=d_obs.detach().numpy()
            d_obs_fpga=d_obs_fpga.flatten().astype(np.float32)
            if (count==0):
                others_time+=time.time()-t6

            # t8=time.time()
            obs_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d_obs_fpga)
            # if (count==0):
                # pcie_hd+=time.time()-t8
            #####################################
            #call kernel

            t0 = time.time()
            krnl_policy(queue, (1,), (1,), obs_buf, b1_buf, b2_buf,res_g) 
            # t1 = time.time()
            # if (count==0):
                # kernel_time_total+=t1-t0
            # kernel_time = time.time() 
            res_np = np.empty_like(streamout)

            t2 = time.time()
            cl.enqueue_copy(queue, res_np, res_g)
            if (count==0):
                kernel_time_total+=time.time()-t0
                # pcie_dh+=time.time()-t2
            logits=torch.from_numpy(res_np)
            c = torch.distributions.Categorical(logits=logits)
            action = int(c.sample().numpy())

            ########################################
            #get action finished

            #####################################
            # create observation and reward from Gym
            # obs shape: (batch, single_shape)
            # act shape: (batch, single_shape)
            test_data_new = RL_data()
            action, action_prob = res_np
            prev_obs = test_data.observation
            t3=time.time()
            test_data_new.observation, test_data_new.reward, done, info = env.step(int(action))
            if (count==0):
                envtime+=time.time()-t2
            action_history.append(action)
            d_obs_history.append(d_obs)
            action_prob_history.append(action_prob)
            reward_history.append(test_data_new.reward)
            rew  += np.array(test_data_new.reward)
            obs_list.append(test_data_new.observation)

            test_data.observation = test_data_new.observation

            if (done or count==iteration_max-1):
                print('Episode %d (%d timesteps) - Reward: %.2f' % (x, count, rew))
                break

        rew_list.append(rew)
        
        if (tup[3]==1):
            t7=time.time()

            # compute advantage
            R = 0
            discounted_rewards = []

            for r in reward_history[::-1]:
                if r != 0: R = 0 # scored/lost a point in pong, so reset reward sum
                R = r + policy.gamma * R
                discounted_rewards.insert(0, R)

            #print(discounted_rewards[:5])

            discounted_rewards = torch.FloatTensor(discounted_rewards)
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / discounted_rewards.std()
            # update policy
            print("training...")
            for _ in range(train_ep):
                # n_batch = 128
                idxs = random.sample(range(len(action_history)), n_batch)
                d_obs_batch = torch.cat([d_obs_history[idx] for idx in idxs], 0)
                action_batch = torch.LongTensor([action_history[idx] for idx in idxs])
                action_prob_batch = torch.FloatTensor([action_prob_history[idx] for idx in idxs])
                advantage_batch = torch.FloatTensor([discounted_rewards[idx] for idx in idxs])
                opt.zero_grad()
                loss = policy(d_obs_batch, action_batch, action_prob_batch, advantage_batch)
                loss.backward()
                opt.step()
                train_time+=time.time()-t7



    print("########## Test completed ##########")

    env.close()

    if (tup[4]==1):
        ###### generate plot of reward
        fig, axs = plt.subplots(1, 1)
        axs.plot(list_episodes, rew_list)
        axs.set_title('Agent 1')
        axs.set(xlabel='Episodes', ylabel='Reward')
        axs.label_outer()
        axs.set_ylim([0,60])
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
    # pcie_hd=pcie_hd/episode_num
    # pcie_dh=pcie_dh/episode_num
    envtime=envtime/episode_num
    train_time=train_time/episode_num
    others_time=others_time/episode_num

    # ################## Profiling Result ###########################
    print("=============Average Execution Time Breakdwon per Iteration============")
    t = PrettyTable(['Inference','Env Step','Training','Others(Sampling,etc)'])
    t.add_row([str(float("{0:.2f}".format(kernel_time_avg*1000)))+" ms",str(float("{0:.2f}".format(envtime*1000)))+" ms",str(float("{0:.2f}".format(train_time*1000)))+" ms",
        str(float("{0:.2f}".format(others_time*1000)))+" ms"])
    print(t)

    # if (tup[5]==1):