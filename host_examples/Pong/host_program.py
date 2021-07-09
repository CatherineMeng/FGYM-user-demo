
import os, sys
import numpy as np
import pyopencl as cl
import time
os.environ["PYOPENCL_CTX"] = '1'
from stable_baselines3.common.env_util import make_vec_env 
import matplotlib.pyplot as plt
import torch
from model import Policy
import gym
import tqdm

################## User input ###########################
# use host_params.in to define the parameters
# episode_num = 3
# iteration_max = 10000
# parallel_env = 4
# environment = 'Pong-v4'
# environment = "PongNoFrameskip-v4"
env = gym.make("PongNoFrameskip-v4")
xclbin_kernel = "mlp_DDR_pong_1.xclbin"
generate_report = False
#########################################################

def read_in_params():
    with open('host_params.in', 'r') as file:
        data = file.read().replace('\n',' ').split(' ')
        # data = data.split(' ')
    n_param = data[1]
    t_param = data[3]
    m_param = data[5]
    k_param = data[7]
    return [n_param,t_param,m_param,k_param]

[n_param,t_param,m_param,k_param] = read_in_params()

#assume this is what these params are
parallel_env = int(n_param)
# iteration_max = int(t_param)
iteration_max = 19000
# episode_num = int(k_param)
episode_num = int(3)

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

def generate_pre_exe_report(observation_flat,parallel_env,env):
    print("\n########################")
    print("Pre-execution report\n")
    print("----------------------------")
    print("Number of parallel environments: ", parallel_env)
    print("Observation vector elements: ", len(observation_flat))
    print("Observation vector bytes: ", observation_flat.nbytes)
    
    action = np.full((parallel_env), 1) #dummy action vector
    action_output = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    reward_flat = reward.flatten()

    print("----------------------------")
    print("Environment Space")
    print("Observation space: ", env.observation_space)
    print("Action space: ", env.action_space)

    print("----------------------------")
    print("Environment State and Observation Shape")
    print("Observation shape: ", observation.shape)
    print("Reward shape: ", reward.shape)

    print("----------------------------")
    print("Environment IO Data sizes")
    print("Observation element type: ", type(observation_flat[0]))
    print("Reward element type: ", type(reward_flat[0]))
    print("Action element type: ", type(action_output))
    print("----------------------------")


# env = make_vec_env(environment, n_envs=parallel_env)
# input_dim = env.observation_space.shape[0]
# output_dim = env.action_space.n
# print("input_dim,output_dim:",input_dim,output_dim)
# observation = env.reset()
# observation_flat = observation.flatten()
input_dim = 12000
output_dim = 2
# size_array = len(observation_flat)*parallel_env

start_time = time.time()

[ctx,queue,mf,krnl_policy] = setup_device()

setup_time = time.time()


if(generate_report):
    generate_pre_exe_report(observation_flat, parallel_env,env)

###### create output buffer, change this if your output is different than the type of observation 
    # output = np.full((parallel_env), np.uint8(1))
streamin = np.array(parallel_env*input_dim*[1]).astype(np.float32)
streamout = np.array(parallel_env*output_dim*[1]).astype(np.float32)
throwaway_time1 = time.time()

res_g = cl.Buffer(ctx, mf.WRITE_ONLY, streamout.nbytes)

setup_time = setup_time - start_time + (time.time() - throwaway_time1) #time it takes to create the buffers and the connection

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
# list_iteration = []
iteration = 0

policy = Policy()
policy.load_state_dict(torch.load('params_6_15.ckpt'))
policy.eval()
w1nparray=policy.layers[0].weight.detach().numpy()[:,:]
w1nparray_part=w1nparray.reshape((4, 128, 12000))
w2nparray=policy.layers[2].weight.detach().numpy() 

obs = env.reset()

test_data.doneVec = np.full((parallel_env), False)
rew=np.full(shape =(parallel_env), fill_value =0,dtype =float)

# prev_obs = None
# d_obs=policy.pre_process(obs, prev_obs)

# # pre-process observation into the required data layout by the fpga kernel
# d_obs_fpga=d_obs.detach().numpy()
# d_obs_fpga=d_obs_fpga.flatten().astype(np.float32)

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


#for x in tqdm.tqdm(range(episode_num)):
for x in range(3):
    print("########## Episode number: ", x, " ##########")
    list_episodes.append(x) 
    test_data.observation = env.reset()
    prev_obs = None

    # print(test_data.observation.shape)
    test_data.doneVec = np.full((parallel_env), False)
    # rew=np.full(shape =(parallel_env), fill_value =0,dtype =float)
    rew=0
    for count in tqdm.tqdm(range(iteration_max)): #how many iterations to complete, will likely finish when 'done' is true from env.step
        # observation = test_data.observation.flatten('F')
        # print("observation_dim: ", test_data.observation.shape,observation.shape)
        d_obs=policy.pre_process(test_data.observation, prev_obs)
        # pre-process observation into the required data layout by the fpga kernel
        d_obs_fpga=d_obs.detach().numpy()
        d_obs_fpga=d_obs_fpga.flatten().astype(np.float32)
        obs_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=d_obs_fpga)
        #####################################
        #call kernel
        throwaway_time = time.time()

        openCL_time = time.time()

        # krnl_vadd(queue, (1,), (1,), obs_buf, reward_buf, res_g, np.int32(size_array),np.int32(parallel_env)) #np.int32(reward[0]), np.int32(size_array),np.int32(parallel_env))
        krnl_policy(queue, (1,), (1,), obs_buf, b1_buf, b2_buf,res_g) 
        kernel_time = time.time() 
        res_np = np.empty_like(streamout)
        cl.enqueue_copy(queue, res_np, res_g)
        # print("res_np:",res_np) 
        logits=torch.from_numpy(res_np)
        c = torch.distributions.Categorical(logits=logits)
        # print(c)
        # print(c.sample())
        action = int(c.sample().numpy())
        # action = int(c.sample().numpy()[0])

        openCL_time_2 = openCL_time - throwaway_time + time.time() - kernel_time

        # test_data.action = res_np
        # test_data.action = np.reshape(res_np, (parallel_env, output_dim))
        # print("res_np_dim: ", res_np.shape)
        # print("action: ", test_data.action.shape)

        vitis_time = kernel_time - openCL_time
        ########################################
        #get action finished

        #####################################
        # create observation and reward from Gym
        # obs shape: (batch, single_shape)
        # act shape: (batch, single_shape)
        test_data_new = RL_data()
        start_time_gym = time.time()

        action, action_prob = res_np
        prev_obs = test_data.observation
        test_data_new.observation, test_data_new.reward, done, info = env.step(int(action))
        rew  += np.array(test_data_new.reward)
        obs_list.append(test_data_new.observation)

        test_data.observation = test_data_new.observation
        gym_time = time.time()

        # test_data_new.reward = test_data_new.reward.astype(np.float32) #observation_flat[0])

        # for i in range(parallel_env):
        #     if done[i] or test_data.doneVec[i]:
        #         if not test_data.doneVec[i]:
        #             print("Episode: ",x+1," Env ", i, " finished after {} timesteps".format(count))
        #         test_data.doneVec[i] = True
        
        # if(test_data.doneVec.all()):
        #     break
        if done:
            print('Episode %d (%d timesteps) - Reward: %.2f' % (x, count, rew))
            # rew_list.append(rew)
            break
        gym_time -= start_time_gym
        total_gym_time += gym_time
        total_vitis_time += vitis_time
        total_openCL_time += openCL_time_2
        kernel_time = kernel_time - openCL_time

    #list_rewards.append(rew)
    rew_list.append(rew)
    # list_iteration.append(iteration)
       




print("########## Test completed ##########")

total_time = total_gym_time + total_vitis_time + setup_time + total_openCL_time

f = open("data_out.txt", "w")
str_data = str(total_time-total_gym_time-total_vitis_time) + "\n" + str(total_time) + "\n"
f.write(str_data)
f.close()

if(test_pf):
    print("Test passed!")
else:
    print("Test failed!")

env.close()

###### generate plot of reward

fig, ax = plt.subplots(constrained_layout=True)

#plt.scatter(list_episodes, list_rewards)
plt.scatter(list_episodes, rew_list)
ax.set_title('Reward Vs Episodes')
ax.set_xlabel('Iterations')
ax.set_ylabel('Reward')
# ax.set_ylim([0,15])
secax = ax.twiny()
secax.set_xticks(list_episodes)
secax.set_xlabel("Episode")

for x in range(episode_num):
    plt.axvline(x, color='black')
plt.savefig("./inf_reward_plot.png")
