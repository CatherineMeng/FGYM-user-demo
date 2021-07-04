
import os, sys
import numpy as np
import pyopencl as cl
import time
import torch
os.environ["PYOPENCL_CTX"] = '1'
from stable_baselines3.common.env_util import make_vec_env 
import matplotlib.pyplot as plt

import gym
import tqdm

################## User input ###########################
# use host_params.in to define the parameters
# episode_num = 3
# iteration_max = 10000
# parallel_env = 4
# environment = 'Pong-v4'
environment = 'CartPole-v0'
# xclbin_kernel = "mlp_DDR_cartpole_16.xclbin"
xclbin_kernel = "top_allprofile.xclbin"
generate_report = True
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
iteration_max = 100
episode_num = int(k_param)

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


env = make_vec_env(environment, n_envs=parallel_env)
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
print("input_dim,output_dim:",input_dim,output_dim)
observation = env.reset()
observation_flat = observation.flatten()

# size_array = len(observation_flat)*parallel_env

start_time = time.time()

[ctx,queue,mf,krnl_vadd] = setup_device()

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
# list_iteration = []
iteration = 0

def get_action(Qout, action_space_len, epsilon):
    # We do not require gradient at this point, because this function will be used either
    # during experience collection or during inference
    
    Q,A = torch.max(torch.from_numpy(Qout), axis=0)
    A = A if torch.rand(1,).item() > epsilon else torch.randint(0,action_space_len,(1,))
    return A

for x in tqdm.tqdm(range(episode_num)):
    print("########## Episode number: ", x, " ##########")
    test_data.observation = env.reset()
    # print(test_data.observation.shape)
    test_data.doneVec = np.full((parallel_env), False)
    rew=np.full(shape =(parallel_env), fill_value =0,dtype =float)
    for count in range(iteration_max): #how many iterations to complete, will likely finish when 'done' is true from env.step
        observation = test_data.observation.flatten('F')
        print("observation_dim: ", test_data.observation.shape,observation.shape)

        #####################################
        #call kernel
        throwaway_time = time.time()

        obs_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=observation)
        # reward_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=test_data.reward)

        openCL_time = time.time()

        # krnl_vadd(queue, (1,), (1,), obs_buf, reward_buf, res_g, np.int32(size_array),np.int32(parallel_env)) #np.int32(reward[0]), np.int32(size_array),np.int32(parallel_env))
        krnl_vadd(queue, (1,), (1,), obs_buf, res_g) 
        
        kernel_time = time.time() 
        
        res_np = np.empty_like(streamout)
        cl.enqueue_copy(queue, res_np, res_g)

        openCL_time_2 = openCL_time - throwaway_time + time.time() - kernel_time

        # test_data.action = res_np
        test_data.action = np.reshape(res_np, (parallel_env, output_dim))
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
        # action = env.action_space.sample()
        for i in range(parallel_env):
            action[i] = get_action(test_data.action, env.action_space.n, epsilon=1)
        test_data_new.observation, test_data_new.reward, done, info = env.step(action)
        print("action: ", action)
        # print("test_data.action: ", test_data.action)
        rew  += np.array(test_data_new.reward)

        test_data.observation = test_data_new.observation
        gym_time = time.time()

        test_data_new.reward = test_data_new.reward.astype(np.float32) #observation_flat[0])

        for i in range(parallel_env):
            if done[i] or test_data.doneVec[i]:
                if not test_data.doneVec[i]:
                    print("Episode: ",x+1," Env ", i, " finished after {} timesteps".format(count))
                test_data.doneVec[i] = True
        
        if(test_data.doneVec.all()):
            break
        gym_time -= start_time_gym
        total_gym_time += gym_time
        total_vitis_time += vitis_time
        total_openCL_time += openCL_time_2
        kernel_time = kernel_time - openCL_time

    list_rewards.append(rew[0])
    # list_iteration.append(iteration)
    list_episodes.append(x)    




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

plt.scatter(list_episodes, list_rewards)
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
