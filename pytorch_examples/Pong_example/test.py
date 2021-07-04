import gym
import numpy as np
import torch
from torch import nn
from model import Policy
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from tqdm import tqdm

env = gym.make('PongNoFrameskip-v4')
env = gym.wrappers.Monitor(env, './tmp', video_callable=lambda ep_id: True, force=True)
env.reset()

policy = Policy()
policy.load_state_dict(torch.load('params_5_10.ckpt'))
policy.eval()

obs_list=[]
for episode in range(1):
    prev_obs = None
    obs = env.reset()
    for t in range(190000):
        #env.render()

        d_obs = policy.pre_process(obs, prev_obs)
        with torch.no_grad():
            action, action_prob = policy(d_obs, deterministic=False)
        
        prev_obs = obs
        obs, reward, done, info = env.step(policy.convert_action(action))
        obs_list.append(obs)
        
        if done:
            print('Episode %d (%d timesteps) - Reward: %.2f' % (episode, t, reward))
            break
        
        #time.sleep(0.033)

env.close()


img = obs_list # some array of images
frames = [] # for storing the generated images
fig = plt.figure()
for i in tqdm(range(len(obs_list))):
    frames.append([plt.imshow(img[i], cmap=cm.Greys_r,animated=True)])

ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True,
                                repeat_delay=1000)
ani.save('./movie_5_11.mp4')
#plt.show()