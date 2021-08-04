
import sys
import gym
from prettytable import PrettyTable

if __name__ == "__main__":
    if(len(sys.argv)!=4 and len(sys.argv)!=5):
        print(len(sys.argv))
        print ("Command Arg Error! \
            required format: python generator.py <device_metadata>.in <algo_cfg>.in env_name --(optional)profile_status")
        sys.exit()

 #    print ("=======================================")
    # print ("==========Obs Space dictionary=========")
    # print ("=======================================")

    print("start")
    # top_instance_name: default is "top" if not input by user
    print ("=======================================")
    print ("==========Device information===========")
    print ("=======================================")
    device_input_file=sys.argv[1]
    fr = open(device_input_file, "r")
    device_name=fr.readline() #returns a string to be put in
    ddr_info=fr.readline().split(" ") 
    hbm_info=fr.readline().split(" ") 
    plram_info=fr.readline().split(" ") #returns a list splitted by space
    
    # plram can be used?
    if (int(plram_info[1])!=0): #=0 means no plram support on the user-provided device
        plram_size=int(plram_info[1]) #in KB
    fr.close()

    env_name=sys.argv[3]
    algo_input_file=sys.argv[2]
    fr1 = open(algo_input_file, "r")
    N=fr1.readline().split(" ")[1]
    T=fr1.readline().split(" ")[1]
    M=fr1.readline().split(" ")[1]
    # K=readline().split(" ")[1]
    flag_plr=0
    # 4 bytes per number, KB
    env = gym.make(env_name)
    obs_spc=env.observation_space
    obs_size=0
    if (len(env.observation_space.shape)>1): #atari games
        obs_size=obs_spc.shape[0]*obs_spc.shape[1]*obs_spc.shape[2]
    else:
        obs_size=obs_spc.shape[0]

    
    required_memory=obs_size*max(int(N),int(M))*4/1024 #in KB
    if (plram_size>required_memory and plram_info[1]!=0):
        flag_plr=1
     #....... Put code here
    fr1.close()


    print ("=======================================")
    print ("===========Model information===========")
    print ("=======================================")
    num_kernel_arg=0
    num_layers=0
    flag=input("DNN weights fit on-chip? T or F")
    if (flag=="T"):
        num_kernel_arg=2 #input and output
    else:
        num_layers=int(input("# layers in the policy DNN model? (0 means no DNN model)"))
        if (num_layers==0):
            num_kernel_arg=2 #input and output
        else:
            num_kernel_arg=num_layers+2 #one port interface for each weight, ttwo ports for input and output
    # Use the above to genertae setArg parts and pre-execution-spec.out 

    # print ("=======================================")
    # print ("========Algorithm information==========")
    # print ("=======================================")
    f=open('./dev.cfg', 'w') 
    # =======Code for generating config file Begins here ==========
    f.write('platform ='+device_name)
    f.write('debug=1')
    f.write("\n\n[connectivity]\n")
    f.write("nk=top:1:top_1\n")

    main_mem="DDR"
    if (int(hbm_info[1])!=0):
        main_mem="HBM"
    In_mem="DDR"
    if (flag_plr==1):
        In_mem="PLRAM"
    f.write("sp=top_1.In:"+In_mem+"[0]\n")
    f.write("sp=top_1.Out:"+main_mem+"[0]\n")
    
    for i in range(num_layers):
        f.write("sp=top_1.W"+str(i)+":"+main_mem+"[0]\n")

    # bitstream_name=sys.argv[3].strip("-")
    
    if (len(sys.argv)==5 and sys.argv[4].strip("-")=="Prof_On"):
        f.write("\n\n[profile]\n")
        f.write("data=all:all:all")

    top_format_sim="void top("+"float *In, "+"float *Out, "
    for i in range(num_layers):
        if (i==num_layers-1):
            top_format_sim+=("float *W"+str(i))
        else:
            top_format_sim+=("float *W"+str(i)+', ')
    top_format_sim+=");"
    # top_format_agg=
    # void top(float *A, float *B1,float *B2,float *O)
    f.write('\n\n# Using the example top module declaration & function ports as the following:\n')
    f.write('#'+top_format_sim)
    f.close()
    print("Pre-Execution Specifications:")

    print ("=======================================")
    print ("===========dev.cfg generated===========")
    print ("=======================================")
    mTab = PrettyTable(["Feature", "Configuration"])

    mTab.add_row(["# Parallel Env (inf batch size)", N])
    mTab.add_row(["Model Update batch size", M])
    mTab.add_row(["# Main Memory Ports", num_kernel_arg])
    mTab.add_row(["Total observation size (KB)", required_memory])
    if (flag_plr==1):
        mTab.add_row(["PLRAM", "Enabled"])
    else:
        mTab.add_row(["PLRAM", "Disabled"])
    if (len(sys.argv)==5 and sys.argv[4]=="Prof_On"):
        mTab.add_row(["Profiling", "Enabled"])
    else:
        mTab.add_row(["Profiling", "Disabled"])
  
    print(mTab)

