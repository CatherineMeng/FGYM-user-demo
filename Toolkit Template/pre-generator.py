
import sys

if __name__ == "__main__":
    if(len(sys.argv)!=4 and len(sys.argv)!=5):
    	print(len(sys.argv))
    	print ("Command Arg Error! \
    		required format: python generator.py <device_metadata>.in <algo_cfg>.in env_name --(optional)profile_status")
    	sys.exit()

 #    print ("=======================================")
	# print ("==========Obs Space dictionary=========")
	# print ("=======================================")


	# top_instance_name: default is "top" if not input by user
	print ("=======================================")
	print ("==========Device information===========")
	print ("=======================================")
	device_input_file=sys.argv[1]
	fr = open(device_input_file, "r")
	device_name=fr.readline() #returns a string to be put in
	ddr_info=readline().split(" ") 
	hbm_info=readline().split(" ") 
	plram_info=readline().split(" ") #returns a list splitted by space
	
	# plram can be used?
	if (int(plram_info[1])!=0): #=0 means no plram support on the user-provided device
		plram_size=int(plram_info[1]) #in KB
	fr.close()

	env_name=sys.argv[3]
	algo_input_file=sys.argv[2]
	fr1 = open(algo_input_file, "r")
	N=readline().split(" ")[1]
	T=readline().split(" ")[1]
	M=readline().split(" ")[1]
	# K=readline().split(" ")[1]
	flag_plr=0
	# 4 bytes per number, KB
	obs_spc=env_name.observation_space
	obs_size=0
	if (len(env_name.observation_space)>1): #atari games
		obs_size=obs_spc[0]*obs_spc[1]*obs_spc[2]
	else:
		obs_size=obs_spc[0]

	
	required_memory=obs_size*max(N,M)*4/1024 #in KB
	if (plram_size>required_memory and plram_info[1]!=0):
		flag_plr=1
	 #....... Put code here
	fr1.close()


	print ("=======================================")
	print ("===========Model information===========")
	print ("=======================================")
	num_kernel_arg=0
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
	f.write('platform =',device_name)
	f.write('debug=1')
	f.write("\n[connectivity]")
	f.write("nk=top:1:top_1")

	main_mem="DDR"
	if (int(hbm_info[1])!=0):
		main_mem="HBM"
	f.write("sp=top_1.In:"+main_mem+"[0]")
	f.write("sp=top_1.Out:"+main_mem+"[0]")
	
	for i in range(num_layers):
		f.write("sp=top_1.W"+str(i)+":"+main_mem+"[0]")

	# bitstream_name=sys.argv[3].strip("-")
	
	if (len(sys.argv)==5 and sys.argv[4].strip("-")=="Prof_On"):
		f.write("[profile]")
		f.write("data=all:all:all")
	# ===========Code for generating host ends =============
	f.close()

	printf("Pre-Execution Specifications:")
	# Look at "4. use tabulate function" in this link: https://www.educba.com/python-print-table/
	# Host Feature					Configuration
	# ------------------------		--------------
	# #parallel env 				16(batch size)
	# #Total observation size 		xxx
	# Main Memory 					PLRAM/DDR...
	# Port width					length of struct(default:batch size)
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

	top_format_sim="void top("+"float *In,"+"float *O"
	for i in range(num_layers):
		top_format_sim+=("float *W"+str(i))
	top_format_sim+=");"
	# top_format_agg=
	# void top(float *A, float *B1,float *B2,float *O)