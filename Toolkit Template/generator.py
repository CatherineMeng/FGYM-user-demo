
import sys

if __name__ == "__main__":
    if(len(sys.argv)!=5 or 6):
    	print ("Command Arg Error! \
    		required format: python generator.py <device_metadata>.in <algo_hps>.in --bitstream_name --env_name --(optional)top_instance_name")
    	sys.exit()
	
	# top_instance_name: default is "top" if not input by user
	print ("=======================================")
	print ("==========Device information===========")
	print ("=======================================")
	device_input_file=sys.argv[1]
	fr = open(device_input_file, "r")
	device_name=fr.readline() #returns a string to be put in
	ddr_info=readline().split(" ") #returns a list splitted by space
	hbm_info=readline().split(" ") #returns a list splitted by space
	plram_info=readline().split(" ") #returns a list splitted by space
	
	# calculating whether plram can be used
	if (int(plram_info[1])!=0): #=0 means no plram support on the user-provided device
		plram_size=int(plram_info[1]) #in KB
	fr.close()

	algo_input_file=sys.argv[1]
	fr1 = open(algo_input_file, "r")
	# read N,T from <algo_hps>.in
	required_memory=env_dict[env_name+"_obs"]*N
	if (plram_size<required_memory) #....... Put code here
	fr1.close()

	print ("=======================================")
	print ("========Algorithm information==========")
	print ("=======================================")
	f=open('host.py', 'w') 
	# =======Code for generating host Begins here ==========
	f.write('import pyopencl')
	f.write('#define BSIZE 16')
	# .... Fill in the rest
	bitstream_name=sys.argv[3].strip("-")
	env_name=sys.argv[4].strip("-")
	#Use the command line arg --bitstream_name to generate the part of loading bitstream. assume user input <name>.xclbin
	#Use the command line arg --env_name and N from algo_hps.in to generate the part of batched gym.step
	# ===========Code for generating host ends =============
	f.close()

	print ("=======================================")
	print ("===========Model information===========")
	print ("=======================================")
	flag=input("Will the entire DNN weights fit on-chip? T or F")
	if (flag=="T"):
		num_kernel_arg=2 #input and output
	else:
		num_layers=int(input("How many layers in the policy DNN model? (0 means no DNN model)"))
		if (num_layers==0):
			num_kernel_arg=2 #input and output
		else:
			num_kernel_arg=num_layers+2 #one port interface for each weight, ttwo ports for input and output
	# Use the above to genertae setArg parts and pre-execution-spec.out 

	printf("Pre-Execution Specifications:")
	# Look at "4. use tabulate function" in this link: https://www.educba.com/python-print-table/
	# Host Feature					Configuration
	# ------------------------		--------------
	# #parallel env 				16(batch size)
	# #Total observation size 		xxx
	# Main Memory 					PLRAM/DDR...
	# Port width					length of struct(default:batch size)

