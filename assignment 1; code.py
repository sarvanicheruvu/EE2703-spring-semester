import sys
	# to get commandline input
def Reverse(lines): 
    return [ele for ele in reversed(lines)] 		
	# function to reverse list 
try:
	with open(sys.argv[1]) as f:
		lines=f.readlines()			
			# taking input from file 
		start=1;end=0
		for i in range(0,len(lines)):
			if '.circuit' in lines[i]:
				start=i
			if '.end' in lines[i]:
				end=i 			
		lines=Reverse(lines[start:end:1])
				# truncating list to between .circuit and .end & reversing it
		if(start>end):
			print("Incorrect file format. Please check.")

		for i in range(0,len(lines)-1):
			# len(lines)-1 to remove .circuit 

			st=lines[i].replace("\n"," ")
				# avoiding unnecessary carriage return
			if st.find("#")!=-1:
				st=st[0:st.index("#")].strip()
					#removing comments, if any
			list1=st.split(' ')		
			print(' '.join(Reverse(list1)),end =" ")	
				# combining elements of the list into one string
			print("\n")
                                                                                                                       
except IOError:
	print("Please input proper file.")
	exit()

