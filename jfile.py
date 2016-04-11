"""
File related library will be saved. 
"""

import pickle


def save_obj(obj, name):
	with open(name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
	with open(name + '.pkl', 'rb') as f:
		return pickle.load(f)

def remove_returns_in_a_phragraph( fname_read, fname_write = None, disp = False):
	"""
	I will read a text file and remove carriage returns in each line unless 
	it is blank line. Then, the modified will be saved back.
	"""
	with open( fname_read, "r") as f:
		lines = f.readlines()

		lines_mod = list()
		line_mod = ""
		for line in lines:
			if line != '\n':
				# If it is a part of a paragraph, it will be concatenated with the previous parts. 
				line_mod += line[:-1] + ' '
			else:
				if len(line_mod) > 0:
					if disp:
						print(line_mod)
						print()
					lines_mod.append(line_mod + '\n')
					lines_mod.append('\n')       
					line_mod = ""
	
	with open( fname_write, "w") as f:
		print("The modified text is saved to", fname_write)
		f.writelines( lines_mod)		