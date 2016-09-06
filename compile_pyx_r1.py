"""
this is a file to compile pyx files and copy to the associated directory 
which is jamespyx_linux and jamespyx_mac
"""

import platform
import os
from os.path import expanduser

def run(): 
	cur_path = os.getcwd()

	# This is the platform dependent part.
	if platform.system() == "Linux":
		home = expanduser("~")
		os.chdir( home + '/Dropbox/Aspuru-Guzik/python_lab/py3/jamespy_py3/')
	elif platform.system() == "Darwin":
		home = expanduser("~")
		if os.path.exists( home + '/Dropbox (Personal)'):
			os.chdir( home + '/Dropbox (Personal)/Aspuru-Guzik/python_lab/py3/jamespy_py3/')
		else:
			os.chdir( home + '/Dropbox (개인 계정)/Aspuru-Guzik/python_lab/py3/jamespy_py3/')		
	elif platform.system() == "Windows":
		home = expanduser("~")
		os.chdir( home + '\Dropbox (개인 계정)\\Aspuru-Guzik\\python_lab\\py3\\jamespy_py3')

	cp_s, mv_s, ext_s = "cp", "mv", "so"
	if platform.system() == "Windows":
		cp_s, mv_s, ext_s = "copy", "move", "pyd"
		os_s = "win"		
	elif platform.system() == "Linux":
		os_s = "linux"
	elif platform.system() == "Darwin":
		os_s = "mac"
	else:
		raise ValueError("Not supporting platform which should be one of Windows, Linux, Darwin(OSX)")
	so_file_l = ["jpyx", "jamc_ch_pyx"]

	# Remove current directory files in order to avoid different os files
	# current file in the target os platfrom is copied back to the directory
	import glob
	for so_file in so_file_l:
		fname = '{0}.*.{1}'.format( so_file, ext_s)
		print("Step 1. for avoiding incompatible binaries...")
		print("Previous infold files are searching:", fname)
		for fn in glob.iglob( fname):
			os.remove( fn)
			print("Previous infold files are removed:", fn)
			break # since iglob is iterator
		fname_j3x = '../jamespyx_{}/j3x/'.format(os_s) + fname
		print("Step 2. for reducing unnecsaary compiling time...")
		print("Previous os-specific files are searching:", fname_j3x)
		for fn_j3x in glob.iglob( fname_j3x):
			os.system('{0} {1} .'.format( cp_s, fn_j3x))
			print("Previous os-specific files are copied:", fn_j3x)
			break

	"""
	Before compiling, try to copy to the jamespy_py3 fold
	to check wether it is needed to compile or not. 
	"""

	#os.system('python3 setup_pyx.py build_ext --inplace')
	os.system('python3 setup_pyx.py build_ext --inplace')
	os.system('python3 setup_jamc_ch_pyx.py build_ext --inplace')

	for so_file in so_file_l:
		os.system('{MV} {SF}.*.{Ext} ../jamespyx_{OS}/j3x/.'.format(
			MV=mv_s, SF=so_file, Ext=ext_s, OS=os_s))

	os.chdir(cur_path)
	# os.getcwd()

	print("pyx code compilation is completed.")

if __name__ == '__main__':
	run()