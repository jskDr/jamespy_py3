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
		os.chdir( home + '/Dropbox (Personal)/Aspuru-Guzik/python_lab/py3/jamespy_py3/')
	elif platform.system() == "Windows":
		home = expanduser("~")
		os.chdir( home + '\Dropbox (개인 계정)\\Aspuru-Guzik\\python_lab\\py3\\jamespy_py3')

	if platform.system() == "Windows":
		if os.path.isfile('jpyx.*.pyd'):
			os.remove('jpyx.*.pyd')	
	else: # Linux, Darwin
		if os.path.isfile('jpyx.*.so'):
			os.remove('jpyx.*.so')

	#os.system('python3 setup_pyx.py build_ext --inplace')
	os.system('python setup_pyx.py build_ext --inplace')

	if platform.system() == "Windows":
		os.system('move jpyx.*.pyd ../jamespyx_win/j3x/.')
	elif platform.system() == "Linux":
		os.system('mv jpyx.*.so ../jamespyx_linux/j3x/.')
	elif platform.system() == "Darwin":
		os.system('mv jpyx.*.so ../jamespyx_mac/j3x/.')
	else:
		raise ValueError("Not supporting platform which should be one of Windows, Linux, Darwin(OSX)")

	os.chdir(cur_path)
	os.getcwd()

	print("pyx code compilation is completed.")

if __name__ == '__main__':
	run()