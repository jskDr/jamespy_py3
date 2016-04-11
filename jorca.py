"""
The utility functions used in ORCA calculations are collected.
Notice some functions are general so that they can be used for other DFT tools.
"""

def findStrings(afile,substr):
	lines=[]
	with open(afile) as astr:
		for line in astr:
			if substr in line:
				 lines.append(line)
	return lines

def findFinalEnergy(afile):
	lines=[]
	substr='FINAL SINGLE POINT ENERGY'
	with open(afile) as astr:
		for line in astr:
			if substr in line:
				tmpline=line.replace(substr,'').strip()
				lines.append(float(tmpline))
	return lines[-1]

def findComputeTime(afile):
    timestr=''
    substr='TOTAL RUN TIME:'
    milisecs=0
    # find string with time
    with open(afile) as astr:
        for line in astr:
            if substr in line:
                timestr=line.replace(substr,'').strip()
    # there is probably a better way of doing this
    timestr=timestr.split()
    milisecs+=int(timestr[-2]) #milisecs
    milisecs+=1000*int(timestr[-4]) #seconds
    milisecs+=60000*int(timestr[-6]) #minutes
    milisecs+=60*60*1000*int(timestr[-8]) #hours
    milisecs+=24*60*60*1000*int(timestr[-10]) #days

    return milisecs

