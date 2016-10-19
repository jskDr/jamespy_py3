# klstm.py
import numpy

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

def create_dataset_nd(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), :].T
		dataX.append(a)
		dataY.append(dataset[i + look_back, :].T)
	return numpy.array(dataX), numpy.array(dataY)

def create_dataset_a(dataset_a, look_back=1):
	"""
	Input
	-----
	dataset_a, 2d nd.array[ Nsample, Ntime]
	"""
	XT, yT = create_dataset_nd( dataset_a.T, look_back=look_back)
	X = XT.T
	X = X.reshape( -1, 1, X.shape[2])
	y = yT.T
	return X, y
