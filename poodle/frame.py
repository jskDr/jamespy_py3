# Python 3
import pandas as pd
import json

class InfoFrame( pd.DataFrame):
	def __init__(self, info_d = {}, **kwargs):
		"""
		DataFrame + Dict
		Parameters
		----------
		The parameters of Pandas DataFrame are used 
		*args : any type
			 all arguments without a keyword
		**kwargs: any type
			 all arguments without a keyword
		"""
		super().__init__(**kwargs)
		self.info_d = info_d

	def to_json_csv(self, json_fname, **kwargs):
		"""
		dict to json with csv file name
		DF to csv
		"""
		d = self.info_d
		d["__csv"] = json_fname + '.csv'  
		super().to_csv( d["__csv"], **kwargs)
		json.dump(d, open( json_fname +'.json','w'))

def read_json_csv( json_fname):
	d = json.load(open( json_fname + '.json'))
	df = pd.read_csv( d["__csv"])
	return InfoFrame( info_d = d, data = df)