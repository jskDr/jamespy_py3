"""
jmongo - mongodb related my codes.
Author: Sungjin Kim, 2015-4-26
License: creative common (cc)
"""

from pymongo import MongoClient
import pandas as pd

class jmc:
	def __init__( self, 
				url = 'molspace.rc.fas.harvard.edu',
				id = 'flow_batt', password = 'muchpotential!', source = 'flow_batt'):
		self.client = MongoClient( url)
		self.client.the_database.authenticate( id, password, source = source)
		self.db = self.client[ source]

	def find( self, collection = 'calculation'):
		self.cursor = self.db.calculation.find()

		return self.cursor
		
	def special_toteng_save_to_csv( self, Ndata = None, fname = 'rafa_total_energy.csv'):
		"""
		Extract data from mongodb especially for total_energy in rafa data and
		save it to csv file.
		"""

		x_list, y_list = [], []
		if Ndata:
			for idx, doc in enumerate(self.cursor):
				if 'total_energy' in doc['properties']:
					y = doc['properties']['total_energy']    
					y_list.append( y)

					cur_mol = self.db['molecule'].find( {"_id":doc['molecule']})
					x = cur_mol[0]['meta_data']['smiles']
					x_list.append( x)

				if idx == Ndata - 1: 
					break
		else:
	    	#if no maximum data size is not given. 
			for doc in self.cursor:
			    if 'total_energy' in doc['properties']:
					y = doc['properties']['total_energy']    
					y_list.append( y)

					cur_mol = self.db['molecule'].find( {"_id":doc['molecule']})
					x = cur_mol[0]['meta_data']['smiles']
					x_list.append( x)

		#Data is saved to csv file.
		xy_dic = {'smiles': x_list, 'total_energy': y_list}
		xy_pdw = pd.DataFrame( xy_dic)
		xy_pdw.to_csv( fname, index = False)