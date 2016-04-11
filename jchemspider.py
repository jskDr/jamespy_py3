#chemspider code
import requests

"""
obabel and babel can be used in the command line
$ obabel ws_no_smiles.inchi -O ws_no_smiles.smi
$ babel ws_no_smiles.inchi -O ws_no_smiles.smi

"""

def post_i2s( inchi_list, flag_confirm = True):
	"""
	post_i2s is more stable than inchi_to_smi
	"""
	tot_err = 0
	smi_list = []
	# text_list = {}
	for ix, ic in enumerate(inchi_list):
		data = {"Content-Type": "application/x-www-form-urlencoded", "inchi": ic}
		r = requests.post( "http://www.chemspider.com/InChI.asmx/InChIToSMILES", data=data)
		# test_list[ ix] = r.content
		smi_list.append( r.content[83:-9])

		if flag_confirm:
			print ix, '>SMILES<', r.content[82:-8],

			#Confirmation of data
			if r.content[82] == '>' and r.content[-9] == '<':
				print 'O.k'    
			else:
				print 'Error!!'
				print r.content	
				tot_err += 1 

	if flag_confirm:
		print "#Total Error is", tot_err

	return smi_list

def inchi_to_smi( inchi_list, flag_confirm = True):
	"""
	[Reference]
	https://www.chemspider.com/InChI.asmx?op=InChIToSMILES
	Since it is web protocol, 
	this should be confirmed for later use --> r.content[83:-9]
	"""
	smi_list = {}
	# text_list = {}
	for ix, ic in enumerate(inchi_list):
		r = requests.get( 'http://www.chemspider.com/InChI.asmx/InChIToSMILES?inchi={} HTTP/1.1'.format( ic))
		# test_list[ ix] = r.content
		smi_list[ ix] = r.content[83:-9]

		if flag_confirm:
			print ix, '>SMILES<', r.content[82:-8],

			#Confirmation of data
			if r.content[82] == '>' and r.content[-9] == '<':
				print 'O.k'    
			else:
				print 'Error!!'
				print r.content	

	return smi_list

def inchi_to_smi_l( inchi_list, flag_confirm = True):
	"""
	[Reference]
	https://www.chemspider.com/InChI.asmx?op=InChIToSMILES
	Since it is web protocol, 
	this should be confirmed for later use --> r.content[83:-9]
	"""
	smi_list = {}
	# text_list = {}
	for ix, ic in enumerate(inchi_list):
		r = requests.get( 'http://www.chemspider.com/InChI.asmx/InChIToSMILES?inchi={} HTTP/1.1'.format( ic))
		# test_list[ ix] = r.content
		smi_list[ ix] = r.content[83:-9]

		if flag_confirm:
			print ix, '>SMILES<', r.content[82:-8],

			#Confirmation of data
			if r.content[82] == '>' and r.content[-9] == '<':
				print 'O.k'    
			else:
				print 'Error!!'
				print r.content	

	return smi_list


def casno_to_inchi( casno_l, flag_confirm = True, engine = 'nist'):
	inch_l = []

	for ii, casno in enumerate(casno_l):
		inchi = casno_to_inchi_each( casno, engine = engine)
		if inchi.startswith('InChI='):
			inch_l.append( inchi)
			print "{} - success".format( ii)		
		else:
			inch_l.append( "")
			print "{} - error".format( ii)

	return inch_l

def _casno_to_inchi_each_r0( casno):
	"""http://webbook.nist.gov/chemistry/"""

	url_cmd = 'http://webbook.nist.gov/cgi/cbook.cgi?ID={}&Units=SI'.format( casno)
	r = requests.post( url_cmd)

	st = r.content.find('<tt>InChI') + 4 #<tt> is added in front of string
	ed = r.content[st:].find('</tt>') + st
	
	return r.content[st:ed]

def casno_to_inchi_each( casno, engine = 'nist'):
	"""
	http://webbook.nist.gov/chemistry/
	http://www.chemnet.com/cas/supplier.cgi?terms=127-07-1&l=en&exact=dict&f=plist&mark=&submit.x=0&submit.y=0
	http://www.chemnet.com/cas/supplier.cgi?terms=541-48-0&l=en&exact=dict&f=plist&mark=&submit.x=0&submit.y=0

	"""

	if engine == "chemnet":
		url_cmd = 'http://www.chemnet.com/cas/supplier.cgi?terms={}&l=en&exact=dict&f=plist&mark=&submit.x=0&submit.y=0'.format( casno)
		r = requests.get( url_cmd)

		st = r.content.find('InChI=') 
		ed = r.content[st:].find('</td>') + st
	else: # engine == "nist" or else
		url_cmd = 'http://webbook.nist.gov/cgi/cbook.cgi?ID={}&Units=SI'.format( casno)
		r = requests.post( url_cmd)

		st = r.content.find('<tt>InChI') + 4 #<tt> is added in front of string
		ed = r.content[st:].find('</tt>') + st
	
	return r.content[st:ed]


def name_to_inchi_each( name):

	url_cmd = 'http://webbook.nist.gov/cgi/cbook.cgi?Name={}&Units=SI'.format( name)
	r = requests.post( url_cmd)

	st = r.content.find('<tt>InChI') + 4 #<tt> is added in front of string
	ed = r.content[st:].find('</tt>') + st
	
	return r.content[st:ed]

def name_to_inchi_each( name, engine = 'nist'):

	if engine == 'nih':
		# http://cactus.nci.nih.gov/chemical/structure/Carbanilide/stdinchi
		url_cmd = 'http://cactus.nci.nih.gov/chemical/structure/{}/stdinchi'.format( name)
		r = requests.post( url_cmd)
		st, ed = 0, None

	else: #engine = 'nist'
		url_cmd = 'http://webbook.nist.gov/cgi/cbook.cgi?Name={}&Units=SI'.format( name)
		r = requests.post( url_cmd)

		st = r.content.find('<tt>InChI') + 4 #<tt> is added in front of string
		ed = r.content[st:].find('</tt>') + st
	
	return r.content[st:ed]

def name_to_inchi( name_l, flag_confirm = True, engine = 'nist'):
	inch_l = []

	for ii, name in enumerate( name_l):
		inchi = name_to_inchi_each( name, engine = engine)
		if inchi.startswith('InChI='):
			inch_l.append( inchi)
			if flag_confirm: print "{} - success".format( ii)		
		else:
			inch_l.append( "")
			if flag_confirm: print "{} - error".format( ii)

	return inch_l

def inchi_to_casno( inchi_l, flag_confirm = True):
	casno_l = []

	for ii, name in enumerate( inchi_l):
		casno = inchi_to_casno_each( name)
		if all ([x.isdigit() for x in casno.split('-')] ):
			casno_l.append( casno)
			if flag_confirm: print "{} - success".format( ii)		
		else:
			casno_l.append( "")
			if flag_confirm: print "{} - error".format( ii)

	return casno_l

def inchi_to_casno_each( inchi):
	# inchi = 'InChI=1S/C14H8O2/c15-13-9-5-1-2-6-10(9)14(16)12-8-4-3-7-11(12)13/h1-8H'
	# http://webbook.nist.gov/cgi/cbook.cgi?InChI=
	url_cmd = 'http://webbook.nist.gov/cgi/cbook.cgi?InChI={}&Units=SI'.format( inchi)
	r = requests.post( url_cmd)

	st_keyword = '<li><strong>CAS Registry Number:</strong>'
	st = r.content.find('<li><strong>CAS Registry Number:</strong>') + len(st_keyword)
	ed = r.content[st:].find('</li>') + st
	
	return r.content[st:ed].strip()

def inchi_to_casno_each_test( inchi):
	# inchi = 'InChI=1S/C14H8O2/c15-13-9-5-1-2-6-10(9)14(16)12-8-4-3-7-11(12)13/h1-8H'
	# http://webbook.nist.gov/cgi/cbook.cgi?InChI=
	url_cmd = 'http://webbook.nist.gov/cgi/cbook.cgi?InChI={}&Units=SI'.format( inchi)
	r = requests.post( url_cmd)

	print r.content

	st_keyword = '<li><strong>CAS Registry Number:</strong>'
	st = r.content.find('<li><strong>CAS Registry Number:</strong>') + len(st_keyword)
	ed = r.content[st:].find('</li>') + st
	
	return r.content[st:ed].strip()

def inchi_to_csid( inchi_l):
    csid_ll = list()
    for ID, inchi in enumerate(inchi_l):
        print ID, inchi, 
        csid_each_l = list()
        for result in cs.search( inchi):
            print( result, result.csid)
            csid_each_l.append( result.csid)
        if len(csid_each_l) == 1:
            csid_ll.append( csid_each_l[0])
        else: 
            csid_ll.append( csid_each_l)
    return csid_ll