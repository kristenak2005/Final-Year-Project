#-*- coding: utf-8 -*-
# Author: Alberto Sainz Dalda <asainz.solarphysics@gmail.com>

""" Routines to save and load data a-la-IDL """

import traceback
import joblib
import os.path
from iris_fitting import saveall as sv
from iris_fitting import find 

#def save(filename_jbl, *list_var):
def save(filename_jbl, *list_var, force=False, verbose=True,
         protocol=2):
  

    saving_data = True 

    #if  os.path.exists(filename_jbl) == True: 
    if  os.path.exists(filename_jbl) == True and force == False:
        # print(os.path.exists(filename_jbl))
        overwrite_yn = input('File {} exists. Do you want to overwrite it?  Y/[n] '.format(filename_jbl))
        if overwrite_yn != 'Y': saving_data = False

    if saving_data ==  True:
        (filename,line_number,function_name,text)=traceback.extract_stack()[-2]
        text = text.replace('\n', '')
        begin=text.find('save(')+len('save(')
        end=text.find(')',begin)
        text=[name.strip() for name in text[begin:end].split(',')]
        dict_var = dict(zip(text[1:],list_var))
    
        if verbose == True:
            print('Saving...')
            for keys in dict_var.keys(): print(keys)
        #  mlm.save(filename_jbl, dict_var,  check_file_exists = False)
        joblib.dump(dict_var, filename_jbl, protocol=protocol)
    else:
        print('Nothing has been done.')
 
    return

def load(filename_jbl, verbose=True, list_var = False):


    out = None

    if os.path.exists(filename_jbl) == True: 
        #dict_var = mlm.load(filename_jbl)
        print
        print('Loading joblib file... {}'.format(filename_jbl))
        print
        dict_var = joblib.load(filename_jbl)

        (filename,line_number,function_name,text_ori)=traceback.extract_stack()[-2]
        end=text_ori.find('=')
        text=text_ori[0:end].replace(' ','')
        text=[name.strip() for name in text[0:end].split(',')]

        count = 0
        txt = '' 
        if verbose  == True:
            print()
            print('Suggested commands:')
            if len(text) > 1 and list_var == False: print('{} = {}'.format(text[0], text_ori[end:]))
            for i in dict_var.keys(): print("{} = {}['{}']".format(i, text[0], i))
            print('del {}'.format(text[0]))
            print()
            txt = txt[:-2]
            print()
        if verbose  == True:
            print('The varible types are:')
            for i in dict_var.keys(): print("{} : {}".format(i, type(dict_var[i])))
            print()
      
        out = dict_var
        if list_var == True: 
            out = []
            for i in dict_var.keys(): out.append(dict_var[i])

    else:
        print('')
        print('File {} does not exist. Nothing has been done.'.format(filename_jbl))

    return out  


def list(ext='.jbl.gz', verbose=True):

    files = find.find('./', '*{}'.format(ext))

    for j,i in enumerate(files):
        print(j, i)
   
    aux =  None 
    print()
    sel = input('Select a file {} to be loaded: '.format(ext))
    print()
    sel = int(sel)
    if sel >= 0 and sel <= j:
        aux = sv.load(files[sel], verbose=verbose)

    return aux     

def arguments():
    """Returns tuple containing dictionary of calling function's
       named arguments and a list of calling function's unnamed
       positional arguments.
    """
    from inspect import getargvalues, stack
    posname, kwname, args = getargvalues(stack()[1][0])[-3:]
    posargs = args.pop(posname, [])
    args.update(args.pop(kwname, []))

    return args, posargs

def make_dict(*expr):
    (filename,line_number,function_name,text)=traceback.extract_stack()[-2]
    begin=text.find('make_dict(')+len('make_dict(')
    end=text.find(')',begin)
    text=[name.strip() for name in text[begin:end].split(',')]

    return dict(zip(text,expr))
