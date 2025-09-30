#-*- coding: utf-8 -*-
# Author: Alberto Sainz Dalda <asainz.solarphysics@gmail.com>

""" Routine to find files recursively in a directory """

import os, fnmatch

def find(path, pattern, only_names = 0):

    result = []
    print('Looking for... {}{}'.format(path, pattern))
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
               result.append(os.path.join(root, name))
        for name in dirs:
            if fnmatch.fnmatch(name, pattern):
               result.append(os.path.join(root, name+'/'))

    if only_names != 0:
       for j in range(len(result)):
           result[j] = result[j].replace(path, '')
           if result[j][0] == '/': result[j] = result[j].replace('/', '')

    result.sort()

    return result

