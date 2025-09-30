# -*- coding: utf-8 -*-
# Author: Alberto Sainz Dalda <asainz.solarphysics@gmail.com>

""" Routines to load and visualize IRIS Level 2 fits data """

import matplotlib.pylab as plt
import numpy as np
from astropy.io import fits
from astropy.io.fits import getheader
import matplotlib.gridspec as gridspec
import os
import time
import tempfile
from iris_fitting import saveall as sv




class atdict(dict):

    """ Make attributes from dictionary keys. Taken from the web """

    __getattr__= dict.__getitem__
    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__


def save(input_obj, filename = None, out_dir = '', only_poi = False, 
         verbose = True, force = False):

    """ Saves a RoI, a SoI or an only-PoI object in a  jbl.gz file """

    if input_obj.filename.find('raster') != -1:
        raster = input_obj
        raster2save = {} 
        poi = {}
        object_methods = [method_name for method_name in dir(raster)
                          if callable(getattr(raster, method_name))]
        aux_memmap_filename = []
        aux_memmap_obj = []
        for s, i in enumerate(raster.windows):
            for j, k in enumerate(raster.raster[i].poi):
                raster2save['{}_poi_{}'.format(i, j)] ={}
                for t in k.keys():
                    raster2save['{}_poi_{}'.format(i, j)][t] = k[t]
            aux_memmap_filename.append('None')
            aux_memmap_obj.append(None)
            if raster.raster[i]['temp_memmap_filename'] != 'None': 
                aux_memmap_filename[s] = raster.raster[i]['temp_memmap_filename']
                aux_memmap_obj[s] = raster.raster[i]['temp_memmap_obj']
                raster.raster[i]['temp_memmap_filename']  = 'None'
                raster.raster[i]['temp_memmap_obj']  = None
            raster2save[i] = dict(raster.raster[i])
            del raster2save[i]['poi']
            if only_poi == True: 
                raster2save[i]['data'] = np.zeros((1,1,1))
        raster2save['filename']  = raster.filename
        raster2save['windows'] = raster.windows
        raster2save['all_windows_in_file'] = raster.all_windows_in_file
        file2save_pos_ini = raster.filename.find('iris_l2_')
        file2save_pos_fin = raster.filename.find('.fits')
        file2save  = 'roi_'+raster.filename[file2save_pos_ini:file2save_pos_fin]+'.jbl.gz'
        if only_poi == True:
            file2save = 'poi_roi_'+raster.filename[file2save_pos_ini:file2save_pos_fin]+'.jbl.gz'
        if filename != None: 
            if filename[-7:] != '.jbl.gz': filename = filename+'.jbl.gz'
            file2save = filename
        if out_dir != '':
            if os.path.isdir(out_dir) == True:
                if out_dir[-1] != '/': out_dir+='/'
                file2save = out_dir+file2save
            else:
                file2save = None
        if file2save != None:
            sv.save(file2save, raster2save, verbose=verbose, force=force)
            for s, i in enumerate(raster.windows):
                if aux_memmap_filename[s] != 'None':
                    raster.raster[i]['temp_memmap_filename'] = aux_memmap_filename[s]
                    raster.raster[i]['temp_memmap_obj'] = aux_memmap_obj[s]

        else:
            print('')
            print('############################## WARNING ##############################\n')
            print('Either the output_dir or the filename are not valid. Nothing '
                  'has been saved.')
            print('')

    if input_obj.filename.find('SJI') != -1:
        sji = input_obj
        sji2save = {} 
        poi = {}
        aux_memmap_filename = 'None'
        for i in sji.window:
            for j, k in enumerate(sji.SJI[i].poi):
                sji2save['{}_poi_{}'.format(i, j)] ={}
                for t in k.keys():
                    sji2save['{}_poi_{}'.format(i, j)][t] = k[t]
            if sji.SJI[i]['temp_memmap_filename'] != 'None': 
                aux_memmap_filename = sji.SJI[i]['temp_memmap_filename']
                aux_memmap_obj = sji.SJI[i]['temp_memmap_obj']
                sji.SJI[i]['temp_memmap_filename']  = 'None'
                sji.SJI[i]['temp_memmap_obj'] = None
            sji2save[i] = dict(sji.SJI[i])
            del sji2save[i]['poi']
            if only_poi == True: 
                soi2save[i]['data'] = np.zeros((1,1,1))
        sji2save['filename']  = sji.filename
        sji2save['window'] = sji.window
        file2save_pos_ini = sji.filename.find('iris_l2_')
        file2save_pos_fin = sji.filename.find('.fits')
        file2save = 'soi_'+sji.filename[file2save_pos_ini:file2save_pos_fin]+'.jbl.gz'
        if only_poi == True:
            file2save = 'poi_soi_'+sji.filename[file2save_pos_ini:file2save_pos_fin]+'.jbl.gz'
        if filename != None: 
            if filename[-7:] != '.jbl.gz': filename = filename+'.jbl.gz'
            file2save = filename
        if out_dir != '':
            if os.path.isdir(out_dir) == True:
                if out_dir[-1] != '/': out_dir+='/'
                file2save = out_dir+file2save
            else:
                file2save = None
        if file2save != None:
            sv.save(file2save, sji2save, verbose=verbose, force=force)
            if aux_memmap_filename != 'None':
                sji.SJI[i]['temp_memmap_filename'] = aux_memmap_filename
                sji.SJI[i]['temp_memmap_obj'] = aux_memmap_obj
        else:
            print('')
            print('############################## WARNING ##############################\n')
            print('Either the output_dir or the filename are not valid. Nothing'
                  'has been saved.')
            print('')

    return


def get_ori_val(array, value):


    """ Returns the index for a given value in an array """

    return np.nanargmin(np.abs(array-value))


def ima_scale(array, minmax=None):

    """ Scales an array to a minimum and a maximum """

    min_array = np.nanmin(array)
    max_array = np.nanmax(array)
    if minmax != None:
        min_array = minmax[0]
        max_array = minmax[1]
    out = array.copy()
    out[0,0,...] = min_array
    out[0,1,...] = max_array

    return out.clip(min_array, max_array)


class raster():


    def __init__(self, filename, window_info = ['Mg II k 2796'], roi = False,
                 memmap = True, verbose = True, set_IRIS = True):

        """ Initialize the raster object or RoI from an IRIS Level 2 data file 
        or a pre-existing RoI """

        if roi == False:
            self.filename = filename
            self.raster = get_raster(self.filename, window_info=window_info,
                          show_imaref=False, verbose = False, memmap = memmap)
            self.windows = list(self.raster.keys())
            self.all_windows_in_file = list(show_lines(filename, only_output=True, verbose = verbose))
            for i in self.windows: 
                self.raster[i] = atdict(self.raster[i])
            self.rebuild(set_IRIS = set_IRIS)
        else:
            raster_input = sv.load(filename, verbose=False)
            raster_input = raster_input['raster2save']
            self.filename = raster_input['filename']
            self.windows =raster_input['windows']
            self.all_windows_in_file = raster_input['all_windows_in_file']
            self.raster = {}
            for i in self.windows: 
                self.raster[i] = atdict(raster_input[i])
            self.rebuild(set_IRIS = False)
            for i in self.windows:
                count_poi = 0
                for k in raster_input.keys():
                    if k.find(i) !=-1 and k.find('poi_'):
                        self.raster[i].poi.insert(count_poi, atdict(raster_input[k]))
                        count_poi+=1
                del self.raster[i].poi[-1]
            if memmap == True: 
                self.tomemmap()
                # for i in self.windows:
                #    out_temp_file = make_temp_file(self.raster[i].data)
                #    datatype = self.raster[i].data.dtype
                #    temp_memmap_file = out_temp_file['temp_file_name']
                #    self.raster[i].data = np.memmap(temp_memmap_file, dtype=datatype, mode='r',
                #                                    shape=self.raster[i].data.shape)
                #    self.raster[i].temp_memmap_obj = out_temp_file['temp_file_obj']
        return

    def flush(self):
        for i in self.windows:
            if self.raster[i].temp_memmap_filename != 'None':
                self.raster[i].data = np.array(self.raster[i].data)
                print('Removing temporary file...', self.raster[i].temp_memmap_filename)
                self.raster[i].temp_memmap_obj.close()
                self.raster[i].temp_memmap_filename = 'None'  
                del self.raster[i].temp_memmap_obj 
                self.raster[i].temp_memmap_obj = None  
        return


    def tomemmap(self):
        for i in self.windows:
            out_temp_file = make_temp_file(self.raster[i].data)
            datatype = self.raster[i].data.dtype
            temp_memmap_file = out_temp_file['temp_file_name']
            self.raster[i].temp_memmap_filename = temp_memmap_file 
            self.raster[i].data = np.memmap(temp_memmap_file, dtype=datatype, mode='r',
                                            shape=self.raster[i].data.shape)
            self.raster[i].temp_memmap_obj = out_temp_file['temp_file_obj']
        return

    def default_figure(self):

        for i in self.windows:
            self.raster[i].figsize = [12,8]
            self.raster[i].tight_layout = True
            self.raster[i].bottom = 0.10
            self.raster[i].top = 0.90
            self.raster[i].left = 0.05
            self.raster[i].right = 0.85
            self.raster[i].wspace = 0.10
            self.raster[i].hspace = 0.10
            self.raster[i].inc_XY = 1
            self.raster[i].alt_inc_XY = 5
            self.raster[i].inc_Z = 1
            self.raster[i].alt_inc_Z = 5
            self.raster[i].title_ax2 = 'Slit_pos'
        return 

    def rebuild(self, set_IRIS = True):

        """ Build/rebuild a RoI """

        dim_XY = self.raster[self.windows[0]].data.shape[0:2]
        self.__count_e = 0
        self.__count_d = 0
        self.__x_pos_ori = int(dim_XY[1]/2) 
        self.__y_pos_ori = int(dim_XY[0]/2) 
        self.__count_windows = 0
        for i in self.windows: 
            self.raster[i].__dim_data = self.raster[i].data.shape
            self.raster[i].extent_display = self.raster[i].extent_opt.copy()
            self.raster[i].extent_display_coords = self.raster[i].extent_opt_coords
            self.raster[i].clip_ima = [5, min(np.nanmax(self.raster[i].data.clip(0)),
                                    np.nanmean(self.raster[i].data.clip(0))*15.0)]
            self.raster[i].cmap = 'afmhot'
            self.raster[i].lim_yplot = [0, min(np.nanmax(self.raster[i].data.clip(0)),
                                    np.nanmean(self.raster[i].data.clip(0))*15.0)]
            self.raster[i].delay = 0.01
            self.raster[i].poi = []
            self.raster[i].__count_poi= 0
            self.raster[i].__move_count_poi = 0
            self.raster[i].__count_coords = 0
            self.raster[i].xlim1 = [self.raster[i].extent_display[0],
                                      self.raster[i].extent_display[1]]
            self.raster[i].ylim1 = [self.raster[i].extent_display[2],
                                      self.raster[i].extent_display[3]]
            self.raster[i].xlim2 = [0, self.raster[i].__dim_data[2]]
            self.raster[i].ylim2 = [0, self.raster[i].__dim_data[0]]
            self.raster[i].xlim3 = [self.raster[i].wl[0], self.raster[i].wl[-1]]
            self.raster[i].ylim3 = self.raster[i].lim_yplot
            self.raster[i].arr_y_ax1 = np.linspace(self.raster[i].extent_display[2], self.raster[i].extent_display[3], self.raster[i].__dim_data[0]) 
            self.raster[i].arr_x_ax1 = np.linspace(self.raster[i].extent_display[0], self.raster[i].extent_display[1], self.raster[i].__dim_data[1]) 
            self.raster[i].arr_x_ax3 = self.raster[i].wl
            self.raster[i].x_pos_ext = self.raster[i].arr_x_ax1[self.__x_pos_ori]
            self.raster[i].y_pos_ext = self.raster[i].arr_y_ax1[self.__y_pos_ori]
            self.raster[i].__z_pos_ori = int(self.raster[i].__dim_data[2]/2)
            self.raster[i].z_pos_ext = self.raster[i].arr_x_ax3[self.raster[i].__z_pos_ori]
            self.raster[i].set_IRIS = set_IRIS 
            if self.raster[i].set_IRIS == True: self.set_IRIS_values(i)
        self.__x_pos_ext = self.raster[self.windows[0]].x_pos_ext
        self.__y_pos_ext = self.raster[self.windows[0]].y_pos_ext
        self.__x_pos = self.__x_pos_ext 
        self.__y_pos = self.__y_pos_ext
        self.default_figure()
        return 



    def set_IRIS_values(self, window_label):

        i = window_label
        if 1:
            if i == 'C II 1336': 
                self.raster[i].z_pos_ext = 1334.54
                aux = get_ori_val(self.raster[i].arr_x_ax3, self.raster[i].z_pos_ext)
                self.raster[i].__z_pos_ori = max(0, aux)
                self.raster[i].clip_ima = [0, 60] 
                self.raster[i].lim_yplot = [-5,100]
                self.raster[i].ylim3 = self.raster[i].lim_yplot
                self.raster[i].cmap =  'afmhot'
            if i == 'Fe XII 1349': 
                self.raster[i].z_pos_ext = 1349 
                aux = get_ori_val(self.raster[i].arr_x_ax3, self.raster[i].z_pos_ext)
                self.raster[i].__z_pos_ori = max(0, aux)
                self.raster[i].clip_ima = [0, 15] 
                self.raster[i].lim_yplot = [-5,25]
                self.raster[i].ylim3 = self.raster[i].lim_yplot
                self.raster[i].cmap =  'copper'
            if i == 'O I 1356': 
                self.raster[i].z_pos_ext = 1355.59 
                aux = get_ori_val(self.raster[i].arr_x_ax3, self.raster[i].z_pos_ext)
                self.raster[i].__z_pos_ori = max(0, aux)
                self.raster[i].clip_ima = [0, 15] 
                self.raster[i].lim_yplot = [-5,25]
                self.raster[i].ylim3 = self.raster[i].lim_yplot
                self.raster[i].cmap =  'copper'
            if i == 'Si IV 1394': 
                self.raster[i].z_pos_ext = 1393.77
                aux = get_ori_val(self.raster[i].arr_x_ax3, self.raster[i].z_pos_ext)
                self.raster[i].__z_pos_ori = max(0, aux)
                self.raster[i].clip_ima = [0, 60] 
                self.raster[i].lim_yplot = [-5,100]
                self.raster[i].ylim3 = self.raster[i].lim_yplot
                self.raster[i].cmap =  'hot'
            if i == 'Si IV 1403': 
                self.raster[i].z_pos_ext = 1402.77 
                aux = get_ori_val(self.raster[i].arr_x_ax3, self.raster[i].z_pos_ext)
                self.raster[i].__z_pos_ori = max(0, aux)
                self.raster[i].clip_ima = [0, 60] 
                self.raster[i].lim_yplot = [-5,100]
                self.raster[i].ylim3 = self.raster[i].lim_yplot
                self.raster[i].cmap =  'hot'
            if i == '2832': 
                self.raster[i].z_pos_ext = 2832.40
                aux = get_ori_val(self.raster[i].arr_x_ax3, self.raster[i].z_pos_ext)
                self.raster[i].__z_pos_ori = max(0, aux)
                self.raster[i].clip_ima = [0, 100] 
                self.raster[i].lim_yplot = [-5,500]
                self.raster[i].ylim3 = self.raster[i].lim_yplot
                self.raster[i].cmap =  'gist_heat'
            if i == '2814': 
                self.raster[i].z_pos_ext = 2814.0
                aux = get_ori_val(self.raster[i].arr_x_ax3, self.raster[i].z_pos_ext)
                self.raster[i].__z_pos_ori = max(0, aux)
                self.raster[i].clip_ima = [0, 200] 
                self.raster[i].lim_yplot = [-5,300]
                self.raster[i].ylim3 = self.raster[i].lim_yplot
                self.raster[i].cmap =  'gist_heat'
            if i == 'Mg II k 2796': 
                self.raster[i].z_pos_ext = 2796.35 
                aux = get_ori_val(self.raster[i].arr_x_ax3, self.raster[i].z_pos_ext)
                if aux < 0:
                    self.raster[i].z_pos_ext = 2803.5
                    aux = get_ori_val(self.raster[i].arr_x_ax3, self.raster[i].z_pos_ext)
                self.raster[i].__z_pos_ori = max(0, aux)
                self.raster[i].clip_ima = [0, 800] 
                self.raster[i].lim_yplot = [-5,600]
                self.raster[i].ylim3 = self.raster[i].lim_yplot
                self.raster[i].cmap =  'afmhot'
        return  


    def update_pads(self, verbose = True):
       
        if verbose == True: print('Updating to...', self.raster[self.__window_label].left,
                                     self.raster[self.__window_label].bottom,
                                     self.raster[self.__window_label].right,
                                     self.raster[self.__window_label].top,
                                     self.raster[self.__window_label].wspace,
                                     self.raster[self.__window_label].hspace)
        self.__fig.subplotpars.update(self.raster[self.__window_label].left,
                                     self.raster[self.__window_label].bottom,
                                     self.raster[self.__window_label].right,
                                     self.raster[self.__window_label].top,
                                     self.raster[self.__window_label].wspace,
                                     self.raster[self.__window_label].hspace)
        return


    def get_pads(self, fig_id, verbose = True):

        if verbose == True: print('Getting ...', fig_id.subplotpars.left,
                                      fig_id.subplotpars.bottom,
                                      fig_id.subplotpars.right,
                                      fig_id.subplotpars.top,
                                      fig_id.subplotpars.wspace,
                                      fig_id.subplotpars.hspace,
                                      fig_id.get_size_inches())
        self.raster[self.__window_label].left = fig_id.subplotpars.left
        self.raster[self.__window_label].bottom = fig_id.subplotpars.bottom
        self.raster[self.__window_label].right = fig_id.subplotpars.right
        self.raster[self.__window_label].top = fig_id.subplotpars.top
        self.raster[self.__window_label].wspace = fig_id.subplotpars.wspace
        self.raster[self.__window_label].hspace = fig_id.subplotpars.hspace
        self.raster[self.__window_label].figsize = fig_id.get_size_inches()
        return 


    def build_figure(self, window_label = None, show = False, filename = None):

        if window_label != None:
            self.__window_label = window_label
            self.__x_pos_ext = self.raster[self.__window_label].x_pos_ext
            self.__y_pos_ext = self.raster[self.__window_label].y_pos_ext
            self.raster[self.__window_label].arr_y_ax1 = np.linspace(self.raster[self.__window_label].extent_display[2], self.raster[self.__window_label].extent_display[3], self.raster[self.__window_label].__dim_data[0]) 
            self.raster[self.__window_label].arr_x_ax1 = np.linspace(self.raster[self.__window_label].extent_display[0], self.raster[self.__window_label].extent_display[1], self.raster[self.__window_label].__dim_data[1]) 
            self._x_pos_ori = get_ori_val(self.raster[self.__window_label].arr_x_ax1, self.__x_pos_ext)
            self.__y_pos_ori = get_ori_val(self.raster[self.__window_label].arr_y_ax1, self.__y_pos_ext)
            self.raster[self.__window_label].__z_pos_ori = get_ori_val(self.raster[self.__window_label].arr_x_ax3, self.raster[self.__window_label].z_pos_ext)
            self.__fig = plt.figure('IRIS raster viewer', figsize=self.raster[self.__window_label].figsize)
            aux = self.__fig.get_size_inches()
            gs = gridspec.GridSpec(5, 2)
            self.__ax1 = self.__fig.add_subplot(gs[0:3,0])
            self.__ax2 = self.__fig.add_subplot(gs[0:3,1])
            self.__ax3 = self.__fig.add_subplot(gs[3:,:])
            if self.raster[self.__window_label].tight_layout == False: 
                self.update_pads()
            ima2show = ima_scale(self.raster[self.__window_label].data[:,:,self.raster[self.__window_label].__z_pos_ori], minmax=self.raster[self.__window_label].clip_ima)
            self.__ima1 = self.__ax1.imshow(ima2show, extent=self.raster[self.__window_label].extent_display, 
                                            origin='lower', cmap=self.raster[self.__window_label].cmap)
            self.__ima1.axes.set_xlabel(self.raster[self.__window_label].extent_display_coords[0])
            self.__ima1.axes.set_ylabel(self.raster[self.__window_label].extent_display_coords[1])
            self.__ima1.axes.set_title("XY map at {0:6.2f} AA - "
                                               "{1}@data".format(self.raster[self.__window_label].z_pos_ext,\
                                                                         self.raster[self.__window_label].__z_pos_ori))
            self.__ima1.axes.set_xlim(self.raster[self.__window_label].xlim1)
            self.__ima1.axes.set_ylim(self.raster[self.__window_label].ylim1)
            self.__clb = self.__ima1.axes.inset_axes([1.05, 0, 0.05, 1])
            cbar = plt.colorbar(self.__ima1, cax=self.__clb)
            cbar.set_label('Intensity [DNs]')
            ima2show = ima_scale(self.raster[self.__window_label].data[:,self.__x_pos_ori,:].clip(*self.raster[self.__window_label].clip_ima))
            self.__ima2 = self.__ax2.imshow(ima2show, origin='lower', 
                                            extent=[0, self.raster[self.__window_label].__dim_data[2]-1,
                                                    0, self.raster[self.__window_label].__dim_data[0]-1],
                                            cmap=self.raster[self.__window_label].cmap)
            self.__ima2.axes.set_xlabel('Wavelength [px]')
            self.__ima2.axes.set_ylabel('Y [px]')
            self.__text_blocked2 = ''
            txt_Slit_Spect = 'Slit Spectra acq. {0:4.1f}@map {1}@data{2}'.format(self.__x_pos, self.__x_pos_ori, self.__text_blocked2)
            txt_Acq_DateTime = 'Slit Spectra acq. {}@data at {}{}'.format(self.__x_pos_ori, self.raster[self.__window_label].date_time_acq_ok[self.__x_pos_ori], self.__text_blocked2)
            if self.raster[self.__window_label].title_ax2 == 'Slit_pos': 
                txt2show_ax2 = txt_Slit_Spect
            if self.raster[self.__window_label].title_ax2 == 'Slit_datetime': 
                txt2show_ax2 = txt_Acq_DateTime
            self.__ima2.axes.set_title(txt2show_ax2)
            self.__ima2.axes.set_xlim(self.raster[self.__window_label].xlim2)
            self.__ima2.axes.set_ylim(self.raster[self.__window_label].ylim2)
            self.__clb2 = self.__ima2.axes.inset_axes([1.05, 0, 0.05, 1])
            cbar2 = plt.colorbar(self.__ima2, cax=self.__clb2)
            #cbar2 = plt.colorbar(self.__ima1, cax=self.__clb2)
            cbar2.set_label('Intensity [DNs]')
            self.__line, = self.__ax3.plot(self.raster[self.__window_label].wl,
                                           self.raster[self.__window_label].data[self.__y_pos_ori,self.__x_pos_ori,:],
                                           color = 'C0')
            self.__ax3.set_title('Spectra at point [{0:4.1f},{1:4.1f}]@map = [{2},{3}]@data'.format(self.__x_pos_ext, \
                                  self.__y_pos_ext, self.__y_pos_ori, self.__x_pos_ori))
            self.__ax3.set_ylim(self.raster[self.__window_label].lim_yplot)
            self.__ax3.set_xlim(self.raster[self.__window_label].xlim3) 
            self.__line.axes.set_xlabel('Wavelength [AA]\n {}'.format(self.__window_label))
            self.__line.axes.set_ylabel('Intensity [DNs]')
            self.__vline1 = self.__ax1.axvline(self.__x_pos_ext, ls = ':')
            self.__hline1 = self.__ax1.axhline(self.__y_pos_ext, ls = ':')
            self.__vline2 = self.__ax2.axvline(self.raster[self.__window_label].__z_pos_ori, ls = ':')
            self.__hline2 = self.__ax2.axhline(self.__y_pos_ori, ls = ':')
            self.__vline3 = self.__ax3.axvline(self.raster[self.__window_label].z_pos_ext, ls = ':')
            self.__in_what_ax = 0
            self.__block_ax1 = False
            self.__block_ax2 = False
            self.__block_ax3 = False
            self.__text_blocked1 =  ''
            # self.__text_blocked2 =  '' It is defined some lines above 
            self.__text_blocked3 =  ''
            self.__count_delay =  0 
            self.__animate = False
            self.__kill_animate = False
            self.__animate_in_ax = 0
            self.__delete_window = None
            aux = self.__fig.get_size_inches()
            if self.raster[self.__window_label].tight_layout == True: plt.tight_layout()
            if show == True: 
                plt.show() 
                self.get_pads(self.__fig, verbose = False)
            if filename != None: 
                self.__fig.savefig(filename)
                plt.close('all')
        return



    def quick_look(self, memmap = False, windows = None):

        """ Interactive display of the data in raster[window].data """

        dim_data = self.raster[self.windows[0]].data.shape
        if dim_data[0] <= 1 and dim_data[1] <= 1: # and dim_data[2] == 1:
            print('')
            print('############################## WARNING ##############################\n'
                  'The input data are only-PoI data.\n'
                  'extract_iris2level.raster.quick_look works with IRIS Level 2 data or RoIs data.\n' 
                  'You can use extract_iris2level.show_poi module to visualize the PoIs.\n')
        else:
            self.__selwindows = self.windows
            if windows != None:
                windows = list(windows)
                self.__selwindows = [i for i in windows if i in self.windows]
            if memmap == True:
                for i in self.__selwindows:
                    out_temp_file = make_temp_file(self.raster[i].data)
                    datatype = self.raster[i].data.dtype
                    temp_file = out_temp_file['temp_file_name']
                    self.raster[i].data = np.memmap(temp_file, dtype=datatype, mode='r',
                                                    shape=self.raster[i].__dim_data)
                    self.raster[i].temp_memmap_obj = out_temp_file['temp_file_obj']
                    self.raster[i].temp_memmap_filename = out_temp_file['temp_file_name']
            window_label = self.__selwindows[0]
            if 'Mg II k 2796' in list(self.raster.keys()) and windows == None:
                window_label = 'Mg II k 2796'
                if self.raster[window_label].set_IRIS == True: self.set_IRIS_values(window_label)
            self.__window_label = window_label


            def show_help():

                """ Shows the help menu for the shortcut keys """

                print("""

            ########## Shortcut keys for extract_irisL2data.raster.quick_look ########## 

            - 'Space bar': start/stop the animation of displaying the data.
              The animation activated on the `XY map` runs either in direction perpendicular
              to the slit, or over time for the sit-and-stare data. The animation
              activated over the `SlitSpectra` runs along the slit. The animation
              activated over the `Spectra` runs along the wavelengths. The animation also
              runs on zoomed areas, and can be zoomed-in while the animation is running.
            - '-/+': control the speed of the SJI animation. By pressing '-/+'
              the animation is displayed slower/faster.
            - 'Left/right/up/down arrows': move manually the focus point
              on the X-Y directions in the `XY map`. That can be done even when the
              panels are blocked, which is useful for a detail inspection.
            - 'z/x': move backward/forward in wavelength.
            - 'n/m': move backward/forward between spectral windows. We can inspect the
              same region in the `XY map` by pressing 'N/M'. See
              :numref:`figure_raster_quicklook_multi`.
            - 'v': fit the sharing axes when a panel has been zoomed-in.
            - 'h/H': by pressing 'h' over a panel we can return it to its initial
              dimensions after being zoomed-in. To change all the panels of a spectral windows press 'H'.
            - 'y': chose between different scales for the `XY map`.
            - 'u/i/o/p': these keys are used to control the contrast intensisty of
              the image displayed in the figure. By pressing 'u/i' the lower limit
              of the intesity image is decreased/increased respectively. By pressing
              'o/p', the higher limit of the intensity image is decreased/increased
              respectively. We can also change the upper/lower limits of the Y axis of the
              plot shows in the `Spectral` with these keys.
            - 'a': add a `Point of IRIS` or `PoI` to the 'SoI'. Thus, the user
              can store a particular slit acquisition. Note that a 'PoI' contains
              the data values for a given position [X, Y, wavelength] in the
              spectral window currently displayed. By pressing 'A', a 'PoI' per
              spectral window and the current wavelegenth in each spectral window is
              saved, i.e. the point information for the [X,Y] is saved simultaneously in
              for all the spectral windows. In particular, a 'PoI' stores the images
              (i.e. data) displayed in the `XY map`, the `SlitSpectra`, and
              the `Spectra` panels. It also contains all the information relevant for the
              display.
              plane [Y,X,selected_step], and some relevant information for the display.
            - 'e/r/t': these keys allow us to erase, return or go to a saved
              'PoI'.
            - 'd': delete all data corresponding to a spectral window from the
              'RoI'.
            - '?' or '/': show this menu.
            - ``q``: quit the visualization tool.

                """)
                return

            self.build_figure(window_label = self.__window_label)

            def take_poi():

                """ Take a point an add it to raster[window] """

                poi = atdict(self.raster[self.__window_label].copy())
                """ Take a point an add it to raster[window] """

                poi = atdict(self.raster[self.__window_label].copy())
                del poi.data
                del poi.poi
                poi.filename = self.filename
                poi.temp_memmap_filename = 'None' #self.raster[self.__window_label].temp_memmap_filename
                poi.temp_memmap_obj = None
                poi.x_pos_ori = self.__x_pos_ori
                poi.y_pos_ori = self.__y_pos_ori
                poi.z_pos_ori = self.raster[self.__window_label].__z_pos_ori
                poi.x_pos_ext = self.__x_pos_ext
                poi.y_pos_ext = self.__y_pos_ext
                poi.z_pos_ext = self.raster[self.__window_label].z_pos_ext
                poi.xlim1 = self.raster[self.__window_label].xlim1  # self.__ax1.get_xlim()
                poi.ylim1 = self.raster[self.__window_label].ylim1  # self.__ax1.get_ylim()
                poi.xlim2 = self.raster[self.__window_label].xlim2  # self.__ax2.get_xlim()
                poi.ylim2 = self.raster[self.__window_label].ylim2  # self.__ax2.get_ylim()
                poi.xlim3 = self.raster[self.__window_label].xlim3
                poi.ylim3 = self.raster[self.__window_label].ylim3
                poi.cmap = self.raster[self.__window_label].cmap
                poi.extent_XY_map = self.raster[self.__window_label].extent_display
                poi.extent_coords_XY_map = self.raster[self.__window_label].extent_display_coords
                poi.extent_SlitSpectra = [0, self.raster[self.__window_label].__dim_data[2], 
                                          0, self.raster[self.__window_label].__dim_data[0]]
                poi.zoom_XY_map = [poi.xlim1[0], poi.xlim1[1], poi.ylim1[0], poi.ylim1[1]]
                poi.zoom_SlitSpectra = [poi.xlim2[0], poi.xlim2[1], poi.ylim2[0], poi.ylim2[1]]
                poi.clip_ima = self.raster[self.__window_label].clip_ima.copy()
                poi.lim_yplot= poi.ylim3
                poi.XY_map = self.raster[self.__window_label].data[:,:,self.raster[self.__window_label].__z_pos_ori] 
                poi.XY_map_label_x = poi.extent_coords_XY_map[0]
                poi.XY_map_label_y = poi.extent_coords_XY_map[1]
                poi.XY_map_label_title = "XY map at {0:6.2f} AA - {1}@data".format(poi.z_pos_ext, poi.z_pos_ori)
                poi.SlitSpectra = self.raster[self.__window_label].data[self.__x_pos_ori,:,:] 
                poi.SlitSpectra_label_x = 'Wavelength [px]'
                poi.SlitSpectra_label_y = 'Y [px]'
                txt_Slit_Spect = 'Slit Spectra acq. {0:4.1f}@map {1}@data{2}'.format(self.__x_pos, self.__x_pos_ori, self.__text_blocked2)
                txt_Acq_DateTime = 'Slit Spectra acq. {}@data at {}{}'.format(self.__x_pos_ori, self.raster[self.__window_label].date_time_acq_ok[self.__x_pos_ori], self.__text_blocked2)
                if self.raster[self.__window_label].title_ax2 == 'Slit_pos': 
                    txt2show_ax2 = txt_Slit_Spect
                if self.raster[self.__window_label].title_ax2 == 'Slit_datetime': 
                    txt2show_ax2 = txt_Acq_DateTime
                poi.SlitSpectra_title = txt2show_ax2
                poi.Spectra = self.raster[self.__window_label].data[self.__y_pos_ori, self.__x_pos_ori, :]
                poi.Spectra_label_x ='Wavelength [AA]\n {}'.format(self.__window_label)
                poi.Spectra_label_y ='Intensity [DNs]'
                poi.Spectra_title = 'Spectra at point [Y, X] = [{0:4.1f},{1:4.1f}]@map = [{2},{3}]@data'.format(
                    poi.x_pos_ext,  poi.y_pos_ext, poi.y_pos_ori, poi.x_pos_ori)
                poi.arr_x_ax1 = self.raster[self.__window_label].arr_x_ax1
                poi.arr_y_ax1 = self.raster[self.__window_label].arr_y_ax1
                poi.arr_x_ax3 = self.raster[self.__window_label].arr_x_ax3
                poi.window_label = self.__window_label
                self.raster[self.__window_label].poi.insert(self.raster[self.__window_label].__count_poi, poi)
                return 


            def set_poi(poi):

                """ Set the values of a raster[window].poi as the active one """

                self.__x_pos_ori = poi.x_pos_ori
                self.__y_pos_ori = poi.y_pos_ori
                self.raster[self.__window_label].__z_pos_ori = poi.z_pos_ori
                self.__x_pos_ext = poi.x_pos_ext 
                self.__y_pos_ext = poi.y_pos_ext 
                self.raster[self.__window_label].z_pos_ext = poi.z_pos_ext 
                self.raster[self.__window_label].clip_ima = poi.clip_ima
                self.raster[self.__window_label].cmap = poi.cmap
                self.raster[self.__window_label].lim_yplot = poi.lim_yplot
                self.raster[self.__window_label].xlim1 = poi.xlim1
                self.raster[self.__window_label].ylim1 = poi.ylim1
                self.raster[self.__window_label].xlim2 = poi.xlim2
                self.raster[self.__window_label].ylim2 = poi.ylim2
                self.raster[self.__window_label].xlim3 = poi.xlim3
                self.raster[self.__window_label].ylim3 = poi.lim_yplot # poi.ylim3
                self.raster[self.__window_label].extent_display = poi.extent_XY_map
                self.raster[self.__window_label].extent_display_coords = poi.extent_coords_XY_map
                self.raster[self.__window_label].arr_x_ax1 = poi.arr_x_ax1
                self.raster[self.__window_label].arr_y_ax1 = poi.arr_y_ax1
                self.raster[self.__window_label].arr_x_ax3 = poi.arr_x_ax3
                return 


            def reset(poi = None, home = None):

                """ Set the values of a raster[window].poi as the active one """

                self.raster[self.__window_label].__dim_data = self.raster[self.__window_label].data.shape
                if home == None or home == 1:
                    self.__clb.remove()
                    ima2show = ima_scale(self.raster[self.__window_label].data[:,:,self.raster[self.__window_label].__z_pos_ori], minmax=self.raster[self.__window_label].clip_ima)
                    self.__ima1 = self.__ax1.imshow(ima2show, origin='lower',
                                                    extent=self.raster[self.__window_label].extent_display,
                                                    cmap=self.raster[self.__window_label].cmap)
                    self.__ima1.axes.set_title("XY map at {0:6.2f} AA - "
                                               "{1}@data{2}".format(self.raster[self.__window_label].z_pos_ext,\
                                                                         self.raster[self.__window_label].__z_pos_ori,\
                                                                         self.__text_blocked1))
                    self.__clb = self.__ima1.axes.inset_axes([1.05, 0, 0.05, 1])
                    cbar = plt.colorbar(self.__ima1, cax=self.__clb)
                    cbar.set_label('Intensity [DNs]')
                if home == None or home == 2:
                    self.__clb2.remove()
                    ima2show = ima_scale(self.raster[self.__window_label].data[:,self.__x_pos_ori,:], minmax=self.raster[self.__window_label].clip_ima)
                    self.__ima2 = self.__ax2.imshow(ima2show, origin='lower',
                                                    cmap=self.raster[self.__window_label].cmap)
                    self.__clb2 = self.__ima2.axes.inset_axes([1.05, 0, 0.05, 1])
                    cbar2 = plt.colorbar(self.__ima2, cax=self.__clb2)
                    cbar2.set_label('Intensity [DNs]')
                if home == None or home == 3:
                    self.__line.remove()
                    self.__line, = self.__ax3.plot(self.raster[self.__window_label].wl,
                                    self.raster[self.__window_label].data[self.__y_pos_ori,self.__x_pos_ori,:],
                                    color='C0')
                    self.__ax3.set_ylim(self.raster[self.__window_label].lim_yplot)
                    self.__ax3.set_xlim([self.raster[self.__window_label].wl[0], self.raster[self.__window_label].wl[-1]])
                    txt_Slit_Spect = 'Slit Spectra acq. {0:4.1f}@map {1}@data{2}'.format(self.__x_pos, self.__x_pos_ori, self.__text_blocked2)
                    txt_Acq_DateTime = 'Slit Spectra acq. {}@data at {}{}'.format(self.__x_pos_ori, self.raster[self.__window_label].date_time_acq_ok[self.__x_pos_ori], self.__text_blocked2)
                    if self.raster[self.__window_label].title_ax2 == 'Slit_pos': 
                        txt2show_ax2 = txt_Slit_Spect
                    if self.raster[self.__window_label].title_ax2 == 'Slit_datetime': 
                        txt2show_ax2 = txt_Acq_DateTime
                    self.__ima2.axes.set_title(txt2show_ax2)
                    self.__ax3.set_title('Spectra at point [{0:4.1f},{1:4.1f}]@map = [{2},{3}]@data{4}'.format(self.__x_pos_ext, \
                                         self.__y_pos_ext, self.__y_pos_ori, self.__x_pos_ori, self.__text_blocked3)) 
                    self.__line.axes.set_xlabel('Wavelength [AA]\n {}'.format(self.__window_label))
                self.__hline1.remove()
                self.__hline2.remove()
                self.__vline1.remove()
                self.__vline2.remove()
                self.__vline3.remove()
                self.__ima1.axes.set_xlim(self.raster[self.__window_label].xlim1)
                self.__ima1.axes.set_ylim(self.raster[self.__window_label].ylim1)
                self.__ima1.axes.set_xlabel(self.raster[self.__window_label].extent_display_coords[0])
                self.__ima1.axes.set_ylabel(self.raster[self.__window_label].extent_display_coords[1])
                self.__ima2.axes.set_xlim(self.raster[self.__window_label].xlim2)
                self.__ima2.axes.set_ylim(self.raster[self.__window_label].ylim2)
                self.__ax3.set_xlim(self.raster[self.__window_label].xlim3)
                self.__ax3.set_ylim(self.raster[self.__window_label].ylim3)
                self.__vline1 = self.__ax1.axvline(self.__x_pos_ext, ls = ':')
                self.__vline2 = self.__ax2.axvline(self.raster[self.__window_label].__z_pos_ori, ls = ':')
                self.__vline3 = self.__ax3.axvline(self.raster[self.__window_label].z_pos_ext, ls = ':')
                self.__hline1 = self.__ax1.axhline(self.__y_pos_ext, ls = ':')
                self.__hline2 = self.__ax2.axhline(self.__y_pos_ori, ls = ':')
                plt.draw()
                return


            def motion(event):

                """ What to do when the mouse moves. """

                self.__in_what_ax = 0
                if self.__ax1.in_axes(event) == True: self.__in_what_ax = 1
                if self.__ax2.in_axes(event) == True: self.__in_what_ax = 2
                if self.__ax3.in_axes(event) == True: self.__in_what_ax = 3
                if self.__animate == False:
                    if self.__in_what_ax > 0:
                        self.__x_pos = event.xdata
                        self.__y_pos = event.ydata
                        self.__onkey = False
                        display()
                return


            def block(ax_id):

                """ Block mouse capture in axes ax_id. """

                if ax_id == 1:
                    self.__block_ax1 = not self.__block_ax1
                    if self.__block_ax1 == True: self.__text_blocked1 = ' - BLOCKED'
                    if self.__block_ax1 == False: self.__text_blocked1 = ''
                    self.__ima1.axes.set_title("XY map at {0:6.2f} AA - "
                                               "{1}@data{2}".format(self.raster[self.__window_label].z_pos_ext,\
                                                                         self.raster[self.__window_label].__z_pos_ori,\
                                                                         self.__text_blocked1))
                    plt.draw()
                if ax_id == 2:
                    self.__block_ax2 = not self.__block_ax2
                    if self.__block_ax2 == True: self.__text_blocked2 = ' - BLOCKED'
                    if self.__block_ax2 == False: self.__text_blocked2 = ''
                    txt_Slit_Spect = 'Slit Spectra acq. {0:4.1f}@map {1}@data{2}'.format(self.__x_pos, self.__x_pos_ori, self.__text_blocked2)
                    txt_Acq_DateTime = 'Slit Spectra acq. {}@data at {}{}'.format(self.__x_pos_ori, self.raster[self.__window_label].date_time_acq_ok[self.__x_pos_ori], self.__text_blocked2)
                    if self.raster[self.__window_label].title_ax2 == 'Slit_pos': 
                        txt2show_ax2 = txt_Slit_Spect
                    if self.raster[self.__window_label].title_ax2 == 'Slit_datetime': 
                        txt2show_ax2 = txt_Acq_DateTime
                    self.__ima2.axes.set_title(txt2show_ax2)
                    plt.draw()
                if ax_id == 3:
                    self.__block_ax3 = not self.__block_ax3
                    if self.__block_ax3 == True: self.__text_blocked3 = ' - BLOCKED'
                    if self.__block_ax3 == False: self.__text_blocked3 = ''
                    self.__ax3.set_title('Spectra at point [{0:4.1f},{1:4.1f}]@map = [{2},{3}]@data{4}'.format(self.__x_pos, \
                                         self.__y_pos_ext, self.__y_pos_ori, self.__x_pos_ori, self.__text_blocked3))
                    plt.draw()


            def click(event):

                """ What to do when the mouse clicks. """

                if self.__ax1.in_axes(event) == True: block(1)
                if self.__ax2.in_axes(event) == True: block(2)
                if self.__ax3.in_axes(event) == True: block(3)


            def onkey(event):

                """ What to do when event key is pressed """

                if event.key != None:
                    if event.key == '?' or event.key == '/': show_help()
                    if event.key == 'm' or event.key == 'n' or event.key == 'M' or event.key == 'N':
                        self.raster[self.__window_label].xlim1 = [*self.__ax1.get_xlim()]
                        self.raster[self.__window_label].ylim1 = [*self.__ax1.get_ylim()]
                        if event.key == 'M' or event.key == 'N':
                            current_xlim1 = self.raster[self.__window_label].xlim1
                            current_ylim1 = self.raster[self.__window_label].ylim1
                            for i in self.__selwindows: 
                                self.raster[i].xlim1 = self.raster[self.__window_label].xlim1
                                self.raster[i].ylim1 = self.raster[self.__window_label].ylim1
                                self.raster[i].arr_y_ax1 = self.raster[self.__window_label].arr_y_ax1
                        if event.key == 'm': self.__count_windows+=1
                        if event.key == 'n': self.__count_windows-=1
                        len_windows = len(self.__selwindows)
                        if self.__count_windows >= len_windows: self.__count_windows=0
                        if self.__count_windows < 0: self.__count_windows=len_windows-1
                        self.__window_label = self.__selwindows[self.__count_windows]
                        reset()
                    if event.key == 'y':
                        current_window_label = self.__window_label
                        ylim1 = [*self.__ax1.get_ylim()]
                        xlim1 = [*self.__ax1.get_xlim()]
                        ylim1_ori_0 = get_ori_val(self.raster[self.__window_label].arr_y_ax1, ylim1[0]) # It's also ylim2_0
                        ylim1_ori_1 = get_ori_val(self.raster[self.__window_label].arr_y_ax1, ylim1[1]) # It's also ylim2_1
                        xlim1_ori_0 = get_ori_val(self.raster[self.__window_label].arr_x_ax1, xlim1[0]) 
                        xlim1_ori_1 = get_ori_val(self.raster[self.__window_label].arr_x_ax1, xlim1[1]) 
                        self.raster[self.__window_label].__count_coords+=1
                        if self.raster[self.__window_label].__count_coords >= len(self.raster[self.__window_label].list_extent_coords):
                            self.raster[self.__window_label].__count_coords = 0
                        self.raster[self.__window_label].extent_display = \
                            self.raster[self.__window_label].list_extent[self.raster[self.__window_label].__count_coords]
                        self.raster[self.__window_label].extent_display_coords = \
                            self.raster[self.__window_label].list_extent_coords[self.raster[self.__window_label].__count_coords]
                        self.__ima1.axes.set_xlabel(self.raster[self.__window_label].extent_display_coords[0])
                        self.__ima1.axes.set_ylabel(self.raster[self.__window_label].extent_display_coords[1])
                        self.raster[self.__window_label].arr_y_ax1 = np.linspace(self.raster[self.__window_label].extent_display[2], self.raster[self.__window_label].extent_display[3], self.raster[self.__window_label].__dim_data[0]) 
                        self.raster[self.__window_label].arr_x_ax1 = np.linspace(self.raster[self.__window_label].extent_display[0], self.raster[self.__window_label].extent_display[1], self.raster[self.__window_label].__dim_data[1]) 
                        self.raster[self.__window_label].xlim1 = tuple(self.raster[self.__window_label].arr_x_ax1[[xlim1_ori_0,
                                                                   xlim1_ori_1]])
                        self.raster[self.__window_label].ylim1 = tuple(self.raster[self.__window_label].arr_y_ax1[[ylim1_ori_0,
                                                                   ylim1_ori_1]])
                        self.__x_pos_ext = self.raster[self.__window_label].arr_x_ax1[self.__x_pos_ori]
                        self.__y_pos_ext = self.raster[self.__window_label].arr_y_ax1[self.__y_pos_ori]
                        self.__block_ax1 = True
                        self.__text_blocked1 = ' - BLOCKED'
                        self.__ima1.axes.set_title("XY map at {0:6.2f} AA - "
                                                   "{1}@data{2}".format(self.raster[self.__window_label].z_pos_ext,\
                                                                             self.raster[self.__window_label].__z_pos_ori,\
                                                                             self.__text_blocked1))
                        reset(home=1)

                    if event.key == 'Y':
                        aux = self.raster[self.__window_label].title_ax2 
                        if aux == 'Slit_pos': 
                            self.raster[self.__window_label].title_ax2 = 'Slit_datetime'
                        if aux == 'Slit_datetime': 
                            self.raster[self.__window_label].title_ax2 = 'Slit_pos'
                        txt_Slit_Spect = 'Slit Spectra acq. {0:4.1f}@map {1}@data{2}'.format(self.__x_pos, self.__x_pos_ori, self.__text_blocked2)
                        txt_Acq_DateTime = 'Slit Spectra acq. {}@data at {}{}'.format(self.__x_pos_ori, self.raster[self.__window_label].date_time_acq_ok[self.__x_pos_ori], self.__text_blocked2)
                        if self.raster[self.__window_label].title_ax2 == 'Slit_pos': 
                            txt2show_ax2 = txt_Slit_Spect
                        if self.raster[self.__window_label].title_ax2 == 'Slit_datetime': 
                            txt2show_ax2 = txt_Acq_DateTime
                        self.__ima2.axes.set_title(txt2show_ax2)
                    if event.key == 'b' or event.key == 'B':
                        if event.key == 'b':
                            if self.__in_what_ax == 1: block(1)
                            if self.__in_what_ax == 2: block(2)
                            if self.__in_what_ax == 3: block(3)
                        else:
                            block(1)
                            block(2)
                            block(3)
                    if event.key == 'a' or event.key == 'A':
                        current_window_label = self.__window_label
                        ylim1 = [*self.__ax1.get_ylim()]
                        xlim1 = [*self.__ax1.get_xlim()]
                        ylim1_ori_0 = get_ori_val(self.raster[self.__window_label].arr_y_ax1, ylim1[0])  # It's also ylim2_0
                        ylim1_ori_1 = get_ori_val(self.raster[self.__window_label].arr_y_ax1, ylim1[1])  # It's also ylim2_1
                        xlim1_ori_0 = get_ori_val(self.raster[self.__window_label].arr_x_ax1, xlim1[0]) 
                        xlim1_ori_1 = get_ori_val(self.raster[self.__window_label].arr_x_ax1, xlim1[1]) 
                        if event.key == 'a': 
                            list_windows = [self.__window_label]
                        else:
                            list_windows =  self.__selwindows
                        for i in list_windows:
                            self.raster[i].xlim1 = [*self.__ax1.get_xlim()]
                            self.raster[i].ylim1 = [*self.__ax1.get_ylim()]
                            if i == current_window_label:
                                self.raster[i].xlim1 = [*self.__ax1.get_xlim()]
                                self.raster[i].ylim1 = [*self.__ax1.get_ylim()]
                                self.raster[i].xlim2 = [*self.__ax2.get_xlim()]
                                self.raster[i].ylim2 = [*self.__ax2.get_ylim()]
                                self.raster[i].xlim3 = [*self.__ax3.get_xlim()]
                                self.raster[i].ylim3 = [*self.__ax3.get_ylim()]
                            else:
                                self.raster[i].xlim1 = [*self.raster[i].arr_x_ax1[[xlim1_ori_0,
                                                               xlim1_ori_1]]]
                                self.raster[i].ylim1 = [*self.raster[i].arr_y_ax1[[ylim1_ori_0,
                                                               ylim1_ori_1]]]
                                self.__x_pos_ext = self.raster[self.__window_label].arr_x_ax1[self.__x_pos_ori]
                                self.__y_pos_ext = self.raster[self.__window_label].arr_y_ax1[self.__y_pos_ori]
                            self.__window_label = i
                            take_poi()
                            self.raster[i].__count_poi+=1
                            print('Saving information at point [Y, X] = [{0:4.1f},{1:4.1f}]@map '
                                  '= [{2},{3}]@data_cube at window {4}'.format(self.__x_pos_ext,
                                             self.__y_pos_ext,
                                             self.__y_pos_ori,
                                             self.__x_pos_ori,
                                             i))
                            self.raster[i].__move_count_poi = self.raster[i].__count_poi-1
                        self.__window_label = current_window_label
                    if (event.key == 'r' or event.key == 't' or  event.key == 'e') \
                        and self.__animate ==  False: 
                        if len(self.raster[self.__window_label].poi) > 0:
                            if event.key == 'r': self.raster[self.__window_label].__move_count_poi-=1
                            if event.key == 't': self.raster[self.__window_label].__move_count_poi+=1
                            if  self.raster[self.__window_label].__move_count_poi > len(self.raster[self.__window_label].poi)-1:
                                self.raster[self.__window_label].__move_count_poi =0
                            if  self.raster[self.__window_label].__move_count_poi < 0:
                                self.raster[self.__window_label].__move_count_poi = len(self.raster[self.__window_label].poi)-1
                            set_poi(self.raster[self.__window_label].poi[self.raster[self.__window_label].__move_count_poi])
                            if self.__count_e == 0:
                                print('Showing PoI #{} of a total of {} saved for '
                                     'window {}'.format(self.raster[self.__window_label].__move_count_poi+1,
                                                       len(self.raster[self.__window_label].poi),
                                                       self.__window_label))
                            reset()
                            if event.key == 'e':
                                self.__count_e+=1
                                if self.__count_e == 1:
                                   print("Are you sure you want to remove PoI #{} of "
                                         "a total of {} saved for  window {}? Press 'e' to "
                                         "comfirm.".format(self.raster[self.__window_label].__move_count_poi+1,
                                         len(self.raster[self.__window_label].poi), self.__window_label))
                                if self.__count_e == 2:
                                    del self.raster[self.__window_label].poi[self.raster[self.__window_label].__move_count_poi]
                                    self.raster[self.__window_label].__move_count_poi-=1
                                    if self.raster[self.__window_label].__move_count_poi < 0: 
                                        self.raster[self.__window_label].__move_count_poi = 0 
                                    print("PoI #{} of a total of {} saved for window {} has been "
                                          "succesfully removed.".format(self.raster[self.__window_label].__move_count_poi,
                                                                        len(self.raster[self.__window_label].poi),
                                                                        self.__window_label))
                                    self.__count_e = 0
                        else:
                            print('There is no PoI saved for window {}'.format(self.__window_label))
                    if event.key != 'e': self.__count_e = 0
                    if event.key == 'd':
                        if len(self.__selwindows) == 1:
                            print('This is the only window available in this raster object.')
                        else:
                            self.__count_d+=1
                            if self.__count_d == 1:
                                   print("Are you sure you want to remove all the "
                                         "information about the spectral window "
                                         "{}? Press 'd' to comfirm.".format(self.__window_label))
                            if self.__count_d == 2:
                                self.__delete_window = self.__window_label
                                self.__count_windows+=1
                                if self.__count_windows >= len(self.__selwindows): self.__count_windows=0
                                self.__window_label = self.__selwindows[self.__count_windows]
                                self.__count_d = 0
                                reset()
                    if event.key != 'd': self.__count_d = 0
                    if event.key == 'h' or event.key == 'H':
                        home = None
                        if (event.key == 'h' and self.__in_what_ax == 1) or event.key == 'H':
                            if event.key == 'h': home = 1
                            self.raster[self.__window_label].xlim1 = (self.raster[self.__window_label].extent_display[0],
                                                    self.raster[self.__window_label].extent_display[1])
                            self.raster[self.__window_label].ylim1 = (self.raster[self.__window_label].extent_display[2],
                                                    self.raster[self.__window_label].extent_display[3])
                            self.raster[self.__window_label].arr_y_ax1 = np.linspace(self.raster[self.__window_label].extent_display[2], self.raster[self.__window_label].extent_display[3], self.raster[self.__window_label].__dim_data[0])
                            self.raster[self.__window_label].arr_x_ax1 = np.linspace(self.raster[self.__window_label].extent_display[0], self.raster[self.__window_label].extent_display[1], self.raster[self.__window_label].__dim_data[1])
                        if (event.key == 'h' and self.__in_what_ax == 2) or event.key == 'H':
                            if event.key == 'h': home = 2
                            self.raster[self.__window_label].xlim2 = (0, self.raster[self.__window_label].__dim_data[2])
                            self.raster[self.__window_label].ylim2 = (0, self.raster[self.__window_label].__dim_data[0])
                        if (event.key == 'h' and self.__in_what_ax == 3) or event.key == 'H':
                            if event.key == 'h': home = 3
                            self.raster[self.__window_label].xlim3 = (self.raster[self.__window_label].wl[0], self.raster[self.__window_label].wl[-1])
                            self.raster[self.__window_label].ylim3 = self.raster[self.__window_label].lim_yplot
                            self.raster[self.__window_label].arr_x_ax3 = self.raster[self.__window_label].wl
                        if event.key == 'H': home = None
                        reset(home = home)
                    if event.key == 'v':
                        if (event.key == 'v' and self.__in_what_ax == 1):
                            ylim1 = [*self.__ax1.get_ylim()]
                            xlim1 = [*self.__ax1.get_xlim()]
                            ylim2_0 = get_ori_val(self.raster[self.__window_label].arr_y_ax1, ylim1[0])
                            ylim2_1 = get_ori_val(self.raster[self.__window_label].arr_y_ax1, ylim1[1])
                            self.raster[self.__window_label].xlim1 = xlim1
                            self.raster[self.__window_label].ylim1 = ylim1
                            self.raster[self.__window_label].ylim2 = (ylim2_0, ylim2_1)
                            reset(home = 2)
                        if (event.key == 'v' and self.__in_what_ax == 2):
                            xlim2 = [*self.__ax2.get_xlim()]
                            xlim2_0 = max(int(xlim2[0]),0)
                            xlim2_1 = min(int(xlim2[1]),self.raster[self.__window_label].__dim_data[2]-1)
                            ylim2 = [*self.__ax2.get_ylim()]
                            ylim2_0 = max(int(ylim2[0]),0)
                            ylim2_1 = min(int(ylim2[1]),self.raster[self.__window_label].__dim_data[0]-1)
                            ylim1_0 = self.raster[self.__window_label].arr_y_ax1[ylim2_0]
                            ylim1_1 = self.raster[self.__window_label].arr_y_ax1[ylim2_1]
                            xlim3_0 = self.raster[self.__window_label].arr_x_ax3[xlim2_0]
                            xlim3_1 = self.raster[self.__window_label].arr_x_ax3[xlim2_1]
                            self.raster[self.__window_label].ylim1 = (ylim1_0, ylim1_1)
                            self.raster[self.__window_label].xlim2 = xlim2
                            self.raster[self.__window_label].ylim2 = ylim2
                            self.raster[self.__window_label].xlim3 = (xlim3_0, xlim3_1)
                            reset(home = None)
                        if (event.key == 'v' and self.__in_what_ax == 3) or event.key == 'V':
                            xlim3 = [*self.__ax3.get_xlim()]
                            xlim2_0 = get_ori_val(self.raster[self.__window_label].arr_x_ax3, xlim3[0])
                            xlim2_1 = get_ori_val(self.raster[self.__window_label].arr_x_ax3, xlim3[1])
                            self.raster[self.__window_label].xlim2 = (xlim2_0, xlim2_1)
                            self.raster[self.__window_label].xlim3 = xlim3
                            reset(home = 2)
                    ylim1 = [*self.__ax1.get_ylim()]
                    xlim1 = [*self.__ax1.get_xlim()]
                    ylim1_ori_0 = get_ori_val(self.raster[self.__window_label].arr_y_ax1, ylim1[0])  # It's also ylim2_0
                    ylim1_ori_1 = get_ori_val(self.raster[self.__window_label].arr_y_ax1, ylim1[1]) # It's also ylim2_1
                    xlim1_ori_0 = get_ori_val(self.raster[self.__window_label].arr_x_ax1, xlim1[0]) 
                    xlim1_ori_1 = get_ori_val(self.raster[self.__window_label].arr_x_ax1, xlim1[1]) 
                    xlim3 = [*self.__ax3.get_xlim()]
                    xlim3_ori_0 = get_ori_val(self.raster[self.__window_label].arr_x_ax3, xlim3[0]) 
                    xlim3_ori_1 = get_ori_val(self.raster[self.__window_label].arr_x_ax3, xlim3[1]) 
                    if event.key == 'left': self.__x_pos_ori-=abs(int(self.raster[self.__window_label].inc_XY))
                    if event.key == 'right': self.__x_pos_ori+=abs(int(self.raster[self.__window_label].inc_XY))
                    if event.key == 'alt+left': self.__x_pos_ori-=abs(int(self.raster[self.__window_label].alt_inc_XY))
                    if event.key == 'alt+right': self.__x_pos_ori+=abs(int(self.raster[self.__window_label].alt_inc_XY))
                    if event.key == 'z': self.raster[self.__window_label].__z_pos_ori-=abs(int(self.raster[self.__window_label].inc_Z))
                    if event.key == 'alt+left': self.__x_pos_ori-=abs(int(self.raster[self.__window_label].alt_inc_XY))
                    if event.key == 'alt+right': self.__x_pos_ori+=abs(int(self.raster[self.__window_label].alt_inc_XY))
                    if event.key == 'z': self.raster[self.__window_label].__z_pos_ori-=abs(int(self.raster[self.__window_label].inc_Z))
                    if event.key == 'x': self.raster[self.__window_label].__z_pos_ori+=abs(int(self.raster[self.__window_label].inc_Z))
                    if event.key == 'Z': self.raster[self.__window_label].__z_pos_ori-=abs(int(self.raster[self.__window_label].alt_inc_Z))
                    if event.key == 'X': self.raster[self.__window_label].__z_pos_ori+=abs(int(self.raster[self.__window_label].alt_inc_Z))
                    # if self.__x_pos_ori < 0 : self.__x_pos_ori =  self.raster[self.__window_label].__dim_data[1]-1
                    # if self.raster[self.__window_label].__z_pos_ori < 0 : self.raster[self.__window_label].__z_pos_ori =  self.raster[self.__window_label].__dim_data[2]-1
                    if self.__x_pos_ori > xlim1_ori_1: self.__x_pos_ori = xlim1_ori_0
                    if self.__x_pos_ori < xlim1_ori_0: self.__x_pos_ori = xlim1_ori_1
                    if self.raster[self.__window_label].__z_pos_ori <  xlim3_ori_0: self.raster[self.__window_label].__z_pos_ori =  xlim3_ori_1
                    if self.raster[self.__window_label].__z_pos_ori >  xlim3_ori_1: self.raster[self.__window_label].__z_pos_ori =  xlim3_ori_0
                    self.__x_pos_ext = self.raster[self.__window_label].arr_x_ax1[self.__x_pos_ori]
                    self.raster[self.__window_label].z_pos_ext = self.raster[self.__window_label].arr_x_ax3[self.raster[self.__window_label].__z_pos_ori]
                    if event.key == 'z' or event.key == 'x' or \
                       event.key == 'Z' or event.key == 'X':
                        self.__vline3.remove()
                        self.__vline2.remove()
                        self.__vline3 = self.__ax3.axvline(self.raster[self.__window_label].z_pos_ext, ls=':')
                        self.__vline2 = self.__ax2.axvline(self.raster[self.__window_label].__z_pos_ori, ls = ':')
                        ima2show = ima_scale(self.raster[self.__window_label].data[:,:,self.raster[self.__window_label].__z_pos_ori], minmax=self.raster[self.__window_label].clip_ima)
                        self.__ima1.set_data(ima2show)
                        self.__ima1.axes.set_title("XY map at {0:6.2f} AA - "
                                                   "{1}@data{2}".format(self.raster[self.__window_label].z_pos_ext,\
                                                                             self.raster[self.__window_label].__z_pos_ori,\
                                                                             self.__text_blocked1))
                    if event.key == 'down': self.__y_pos_ori-=abs(int(self.raster[self.__window_label].inc_XY))
                    if event.key == 'up': self.__y_pos_ori+=abs(int(self.raster[self.__window_label].inc_XY))
                    if event.key == 'alt+down': self.__y_pos_ori-=abs(int(self.raster[self.__window_label].alt_inc_XY))
                    if event.key == 'alt+up': self.__y_pos_ori+=abs(int(self.raster[self.__window_label].alt_inc_XY))
                    if self.__y_pos_ori > ylim1_ori_1 : self.__y_pos_ori= ylim1_ori_0
                    if self.__y_pos_ori <  ylim1_ori_0: self.__y_pos_ori =  ylim1_ori_1
                    self.__y_pos_ext = self.raster[self.__window_label].arr_y_ax1[self.__y_pos_ori]
                    if event.key == 'up' or event.key == 'alt+up' or \
                       event.key == 'down' or event.key == 'alt+down':
                        self.__hline1.remove()
                        self.__hline2.remove()
                        self.__hline1 = self.__ax1.axhline(self.__y_pos_ext, ls = ':')
                        self.__hline2 = self.__ax2.axhline(self.__y_pos_ori, ls = ':')
                        self.__line.set_ydata(self.raster[self.__window_label].data[self.__y_pos_ori,
                                                                               self.__x_pos_ori,:])
                    if event.key == 'right' or event.key == 'alt+right' or \
                       event.key == 'left' or event.key == 'alt+left':
                        self.__vline1.remove()
                        self.__vline1 = self.__ax1.axvline(self.__x_pos_ext, ls=':')
                        self.__line.set_ydata(self.raster[self.__window_label].data[self.__y_pos_ori,
                                                                               self.__x_pos_ori,:])
                    if event.key == '+' or event.key == '=':  self.__count_delay-=2
                    if event.key == '-':  self.__count_delay+=2
                    if self.__in_what_ax != 3:
                        if event.key == 'u': self.raster[self.__window_label].clip_ima[0]*=0.9
                        if event.key == 'i': 
                            self.raster[self.__window_label].clip_ima[0]*=1.1
                            if self.raster[self.__window_label].clip_ima[0] >= self.raster[self.__window_label].clip_ima[1]:
                                self.raster[self.__window_label].clip_ima[0] = self.raster[self.__window_label].clip_ima[1]*0.9
                        if event.key == 'o': 
                            self.raster[self.__window_label].clip_ima[1]*=0.9
                            if self.raster[self.__window_label].clip_ima[1] <= self.raster[self.__window_label].clip_ima[0]:
                                self.raster[self.__window_label].clip_ima[1] = self.raster[self.__window_label].clip_ima[0]*1.1
                        if event.key == 'p': self.raster[self.__window_label].clip_ima[1]*=1.1
                    if self.__in_what_ax == 3: #  and self.__animate_in_ax != 3:
                        if event.key == 'u': self.raster[self.__window_label].lim_yplot[0]*=0.9
                        if event.key == 'i': 
                            self.raster[self.__window_label].lim_yplot[0]*=1.1
                            if self.raster[self.__window_label].lim_yplot[0] >= self.raster[self.__window_label].lim_yplot[1]:
                                self.raster[self.__window_label].lim_yplot[0] = self.raster[self.__window_label].lim_yplot[1]*0.9
                        if event.key == 'o': 
                            self.raster[self.__window_label].lim_yplot[1]*=0.9
                            if self.raster[self.__window_label].lim_yplot[1] <= self.raster[self.__window_label].lim_yplot[0]:
                                self.raster[self.__window_label].lim_yplot[1] = self.raster[self.__window_label].lim_yplot[0]*1.1
                        if event.key == 'p': self.raster[self.__window_label].lim_yplot[1]*=1.1
                        self.__ax3.set_ylim(self.raster[self.__window_label].lim_yplot)

                    if self.raster[self.__window_label].clip_ima[0] == 0: self.raster[self.__window_label].clip_ima[0] = 5
                    opt_ima_scale = ['u','i','o','p']
                    if event.key in opt_ima_scale and self.__in_what_ax != 3:
                        self.__ima1.set_clim(self.raster[self.__window_label].clip_ima)
                        self.__ima2.set_clim(self.raster[self.__window_label].clip_ima)
                    ima2show = ima_scale(self.raster[self.__window_label].data[:,self.__x_pos_ori,:], minmax=self.raster[self.__window_label].clip_ima)
                    self.__ima2.set_data(ima2show)
                    ima2show = ima_scale(self.raster[self.__window_label].data[:,:,self.raster[self.__window_label].__z_pos_ori], minmax=self.raster[self.__window_label].clip_ima)
                    self.__ima1.set_data(ima2show)
                    plt.draw()
                    if self.__count_delay <= 0:  self.__count_delay = 1
                    self.__delay = self.raster[self.__window_label].delay*self.__count_delay
                    if event.key == ' ': 
                        self.__animate = not self.__animate
                        if self.__animate == True:
                            if self.__in_what_ax == 1: self.__animate_in_ax = 1
                            if self.__in_what_ax == 2: self.__animate_in_ax = 2
                            if self.__in_what_ax == 3: self.__animate_in_ax = 3
                    if event.key == 'q':
                        if self.__animate == True: 
                             self.__animate = False
                             self.__fig.canvas.stop_event_loop()
                             plt.draw()
                    while self.__animate == True and self.__animate_in_ax == 1:
                            self.__vline1.remove()
                            self.__x_pos_ext = self.raster[self.__window_label].arr_x_ax1[self.__x_pos_ori]
                            ima2show = self.raster[self.__window_label].data[:,self.__x_pos_ori,:]
                            self.__ima2.set_data(ima2show)
                            self.__vline1 = self.__ax1.axvline(self.__x_pos_ext, ls = ':')
                            self.__line.set_ydata(self.raster[self.__window_label].data[self.__y_pos_ori,
                                                                               self.__x_pos_ori,:])
                            txt_Slit_Spect = 'Slit Spectra acq. {0:4.1f}@map {1}@data{2}'.format(self.__x_pos, self.__x_pos_ori, self.__text_blocked2)
                            txt_Acq_DateTime = 'Slit Spectra acq. {}@data at {}{}'.format(self.__x_pos_ori, self.raster[self.__window_label].date_time_acq_ok[self.__x_pos_ori], self.__text_blocked2)
                            if self.raster[self.__window_label].title_ax2 == 'Slit_pos': 
                                txt2show_ax2 = txt_Slit_Spect
                            if self.raster[self.__window_label].title_ax2 == 'Slit_datetime': 
                                txt2show_ax2 = txt_Acq_DateTime
                            self.__ima2.axes.set_title(txt2show_ax2)
                            self.__ax3.set_title('Spectra at point [{0:4.1f},{1:4.1f}]@map = [{2},{3}]@data{4}'.format(self.__x_pos_ext, \
                                                 self.__y_pos_ext, self.__y_pos_ori, self.__x_pos_ori, self.__text_blocked3)) 
                            self.__fig.canvas.start_event_loop(self.__delay)
                            self.__x_pos_ori+=1
                            if self.__x_pos_ori > xlim1_ori_1: self.__x_pos_ori = xlim1_ori_0
                            if self.__x_pos_ori < xlim1_ori_0: self.__x_pos_ori = xlim1_ori_1
                            plt.draw()
                    while self.__animate == True and self.__animate_in_ax == 2:
                            self.__hline1.remove()
                            self.__hline2.remove()
                            self.__y_pos_ext = self.raster[self.__window_label].arr_y_ax1[self.__y_pos_ori]
                            self.__hline1 = self.__ax1.axhline(self.__y_pos_ext, ls = ':')
                            self.__hline2 = self.__ax2.axhline(self.__y_pos_ori, ls = ':')
                            self.__line.set_ydata(self.raster[self.__window_label].data[self.__y_pos_ori, self.__x_pos_ori, :])
                            self.__fig.canvas.start_event_loop(self.__delay)
                            self.__y_pos_ori+=1
                            self.__ax3.set_title('Spectra at point [{0:4.1f},{1:4.1f}]@map = [{2},{3}]@data{4}'.format(self.__x_pos_ext, \
                                                 self.__y_pos_ext, self.__y_pos_ori, self.__x_pos_ori, self.__text_blocked3)) 
                            if self.__y_pos_ori > ylim1_ori_1 : self.__y_pos_ori= ylim1_ori_0
                            if self.__y_pos_ori <  ylim1_ori_0: self.__y_pos_ori =  ylim1_ori_1
                            plt.draw()
                    while self.__animate == True and self.__animate_in_ax == 3:
                            self.__vline2.remove()
                            self.__vline3.remove()
                            ima2show = self.raster[self.__window_label].data[:,:,self.raster[self.__window_label].__z_pos_ori]
                            self.__ima1.set_data(ima2show)
                            self.__vline2 = self.__ax2.axvline(self.raster[self.__window_label].__z_pos_ori, ls = ':')
                            self.__vline3 = self.__ax3.axvline(self.raster[self.__window_label].arr_x_ax3[self.raster[self.__window_label].__z_pos_ori], ls=':')
                            self.__ima1.axes.set_title("XY map at {0:6.2f} AA - "
                                                       "{1}@data{2}".format(self.raster[self.__window_label].z_pos_ext,\
                                                                                 self.raster[self.__window_label].__z_pos_ori,\
                                                                                 self.__text_blocked1))
                            self.raster[self.__window_label].__z_pos_ori+=1
                            if self.raster[self.__window_label].__z_pos_ori <  xlim3_ori_0: self.raster[self.__window_label].__z_pos_ori =  xlim3_ori_1
                            if self.raster[self.__window_label].__z_pos_ori >  xlim3_ori_1: self.raster[self.__window_label].__z_pos_ori =  xlim3_ori_0
                            self.__fig.canvas.start_event_loop(self.__delay)
                            plt.draw()
                    else:
                        if event.key != 'q': 
                            self.__onkey = True
                            display()
                        else:
                            #if memmap == True:
                            #    for i in self.__selwindows:
                            #        self.raster[i].data = np.array(self.raster[i].data)
                            #        print('Removing temporary file...', self.raster[i].__temp_file.name)
                            #        self.raster[i].__temp_file.close()
                            #        del self.raster[i].__temp_file # = self.raster[i].__temp_file.name
                            #plt.tight_layout()
                            plt.draw()
                            cfig= plt.gcf()
                            self.get_pads(self.__fig, verbose = False)
                            for i in self.__selwindows: self.raster[i].figsize = self.raster[self.__window_label].figsize 
                            plt.close('all')
                return 


            def display():

                """ What to display and how to do it """

                if self.__in_what_ax == 1 and (self.__block_ax1 == False or self.__onkey == True): 
                    self.__hline1.remove()
                    self.__vline1.remove()
                    self.__hline2.remove()
                    if self.__onkey  == False: 
                        self.__x_pos_ext = self.__x_pos
                        self.__y_pos_ext = self.__y_pos
                        self.__x_pos_ori = get_ori_val(self.raster[self.__window_label].arr_x_ax1, self.__x_pos)
                        self.__y_pos_ori = get_ori_val(self.raster[self.__window_label].arr_y_ax1, self.__y_pos)
                    self.__hline1 = self.__ax1.axhline(self.__y_pos_ext, ls = ':')
                    self.__vline1 = self.__ax1.axvline(self.__x_pos_ext, ls = ':')
                    self.__hline2 = self.__ax2.axhline(self.__y_pos_ori, ls = ':')
                    ima2show = self.raster[self.__window_label].data[:,self.__x_pos_ori,:]
                    self.__ima2.set_data(ima2show)
                    self.__line.set_ydata(self.raster[self.__window_label].data[self.__y_pos_ori,
                                                                       self.__x_pos_ori,:])
                    txt_Slit_Spect = 'Slit Spectra acq. {0:4.1f}@map {1}@data{2}'.format(self.__x_pos, self.__x_pos_ori, self.__text_blocked2)
                    txt_Acq_DateTime = 'Slit Spectra acq. {}@data at {}{}'.format(self.__x_pos_ori, self.raster[self.__window_label].date_time_acq_ok[self.__x_pos_ori], self.__text_blocked2)
                    if self.raster[self.__window_label].title_ax2 == 'Slit_pos': 
                        txt2show_ax2 = txt_Slit_Spect
                    if self.raster[self.__window_label].title_ax2 == 'Slit_datetime': 
                        txt2show_ax2 = txt_Acq_DateTime
                    self.__ima2.axes.set_title(txt2show_ax2)
                    self.__ax3.set_title('Spectra at point [{0:4.1f},{1:4.1f}]@map = [{2},{3}]@data{4}'.format(self.__x_pos_ext, \
                                         self.__y_pos_ext, self.__y_pos_ori, self.__x_pos_ori, self.__text_blocked3)) 
                if self.__in_what_ax == 2 and (self.__block_ax2 == False or self.__onkey == True): 
                    self.__hline1.remove()
                    self.__vline2.remove()
                    self.__hline2.remove()
                    self.__vline3.remove()
                    if self.__onkey  == False: 
                        self.raster[self.__window_label].__z_pos_ori = int(self.__x_pos)
                        self.raster[self.__window_label].z_pos_ext = self.raster[self.__window_label].arr_x_ax3[int(self.__x_pos)]
                        self.__y_pos = min(int(self.__y_pos), self.raster[self.__window_label].__dim_data[0]-1)
                        self.__y_pos = max(0,self.__y_pos)
                        self.__y_pos_ori = self.__y_pos
                        self.__y_pos_ext = self.raster[self.__window_label].arr_y_ax1[self.__y_pos] 
                    self.__ima1.set_data(self.raster[self.__window_label].data[:,:,self.raster[self.__window_label].__z_pos_ori])
                    self.__ima1.axes.set_title("XY map at {0:6.2f} AA - "
                                               "{1}@data{2}".format(self.raster[self.__window_label].z_pos_ext,\
                                                                         self.raster[self.__window_label].__z_pos_ori,\
                                                                         self.__text_blocked1))
                    self.__ax3.set_title('Spectra at point [{0:4.1f},{1:4.1f}]@map = [{2},{3}]@data{4}'.format(self.__x_pos, \
                                         self.__y_pos_ext, self.__y_pos_ori, self.raster[self.__window_label].__z_pos_ori, self.__text_blocked3)) 
                    min_value = np.nanmin(self.raster[self.__window_label].data[:,:,self.raster[self.__window_label].__z_pos_ori])
                    max_value = np.nanmax(self.raster[self.__window_label].data[:,:,self.raster[self.__window_label].__z_pos_ori])
                    self.__hline1 = self.__ax1.axhline(self.__y_pos_ext, ls = ':')
                    self.__vline2 = self.__ax2.axvline(self.raster[self.__window_label].__z_pos_ori, ls = ':')
                    self.__hline2 = self.__ax2.axhline(self.__y_pos_ori, ls = ':')
                    self.__vline3 = self.__ax3.axvline(self.raster[self.__window_label].wl[self.raster[self.__window_label].__z_pos_ori], ls=':')
                    self.__line.set_ydata(self.raster[self.__window_label].data[self.__y_pos_ori, self.__x_pos_ori, :])
                    txt_Slit_Spect = 'Slit Spectra acq. {0:4.1f}@map {1}@data{2}'.format(self.__x_pos, self.__x_pos_ori, self.__text_blocked2)
                    txt_Acq_DateTime = 'Slit Spectra acq. {}@data at {}{}'.format(self.__x_pos_ori, self.raster[self.__window_label].date_time_acq_ok[self.__x_pos_ori], self.__text_blocked2)
                    if self.raster[self.__window_label].title_ax2 == 'Slit_pos': 
                        txt2show_ax2 = txt_Slit_Spect
                    if self.raster[self.__window_label].title_ax2 == 'Slit_datetime': 
                        txt2show_ax2 = txt_Acq_DateTime
                    self.__ima2.axes.set_title(txt2show_ax2)
                    self.__ax3.set_title('Spectra at point [{0:4.1f},{1:4.1f}]@map = [{2},{3}]@data{4}'.format(self.__x_pos_ext, \
                                         self.__y_pos_ext, self.__y_pos_ori, self.__x_pos_ori, self.__text_blocked3)) 
                if self.__in_what_ax == 3 and (self.__block_ax3 == False or self.__onkey == True):
                    self.__vline2.remove()
                    self.__vline3.remove()
                    if self.__onkey  == False: 
                        self.raster[self.__window_label].z_pos_ext = self.__x_pos
                        self.raster[self.__window_label].__z_pos_ori = get_ori_val(self.raster[self.__window_label].arr_x_ax3, self.raster[self.__window_label].z_pos_ext)
                    self.__vline3 = self.__ax3.axvline(self.raster[self.__window_label].z_pos_ext, ls=':')
                    self.__vline2 = self.__ax2.axvline(self.raster[self.__window_label].__z_pos_ori, ls = ':')
                    ima2show = self.raster[self.__window_label].data[:,:,self.raster[self.__window_label].__z_pos_ori]
                    self.__ima1.set_data(ima2show)
                    self.__ima1.axes.set_title("XY map at {0:6.2f} AA - "
                                               "{1}@data{2}".format(self.raster[self.__window_label].z_pos_ext,\
                                                                         self.raster[self.__window_label].__z_pos_ori,\
                                                                         self.__text_blocked1))
                    self.__ax3.set_title('Spectra at point [{0:4.1f},{1:4.1f}]@map = [{2},{3}]@data{4}'.format(self.__x_pos_ext, \
                                         self.__y_pos_ext, self.__y_pos_ori, self.__x_pos_ori, self.__text_blocked3)) 
                plt.draw()
                if self.__delete_window != None:
                    print('Removing all data from {} window.'.format(self.__delete_window))
                    self.__selwindows.remove(self.__delete_window)
                    del self.raster[self.__delete_window] #.data
                    if self.raster[self.__delete_window].temp_memmap_filename != 'None':
                        print('Removing temporary file...', self.raster[self.__delete_window].temp_memmap_filename)
                        self.raster[i].temp_memmap_obj.close()
            self.__cid_motion = self.__fig.canvas.mpl_connect('motion_notify_event', motion)
            self.__cid_onkey  = self.__fig.canvas.mpl_connect('key_press_event', onkey)
            self.__cid_click = self.__fig.canvas.mpl_connect('button_press_event', click)
            plt.show()
            plt.draw()
            cfig = plt.gcf()
            self.get_pads(self.__fig, verbose = False)
            for i in self.__selwindows: self.raster[i].figsize = self.raster[self.__window_label].figsize 

            return 


def make_temp_file(input_data):

    """ Make a temporary file. It returns the TemporaryFile object and its name
    in the filesystem."""

    dim_data = input_data.shape
    dtype_data = input_data.dtype
    temp_file_obj = tempfile.NamedTemporaryFile(delete = True) 
    filename = temp_file_obj.name
    print('Creating temporary file... ', filename)
    fpath = np.memmap(filename, dtype=dtype_data, mode='w+', shape=(dim_data))
    fpath[:] = input_data[:]
    del fpath

    # return fpath
    return {'temp_file_obj': temp_file_obj, 'temp_file_name':filename}


def remove_all_poi(raster):

    """ Remove all PoI from a RoI """

    for window_label in raster.windows:
        howmany_poi = len(raster.raster[window_label].poi)
        total_howmany_poi = howmany_poi
        while howmany_poi != 0:
            print('Removing PoI # {} of a total of {} in the window '
                  '{}'.format(howmany_poi, total_howmany_poi, window_label))
            del raster.raster[window_label].poi[howmany_poi-1]
            howmany_poi-=1

    return 


def show_poi(poi, show_Spectra = False, show_XY_map = False, show_SlitSpectra = False, 
             show_SJI_slit = True, colorbar1 = True, colorbar2 = False):

    """ Shows panels associated to a PoI """

    if poi.filename.find('raster') != -1:
        if show_XY_map == True:    
            dim_data = poi.XY_map.shape
            w = dim_data[1]
            h = dim_data[0]
            ratio = h/w
            if ratio >= 1:  h_size = max(10, 10*h/w)
            if ratio < 1: h_size = min(10, 10*h/w)
            fig = plt.figure(figsize=(10,h_size))
            ax1 = fig.add_subplot()
            ima1 = ax1.imshow(poi.XY_map.clip(*poi.clip_ima), extent = poi.extent_XY_map,
                              origin='lower', cmap=poi.cmap)
            ima1.axes.set_xlim(poi.xlim1)
            ima1.axes.set_ylim(poi.ylim1)
            ima1.axes.set_xlabel(poi.XY_map_label_x)
            ima1.axes.set_ylabel(poi.XY_map_label_x)
            ima1.axes.set_title(poi.XY_map_label_title)
            ima1.axes.axvline(poi.x_pos_ext, ls=':')
            ima1.axes.axhline(poi.y_pos_ext, ls=':')
            if colorbar1 == True: 
                clb = ima1.axes.inset_axes([1.05, 0, 0.05, 1])
                cbar = plt.colorbar(ima1, cax=clb)
                cbar.set_label('Intensity [DNs]')
            plt.show()
        if show_SlitSpectra== True:
            dim_data = poi.SlitSpectra.shape
            w = dim_data[1]
            h = dim_data[0]
            ratio = h/w
            if ratio >= 1:  h_size = max(10, 10*h/w)
            if ratio < 1: h_size = min(10, 10*h/w)
            fig = plt.figure(figsize=(10,h_size))
            ax1 = fig.add_subplot()
            ima2 = ax1.imshow(poi.SlitSpectra.clip(*poi.clip_ima), extent = poi.extent_SlitSpectra,
                              origin='lower', cmap=poi.cmap)
            ima2.axes.set_xlim(poi.xlim2)
            ima2.axes.set_ylim(poi.ylim2)
            if colorbar2 == True:
                clb2 = ima2.axes.inset_axes([1.05, 0, 0.05, 1])
                cbar2 = plt.colorbar(ima2, cax=clb2)
                cbar2.set_label('Intensity [DNs]')
            ima2.axes.set_xlim(poi.xlim2)
            ima2.axes.set_ylim(poi.ylim2)
            ima2.axes.set_xlabel(poi.SlitSpectra_label_x)
            ima2.axes.set_ylabel(poi.SlitSpectra_label_y)
            ima2.axes.set_title(poi.SlitSpectra_title)
            ima2.axes.axvline(poi.z_pos_ori, ls=':')
            ima2.axes.axhline(poi.y_pos_ori, ls=':')
            plt.show()
        if show_Spectra == True:
            fig = plt.figure(figsize=(10,3))
            plt.plot(poi.wl, poi.Spectra)
            plt.ylim(poi.ylim3)
            plt.xlabel(poi.Spectra_label_x)
            plt.ylabel(poi.Spectra_label_y)
            plt.axvline(poi.z_pos_ext, ls=':')
            plt.title(poi.Spectra_title)
            plt.tight_layout()
            plt.show()

        if show_Spectra ==  False and show_XY_map == False and show_SlitSpectra== False:
            fig = plt.figure(1, figsize=(12,8))
            gs = gridspec.GridSpec(5, 2)
            ax1 = fig.add_subplot(gs[0:3,0])
            ax2 = fig.add_subplot(gs[0:3,1])
            ax3 = fig.add_subplot(gs[3:,:])
            ima1 = ax1.imshow(poi.XY_map.clip(*poi.clip_ima),
                              extent=poi.extent_display, origin='lower',
                              cmap=poi.cmap)
            ima1.axes.set_xlabel(poi.XY_map_label_x)
            ima1.axes.set_ylabel(poi.XY_map_label_y)
            ima1.axes.set_title(poi.XY_map_label_title)
            ima1.axes.set_xlim(poi.xlim1)
            ima1.axes.set_ylim(poi.ylim1)
            if colorbar1 == True:
                clb = ima1.axes.inset_axes([1.05, 0, 0.05, 1])
                cbar = plt.colorbar(ima1, cax=clb)
                cbar.set_label('Intensity [DNs]')
            ima2 = ax2.imshow(poi.SlitSpectra.clip(*poi.clip_ima),
                              origin='lower', extent=poi.extent_SlitSpectra,
                              cmap=poi.cmap)
            ima2.axes.set_xlabel(poi.SlitSpectra_label_x)
            ima2.axes.set_ylabel(poi.SlitSpectra_label_y)
            ima2.axes.set_title(poi.SlitSpectra_title)
            ima2.axes.set_xlim(poi.xlim2)
            ima2.axes.set_ylim(poi.ylim2)
            if colorbar2 == True:
                clb2 = ima2.axes.inset_axes([1.05, 0, 0.05, 1])
                cbar2 = plt.colorbar(ima1, cax=clb2)
                cbar2.set_label('Intensity [DNs]')
            line, = ax3.plot(poi.wl, poi.Spectra, color = 'C0')
            ax3.set_title('Spectra at point [{0:4.1f},{1:4.1f}]@map = '
                          '[{2},{3}]@data'.format(poi.x_pos_ext,
                                                       poi.y_pos_ext,
                                                       poi.x_pos_ori,
                                                       poi.x_pos_ori))
            ax3.set_xlim(poi.xlim3)
            ax3.set_ylim(poi.ylim3)
            line.axes.set_xlabel('Wavelength [AA]\n {}'.format(poi['TDESCT']))
            line.axes.set_ylabel('Intensity [DNs]')
            vline1 = ax1.axvline(poi.x_pos_ext, ls = ':')
            hline1 = ax1.axhline(poi.y_pos_ext, ls = ':')
            vline2 = ax2.axvline(poi.z_pos_ori, ls = ':')
            hline2 = ax2.axhline(poi.y_pos_ori, ls = ':')
            vline3 = ax3.axvline(poi.z_pos_ext, ls = ':')
            plt.tight_layout()
            plt.show()


    if poi.filename.find('SJI') != -1:
            dim_data = poi.XY_map.shape
            w = dim_data[1]
            h = dim_data[0]
            ratio = h/w
            if ratio >= 1:  h_size = max(10, 10*h/w)
            if ratio < 1: h_size = min(10, 10*h/w)
            fig = plt.figure(figsize=(10,h_size))
            ax1 = fig.add_subplot()
            ima1 = ax1.imshow(poi.XY_map.clip(*poi.clip_ima), extent = poi.extent_display,
                              origin='lower', cmap=poi.cmap)
            ima1.axes.set_xlabel(poi.XY_map_label_x)
            ima1.axes.set_ylabel(poi.XY_map_label_y)
            ima1.axes.set_title(poi.XY_map_label_title)
            ima1.axes.set_xlim(poi.xlim)
            ima1.axes.set_ylim(poi.ylim)
            if show_SJI_slit == True: ima1.axes.axvline(poi.slit_pos_X, ls=':',
                                                        linewidth=poi.slit_pos_linewidth)
            if colorbar1 == True:
                clb = ima1.axes.inset_axes([1.05, 0, 0.05, 1])
                cbar = plt.colorbar(ima1, cax=clb)
                cbar.set_label('Intensity [DNs]')
            plt.tight_layout()
            plt.show()

    return


def load(filename, roi_file = False, soi_file = False, verbose = False, memmap =
         False, set_IRIS = True, **kwargs):

    """ Load filename, whcih can a raster, a SJI, a RoI, a SoI, or an only-PoI
    file. """

    out =  None
    file_type = None

    if os.path.isfile(filename) == True:
        aux = os.popen('file {}'.format(filename)).read()
        if aux.find('FITS image data') != -1: file_type = 'fits'
        if aux.find('gzip compressed data') != -1: file_type = 'gzip'

        if file_type == 'fits':
            is_a_sji, is_a_raster = determine_iris_file_type(filename, verbose = verbose)
            if is_a_sji == True: 
                out = SJI(filename, verbose = verbose, memmap = memmap, set_IRIS = set_IRIS)
                if verbose == True:
                    print("The SJI data are passed to the output variable \n"
                          "(e.g iris_sji) as a dictionary with the following entries:\n ")
                    for i in out.SJI.keys(): print(' - iris_sji.SJI[\'{}\']'.format(i))
                    print('\n')
            if is_a_raster == True: 
                out = raster(filename,  verbose = verbose, memmap = memmap,
                             set_IRIS = set_IRIS, **kwargs)
                if verbose == True:
                    print('\n')
                    print("The selected data are passed to the output variable \n"
                          "(e.g iris_raster) as a dictionary with the following entries:\n ")
                    for i in out.raster.keys(): print(' - iris_raster.raster[\'{}\']'.format(i))
                    print('\n')

        if file_type == 'gzip':
            if roi_file  == False and soi_file == False:
                try:
                    if aux.find('roi') != -1:
                        if verbose == True: print('Loading RoI IRIS L2 data file... \n')
                        out = raster(filename,  roi=True, memmap = memmap)
                    if aux.find('soi') != -1:
                        if verbose == True: print('Loading SoI IRIS L2 data file... \n')
                        out = SJI(filename,  soi=True, memmap=memmap)
                except:
                    print("The file introduced has not been properly identified. \n"
                          "Let's give it a chance.\n")
                    out = raster(filename,  **kwargs)
            if roi_file  != False:
                try: 
                    if verbose == True: print('Loading RoI IRIS L2 data file... \n')
                    out = raster(filename,  roi=True)
                except:
                    print('')
                    print('The file passed through roi_file keyword is not a RoI file')
                    print('')
            if soi_file  != False:
                try: 
                    if verbose == True: print('Loading SoI IRIS L2 data file... \n')
                    out = SJI(filename,  soi=True, memmap=memmap)
                except:
                    print('')
                    print('The file passed through soi_file keyword is not a SoI file')
                    print('')
    else:
        print('\n')
        print('WARNING: The file {} does not exist. Nothing has been done'.format(filename))
        print('\n')

    return out

def determine_iris_file_type(filename, verbose = False):

    """ Determines if filename is a raster or a SJI IRIS Level 2 data file."""

    is_a_sji = False
    is_a_raster = False

    if os.path.isfile(filename) == True:
        aux = os.popen('file {}'.format(filename)).read()
        if aux.find('FITS image data') != -1:
            try: 
                hdr = only_header(filename, verbose = False)
                if hdr['INSTRUME'] == 'SJI': is_a_sji = True
                if hdr['INSTRUME'] == 'SPEC': is_a_raster = True
                if is_a_sji  == True: 
                    if verbose == True:
                        print('\n')
                        print('The provided file is a SJI IRIS Level 2 data file containing {} data.'.format(hdr['TDESC1']))
                    lines = show_lines(filename, verbose = verbose)
                    if verbose == True:
                        print('\n')
                if is_a_raster  == True: 
                    if verbose == True:
                        print('\n')
                        print('The provided file is a raster IRIS Level 2 data file.') 
                    lines = show_lines(filename, verbose = verbose)
                if is_a_sji == False and is_a_raster == False:
                    if verbose == True:
                        print('\n')
                        print('WARNING: The provided file ({}) is neither a raster nor a SJI IRIS Level 2 data file'.format(filename))
                        print('\n')
            except:
                    print('\n')
                    print('WARNING: The provided file ({}) is neither a raster nor a SJI file or it is corrupted.'.format(filename))
                    print('\n')
                    pass
    else:
        print('\n')
        print('WARNING: The file {} does not exist. Nothing has been done'.format(filename))
        print('\n')
    return is_a_sji, is_a_raster


def get_raster(filename, window_info = ['Mg II k 2796'], verbose =  False,
        showtime = False, show_imaref=True, showY = False, option_extent = 0,
        no_data = False, memmap = False):
  
    """ Get data and some information of the headers from a raster IRIS Level 2
    data file.  It is an ancient method, originally called 'get_data_l2',  
    but it is useful. """


    hdulist = fits.open(filename)

    hdr = hdulist[0].header
    nwin  = hdr['NWIN']

    extra_info = hdulist[-1].data
    date_time_fuv = []
    date_time_nuv = []
    date_time_fuv_ok = []
    date_time_nuv_ok = []
    for i in extra_info: 
        aux_info = i[5]
        pos_type = aux_info.find('_fuv')
        datetime = aux_info[pos_type-17:pos_type]
        date_time_fuv.append(datetime)
        ok_date = '{}-{}-{} {}:{}:{}'.format(datetime[0:4], datetime[4:6], 
                                             datetime[6:8], datetime[9:11], datetime[11:13], datetime[13:15])
        date_time_fuv_ok.append(ok_date)
        aux_info = i[6][0:66]
        pos_type = aux_info.find('_nuv')
        datetime = aux_info[pos_type-17:pos_type]
        date_time_nuv.append(datetime)
        ok_date = '{}-{}-{} {}:{}:{}'.format(datetime[0:4], datetime[4:6],
                  datetime[6:8], datetime[9:11], datetime[11:13], datetime[13:15])
        date_time_nuv_ok.append(ok_date)


    out = {}

    all_windows = [hdr['TDESC{}'.format(j+1)] for j in range(hdr['nwin'])]

    if verbose == True:
       print()
       print('Available data are stored in windows labeled as:')
       for i in range(nwin): print(i+1, hdr['TDESC{}'.format(i+1)])

    if window_info[0] == 'all': window_info = all_windows

    for label_window_info in window_info:
        label_window_ok = label_window_info.replace(' ', '_')
        for i in range(nwin): 
            if hdr['TDESC{}'.format(i+1)].find(label_window_info) != -1: 
                windx = i+1
                number_ext = i+1


        if verbose == True:
           print()
           print('Reading data for requested window = {0} {1}'.format(windx,hdr['TDESC{}'.format(windx)]))

        data = hdulist[windx].data
        data = np.transpose(data, [1,0,2])
        temp_memmap_filename = 'None' 
        temp_memmap_obj = None
        if memmap == True:
           out_temp_file = make_temp_file(data)
           datatype = data.dtype
           dim_data = data.shape
           temp_memmap_filename = out_temp_file['temp_file_name']
           data = np.memmap(temp_memmap_filename, dtype=datatype, mode='r',
                            shape=dim_data)
           temp_memmap_obj = out_temp_file['temp_file_obj']

        if verbose == True:
           print('Size of the read data from window = {0} {1}: {2}'.format(windx,hdr['TDESC{}'.format(windx)], data.shape))
           print()


        hdr_windx_0 = hdr.cards['*{}'.format(windx)]
        hdr_windx = getheader(filename, windx)

        DATE_OBS = hdr['DATE_OBS']
        DATE_END = hdr['DATE_END']
        TDET   = hdr['TDET{}'.format(windx)]
        TDESCT = hdr['TDESC{}'.format(windx)]
        TWAVE  = hdr['TWAVE{}'.format(windx)]
        TWMIN  = hdr['TWMIN{}'.format(windx)]
        TWMAX  = hdr['TWMAX{}'.format(windx)]
        SPCSCL = hdr_windx['CDELT1']
        SPXSCL = hdr_windx['CDELT3']
        SPYSCL = hdr_windx['CDELT2']
        POS_X  = hdr_windx['CRVAL3']
        POS_Y  = hdr_windx['CRVAL2']
        EXPTIME = hdr['EXPTIME']
        STEPT_AV = hdr['STEPT_AV']

        wl = (np.array(range(hdulist[windx].header['NAXIS1'])) * SPCSCL) + TWMIN

        pos_ref = filename.find('iris_l2_')
        date_in_filename = filename[pos_ref+8:pos_ref+23]
        iris_obs_code    = filename[pos_ref+24:pos_ref+34]
        raster_info      = filename[pos_ref+35:pos_ref+53] 

        data_aux =  hdulist[-2].data

        if 'NUV' in TDET:
           binwl = data_aux[0,7]
           binxy = data_aux[0,8]
        if 'FUV' in TDET:
           binwl =  data_aux[0,5]
           binxy = data_aux[0,6]

        del data_aux

        extent_arcsec_arcsec = [0, data.shape[1]*SPXSCL, 0, data.shape[0]*SPYSCL]
        extent_px_px = [0, data.shape[1]*1., 0, data.shape[0]*1.] 
        extent_px_arcsec  = [0, data.shape[1]*1., 0, data.shape[0]*SPYSCL]
        extent_time_arcsec = [0, data.shape[1]*STEPT_AV, 0, data.shape[0]*SPYSCL]
        extent_time_px     = [0, data.shape[1]*STEPT_AV, 0, data.shape[0]]

        min_extent = np.zeros(4)
        list_extent = [extent_arcsec_arcsec, extent_px_px,
                       extent_px_arcsec,extent_time_arcsec, extent_time_px]
        list_extent_coords = [('X [arcsec]','Y [arcsec]'), ('X [px]','Y [px]'),
                              ('X [px]','Y [arcsec]'),('time [s]','Y [arcsec]'),
                              ('time [s]','Y [px]')]

        min_extent = list(map(lambda a: abs(a[3] - a[1]), list_extent))
        min_extent = np.array(min_extent)

        if extent_arcsec_arcsec[1] == 0: min_extent[0] = 1e9

        extent_opt = list_extent[np.argmin(min_extent)]
        extent_opt_coords = list_extent_coords[np.argmin(min_extent)]

        if show_imaref == True:
           extent = extent_arcsec_arcsec 
           ima_ref = data[:,:,50].clip(min=0, max=np.mean(data[:,:,50]))
           plt.figure()
           plt.subplot(2,1,1)
           if SPXSCL == 0.: 
              extent = extent_px_px 
              if showY == True: extent =  extent_px_arcsec
              if showtime == True:  
                 extent = extent_time_px
                 if showY == True: extent = extent_time_arcsec
           count = 1       
           if verbose == True:
               print('Extent coordinates options (option_extent):')
               for i in list_extent_coords: 
                   print(count, i)
                   count+=1
           extent = extent_opt    
           extent_labels =  extent_opt_coords 
           if option_extent !=0: 
               extent = list_extent [option_extent-1]
               extent_labels = list_extent_coords [option_extent-1]
           plt.imshow(ima_ref, origin='lower', extent=extent)
           plt.xlabel(extent_labels[0])
           plt.ylabel(extent_labels[1])
           plt.subplot(2,1,2)
           plt.plot(wl, data[0,0,:].clip(min=0, max=5000))
           plt.tight_layout()
           plt.show()

        if no_data == True: data = None
        
        if TDET == 'NUV':
            date_time_acq = date_time_nuv
            date_time_acq_ok  = date_time_nuv_ok 

        if 'FUV' in TDET:
            date_time_acq = date_time_fuv
            date_time_acq_ok  = date_time_fuv_ok 

        out_win = {"data":data, "wl":wl, "date_in_filename": date_in_filename, 
               "iris_obs_code": iris_obs_code, "raster_info": raster_info, 
               "DATE_OBS":DATE_OBS, "DATE_END":DATE_END, "TDET":TDET, "TDESCT":TDESCT,
               "TWAVE":TWAVE, "TWMIN":TWMIN, "TWMAX":TWMAX, "SPCSCL":SPCSCL,
               "SPXSCL":SPXSCL, "SPYSCL":SPYSCL, "EXPTIME":EXPTIME, "STEPT_AV":STEPT_AV,
               "POS_X":POS_X, "POS_Y":POS_Y,
               "date_time_acq": date_time_acq, 
               "date_time_acq_ok": date_time_acq_ok, 
               "number_ext": number_ext,
               "binxy": binxy,
               "binwl": binwl, 
               "extent_arcsec_arcsec":extent_arcsec_arcsec,
               "extent_px_px":extent_px_px,
               "extent_px_arcsec":extent_px_arcsec,
               "extent_time_px":extent_time_px,
               "extent_time_arcsec":extent_time_arcsec,
               "extent_opt":extent_opt,
               "extent_opt_coords":extent_opt_coords,
               "list_extent":list_extent, 
               "list_extent_coords":list_extent_coords,
               "temp_memmap_filename":temp_memmap_filename,
               "temp_memmap_obj":temp_memmap_obj}

        out[label_window_info] = out_win
        if verbose == True:
           print('Output is a dictionary with the following keywords:')
           print('(Upper-case keys are directly extracted from the original header)')
           for i in out[label_window_info].keys(): print('  - {}'.format(i))
           print()
        
    hdulist.close()
    return out


def info_fits(filename):

    """ Returns information about the extension of a raster or SJI IRIS Level 2 data file. """

    hdulist = fits.open(filename)
    print()
    hdulist.info()
    hdr  = hdulist[0].header
    print()
    print('Observation description: ', hdr['OBS_DESC'])
    print()
    nwin  = hdr['NWIN']
    for i in range(nwin):
        if 'SJI' in hdr['TDET{}'.format(i+1)]:
            print('Extension No. {0} stores data and header of {1}: '
                  '{2:.2f} - {3:.2f} AA'.format(i+1, hdr['TDESC{}'.format(i+1)], 
                                                      hdr['TWMIN{}'.format(i+1)],
                                                      hdr['TWMAX{}'.format(i+1)]))
        else:
            print('Extension No. {0} stores data and header of {1}:\t '
                  '{2:.2f} - {3:.2f} AA ({4})'.format(i+1, hdr['TDESC{}'.format(i+1)], 
                                                      hdr['TWMIN{}'.format(i+1)],
                                                      hdr['TWMAX{}'.format(i+1)],
                                                      hdr['TDET{}'.format(i+1)][0:3]))

    i = 0
    i_ext = 0
    if 'SJI' in hdr['TDET{}'.format(i+1)]: i_ext = -1
    print()
    print('To get the main header use :')
    print('hdr = extract_iris2level.only_header(filename)')
    print()
    print('To get header corresponding to data of {} use :'.format(hdr['TDESC{}'.format(i+1)]))
    print('hdr = extract_irisL2data.only_header(filename, extension = {})'.format(i_ext+1) ) 
    print()
    print('To get the data of {} use :'.format(hdr['TDESC{}'.format(i+1)]))
    print('data = extract_irisL2data.only_data(filename, extension = {})'.format(i_ext+1) ) 

    return

def get_extinfo(filename, window_info = ['Mg II k 2796']):

    """ Returns a list which elements have the information about
    the label_window_info, its extension number, and its detector
    descriptor """

    hdulist = fits.open(filename)
    print()
    hdr  = hdulist[0].header

    nwin = hdr['NWIN']

    out = []
    for label_window_info in window_info:
        label_window_ok = label_window_info.replace(' ', '_')
        for i in range(nwin):
            if hdr['TDESC{}'.format(i+1)].find(label_window_info) != -1: 
                out.append([label_window_info,  i+1, hdr['TDET{}'.format(i+1)]])
    return out

def only_header(filename, extension = 0, verbose = False):

    """ Returns the main header (index No. 0, default) or auxiliar headers of a 
    raster or SJI IRIS Level 2 data file. """

    hdulist = fits.open(filename)
    if verbose == True:
        info_fits(filename)
        print()
        print('Returning header for extension No. {}'.format(extension))

    return hdulist[extension].header 

def only_data(filename, extension = 1, memmap = False, verbose = False):

    """ Returns the data from an extension of  a raster or SJI IRIS Level 2 data file. """

    hdulist = fits.open(filename)
    if verbose == True:
        info_fits(filename)
        print()
        print('Returning the data stored in extension No. {}'.format(extension))

    data = hdulist[extension].data

    if hdulist[0].header['TDET1'] == 'SJI': 
        if (extension == len(hdulist)-1 or extension == -1):
            data_ok = []
            f = data.formats
            f_ok = []
            for i in f: f_ok.append(i.replace('A', '<U'))
            n = data.names
            n[-1] = 'SJItemp'
            dt = [(j, k) for j, k in zip(n, f_ok)]
            data = np.array(data, dtype=dt)
            data = data.view(np.recarray)
    else:
        if (extension == len(hdulist)-1 or extension == -1):
            data_ok = []
            for i in data:
                data_ok.append((*i[0:-1], i[-1][:66], i[-1][66:121], i[-1][121:]))
            f = data.formats
            f_ok = []
            for i in f: f_ok.append(i.replace('A', '<U'))
            f_ok.append('<U55')
            f_ok.append('<U55')
            n = data.names
            n.append('FUVtemp')
            n.append('NUVtemp')
            dt = [(j, k) for j, k in zip(n, f_ok)]
            data = np.array(data_ok, dtype=dt)
            data = data.view(np.recarray)

    try:
        dim_data = data.shape
        temp_file_obj = 'None'
        temp_file_name = 'None'
        if memmap == True:
           out_temp_file = make_temp_file(data)
           temp_file = out_temp_file['temp_file_name']
           data = np.memmap(temp_file, dtype=data.dtype, mode='r', 
                                 shape=dim_data)
           temp_file_name = out_temp_file['temp_file_obj']
    except:
        print('NoneType/NoneDimension data in extension {}'.format(extension))
     
    return data #, temp_file_obj, temp_file_name


def show_lines(filename, verbose = False, only_output = False):

    """ Shows the spectral windows observed - and its size - stored in a raster or
    SJI IRIS Level 2 data file. It returns a list with string labels of the
    observed spectral windows stored in the file."""

    out = np.array([])

    if os.path.isfile(filename) == True:
        hdulist = fits.open(filename)
        if only_output == False:
            if verbose == True: 
                print()
                print('Extracting information from file {}... '.format(filename))
        hdr = hdulist[0].header
        nwin  = hdr['NWIN']

        if only_output == False:
            if verbose == True: 
                if hdr['INSTRUME'] == 'SJI':
                    print()
                    print('Available data with size Y x X x Image are stored in a windows labeled as:')
                    print()
                    print('-------------------------------------------------------------')
                    print('Index --- Window label --- Y x X x Im --- Spectral range [AA]')
                    print('-------------------------------------------------------------')
                else:
                    print()
                    print('Available data with size Y x X x Wavelength are stored in windows labeled as:')
                    print()
                    print('--------------------------------------------------------------------')
                    print('Index --- Window label --- Y x X x WL --- Spectral range [AA] (band)')
                    print('--------------------------------------------------------------------')
            list_observed_wls = []
            for i in range(nwin):
                window_label =  hdr['TDESC{}'.format(i+1)]
                list_observed_wls.append(window_label)
                if verbose == True: 
                    if hdr['INSTRUME'] == 'SJI':
                        print('  {:>} \t    {:<12}   {:>}x{:>}x{:>} \t  {:>.2f} - {:>.2f}'.format(i, window_label,
                                                               hdulist[i].header['NAXIS2'],
                                                               hdulist[i].header['NAXIS1'],
                                                               hdulist[i].header['NAXIS3'],
                                                               hdulist[0].header['TWMIN'+str(i+1)],
                                                               hdulist[0].header['TWMAX'+str(i+1)]
                                                               ))
                    else:
                        print('  {:>} \t {:<12} \t   {:>}x{:>}x{:>} \t   {:>.2f} - {:>.2f}  ({})'.format(i, window_label,
                                                               hdulist[i+1].header['NAXIS2'],
                                                               hdulist[i+1].header['NAXIS3'],
                                                               hdulist[i+1].header['NAXIS1'],
                                                               hdulist[0].header['TWMIN'+str(i+1)],
                                                               hdulist[0].header['TWMAX'+str(i+1)],
                                                               hdulist[0].header['TDET{}'.format(i+1)][0:3]
                                                               ))
            if verbose == True: 
                if hdr['INSTRUME'] == 'SJI':
                    print('-------------------------------------------------------------')
                else:
                    print('--------------------------------------------------------------------')
            if verbose == True: 
                print()
                print('Observation description: ', hdr['OBS_DESC'])
                print()
        else:
            list_observed_wls = []
            for i in range(nwin):
                window_label =  hdr['TDESC{}'.format(i+1)]
                list_observed_wls.append(window_label)

        out = np.array(list_observed_wls)
    else:
        print()
        print('WARNING: File {} does not exist. Nothing has been done.'.format(filename))
        print()

    return out



class SJI():

    def __init__(self, filename, verbose = False, soi = False, 
                 memmap = True, set_IRIS = True):


        """ Initialize the SJI object """

        if soi == False:
            self.filename = filename
            self.SJI = get_SJI(self.filename,  show_imaref=False, verbose = verbose, memmap = memmap)
            self.window = np.array(list(self.SJI.keys()))
            self.rebuild(set_IRIS = set_IRIS)
        else:
            sji_input = sv.load(filename, verbose=False)
            sji_input = sji_input['sji2save']
            self.filename = sji_input['filename']
            self.window =sji_input['window']
            self.SJI = {}
            for i in self.window: 
                self.SJI[i] = atdict(sji_input[i])
            self.SJI[self.window[0]].poi = []
            for i in self.window:
                count_poi = 0
                for k in sji_input.keys():
                    if k.find(i) !=-1 and k.find('poi_'):
                        self.SJI[i].poi.insert(count_poi, atdict(sji_input[k]))
                        count_poi+=1
                del self.SJI[i].poi[-1]
            if memmap == True:
                self.tomemmap()
        return

    def rebuild(self, set_IRIS = True):
            self.SJI[self.window[0]] = atdict(self.SJI[self.window[0]])
            self.SJI[self.window[0]].clip_ima = [5, min(np.nanmax(self.SJI[self.window[0]].data.clip(0)), 
                                        np.nanmean(self.SJI[self.window[0]].data.clip(0))*15.0)]
            self.SJI[self.window[0]].cmap = 'afmhot'
            self.SJI[self.window[0]].slit_acqnum = 0
            self.SJI[self.window[0]].__clip_ima_ini = self.SJI[self.window[0]].clip_ima.copy()
            self.SJI[self.window[0]].poi =[]
            self.SJI[self.window[0]].__animate = False
            self.SJI[self.window[0]].__delay = 0.001
            self.SJI[self.window[0]].__dim_data = self.SJI[self.window[0]].data.shape
            self.SJI[self.window[0]].__count_e = 0
            self.SJI[self.window[0]].show_slit = False 
            self.SJI[self.window[0]].__show_title = True 
            self.SJI[self.window[0]].__linewidth = 2
            self.SJI[self.window[0]].xlim1 = [0, self.SJI[self.window[0]].__dim_data[1]]
            self.SJI[self.window[0]].ylim1 = [0, self.SJI[self.window[0]].__dim_data[0]]
            self.SJI[self.window[0]].xlim2 = [0, self.SJI[self.window[0]].__dim_data[2]]
            self.SJI[self.window[0]].ylim2 = [0, int(self.SJI[self.window[0]].__dim_data[2]/8)]
            self.SJI[self.window[0]].extent_display =  self.SJI[self.window[0]].extent_opt
            self.SJI[self.window[0]].extent_display_coords = self.SJI[self.window[0]].extent_opt_coords
            self.SJI[self.window[0]].__count_poi = 0
            self.SJI[self.window[0]].__move_count_poi = 0
            self.SJI[self.window[0]].set_IRIS = set_IRIS
            if self.SJI[self.window[0]].set_IRIS  == True: self.set_IRIS_values(self.window[0])
            dim_data = self.SJI[self.window[0]].data.shape
            w = dim_data[1]
            h = dim_data[0]
            ratio = h/w
            if ratio >= 1:  h_size = max(10, 10*h/w)
            if ratio < 1: h_size = min(10, 10*h/w)
            self.SJI[self.window[0]].figsize = [10,h_size]
            self.SJI[self.window[0]].__figsize_ori = self.SJI[self.window[0]].figsize

            return

    def set_IRIS_values(self, window_label):

        if window_label == 'SJI_1330': self.SJI[self.window[0]].cmap = 'copper'
        if window_label == 'SJI_1400': self.SJI[self.window[0]].cmap = 'hot'
        if window_label == 'SJI_2832': self.SJI[self.window[0]].cmap = 'gist_heat'
        if window_label == 'SJI_2796': self.SJI[self.window[0]].cmap = 'afmhot'


    def flush(self):
        for i in self.window:
            if self.SJI[i].temp_memmap_filename != 'None':
                self.SJI[i].data = np.array(self.SJI[i].data)
                print('Removing temporary file...', self.SJI[i].temp_memmap_filename)
                self.SJI[i].temp_memmap_obj.close()
                del self.SJI[i].temp_memmap_obj # = self.SJI[i].__temp_file.name
                self.SJI[i].temp_memmap_filename = 'None'
                self.SJI[i].temp_memmap_obj = None

        return


    def tomemmap(self):
        for i in self.window:
            out_temp_file = make_temp_file(self.SJI[i].data)
            datatype = self.SJI[i].data.dtype
            temp_memmap_file = out_temp_file['temp_file_name']
            self.SJI[i].data = np.memmap(temp_memmap_file, dtype=datatype, mode='r',
                                            shape=self.SJI[i].data.shape)
            self.SJI[i].temp_memmap_obj = out_temp_file['temp_file_obj']
            self.SJI[i].temp_memmap_filename = temp_memmap_file
        return

    def build_figure(self, show = False, filename = None):

            dim_data = self.SJI[self.window[0]].data.shape
            extent = self.SJI[self.window[0]].extent_arcsec_arcsec
            extent_opt = self.SJI[self.window[0]].extent_opt
            extent_labels =  self.SJI[self.window[0]].extent_opt_coords
            self.__fig=plt.figure('IRIS slit viewer', figsize = self.SJI[self.window[0]].figsize)
            gs = gridspec.GridSpec(5, 1)
            self.__ax = self.__fig.add_subplot(gs[0:4,0])
            self.__ax2 = self.__fig.add_subplot(gs[4:,0])
            ima2show = ima_scale(self.SJI[self.window[0]].data[:,:,self.SJI[self.window[0]].slit_acqnum], minmax=self.SJI[self.window[0]].clip_ima)
            self.__im = self.__ax.imshow(ima2show, origin='lower',
                                         extent=self.SJI[self.window[0]].extent_display,
                                         cmap=self.SJI[self.window[0]].cmap)
            self.__im.axes.set_xlim(self.SJI[self.window[0]].xlim1)
            self.__im.axes.set_ylim(self.SJI[self.window[0]].ylim1)
            self.__im.set_clim(self.SJI[self.window[0]].clip_ima)
            self.__im.axes.set_xlabel(extent_labels[0])
            self.__im.axes.set_ylabel(extent_labels[1])
            if self.SJI[self.window[0]].show_slit == True:
                self.__vline = self.__ax.axvline(self.SJI[self.window[0]].SLTPX1IX[self.SJI[self.window[0]].slit_acqnum],
                               ls=":", linewidth=self.SJI[self.window[0]].__linewidth)
            self.__clb = self.__ax.inset_axes([1.05, 0, 0.05, 1])
            cbar = plt.colorbar(self.__im, cax=self.__clb)
            cbar.set_label('Intensity [DNs]')
            self.arr_x_ax = np.linspace(0, dim_data[2],
                                        int(self.SJI[self.window[0]].xlim1[1] - 
                                            self.SJI[self.window[0]].xlim1[0]))
            self.__bar = np.zeros((max(1,int(dim_data[2]/8)), dim_data[2]))
            self.__bar[:,self.SJI[self.window[0]].slit_acqnum] = 1
            self.__im2 = self.__ax2.imshow(self.__bar) 
            self.__ax2.set_xlim(self.SJI[self.window[0]].xlim2) 
            self.__ax2.yaxis.set_ticks([],[])
            self.__ax2.set_xlabel('Image No.')
            self.__title = self.__ax.set_title('Image {} of {}  - {} - {}'.format(self.SJI[self.window[0]].slit_acqnum, dim_data[2]-1,
                                                                       self.window[0],
                                                                       self.SJI[self.window[0]].date_time_acq_ok[self.SJI[self.window[0]].slit_acqnum]))
            plt.tight_layout()
            if show == True:
                plt.show() 
                self.SJI[self.window[0]].figsize = self.__fig.get_size_inches()
            if filename != None:
                self.__fig.savefig(filename)
                plt.close('all')


            return

    def quick_look(self, memmap = False, loop = None, blit = True):

        """ Interactive display of the data in SJI[window].data """

        dim_data = self.SJI[self.window[0]].data.shape
        if dim_data[0] <= 1 and dim_data[1] <= 1: # and dim_data[2] == 1:
            print('')
            print('############################## WARNING ##############################\n'
                  'The input data are only-PoI data.\n'
                  'extract_iris2level.raster.quick_look works with IRIS Level 2 data or RoIs data.\n' 
                  'You can use extract_iris2level.show_poi module to visualize the PoIs.\n')
        else:

            if memmap == True:
                out_temp_file = make_temp_file(self.SJI[self.window[0]].data)
                datatype = self.SJI[self.window[0]].data.dtype
                temp_file = out_temp_file['temp_file_name']
                self.SJI[self.window[0]].data = np.memmap(temp_file, dtype=datatype, mode='r', 
                                                          shape=self.SJI[self.window[0]].__dim_data)
                self.SJI[self.window[0]].temp_memmap_obj = out_temp_file['temp_file_obj']
                self.SJI[self.window[0]].temp_memmap_filename = temp_file
            self.__stop_loop = False
            no_times = 0

            self.build_figure()

            plt.draw()
            self.SJI[self.window[0]].slit_acqnum_delay =  0

            def show_help():

                """ Shows the help menu gort the shortcut keys """

                print("""

            ########## Shortcut keys for extract_irisL2data.SJI.quick_look ########## 

            - 'Space bar': start/stop the animation of displaying the data
              corresponding to the steps in the 3rd dimension.
            - '-/+': controls the speed of the SJI animation. By pressing '-/+'
              the animation is displayed slower/faster.
            - 'Left/right arrow': allows to move backward/forward step-by-step
              along the 3rd dimension.
            - 'v/V': show/hide a dashed vertical line over the slit.
              by pressing 'V', the line overplotted is thicker.
            - 'H': returns to the initial display parameters. This affects to
              the image displayed (e.g. after being zoomed-in) and to the contrast
              intensity levels of the image.
            - 'u/i/o/p': these keys are used to control the contrast intensisty of
              the image displayed in the figure. By pressing 'u/i' the lower limit
              of the intesity image is decreased/increased respectively. By pressing
              'o/p', the higher limit of the intensity image is decreased/increased
              respectively.
            - 'a': add a `Point of IRIS` or `PoI` to the 'SoI'. Thus, the user
              can store a particular slit acquisition. Note that a 'PoI' contains
              the data values for a given step, i.e. the 2D data corresponding to the
              plane [Y,X,selected_step], and some relevant information for the display.
            - 'e/r/t': these keys allow us to erase, return or go to a saved
              'PoI'.
            - '?' or '/': show this menu
            - 'q': quit the visualization tool.

                    """)
                return 

            def take_poi():

                """ Take a point an add it to raster[window] """

                poi = atdict(self.SJI[self.window[0]].copy())
                del poi.data
                del poi.poi
                poi.temp_memmap_file = 'None' #self.SJI[self.window[0]].temp_memmap_filename
                poi.temp_memmap_obj = None
                poi.filename = self.filename
                poi.slit_acqnum = self.SJI[self.window[0]].slit_acqnum
                poi.slit_pos_X = self.SJI[self.window[0]].SLTPX1IX[self.SJI[self.window[0]].slit_acqnum]
                poi.show_slit = self.SJI[self.window[0]].show_slit
                poi.slit_pos_linewidth = self.SJI[self.window[0]].__linewidth
                poi.XY_map = self.SJI[self.window[0]].data[:,:,self.SJI[self.window[0]].slit_acqnum]
                poi.XY_map_label_title = 'Image {} of {}  - {} - {}'.format(self.SJI[self.window[0]].slit_acqnum, dim_data[2]-1, 
                                                                            self.window[0],
                                                                            self.SJI[self.window[0]].date_time_acq_ok[self.SJI[self.window[0]].slit_acqnum])
                poi.XY_map_label_x = self.SJI[self.window[0]].extent_display_coords[0]
                poi.XY_map_label_y = self.SJI[self.window[0]].extent_display_coords[1]
                poi.clip_ima = self.SJI[self.window[0]].clip_ima.copy()
                poi.cmap = self.SJI[self.window[0]].cmap
                poi.extent_display = self.SJI[self.window[0]].extent_display
                poi.extent_display_coords = self.SJI[self.window[0]].extent_display_coords
                poi.xlim = [*self.__ax.get_xlim()] # self.SJI[self.window[0]].xlim1
                poi.ylim = [*self.__ax.get_ylim()] # self.SJI[self.window[0]].ylim1
                poi.xlim2 = [*self.__ax2.get_xlim()] # self.SJI[self.window[0]].xlim2  # self.__ax2.get_xlim()
                poi.time_stamp =  self.SJI[self.window[0]].date_time_acq_ok[self.SJI[self.window[0]].slit_acqnum]
                self.SJI[self.window[0]].poi.insert(self.SJI[self.window[0]].__count_poi, poi)

            def set_poi(poi):

                """ Set the values of a raster[window].poi as the active one """

                self.SJI[self.window[0]].slit_acqnum = poi.slit_acqnum
                self.SJI[self.window[0]].show_slit = poi.show_slit
                self.SJI[self.window[0]].clip_ima = poi.clip_ima.copy()
                self.SJI[self.window[0]].cmap = poi.cmap
                self.SJI[self.window[0]].date_time_acq_ok[self.SJI[self.window[0]].slit_acqnum] = poi.time_stamp
                self.SJI[self.window[0]].__linewidth = poi.slit_pos_linewidth
                self.SJI[self.window[0]].xlim1 = poi.xlim
                self.SJI[self.window[0]].ylim1 = poi.ylim
                self.SJI[self.window[0]].xlim2 = poi.xlim2
                self.SJI[self.window[0]].ylim2 = poi.ylim2

            def onkey(event):

                """ What to do when event key is pressed """

                if event.key == '?' or event.key == '/': show_help()
                if event.key == 'left' or event.key == 'right': self.__bar[:,self.SJI[self.window[0]].slit_acqnum] = 0
                if event.key == 'left': self.SJI[self.window[0]].slit_acqnum-=1
                if event.key == 'right': self.SJI[self.window[0]].slit_acqnum+=1
                if self.SJI[self.window[0]].slit_acqnum > np.min((dim_data[2]-1, int(self.SJI[self.window[0]].xlim2[1]))):
                    self.SJI[self.window[0]].slit_acqnum = int(self.SJI[self.window[0]].xlim2[0])
                if self.SJI[self.window[0]].slit_acqnum <  int(self.SJI[self.window[0]].xlim2[0]):
                    self.SJI[self.window[0]].slit_acqnum = np.min((dim_data[2]-1, int(self.SJI[self.window[0]].xlim2[1])))
                if event.key == '-':  self.SJI[self.window[0]].slit_acqnum_delay+=2
                if event.key == '+' or event.key == '=':  self.SJI[self.window[0]].slit_acqnum_delay-=2
                if self.SJI[self.window[0]].slit_acqnum_delay <= 0:  self.SJI[self.window[0]].slit_acqnum_delay = 1
                delay = self.SJI[self.window[0]].__delay*self.SJI[self.window[0]].slit_acqnum_delay
                if event.key == 'u': self.SJI[self.window[0]].clip_ima[0]*=0.9
                if event.key == 'i': self.SJI[self.window[0]].clip_ima[0]*=1.1
                if event.key == 'o': self.SJI[self.window[0]].clip_ima[1]*=0.9
                if event.key == 'p': self.SJI[self.window[0]].clip_ima[1]*=1.1
                if event.key == 'h' or event.key == 'H': 
                    self.SJI[self.window[0]].clip_ima = self.SJI[self.window[0]].__clip_ima_ini
                    self.SJI[self.window[0]].xlim1 = [self.SJI[self.window[0]].extent_display[0],
                                                      self.SJI[self.window[0]].extent_display[1]]
                    self.SJI[self.window[0]].ylim1 = [self.SJI[self.window[0]].extent_display[2],
                                                      self.SJI[self.window[0]].extent_display[3]]
                    self.SJI[self.window[0]].xlim2 = [0, self.SJI[self.window[0]].__dim_data[2]]
                    self.SJI[self.window[0]].ylim2 = [0, max(1,int(self.SJI[self.window[0]].__dim_data[2]/8))]
                    self.__im.axes.set_xlim(self.SJI[self.window[0]].xlim1)
                    self.__im.axes.set_ylim(self.SJI[self.window[0]].ylim1)
                    self.__im2.axes.set_xlim(self.SJI[self.window[0]].xlim2)
                    self.__im2.axes.set_ylim(self.SJI[self.window[0]].ylim2)
                    if event.key == 'H':
                        if self.SJI[self.window[0]].figsize != self.SJI[self.window[0]].__figsize_ori:
                            self.__fig.set_size_inches(self.SJI[self.window[0]].__figsize_ori,
                                                       forward = True)
                            plt.tight_layout()
                if event.key == 'a':
                    take_poi()
                    self.SJI[self.window[0]].__count_poi+=1
                    print('Saving PoI # {} of a total of {} for window'.format(
                          self.SJI[self.window[0]].__count_poi,
                          len(self.SJI[self.window[0]].poi),
                          self.window[0]))
                if (event.key == 'r' or event.key == 't' or event.key == 'e') and \
                    self.SJI[self.window[0]].__animate == False:
                    if len(self.SJI[self.window[0]].poi) > 0:
                        self.__bar[:,self.SJI[self.window[0]].slit_acqnum] = 0
                        if event.key == 'r': self.SJI[self.window[0]].__move_count_poi-=1
                        if event.key == 't': self.SJI[self.window[0]].__move_count_poi+=1
                        if  self.SJI[self.window[0]].__move_count_poi > len(self.SJI[self.window[0]].poi)-1:
                            self.SJI[self.window[0]].__move_count_poi =0
                        if  self.SJI[self.window[0]].__move_count_poi < 0:
                            self.SJI[self.window[0]].__move_count_poi = len(self.SJI[self.window[0]].poi)-1
                        set_poi(self.SJI[self.window[0]].poi[self.SJI[self.window[0]].__move_count_poi])
                        print('Showing PoI #{} of a total of {} saved for '
                              'window {}'.format(self.SJI[self.window[0]].__move_count_poi+1,
                                                 len(self.SJI[self.window[0]].poi),
                                                 self.window[0]))
                        self.__im.axes.set_xlim(self.SJI[self.window[0]].xlim1)
                        self.__im.axes.set_ylim(self.SJI[self.window[0]].ylim1)
                        self.__im2.axes.set_xlim(self.SJI[self.window[0]].xlim2)
                        if event.key == 'e':
                            self.SJI[self.window[0]].__count_e+=1
                            if self.SJI[self.window[0]].__count_e == 1:
                               print("Are you sure you want to remove PoI #{} of "
                                     "a total of {} saved for  window {}? Press 'e' to "
                                     "comfirm.".format(self.SJI[self.window[0]].__move_count_poi+1,
                                     len(self.SJI[self.window[0]].poi), self.window[0]))
                            if self.SJI[self.window[0]].__count_e == 2:
                                del self.SJI[self.window[0]].poi[self.SJI[self.window[0]].__move_count_poi]
                                self.SJI[self.window[0]].__move_count_poi-=1
                                if self.SJI[self.window[0]].__move_count_poi < 0:
                                    self.SJI[self.window[0]].__move_count_poi = 0
                                print("PoI #{} of a total of {} saved for window {} has been "
                                      "succesfully removed.".format(self.SJI[self.window[0]].__move_count_poi,
                                                                    len(self.SJI[self.window[0]].poi),
                                                                    self.window[0]))
                                self.SJI[self.window[0]].__count_e = 0
                    else:
                        print('There is no PoI saved for window {}'.format(self.window[0]))
                    if event.key != 'e': self.SJI[self.window[0]].__count_e = 0
                ima2show = ima_scale(self.SJI[self.window[0]].data[:,:,self.SJI[self.window[0]].slit_acqnum], minmax=self.SJI[self.window[0]].clip_ima)
                self.__im.set_data(ima2show)
                self.__bar[:,self.SJI[self.window[0]].slit_acqnum] = 1
                self.__im2.set_data(self.__bar)
                opt_ima_scale = ['u','i','o','p','r','t','h']
                if event.key in opt_ima_scale:
                    self.__im.set_clim(self.SJI[self.window[0]].clip_ima)
                    if  self.SJI[self.window[0]].__animate == True: 
                        ima2show = ima_scale(self.SJI[self.window[0]].data[:,:,self.SJI[self.window[0]].slit_acqnum], minmax=self.SJI[self.window[0]].clip_ima)
                        self.__im.set_data(ima2show)
                if event.key == 'd': 
                    self.SJI[self.window[0]].__show_title = not self.SJI[self.window[0]].__show_title
                    if self.SJI[self.window[0]].__show_title ==  False:
                        self.__title= self.__ax.set_title(u'\u2588'*120)
                        self.__title.set_color('white')
                if event.key == 'v' or event.key == 'V': 
                    self.SJI[self.window[0]].__linewidth = 2
                    if event.key == 'V': self.SJI[self.window[0]].__linewidth = 4
                    self.SJI[self.window[0]].show_slit = not self.SJI[self.window[0]].show_slit
                if self.SJI[self.window[0]].show_slit == True: 
                    try:
                        self.__vline.remove()
                        self.__vline = self.__ax.axvline(self.SJI[self.window[0]].SLTPX1IX[self.SJI[self.window[0]].slit_acqnum],
                                ls=":", linewidth=self.SJI[self.window[0]].__linewidth)
                    except:
                        self.__vline = self.__ax.axvline(self.SJI[self.window[0]].SLTPX1IX[self.SJI[self.window[0]].slit_acqnum],
                                ls=":", linewidth=self.SJI[self.window[0]].__linewidth)
                        pass
                else:
                    try:
                        self.__vline.remove()
                    except:
                        pass
                self.__title = self.__ax.set_title('Image {} of {}  - {} - {}'.format(self.SJI[self.window[0]].slit_acqnum, dim_data[2]-1,
                                                                       self.window[0],
                                                                       self.SJI[self.window[0]].date_time_acq_ok[self.SJI[self.window[0]].slit_acqnum]))
                plt.draw()
                if event.key == ' ': self.SJI[self.window[0]].__animate = not self.SJI[self.window[0]].__animate
                if event.key == 'q': 
                    self.SJI[self.window[0]].__animate =  False
                    self.__fig.canvas.stop_event_loop()
                    plt.draw()
                self.SJI[self.window[0]].xlim2 = [*self.__ax2.get_xlim()]
                while self.SJI[self.window[0]].__animate == True: 
                        if self.SJI[self.window[0]].slit_acqnum > np.min((dim_data[2]-1, int(self.SJI[self.window[0]].xlim2[1]))):
                            self.SJI[self.window[0]].slit_acqnum = int(self.SJI[self.window[0]].xlim2[0])
                        if self.SJI[self.window[0]].slit_acqnum == 0: tstart = time.time()
                        ima2show = self.SJI[self.window[0]].data[:,:,self.SJI[self.window[0]].slit_acqnum]
                        self.__im2.set_data(self.__bar)
                        self.__ax2.draw_artist(self.__im2)
                        self.__fig.canvas.blit(self.__ax2.bbox)
                        if self.SJI[self.window[0]].show_slit == True:
                            self.__vline.remove()
                            self.__vline = self.__ax.axvline(self.SJI[self.window[0]].SLTPX1IX[self.SJI[self.window[0]].slit_acqnum],
                                            ls=":", linewidth=self.SJI[self.window[0]].__linewidth)
                            self.__bar[:,self.SJI[self.window[0]].slit_acqnum] = 0
                        if blit == False:
                            self.__im.set_data(ima2show)
                            self.__ax.set_title('Image {} of {}  - {} - {}'.format(self.SJI[self.window[0]].slit_acqnum, dim_data[2]-1,
                                                                                 self.window[0],
                                                                                 self.SJI[self.window[0]].date_time_acq_ok[self.SJI[self.window[0]].slit_acqnum]))
                            self.__fig.canvas.start_event_loop(delay)
                            self.__fig.canvas.flush_events()
                            plt.draw()
                        else:
                            self.__im.set_data(ima2show)
                            self.__ax.draw_artist(self.__im)
                            self.__fig.canvas.blit(self.__ax.bbox)
                            self.__bar[:,self.SJI[self.window[0]].slit_acqnum] = 1 
                            self.__im2.set_data(self.__bar)
                            self.__ax2.draw_artist(self.__im2)
                            self.__fig.canvas.blit(self.__ax2.bbox)
                            self.__bar[:,self.SJI[self.window[0]].slit_acqnum] = 0
                            if self.SJI[self.window[0]].__show_title == True:
                                self.__title= self.__ax.set_title(u'\u2588'*120)
                                self.__title.set_color('white')
                                self.__ax.draw_artist(self.__title)
                                self.__title= self.__ax.set_title('Image {} of {}  - {} - {}'.format(self.SJI[self.window[0]].slit_acqnum, dim_data[2]-1,
                                                                           self.window[0],
                                                                           self.SJI[self.window[0]].date_time_acq_ok[self.SJI[self.window[0]].slit_acqnum]))
                                self.__title.set_color('black')
                                self.__ax.draw_artist(self.__title)
                                self.__fig.canvas.blit(self.__title.get_window_extent())
                            if self.SJI[self.window[0]].show_slit == True:
                                self.__ax.draw_artist(self.__vline)
                                self.__fig.canvas.blit(self.__vline.clipbox)
                            self.__fig.canvas.start_event_loop(delay)
                        self.SJI[self.window[0]].slit_acqnum+=1
                        if self.SJI[self.window[0]].slit_acqnum > np.min((dim_data[2]-1, int(self.SJI[self.window[0]].xlim2[1]))):
                            self.SJI[self.window[0]].slit_acqnum = int(self.SJI[self.window[0]].xlim2[0])
                if event.key == 'q': 
                    if memmap == True:
                        self.SJI[self.window[0]].data = np.array(self.SJI[self.window[0]].data)
                        print('Removing temporary file...', self.SJI[self.window[0]].temp_memmap_file)
                        self.SJI[self.window[0]].temp_memmap_obj.close()
                        del self.SJI[self.window[0]].temp_memmap_obj
                    self.SJI[self.window[0]].figsize = self.__fig.get_size_inches()
                    plt.close('all')
                return

            def display():

                """ Simple display using blit """

                ima2show = self.SJI[self.window[0]].data[:,:,self.SJI[self.window[0]].slit_acqnum] 
                self.__im.set_data(ima2show)
                self.__ax.draw_artist(self.__im)
                self.__fig.canvas.blit(self.__ax.bbox)
                self.__im2.set_data(self.__bar)
                self.__ax2.draw_artist(self.__im2)
                self.__fig.canvas.blit(self.__ax2.bbox)
                if self.SJI[self.window[0]].__show_title == True:
                    self.__title= self.__ax.set_title(u'\u2588'*120)
                    self.__title.set_color('white')
                    self.__ax.draw_artist(self.__title)
                    self.__title= self.__ax.set_title('Image {} of {}  - {} - {}'.format(self.SJI[self.window[0]].slit_acqnum, dim_data[2]-1,
                                                               self.window[0],
                                                               self.SJI[self.window[0]].date_time_acq_ok[self.SJI[self.window[0]].slit_acqnum]))
                    self.__title.set_color('black')
                    self.__ax.draw_artist(self.__title)
                    self.__fig.canvas.blit(self.__title.get_window_extent())
                if self.SJI[self.window[0]].show_slit == True:
                    try:
                        self.__vline.remove()
                        self.__vline = self.__ax.axvline(self.SJI[self.window[0]].SLTPX1IX[self.SJI[self.window[0]].slit_acqnum],
                                ls=":", linewidth=self.SJI[self.window[0]].__linewidth)
                    except:
                        self.__vline = self.__ax.axvline(self.SJI[self.window[0]].SLTPX1IX[self.SJI[self.window[0]].slit_acqnum],
                                ls=":", linewidth=self.SJI[self.window[0]].__linewidth)
                        pass
                    self.__ax.draw_artist(self.__vline)
                    self.__fig.canvas.blit(self.__vline.clipbox)
                return 


            def motion(event):

                """ What to do when the mouse moves. It only has an effect on
                    the slider bar """

                if self.__ax2.in_axes(event) == True: 
                        self.__bar[:,self.SJI[self.window[0]].slit_acqnum] = 0
                        self.SJI[self.window[0]].slit_acqnum = int(event.xdata)
                        self.__bar[:,self.SJI[self.window[0]].slit_acqnum] = 1
                        display()
                return
            self.__cid_onlkey = self.__fig.canvas.mpl_connect('key_press_event', onkey)
            self.__cid_motion = self.__fig.canvas.mpl_connect('motion_notify_event', motion)
            plt.show()

        return



def get_SJI(filename, verbose =  False, showtime = False, show_imaref = True,
        option_extent = False, no_data = False, memmap= True):

    """ Get SJI data and some information of the headers. 
        It is an ancient method, but it is useful. """

    global count, data, im

    hdulist = fits.open(filename)
    hdr = hdulist[0].header

    data = hdulist[0].data
    data = np.transpose(data, [1,2,0])
    temp_memmap_filename = 'None' 
    temp_memmap_obj = None
    if memmap == True:
       out_temp_file = make_temp_file(data)
       datatype = data.dtype
       dim_data = data.shape
       temp_memmap_filename = out_temp_file['temp_file_name']
       data = np.memmap(temp_memmap_filename, dtype=datatype, mode='r',
                        shape=dim_data)
       temp_memmap_obj = out_temp_file['temp_file_obj']

    extra_info = hdulist[2].data

    date_time_acq = []
    date_time_acq_ok = []
    for i in extra_info: 
        aux_info = i[3]
        pos_sji = aux_info.find('_sji')
        datetime = aux_info[pos_sji-17:pos_sji]
        date_time_acq.append(datetime)
        ok_date = '{}-{}-{} {}:{}:{}'.format(datetime[0:4], datetime[4:6],
                  datetime[6:8], datetime[9:11], datetime[11:13], datetime[13:15])
        date_time_acq_ok.append(ok_date)

    extra_info = hdulist[1].data
    
    SLTPX1IX = []
    SLTPX2IX = []
    XCENIX = []
    YCENIX = []

    for i in extra_info: 
        SLTPX1IX.append(i[4])
        SLTPX2IX.append(i[5])
        XCENIX.append(i[10])
        YCENIX.append(i[11])

    sji_label = hdr['TDESC1']

    windx = 1
    hdr_windx = hdr 

    DATE_OBS = hdr['DATE_OBS']
    DATE_END = hdr['DATE_END']
    TDET   = hdr['TDET{}'.format(windx)]
    TDESCT = hdr['TDESC{}'.format(windx)]
    TWAVE  = hdr['TWAVE{}'.format(windx)]
    TWMIN  = hdr['TWMIN{}'.format(windx)]
    TWMAX  = hdr['TWMAX{}'.format(windx)]
    SPCSCL = hdr_windx['CDELT1']
    SPXSCL = hdr_windx['CDELT3']
    SPYSCL = hdr_windx['CDELT2']
    POS_X  = hdr_windx['CRVAL3']
    POS_Y  = hdr_windx['CRVAL2']

    out = {}

    wl = (np.array(range(hdulist[windx].header['NAXIS1'])) * SPCSCL) + TWMIN

    pos_ref = filename.find('iris_l2_')
    date_in_filename = filename[pos_ref+8:pos_ref+23]
    iris_obs_code    = filename[pos_ref+24:pos_ref+34]
    raster_info      = filename[pos_ref+35:pos_ref+53]

    extent_arcsec_arcsec = [0, data.shape[1]*SPXSCL, 0, data.shape[0]*SPYSCL]
    extent_px_px = [0, data.shape[1]*1., 0, data.shape[0]*1.]
    extent_px_arcsec  = [0, data.shape[1]*1., 0, data.shape[0]*SPYSCL]

    min_extent = np.zeros(4)
    list_extent = [extent_arcsec_arcsec, extent_px_px,
                   extent_px_arcsec]
    list_extent_coords = [('X [arcsec]','Y [arcsec]'), ('X [px]','Y [px]'),
                          ('X [px]','Y [arcsec]')]


    min_extent = list(map(lambda a: abs(a[3] - a[1]), list_extent))
    min_extent = np.array(min_extent)

    if extent_arcsec_arcsec[1] == 0: min_extent[0] = 1e9

    extent_opt = list_extent[np.argmin(min_extent)]
    extent_opt_coords = list_extent_coords[np.argmin(min_extent)]

    if show_imaref == True:
        count = 0
        no_times = 0
        dim_data = data.shape
        extent = extent_arcsec_arcsec
        if verbose == True:
           print('Extent coordinates options (option_extent):')
           for i in list_extent_coords:
               print(count, i)
               count+=1
        count = 0
        extent = extent_opt
        extent_labels =  extent_opt_coords
        if option_extent !=0:
           extent = list_extent [option_extent-1]
           extent_labels = list_extent_coords [option_extent-1]
        fig=plt.figure()
        print(list_extent)
        plt.subplot(1,1,1)
        im = plt.imshow(data[:,:,count], origin='lower', extent=extent_opt)
        plt.xlabel(extent_labels[0])
        plt.ylabel(extent_labels[1])
        plt.tight_layout()
        plt.title('Image {} of {}  - {}'.format(count, dim_data[2],
                                                date_time_acq[count]))
        plt.show()
        def onkey(event):
            global count, data, im 
            if event.key == 'left': count-=1
            if event.key == 'right': count+=1
            if count == dim_data[2]: count=0
            if count < 0 : count =  dim_data[2]-1
            im.set_data(data[:,:,count]) 
            plt.title('Image {} of {}  - {}'.format(count, dim_data[2],
                                                    date_time_acq[count]))
            plt.draw()
            if event.key == 'q': plt.close(1)
            return 
        cid = fig.canvas.mpl_connect('key_press_event', onkey)

    if no_data == True: data = None

    out_win = {"data":data, "wl":wl, "date_in_filename": date_in_filename,
           "iris_obs_code": iris_obs_code, "raster_info": raster_info,
           "DATE_OBS":DATE_OBS, "DATE_END":DATE_END, "TDET":TDET, "TDESCT":TDESCT,
           "TWAVE":TWAVE, "TWMIN":TWMIN, "TWMAX":TWMAX, "SPCSCL":SPCSCL,
           "SPXSCL":SPXSCL, "SPYSCL":SPYSCL, 
           "POS_X":POS_X, "POS_Y":POS_Y,
           "SLTPX1IX":SLTPX1IX, "SLTPX2IX":SLTPX2IX, 
           "XCENIX":XCENIX, "YCENIX":YCENIX,     
           "date_time_acq":date_time_acq,
           "date_time_acq_ok":date_time_acq_ok,
           "extent_arcsec_arcsec":extent_arcsec_arcsec,
           "extent_px_px":extent_px_px,
           "extent_px_arcsec":extent_px_arcsec,
           "extent_opt":extent_opt,
           "extent_opt_coords":extent_opt_coords,
           "list_extent":list_extent,
           "list_extent_coords":list_extent_coords,
           "temp_memmap_filename":temp_memmap_filename,
           "temp_memmap_obj":temp_memmap_obj}

    out[sji_label] = out_win
    hdulist.close()

    return out




