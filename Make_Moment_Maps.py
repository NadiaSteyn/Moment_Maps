#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make Moment Maps
Created on Wed May  6 11:37:46 2020

This program will generate moment maps of a given list of detections 

"""
#my first git commit
#my second commit
#I want the GitHub action to run once I push this change

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits  # We use fits to open the actual data file
from spectral_cube import SpectralCube
import matplotlib.gridspec as gridspec
import warnings
from astropy.wcs import WCS
from matplotlib.patches import Ellipse
import matplotlib
from astropy.visualization import simple_norm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredEllipse
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDrawingArea
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', FITSFixedWarning)
#module add python/3.8.3
#comment out block: cmd + /
#cancel the run control+c

print("git commit")

from matplotlib.offsetbox import (AnchoredOffsetbox, AuxTransformBox, DrawingArea, TextArea, VPacker)
class AnchoredEllipseBeam(AnchoredOffsetbox): #class that I need for plotting ellipse
    def __init__(self, transform, width, height, angle, loc = 'lower left',
                 pad=0.5, borderpad=0.1, prop=None, frameon=False):
        """
        Draw an ellipse the size in data coordinate of the give axes.
        pad, borderpad in fraction of the legend font size (or prop)
        """
        self._box = AuxTransformBox(transform)
        self.ellipse = Ellipse((0, 0), width, height, angle,fill=False,color='k',lw=2)
        self._box.add_artist(self.ellipse)
        super().__init__(loc, pad=pad, borderpad=borderpad,
                         child=self._box, prop=prop, frameon=frameon)
        
######################################
#Input
######################################

#Choose FITS cube:
hdu = fits.open('/idia/projects/vela/V1_GA_CARACal/mosaics_nadia/T12/output/mosaics/mosaic_T12.fits') # 12" x 12"
# beamsize = 12 # if you dont have the beamsize in the header

# hdu[0].header.set('BMAJ',beamsize/3600)
# hdu[0].header.set('BMIN',beamsize/3600) # Later the GPS mosaics will have this info in the header
cube = SpectralCube.read(hdu) # Initiate a SpectralCube

#convert cube to VRAD m/s:
print('CONVERTING CUBE INTO VELOCITY UNITS...')
cube = cube.with_spectral_unit(u.m/u.s,velocity_convention='optical',rest_value=1420.4*u.MHz)
print(f"Cube now has spectral unit {cube.header['CTYPE3']} {cube.header['CUNIT3']}")
hdu.close()

#Cube now has dimensions [Vel, Dec, RA]
_, Dec, _ = cube.world[0, :, 0]  #extract Dec world coordinates from cube
_, _, RA = cube.world[0, 0, :]  #extract RA world coordinates from cube

rel = np.ones(1000000) # a fake array of rels in case the the user doesn't have rels
ID = np.ones(1000000)

nu_lab = 1420405751.7667  #21cm line in HZ
c=3e5
def frequency_points(freq,convention):
    if convention == 'Radio':
        return c*(nu_lab - freq)/nu_lab 
    
    elif convention  == 'Optical':
        return c*(nu_lab - freq)/freq
    
#Choose detection list:
infile = '/users/nadia/HI_flux/T12_x,y_new_03_Sep_2021.txt'
x, y, z, vel \
= np.loadtxt(infile, usecols = (0,1,2,3), unpack=True)
ID = ID.astype(int)
#vel = np.round(frequency_points(freq,'Optical'),0) # if you have freq instead of vel
ID = np.loadtxt(infile,usecols=4,dtype=str,unpack=True)


xpad=80
ypad=80
zpad=2
radii=60 #radius used around source for HI profile
zpad_HI=16
label_name="T12"
######################################

#Major and Minor axes (to draw the beam size):
def check_for_beam_info(cube):
    try: 
        bmaj_test = cube.header['BMAJ']
        bmin_test = cube.header['BMIN']
        return True
    except KeyError:
        print('No beam info found in Header')
        return False

def get_beam_info(cube):
    beam_major = cube.header['BMAJ']
    beam_minor = cube.header['BMIN']
    return beam_major,beam_minor

def convert_arcsecs_to_pix(cube,arcssec_value):
    deg_per_pix = cube.header['CDELT2']
    arcsecs_per_pix = deg_per_pix * 3600
    return arcssec_value/arcsecs_per_pix


if type(x) == float or type(x) == int:
    x,y,z,vel = np.array([x]), np.array([y]), np.array([z]), np.array([vel])

if type(rel) == float or type(rel)==int:
    rel = np.array([rel])
else:    
    rel = rel[:len(x)]

if type(ID)==float or type(ID)==int:
    ID = np.array([ID])
else:
    ID = ID[:len(x)]

pos_vels = []
neg_vels = []
pos_test = np.where(vel > -300)[0]
for i in range(len(vel)):
    if vel[i]>-300: 
        pos_vels.append(i)
    else: 
        neg_vels.append(i)
print(f'positive velocities: {len(pos_vels)}')
print(f'negative velocities: {len(neg_vels)}')

######################################
#Functions
######################################

def zerofy(number):
    if number <0:
        return 0 
    else:
        return number

def reduce_fits_file(fits_file,x_center,y_center,z_center,xpad,ypad,zpad): #if centre - pad is a negative number, the subcube is out of range. In this case we want to use whatever padding is available, so we make centre - pad = 0.
    sub_array_3d = fits_file[zerofy(int(round(z_center-zpad))):int(round(z_center+zpad+1)),
                             zerofy(int(round(y_center-ypad))):int(round(y_center+ypad+1)), 
                             zerofy(int(round(x_center-xpad))):int(round(x_center+xpad+1))]
    return sub_array_3d

def noise_cube(cube,x,y,z,xpad,ypad,zpad): # for the masking
    sub_cube = reduce_fits_file(cube, x, y, z, xpad, ypad, zpad) # Cutting the main fits file into a sub-cube
    ### This is how we determine what method we want to use for noise
    noise_moment_0 = sub_cube.moment(order=0)
    noise= np.nanmean(noise_moment_0.hdu.data)  # nanmean takes the mean ignoring the nans 
    sigma = np.nanstd(noise_moment_0.hdu.data)
    return noise,sigma

def get_noise(cube,x,y,z,xpad,ypad,zpad):  # function to look at the 8 squares around the main square 
    means=[] # lists to store the mean and sigma values 
    sigmas=[]
    x_vals = x+np.array([-1,-1,-1,0,0,1,1,1])*2*xpad    # x and y positions of the 8 squares 
    y_vals = y+np.array([1,0,-1,1,-1,1,0,-1])*2*ypad
    for i in range(len(x_vals)):
        val = noise_cube(cube, x_vals[i], y_vals[i], z, xpad, ypad, zpad)   # work out the noise and sigma for that cube
        if np.isnan(val[0]).any() == False and np.isnan(val[1]).any()==False:  # if both noise and sigma are NOT NANS then continue 
            means.append(val[0])
            sigmas.append(val[1])
            
    means = np.array(means)
    sigmas = np.array(sigmas)
    msk = np.abs(means-np.mean(means))<np.std(means)    # work out which (if any) of the means are too high (higher than 1 sigma). These could have sources in them
    means = means[msk]   #apply the mask to the means and the sigmas (keeping all the noise squares that have a mean<1 sigma. i.e. discard the squares that have sources)
    sigmas = sigmas[msk]
    noise = np.mean(means)  #average over all the appropraite squares that are left 
    sigma= np.mean(sigmas)
    return noise,sigma  #return the noise and sigma

def drawEllipse(cube,ax):      
    if check_for_beam_info(cube)==True:
        majorBeam, minorBeam = get_beam_info(cube)
        majorBeamPixel, minorBeamPixel = convert_arcsecs_to_pix(cube,majorBeam*3600),convert_arcsecs_to_pix(cube,minorBeam*3600)
        aeb = AnchoredEllipseBeam(ax.transData,width=majorBeamPixel,height=minorBeamPixel,angle=0)
    else:
        aeb = AnchoredEllipseBeam(ax.transData,width=0,height=0,angle=0)
    ax.add_artist(aeb)

def cut_mom_map(cube,x,y,z,xpad,ypad,zpad,pos=111,label=None,ID=None,vel_label=None,frame='world',clean=False, cleaning_factor=1,order=0,vmin=None,vmax=None,HI_profile=False,HI_radii=radii): #I'm using label as the reliability
    
    current_wcs_3d = cube.wcs
    current_ra,current_dec,current_vel = current_wcs_3d.pixel_to_world_values(x,y,z)
    current_vel = current_vel/1000 # in km/s
    print('Cutting moment map...')
    sub_cube = reduce_fits_file(cube, x, y, z, xpad, ypad, zpad) # Cutting the main fits file into a sub-cube
    current_sub_cube_wcs_3d = sub_cube.wcs
    sub_cube_center_x,sub_cube_center_y,_ = current_sub_cube_wcs_3d.world_to_pixel_values(current_ra,current_dec,current_vel)
    #going from the given pixel, to ra/dec, to pixel in the sub cube
    if order ==0:
        moment_map = sub_cube.with_spectral_unit(u.km/u.s).moment(order=order)  # changing units from m/s to km/s. Units will be cube units * spectral unit (Jy km/s / beam)
    else:
        moment_map=sub_cube.with_spectral_unit(u.km/u.s).moment(order=1,axis=0,how='cube')
    
    if clean==True:
        print(f'cleaning with factor {cleaning_factor}') # removing cleaning_factor * sigma data
        noise,sigma = get_noise(cube,x,y,z,xpad,ypad,zpad)
        tolerance = sigma * cleaning_factor
        diff = moment_map.hdu.data - noise
        cut = np.where(diff < tolerance)
        
        moment_map_clean = moment_map.copy() #This is the cut-out of the inside of the mom-1 galaxy, using the outline from mom-0
    
        for i in range(len(cut[0])):
           moment_map_clean.hdu.data[cut[0][i]][cut[1][i]] = np.nan #change all the pixels that are <tolerence into NaNs
        img = moment_map_clean.hdu.data
    else:
        img = moment_map.hdu.data
        #hi_column_density = moment_map_clean * 1.82 * 10**18 / (u.cm * u.cm) * u.s / u.K / u.km 
    
    if HI_profile == True:
        gs = gridspec.GridSpec(ncols=20,nrows=25,figure=fig)
        gs.update(wspace=0)
        # Initiate an axis object with WCS projection information:
        ax = fig.add_subplot(gs[0:17,0:16], projection=moment_map.wcs) # POINTS ARE NOT IN RA/DEC but the AXES ARE
        ax_HI = fig.add_subplot(gs[19:,0:20])
        hx,hy,current_vel_HI,HI_centre,HI_radius=get_HI_profile(cube, x, y, z, xpad, ypad, radius=HI_radii)
        ax_HI.plot(hx, hy,'.-', color='k') # Display the HI profile
        ax_HI.axvline(current_vel,ls='--',color='k',alpha=0.5)
        ax_HI.axhline(0,ls='--',color='k',alpha=0.5)
        #plt.legend()

        # Add axes labels
        ax_HI.set_xlabel("Velocity (km/s)", fontsize=10, labelpad=10)
        ax_HI.set_ylabel("Jy/beam", fontsize=10, labelpad=10)
        ax_HI.minorticks_on()
        ax_HI.tick_params(axis='both',labelsize=10)
        ax_HI.tick_params(which='both', width=2)
        ax_HI.tick_params(which='major', length=8, direction='in')#,right=True)
        ax_HI.tick_params(which='minor', length=4, direction='in')#,right=True)
        
        #ax.scatter(HI_centre[0],HI_centre[1]) #x and y of closest pixel to the ra/dec of the object
        ax.scatter(sub_cube_center_x,sub_cube_center_y,color='k',marker='+') #the exact x
        circle = plt.Circle(HI_centre,HI_radius,facecolor=None,color='k',fill=False,ls='--')
        #circle_half = plt.Circle(HI_centre,HI_radius/2,facecolor=None,color='k',fill=False,ls=':')
        #circle_quarter = plt.Circle(HI_centre,HI_radius/4,facecolor=None,color='k',fill=False,ls=':')
        ax.add_artist(circle)
        #ax.add_artist(circle_half)
        #ax.add_artist(circle_quarter)
        
    else:
        ax = fig.add_subplot(pos,projection=moment_map.wcs)

    # Display the moment map image
    if order==0: #moment0
        if vmin==None and vmax==None: #if both vmin & vmax are not specified:
            im = ax.imshow(img,cmap='YlOrRd') #making the 2D histogram
        elif vmin!=None and vmax==None:
            im = ax.imshow(img,cmap='YlOrRd', vmin=vmin) #cmap='YlOrRd' #cmap='RdBu'
        elif vmax!=None and vmin==None:
            im = ax.imshow(img,cmap='YlOrRd', vmax=vmax)
        elif vmin!=None and vmax!=None:
            im = ax.imshow(img,cmap='YlOrRd', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im,ax=ax, pad=0.01)
        cbar.set_label('Brightness (Jy km/s /beam)', size=10, rotation=90, labelpad=11) 
        cbar.ax.tick_params(labelsize=10)
        
    elif order==1: #moment1
        if vmin==None and vmax==None: #if both vmin & vmax are not specified:
            print("***WARNING*** It is highly recommended to specify a relevant vmin and vmax for moment-1 maps")
            im = ax.imshow(img,cmap='RdBu')
        elif vmin!=None and vmax==None:
            print("***WARNING*** It is highly recommended to specify a relevant vmin and vmax for moment-1 maps")
            im = ax.imshow(img,cmap='RdBu', vmin=vmin)
        elif vmax!=None and vmin==None:
            print("***WARNING*** It is highly recommended to specify a relevant vmin and vmax for moment-1 maps")
            im = ax.imshow(img,cmap='RdBu', vmax=vmax)
        elif vmin!=None and vmax!=None:
            im = ax.imshow(img,cmap='RdBu', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im,ax = ax,pad=0.01)
        cbar.set_label('Velocity (km/s)', size=10, labelpad=-0.08) 
        cbar.ax.tick_params(labelsize=10)

    # Add axes labels
    ax.set_xlabel("RA (deg)", fontsize=10, labelpad=0.7)
    ax.set_ylabel("Dec (deg)", fontsize=10, labelpad=-0.5)
    
    # Add ellipse:
    drawEllipse(cube, ax)
    
    #Ticks:
    ax.minorticks_on()
    #ax.tick_params(axis='both',labelsize=10)
    #ax.tick_params(which='both', width=1, labelsize=10)
    #ax.tick_params(which='both',direction='in')
    ax.tick_params(which='major', length=5)#, right=True)
    ax.tick_params(which='minor', length=2)#, right=True)
    
    # Add labels:
    if grid == True:
        x_box,y_box = 0.03,0.87
    else: x_box,y_box = 0.015,0.94
    if label == None and ID == None and frame=='cartesian':
        plt.text(x_box,y_box,f'Z_chan = {int(round(z))}',ha ='left',va='center',transform = ax.transAxes,fontsize=10,weight='bold',color='k',bbox=dict(facecolor='w',alpha=0.5))
    elif label != None and ID == None and frame=='cartesian':
        plt.text(x_box,y_box,f'Z_chan = {int(round(z))} \n{label_name} = {round(label,4)}',ha ='left',va='center',transform = ax.transAxes,fontsize=10,weight='bold',color='k',bbox=dict(facecolor='w',alpha=0.5))
    elif label != None and ID != None and frame=='cartesian':
        plt.text(x_box,y_box,f'Z_chan = {int(round(z))} \n{label_name} = {round(label,4)} \nID = {ID}',ha ='left',va='center',transform = ax.transAxes,fontsize=10,weight='bold',color='k',bbox=dict(facecolor='w',alpha=0.5))
        
    elif label !=None and ID != None and frame == 'world':
        if vel_label == None:
            plt.text(x_box,y_box,f'Vel = {int(round(current_vel))} km/s \n{label_name} = {label} \nID = {ID}',ha ='left',va='center',transform = ax.transAxes,fontsize=10,weight='bold',color='k',bbox=dict(facecolor='w',alpha=0.5))
        else:
            plt.text(x_box,y_box,f'Vel = {int(vel_label)} km/s \n{label_name} = {label} \nID = {ID}',ha ='left',va='center',transform = ax.transAxes,fontsize=10,weight='bold',color='k',bbox=dict(facecolor='w',alpha=0.5))
            
            
    elif label != None and ID ==None and frame == 'world':
        if vel_label==None:
            plt.text(x_box,y_box,f'Vel = {int(round(current_vel))} km/s \n{label_name} = {label}',ha ='left',va='center',transform = ax.transAxes,fontsize=10,weight='bold',color='k',bbox=dict(facecolor='w',alpha=0.5))
        else:
            plt.text(x_box,y_box,f'Vel = {int(vel_label)} km/s \n{label_name} = {label}',ha ='left',va='center',transform = ax.transAxes,fontsize=10,weight='bold',color='k',bbox=dict(facecolor='w',alpha=0.5))
                
                
    elif label == None and ID == None and frame == 'world':
        if vel_label==None:
            plt.text(x_box,y_box,f'Vel = {int(round(current_vel))} km/s',ha ='left',va='center',transform = ax.transAxes,fontsize=10,weight='bold',color='k',bbox=dict(facecolor='w',alpha=0.5))
        else:
            plt.text(x_box,y_box,f'Vel = {int(vel_label)} km/s',ha ='left',va='center',transform = ax.transAxes,fontsize=10,weight='bold',color='k',bbox=dict(facecolor='w',alpha=0.5))
            
            
    elif label == None and ID != None and frame == 'world':
        if vel_label == None:
            plt.text(x_box,y_box,f'Vel = {int(round(current_vel))} km/s \nID = {ID}',ha ='left',va='center',transform = ax.transAxes,fontsize=10,weight='bold',color='k',bbox=dict(facecolor='w',alpha=0.5))
        else:
            plt.text(x_box,y_box,f'Vel = {int(vel_label)} km/s \nID = {ID}',ha ='left',va='center',transform = ax.transAxes,fontsize=10,weight='bold',color='k',bbox=dict(facecolor='w',alpha=0.5))
        
        
    elif label == None and ID != None and frame=='cartesian':
        plt.text(x_box,y_box,f'Z_chan = {int(round(z))} \nID = {ID}',ha ='left',va='center',transform = ax.transAxes,fontsize=10,weight='bold',color='k',bbox=dict(facecolor='w',alpha=0.5))
        
def sum_region(centre, radius, data_array):
    xs = []
    ys = []
    for i in range(len(data_array)):
        for j in range(len(data_array[0])):
            xs.append(i)
            ys.append(j)
    xs = np.array(xs)
    ys = np.array(ys)
    rs = np.sqrt((centre[0]-xs)**2+(centre[1]-ys)**2) #work out how far away each pixel is from the centre
    cut = np.where(rs<radius)[0]
    xs = xs[cut]
    ys = ys[cut]
    rs = rs[cut]
    
    val = []
    for i in range(len(xs)):
        val.append(data_array[ys[i]][xs[i]])
            
    sum_val = np.nansum(val)
    return sum_val
    
def get_HI_profile(cube,x,y,z,xpad,ypad,radius):
    current_wcs_3d = cube.wcs
    current_ra,current_dec,current_vel = current_wcs_3d.pixel_to_world_values(x,y,z)
    sub_cube = reduce_fits_file(cube, x, y, z, xpad, ypad, zpad_HI) # Cutting the main fits file into a sub-cube
    current_wcs_3d = sub_cube.wcs
    centre_sub_x,centre_sub_y,_ = current_wcs_3d.world_to_pixel_values(current_ra,current_dec,current_vel)
    sub_cube_data = sub_cube.hdu.data
    degree_per_pixel = np.abs(float(sub_cube.header['CDELT1']))
    arcsec_per_pixel = degree_per_pixel*3600 #how many arcsecs in a pixel
    pixel_radius = radius #radius/arcsec_per_pixel #nearest pixel to the desired arcsec
    z_len = len(sub_cube_data)
    xs,ys = np.ones(z_len),np.ones(z_len)
    centre_pix = (int(np.round(centre_sub_x)),int(np.round(centre_sub_y))) # this is the central pixel of the subcube 
    zs = np.arange(z_len)
    _,_,current_vel_range = current_wcs_3d.pixel_to_world_values(xs,ys,zs)
    
    line_intensity_max = [sum_region((centre_sub_x,centre_sub_y),pixel_radius,z_channel) for z_channel in sub_cube_data]     #loop that gives me a sum of everything in the z channel range, in the radius around the (actual) centre
    return current_vel_range/1000,line_intensity_max,current_vel/1000, centre_pix, pixel_radius

######################################
# Plots
######################################

pos_only = True #plot only the detections with positive velocity
grid = False #plot moment maps in grids of 9
sorting = False

if pos_only==True:
    print("taking positive vels only")
    x=x[pos_vels]
    y=y[pos_vels]
    z=z[pos_vels]
    ID=ID[pos_vels]
    rel=rel[pos_vels]
#     ra_sof=ra_sof[pos_vels]
#     dec_sof=dec_sof[pos_vels]
    vel=vel[pos_vels]

if sorting == True: #sort the images
    arg = np.argsort(z)[::-1] #reversing the order
    z,x,y = z[arg],x[arg],y[arg]

###########################
#                         #
#    grid of 9 images     #
#                         #
###########################
    
if grid==True: # mom-maps ONLY, no HI profiles

    if len(ID)%9 == 0: #if len(ID) is a multiple of 9
        max_frame=int(len(ID)/9)
    else:
        max_frame=int(np.trunc(len(ID)/9)+1)#if len(ID) is not a multiple of 9
    frame_counter=1
    file_name = infile.split('/')[-1]
    i = 0
    while i < len(x):
        fig =plt.figure(figsize=(15, 10))
        #plt.rcParams.update({'font.size': 5})
        plt.tight_layout() 
        for counter in range(9): #a grid of 9 plots in a figure
            pos=330+counter+1
            cut_mom_map(cube,x[i],y[i],z[i],xpad,ypad,zpad,pos=pos,order=0,HI_radii=radii,vel_label=vel[i],ID=ID[i])
            i+=1
            if i == len(x):
                break
        print('Saving: ',f'MomMaps_{file_name}_{frame_counter}of{max_frame}.png')
        plt.savefig(f'MomMaps_{file_name}_{frame_counter}of{max_frame}.png')
        frame_counter+=1

###########################
#                         #
#    individual images    #
#                         #
###########################

else: # HI profiles optional  

    print(f'printing {len(x)} moment maps')
    print()
    for i in range(len(x)):
        fig = plt.figure(figsize=(10,10))
        plt.tight_layout()
        cut_mom_map(cube,x[i],y[i],z[i],xpad,ypad,zpad,HI_profile=True,HI_radii=radii,vel_label=vel[i],ID=ID[i]) 
        plt.savefig(f'Mom0_{i+1}_vel={int(vel[i])}.png')
        print(f'Saved Mom0_{i+1}_vel={int(vel[i])}.png')
        print() 
        plt.close()

# To download a folder of PNGs to your local computer:
# navigate into the folder of PNGs
# zip <name>.zip *
# right-click download the zip folder