#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make Moment Maps
Created on Wed May  6 11:37:46 2020

This program will generate moment maps of a given list of detections 

"""

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits  # We use fits to open the actual data file
from spectral_cube import SpectralCube
import matplotlib.gridspec as gridspec
import warnings
from astropy.wcs import WCS
from astropy.wcs import FITSFixedWarning
warnings.simplefilter('ignore', FITSFixedWarning)
#module add python/3.8.3
#comment out block: cmd + /
#cancel the run control+c

######################################
#FITS cube
######################################

#Choose FITS cube:

hdu = fits.open('/carta_share/users/nadia/Nadia_GA/1538811061_HI_mosaic_heliocen.fits') # shape: 3788, 3474, 573 
cube = SpectralCube.read(hdu) # Initiate a SpectralCube

#convert cube to VRAD m/s:
print('CONVERTING CUBE INTO VELOCITY UNITS...')
cube = cube.with_spectral_unit(u.m/u.s,velocity_convention='radio',rest_value=1420.4*u.MHz)
print(f"Cube now has spectral unit {cube.header['CTYPE3']} {cube.header['CUNIT3']}")
hdu.close()

#Cube now has dimensions [Vel, Dec, RA]
_, Dec, _ = cube.world[0, :, 0]  #extract Dec world coordinates from cube
_, _, RA = cube.world[0, 0, :]  #extract RA world coordinates from cube


######################################
#Input
######################################

rel = np.ones(1000000) # a fake array of rels in case the the user doesn't have rels
ID = np.ones(1000000)

#Choose detection list:
infile = '/users/nadia/moment_maps/1538811061_mosaic/Visual_sources/1538811061_visual.txt'
vel, x, y, z = np.loadtxt(infile,usecols=(6,7,8,9),unpack=True) #vel in km/s
vel = np.around(vel,0)

#vel = 300000*((-freq + 1420.4e6)/(1420.4e6)) #if your list has freq
xpad=35
ypad=35
zpad=2
radii=15 #radius used around source for HI profile

######################################

if type(x) == float or type(x) == int:
    x,y,z = np.array([x]), np.array([y]), np.array([z])

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


def cut_mom_map(cube,x,y,z,xpad,ypad,zpad,pos=111,label=None,ID=None,frame='world',clean=False, cleaning_factor=1,order=0,vmin=None,vmax=None,HI_profile=False,HI_radii=radii):
    
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
    
    # Initiate an axis object with WCS projection information
    
    if HI_profile == True:
        gs = gridspec.GridSpec(ncols=20,nrows=25,figure=fig)
        gs.update(wspace=0)
        ax = fig.add_subplot(gs[0:17,0:16], projection=moment_map.wcs) # POINTS ARE NOT IN RA/DEC but the AXES ARE
        ax_HI = fig.add_subplot(gs[19:,0:20])
        hx,hy,current_vel_HI,HI_centre,HI_radius=get_HI_profile(cube, x, y, z, xpad, ypad,radius=HI_radii)
        # Display the HI profile
        ax_HI.plot(hx, hy,'o-', color='k')
        ax_HI.axvline(current_vel,ls='--',color='k',alpha=0.5)
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
        cbar.set_label('Brightness (Jy km/s /beam)', size=10, rotation=90,labelpad=11) 
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
    ax.minorticks_on()
    #ax.tick_params(axis='both',labelsize=10)
    ax.tick_params(which='both', width=1, labelsize=10)
    ax.tick_params(which='both',direction='in')
    ax.tick_params(which='major', length=5)#, right=True)
    ax.tick_params(which='minor', length=2)#, right=True)
        
    # Add labels:
    x_box,y_box = 0.03,0.87
    if label == None and ID == None and frame=='cartesian':
        plt.text(x_box,y_box,f'Z_chan = {int(round(z))}',ha ='left',va='center',transform = ax.transAxes,fontsize=10,weight='bold',color='k',bbox=dict(facecolor='w',alpha=0.5))
    elif label != None and ID == None and frame=='cartesian':
        plt.text(x_box,y_box,f'Z_chan = {int(round(z))} \nrel = {round(label,4)}',ha ='left',va='center',transform = ax.transAxes,fontsize=10,weight='bold',color='k',bbox=dict(facecolor='w',alpha=0.5))
    elif label != None and ID != None and frame=='cartesian':
        plt.text(x_box,y_box,f'Z_chan = {int(round(z))} \nrel = {round(label,4)} \nID = {int(ID)}',ha ='left',va='center',transform = ax.transAxes,fontsize=10,weight='bold',color='k',bbox=dict(facecolor='w',alpha=0.5))
    elif label !=None and ID != None and frame == 'world':
        plt.text(x_box,y_box,f'Vel = {int(round(current_vel))} km/s \nrel = {round(label,4)} \nID = {int(ID)}',ha ='left',va='center',transform = ax.transAxes,fontsize=10,weight='bold',color='k',bbox=dict(facecolor='w',alpha=0.5))
    elif label != None and ID ==None and frame == 'world':
        plt.text(x_box,y_box,f'Vel = {int(round(current_vel))} km/s \nrel = {round(label,4)}',ha ='left',va='center',transform = ax.transAxes,fontsize=10,weight='bold',color='k',bbox=dict(facecolor='w',alpha=0.5))
    elif label == None and ID == None and frame == 'world':
        plt.text(x_box,y_box,f'Vel = {int(round(current_vel))} km/s',ha ='left',va='center',transform = ax.transAxes,fontsize=10,weight='bold',color='k',bbox=dict(facecolor='w',alpha=0.5))
    elif label == None and ID != None and frame == 'world':
        plt.text(x_box,y_box,f'Vel = {int(round(current_vel))} km/s \nID = {int(ID)}',ha ='left',va='center',transform = ax.transAxes,fontsize=10,weight='bold',color='k',bbox=dict(facecolor='w',alpha=0.5))
    elif label == None and ID != None and frame=='cartesian':
        plt.text(x_box,y_box,f'Z_chan = {int(round(z))} \nID = {int(ID)}',ha ='left',va='center',transform = ax.transAxes,fontsize=10,weight='bold',color='k',bbox=dict(facecolor='w',alpha=0.5))

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
    sub_cube = reduce_fits_file(cube, x, y, z, xpad, ypad, zpad) # Cutting the main fits file into a sub-cube
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
    
    current_vel_range = current_vel_range
    
    line_intensity_max = [sum_region((centre_sub_x,centre_sub_y),pixel_radius,z_channel) for z_channel in sub_cube_data] #loop that gives me a sum of everything in the z channel range, in the radius around the ACTUAL centre
    return current_vel_range/1000,line_intensity_max,current_vel/1000, centre_pix, pixel_radius

######################################
# Plots
######################################

pos_only = False #plot only the detections with positive velocity
grid = True #plot moment maps in grids of 9
sorting = False

if pos_only==True:
    print("taking positives only")
    x=x[pos_vels]
    y=y[pos_vels]
    z=z[pos_vels]
    ID=ID[pos_vels]
    rel=rel[pos_vels]
    ra_sof=ra_sof[pos_vels]
    dec_sof=dec_sof[pos_vels]
    vel=vel[pos_vels]

if sorting == True:
    arg = np.argsort(z)[::-1]	#sort the images
    z,x,y = z[arg],x[arg],y[arg]

if grid==True:
### Print images in grids of 9 (mom-maps ONLY - can't accommodate HI profiles at this stage)

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
            cut_mom_map(cube,x[i],y[i],z[i],xpad,ypad,zpad,pos=pos,order=0,vmin=0,HI_radii=radii) #label=rel[i],ID=ID[i]
            i+=1
            if i == len(x):
                break
        print('Saving: ',f'MomMaps_{file_name}_{frame_counter}of{max_frame}.png')
        plt.savefig(f'MomMaps_{file_name}_{frame_counter}of{max_frame}.png')
        frame_counter+=1

else: # Print individual images (HI profiles optional)  

    print(f'printing {len(x)} moment maps')
    print()
    for i in range(len(x)):
        fig = plt.figure(figsize=(10,10))
        plt.tight_layout()
        cut_mom_map(cube,x[i],y[i],z[i],xpad,ypad,zpad,vmin=0,HI_profile=True,HI_radii=radii)
        #remove label=rel[i],ID=ID[i] if not relevant
        plt.savefig(f'Mom_HI_{i+1}.png')
        print(f'Saved Mom_HI_{i+1}.png')
        print() 
        plt.close()
