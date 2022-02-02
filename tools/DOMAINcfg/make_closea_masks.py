#!/usr/local/sci/bin/python2.7

'''
Routine to create closea mask fields based on old NEMO closea index definitions.
Details of the grid and the bathymetry are read in from the domain_cfg.nc file and
the closea_mask* fields are appended to the same domain_cfg.nc file. 

To use this routine:

  1. Provide domain_cfg.nc file for your configuration.

  2. Define closed seas for your configuration in Section 2 
     using indices in the old NEMO style. (Read the comments on 
     indexing in Section 2!). Examples are given for eORCA025 
     (UK version) for the three different options:
        - just defining closed seas (and distribute fluxes over global ocean)
        - defining closed seas with a RNF mapping for the American Great Lakes to the St Laurence Seaway
        - defining closed seas with an EMPMR mapping for the American Great Lakes to the St Laurence Seaway

  3. Choose whether to mask the closea_mask* fields. Not required
     but makes the fields easier to check.

  4. Module can be run in python or from linux command line if you
     change the top line to point to your python installation. If 
     using from command line, type "make_closea_masks.py --help"
     for usage.

@author: Dave Storkey
@date: Dec 2017
'''
import netCDF4 as nc
import numpy as np
import numpy.ma as ma

def make_closea_masks(config=None,domcfg_file=None,mask=None):

#=========================
# 1. Read in domcfg file
#=========================

    if config is None:
        raise Exception('configuration must be specified')

    if domcfg_file is None:
        raise Exception('domain_cfg file must be specified')

    if mask is None:
        mask=False

    domcfg = nc.Dataset(domcfg_file,'r+')
    lon = domcfg.variables['nav_lon'][:]
    lat = domcfg.variables['nav_lat'][:]
    top_level = domcfg.variables['top_level'][0][:]

    nx = top_level.shape[1]
    ny = top_level.shape[0]

    # Generate 2D "i" and "j" fields for use in "where" statements.
    # These are the Fortran indices, counting from 1, so we have to
    # add 1 to np.arange because python counts from 0.

    ones_2d = np.ones((ny,nx))
    ii1d = np.arange(nx)+1
    jj1d = np.arange(ny)+1
    ii2d = ii1d * ones_2d
    jj2d = np.transpose(jj1d*np.transpose(ones_2d)) 
 
#=====================================
# 2. Closea definitions (old style)
#=====================================

    # NB. The model i and j indices defined here are Fortran indices, 
    #     ie. counting from 1 as in the NEMO code. Also the indices
    #     of the arrays (ncsi1 etc) count from 1 in order to match
    #     the Fortran code.
    #     This means that you can cut and paste the definitions from
    #     the NEMO code and change round brackets to square brackets. 
    #     But BEWARE: Fortran array(a:b) == Python array[a:b+1] !!!
    #

    # If use_runoff_box = True then specify runoff area as all sea points within
    # a rectangular area. If use_runoff_box = False then specify a list of points
    # as in the old NEMO code. Default to false.
    use_runoff_box = False

    #================================================================
    if config == 'ORCA2':

        num_closea = 4
        max_runoff_points = 4

        ncsnr = np.zeros(num_closea+1,dtype=np.int)                     ; ncstt = np.zeros(num_closea+1,dtype=np.int)
        ncsi1 = np.zeros(num_closea+1,dtype=np.int)                     ; ncsj1 = np.zeros(num_closea+1,dtype=np.int)
        ncsi2 = np.zeros(num_closea+1,dtype=np.int)                     ; ncsj2 = np.zeros(num_closea+1,dtype=np.int)
        ncsir = np.zeros((num_closea+1,max_runoff_points+1),dtype=np.int) ; ncsjr = np.zeros((num_closea+1,max_runoff_points+1),dtype=np.int)

        # Caspian Sea (spread over globe)
        ncsnr[1]   =   1  ;  ncstt[1]   =   0   
        ncsi1[1]   =  11  ;  ncsj1[1]   = 103
        ncsi2[1]   =  17  ;  ncsj2[1]   = 112

        # Great Lakes - North America - put at St Laurent mouth
        ncsnr[2]   =   1  ;  ncstt[2]   =   2 
        ncsi1[2]   =  97  ;  ncsj1[2]   = 107
        ncsi2[2]   = 103  ;  ncsj2[2]   = 111
        ncsir[2,1] = 110  ;  ncsjr[2,1] = 111           

        # Black Sea (crossed by the cyclic boundary condition)
        # put in Med Sea (north of Aegean Sea)
        ncsnr[3:5] =   4  ;  ncstt[3:5] =   2           
        ncsir[3:5,1] = 171;  ncsjr[3:5,1] = 106     
        ncsir[3:5,2] = 170;  ncsjr[3:5,2] = 106 
        ncsir[3:5,3] = 171;  ncsjr[3:5,3] = 105 
        ncsir[3:5,4] = 170;  ncsjr[3:5,4] = 105 
        # west part of the Black Sea      
        ncsi1[3]   = 174  ;  ncsj1[3]   = 107      
        ncsi2[3]   = 181  ;  ncsj2[3]   = 112      
        # east part of the Black Sea      
        ncsi1[4]   =   2  ;  ncsj1[4]   = 107      
        ncsi2[4]   =   6  ;  ncsj2[4]   = 112      

    #================================================================
    elif config == 'eORCA1':

        num_closea = 1
        max_runoff_points = 1

        ncsnr = np.zeros(num_closea+1,dtype=np.int)                     ; ncstt = np.zeros(num_closea+1,dtype=np.int)
        ncsi1 = np.zeros(num_closea+1,dtype=np.int)                     ; ncsj1 = np.zeros(num_closea+1,dtype=np.int)
        ncsi2 = np.zeros(num_closea+1,dtype=np.int)                     ; ncsj2 = np.zeros(num_closea+1,dtype=np.int)
        ncsir = np.zeros((num_closea+1,max_runoff_points+1),dtype=np.int) ; ncsjr = np.zeros((num_closea+1,max_runoff_points+1),dtype=np.int)

        # Caspian Sea  (spread over the globe)
        ncsnr[1]   = 1    ; ncstt[1]   = 0           
        ncsi1[1]   = 332  ; ncsj1[1]   = 243
        ncsi2[1]   = 344  ; ncsj2[1]   = 275

    #================================================================
    elif config == 'eORCA025_UK':

        num_closea = 10
        max_runoff_points = 1

        ncsnr = np.zeros(num_closea+1,dtype=np.int)                     ; ncstt = np.zeros(num_closea+1,dtype=np.int)
        ncsi1 = np.zeros(num_closea+1,dtype=np.int)                     ; ncsj1 = np.zeros(num_closea+1,dtype=np.int)
        ncsi2 = np.zeros(num_closea+1,dtype=np.int)                     ; ncsj2 = np.zeros(num_closea+1,dtype=np.int)
        ncsir = np.zeros((num_closea+1,max_runoff_points+1),dtype=np.int) ; ncsjr = np.zeros((num_closea+1,max_runoff_points+1),dtype=np.int)

        # Caspian Sea
        ncsnr[1]   = 1    ; ncstt[1]   = 0     
        ncsi1[1]   = 1330 ; ncsj1[1]   = 831
        ncsi2[1]   = 1375 ; ncsj2[1]   = 981

        # Aral Sea
        ncsnr[2]   = 1    ; ncstt[2]   = 0     
        ncsi1[2]   = 1376 ; ncsj1[2]   = 900
        ncsi2[2]   = 1400 ; ncsj2[2]   = 981

        # Azov Sea
        ncsnr[3]   = 1    ; ncstt[3]   = 0     
        ncsi1[3]   = 1284 ; ncsj1[3]   = 908
        ncsi2[3]   = 1304 ; ncsj2[3]   = 933

        # Lake Superior
        ncsnr[4]   = 1    ; ncstt[4]   = 0     
        ncsi1[4]   = 781  ; ncsj1[4]   = 905 
        ncsi2[4]   = 815  ; ncsj2[4]   = 926 

        # Lake Michigan
        ncsnr[5]   = 1    ; ncstt[5]   = 0     
        ncsi1[5]   = 795  ; ncsj1[5]   = 871             
        ncsi2[5]   = 813  ; ncsj2[5]   = 905 

        # Lake Huron part 1
        ncsnr[6]   = 1    ; ncstt[6]   = 0     
        ncsi1[6]   = 814  ; ncsj1[6]   = 882             
        ncsi2[6]   = 825  ; ncsj2[6]   = 905 

        # Lake Huron part 2
        ncsnr[7]   = 1    ; ncstt[7]   = 0     
        ncsi1[7]   = 826  ; ncsj1[7]   = 889             
        ncsi2[7]   = 833  ; ncsj2[7]   = 905 

        # Lake Erie
        ncsnr[8]   = 1    ; ncstt[8]   = 0     
        ncsi1[8]   = 816  ; ncsj1[8]   = 871             
        ncsi2[8]   = 837  ; ncsj2[8]   = 881 

        # Lake Ontario
        ncsnr[9]   = 1    ; ncstt[9]   = 0     
        ncsi1[9]   = 831  ; ncsj1[9]   = 882             
        ncsi2[9]   = 847  ; ncsj2[9]   = 889 

        # Lake Victoria
        ncsnr[10]   = 1    ; ncstt[10]   = 0   
        ncsi1[10]   = 1274 ; ncsj1[10]   = 672 
        ncsi2[10]   = 1289 ; ncsj2[10]   = 687 

    #================================================================
    elif config == 'eORCA025_UK_rnf':

        num_closea = 10
        max_runoff_points = 1
        use_runoff_box = True

        ncsnr = np.zeros(num_closea+1,dtype=np.int)                     ; ncstt = np.zeros(num_closea+1,dtype=np.int)
        ncsi1 = np.zeros(num_closea+1,dtype=np.int)                     ; ncsj1 = np.zeros(num_closea+1,dtype=np.int)
        ncsi2 = np.zeros(num_closea+1,dtype=np.int)                     ; ncsj2 = np.zeros(num_closea+1,dtype=np.int)
        ncsir1 = np.zeros(num_closea+1,dtype=np.int)                    ; ncsjr1 = np.zeros(num_closea+1,dtype=np.int)
        ncsir2 = np.zeros(num_closea+1,dtype=np.int)                    ; ncsjr2 = np.zeros(num_closea+1,dtype=np.int)
        ncsir = np.zeros((num_closea+1,max_runoff_points+1),dtype=np.int) ; ncsjr = np.zeros((num_closea+1,max_runoff_points+1),dtype=np.int)

        # Caspian Sea
        ncsnr[1]   = 1    ; ncstt[1]   = 0     
        ncsi1[1]   = 1330 ; ncsj1[1]   = 831
        ncsi2[1]   = 1375 ; ncsj2[1]   = 981

        # Aral Sea
        ncsnr[2]   = 1    ; ncstt[2]   = 0     
        ncsi1[2]   = 1376 ; ncsj1[2]   = 900
        ncsi2[2]   = 1400 ; ncsj2[2]   = 981

        # Azov Sea
        ncsnr[3]   = 1    ; ncstt[3]   = 0     
        ncsi1[3]   = 1284 ; ncsj1[3]   = 908
        ncsi2[3]   = 1304 ; ncsj2[3]   = 933

        # Lake Superior
        ncsnr[4]   = 1    ; ncstt[4]   = 1     
        ncsi1[4]   = 781  ; ncsj1[4]   = 905 
        ncsi2[4]   = 815  ; ncsj2[4]   = 926 
        # runff points the St Laurence Seaway for all Great Lakes
        ncsir1[4:10]   = 873 ; ncsjr1[4:10]   = 909 
        ncsir2[4:10]   = 884 ; ncsjr2[4:10]   = 920 

        # Lake Michigan
        ncsnr[5]   = 1    ; ncstt[5]   = 1     
        ncsi1[5]   = 795  ; ncsj1[5]   = 871             
        ncsi2[5]   = 813  ; ncsj2[5]   = 905 

        # Lake Huron part 1
        ncsnr[6]   = 1    ; ncstt[6]   = 1     
        ncsi1[6]   = 814  ; ncsj1[6]   = 882             
        ncsi2[6]   = 825  ; ncsj2[6]   = 905 

        # Lake Huron part 2
        ncsnr[7]   = 1    ; ncstt[7]   = 1     
        ncsi1[7]   = 826  ; ncsj1[7]   = 889             
        ncsi2[7]   = 833  ; ncsj2[7]   = 905 

        # Lake Erie
        ncsnr[8]   = 1    ; ncstt[8]   = 1     
        ncsi1[8]   = 816  ; ncsj1[8]   = 871             
        ncsi2[8]   = 837  ; ncsj2[8]   = 881 

        # Lake Ontario
        ncsnr[9]   = 1    ; ncstt[9]   = 1     
        ncsi1[9]   = 831  ; ncsj1[9]   = 882             
        ncsi2[9]   = 847  ; ncsj2[9]   = 889 

        # Lake Victoria
        ncsnr[10]   = 1    ; ncstt[10]   = 0   
        ncsi1[10]   = 1274 ; ncsj1[10]   = 672 
        ncsi2[10]   = 1289 ; ncsj2[10]   = 687 

    #================================================================
    elif config == 'eORCA025_UK_empmr':

        num_closea = 10
        max_runoff_points = 1
        use_runoff_box = True

        ncsnr = np.zeros(num_closea+1,dtype=np.int)                     ; ncstt = np.zeros(num_closea+1,dtype=np.int)
        ncsi1 = np.zeros(num_closea+1,dtype=np.int)                     ; ncsj1 = np.zeros(num_closea+1,dtype=np.int)
        ncsi2 = np.zeros(num_closea+1,dtype=np.int)                     ; ncsj2 = np.zeros(num_closea+1,dtype=np.int)
        ncsir1 = np.zeros(num_closea+1,dtype=np.int)                    ; ncsjr1 = np.zeros(num_closea+1,dtype=np.int)
        ncsir2 = np.zeros(num_closea+1,dtype=np.int)                    ; ncsjr2 = np.zeros(num_closea+1,dtype=np.int)
        ncsir = np.zeros((num_closea+1,max_runoff_points+1),dtype=np.int) ; ncsjr = np.zeros((num_closea+1,max_runoff_points+1),dtype=np.int)

        # Caspian Sea
        ncsnr[1]   = 1    ; ncstt[1]   = 0     
        ncsi1[1]   = 1330 ; ncsj1[1]   = 831
        ncsi2[1]   = 1375 ; ncsj2[1]   = 981

        # Aral Sea
        ncsnr[2]   = 1    ; ncstt[2]   = 0     
        ncsi1[2]   = 1376 ; ncsj1[2]   = 900
        ncsi2[2]   = 1400 ; ncsj2[2]   = 981

        # Azov Sea
        ncsnr[3]   = 1    ; ncstt[3]   = 0     
        ncsi1[3]   = 1284 ; ncsj1[3]   = 908
        ncsi2[3]   = 1304 ; ncsj2[3]   = 933

        # Lake Superior
        ncsnr[4]   = 1    ; ncstt[4]   = 2     
        ncsi1[4]   = 781  ; ncsj1[4]   = 905
        ncsi2[4]   = 815  ; ncsj2[4]   = 926 
        # runff points the St Laurence Seaway for all Great Lakes
        ncsir1[4:10]   = 873 ; ncsjr1[4:10]   = 909 
        ncsir2[4:10]   = 884 ; ncsjr2[4:10]   = 920 

        # Lake Michigan
        ncsnr[5]   = 1    ; ncstt[5]   = 2     
        ncsi1[5]   = 795  ; ncsj1[5]   = 871             
        ncsi2[5]   = 813  ; ncsj2[5]   = 905 

        # Lake Huron part 1
        ncsnr[6]   = 1    ; ncstt[6]   = 2     
        ncsi1[6]   = 814  ; ncsj1[6]   = 882             
        ncsi2[6]   = 825  ; ncsj2[6]   = 905 

        # Lake Huron part 2
        ncsnr[7]   = 1    ; ncstt[7]   = 2     
        ncsi1[7]   = 826  ; ncsj1[7]   = 889             
        ncsi2[7]   = 833  ; ncsj2[7]   = 905 

        # Lake Erie
        ncsnr[8]   = 1    ; ncstt[8]   = 2     
        ncsi1[8]   = 816  ; ncsj1[8]   = 871             
        ncsi2[8]   = 837  ; ncsj2[8]   = 881 

        # Lake Ontario
        ncsnr[9]   = 1    ; ncstt[9]   = 2     
        ncsi1[9]   = 831  ; ncsj1[9]   = 882             
        ncsi2[9]   = 847  ; ncsj2[9]   = 889 

        # Lake Victoria
        ncsnr[10]   = 1    ; ncstt[10]   = 0   
        ncsi1[10]   = 1274 ; ncsj1[10]   = 672 
        ncsi2[10]   = 1289 ; ncsj2[10]   = 687 

#=====================================
# 3. Generate mask fields
#=====================================

    rnf_count = 0
    empmr_count = 0

    closea_mask = ma.zeros(top_level.shape,dtype=np.int)
    temp_mask_rnf = ma.zeros(top_level.shape,dtype=np.int)
    temp_mask_empmr = ma.zeros(top_level.shape,dtype=np.int)
    closea_mask_rnf = ma.zeros(top_level.shape,dtype=np.int)
    closea_mask_empmr = ma.zeros(top_level.shape,dtype=np.int)

    for ics in range(num_closea):
        closea_mask = ma.where( ( ii2d[:] >= ncsi1[ics+1] ) & ( ii2d[:] <= ncsi2[ics+1] ) &
                                ( jj2d[:] >= ncsj1[ics+1] ) & ( jj2d[:] <= ncsj2[ics+1] ) &
                                ( top_level == 1 ), ics+1, closea_mask)
        if ncstt[ics+1] == 1:
            rnf_count = rnf_count + 1
            temp_mask_rnf[:] = 0
            if use_runoff_box:
                temp_mask_rnf = ma.where( ( ii2d[:] >= ncsir1[ics+1] ) & ( ii2d[:] <= ncsir2[ics+1] ) &
                                       ( jj2d[:] >= ncsjr1[ics+1] ) & ( jj2d[:] <= ncsjr2[ics+1] ) &
                                       ( top_level == 1 ), rnf_count, 0)
            else:
                for ir in range(ncsnr[ics+1]):
                    temp_mask_rnf[ncsjr[ics+1],ncsjr[ics+1]] = rnf_count
 
            temp_mask_rnf = ma.where( closea_mask_rnf > 0, ma.minimum(temp_mask_rnf,closea_mask_rnf), temp_mask_rnf)
            min_rnf = ma.amin(temp_mask_rnf[ma.where(temp_mask_rnf > 0)])
            max_rnf = ma.amax(temp_mask_rnf[ma.where(temp_mask_rnf > 0)])
            if min_rnf != max_rnf:
                print 'min_rnf, max_rnf : ',min_rnf,max_rnf
                raise Exception('Partially overlapping target rnf areas for two closed seas.')
            else:
                # source area:
                closea_mask_rnf[ma.where(closea_mask==ics+1)] = min_rnf
                # target area:
                closea_mask_rnf[ma.where(temp_mask_rnf>0)] = min_rnf
                # reset rnf_count:
                rnf_count = min_rnf
                    
        if ncstt[ics+1] == 2:
            empmr_count = empmr_count + 1
            temp_mask_empmr[:] = 0
            if use_runoff_box:
                temp_mask_empmr = ma.where( ( ii2d[:] >= ncsir1[ics+1] ) & ( ii2d[:] <= ncsir2[ics+1] ) &
                                          ( jj2d[:] >= ncsjr1[ics+1] ) & ( jj2d[:] <= ncsjr2[ics+1] ) &
                                          ( top_level == 1 ), empmr_count, 0)
            else:
                for ir in range(ncsnr[ics+1]):
                    temp_mask_empmr[ncsjr[ics+1],ncsjr[ics+1]] = empmr_count

            temp_mask_empmr = ma.where( closea_mask_empmr > 0, ma.minimum(temp_mask_empmr,closea_mask_empmr), temp_mask_empmr)
            min_empmr = ma.amin(temp_mask_empmr[ma.where(temp_mask_empmr > 0)])
            max_empmr = ma.amax(temp_mask_empmr[ma.where(temp_mask_empmr > 0)])
            if min_empmr != max_empmr:
                raise Exception('Partially overlapping target empmr areas for two closed seas.')
            else:
                # source area:
                closea_mask_empmr[ma.where(closea_mask==ics+1)] = min_empmr
                # target area:
                closea_mask_empmr[ma.where(temp_mask_empmr>0)] = min_empmr
                # reset empmr_count:
                empmr_count = min_empmr
                    
    if mask:
        # apply land-sea mask if required
        closea_mask.mask = np.where(top_level==0,True,False)
        closea_mask_rnf.mask = np.where(top_level==0,True,False)
        closea_mask_empmr.mask = np.where(top_level==0,True,False)

#=====================================
# 4. Append masks to domain_cfg file.
#=====================================

    domcfg.createVariable('closea_mask',datatype='i',dimensions=('y','x'),fill_value=closea_mask.fill_value,chunksizes=(1000,1000))
    domcfg.variables['closea_mask'][:]=closea_mask
    if rnf_count > 0:
        domcfg.createVariable('closea_mask_rnf',datatype='i',dimensions=('y','x'),fill_value=closea_mask_rnf.fill_value,chunksizes=(1000,1000))
        domcfg.variables['closea_mask_rnf'][:]=closea_mask_rnf
    if empmr_count > 0:
        domcfg.createVariable('closea_mask_empmr',datatype='i',dimensions=('y','x'),fill_value=closea_mask_empmr.fill_value,chunksizes=(1000,1000))
        domcfg.variables['closea_mask_empmr'][:]=closea_mask_empmr

    domcfg.close()


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", action="store",dest="config",default=None,
                    help="configuration: eORCA1, eORCA025_UK")
    parser.add_argument("-d", "--domcfg", action="store",dest="domcfg_file",default=None,
                    help="domcfg file (input)")
    parser.add_argument("-m", "--mask", action="store_true",dest="mask",default=False,
                    help="mask output file based on top_level in domcfg file")

    args = parser.parse_args()

    make_closea_masks(config=args.config,domcfg_file=args.domcfg_file,mask=args.mask)
