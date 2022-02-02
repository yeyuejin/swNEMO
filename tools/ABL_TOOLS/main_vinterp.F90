PROGRAM main
   !!======================================================================
   !!                     ***  PROGRAM main  ***
   !!
   !! ** Purpose : Vertical interpolation of ECMWF dataset on a given fixed 
   !!              vertical grid
   !! 
   !!======================================================================
   !! History : 2016-10  (F. Lemarié)  Original code
   !!   
   !!----------------------------------------------------------------------
   USE module_io       ! I/O routines
   USE module_interp   ! vertical interpolation routines
   USE module_grid     ! compute input and output grids 
   !!
   IMPLICIT NONE
   !!----------------------------------------------------------------------
   !! 
   !!  
   !! 
   !!----------------------------------------------------------------------
   !
   INTEGER                                  :: ji,jj,jk,kt, jk_in, nhym, nhyi
   INTEGER                                  :: jpka_in, jpka      ! number of vertical levels for input and target grids 
   INTEGER                                  :: jpi , jpj          ! number of grid points in x and y directions     
   INTEGER                                  :: iloc, jloc         ! grid indexes for c1d case
   INTEGER                                  :: status             
   INTEGER                                  :: jptime,ctrl
   INTEGER                                  :: ioerr
   INTEGER, ALLOCATABLE, DIMENSION(:,:  )   :: ind  
   INTEGER, PARAMETER                       :: stdout  = 6
   INTEGER, PARAMETER                       :: jp_weno = 1
   INTEGER, PARAMETER                       :: jp_spln = 2   
   !!
   REAL(8)                                  :: hc,hmax,theta_s,z1 ! parameters related to the target vertical grid
   REAL(8)                                  :: cff
   !!  
   REAL(8), ALLOCATABLE, DIMENSION(:      ) :: A_w                ! A coefficients to reconstruct ECMWF grid
   REAL(8), ALLOCATABLE, DIMENSION(:      ) :: A_wa               ! A coefficients to reconstruct ECMWF grid
   REAL(8), ALLOCATABLE, DIMENSION(:      ) :: B_w                ! B coefficients to reconstruct ECMWF grid
   REAL(8), ALLOCATABLE, DIMENSION(:      ) :: B_wa               ! B coefficients to reconstruct ECMWF grid
   REAL(8), ALLOCATABLE, DIMENSION(:      ) :: tmp1d, tmp_fullw, tmp_fullm ! temporary/working 1D arrays
   REAL(8), ALLOCATABLE, DIMENSION(:      ) :: e3t,e3w            ! thickness of vertical layers in target grid
   REAL(8), ALLOCATABLE, DIMENSION(:      ) :: ght,ghw            ! altitude of vertical grid points 
   REAL(8), ALLOCATABLE, DIMENSION(:,:    ) :: e3_bak
   REAL(8), ALLOCATABLE, DIMENSION(:,:,:  ) :: ghw_in             ! altitude of cell interfaces  of ECMWF grid
   REAL(8), ALLOCATABLE, DIMENSION(:,:,:  ) :: e3t_in             ! thickness of vertical layers in ECMWF grid
   REAL(8), ALLOCATABLE, DIMENSION(:,:,:,:) :: humi
   REAL(8), ALLOCATABLE, DIMENSION(:,:,:,:) :: tair, tpot
   REAL(8), ALLOCATABLE, DIMENSION(:,:,:,:) :: varout, varc1d 
   REAL(8), ALLOCATABLE, DIMENSION(:,:    ) :: slp, zsurf
   REAL(8), ALLOCATABLE, DIMENSION(:,:    ) :: tmask, tmask2      ! land-sea mask  
   !!
   CHARACTER(len=500)                       :: file_u,file_v,file_hpg,file_geos  ! ECMWF files containing wind components 
   CHARACTER(len=500)                       :: file_t,file_q, file_m             ! ECMWF files containing tracers and mask 
   CHARACTER(len=500)                       :: file_z,file_p,cn_dir,file_in      ! ECMWF files containing surface geopot and pressure
   CHARACTER(len=500)                       :: grd_file, abl_file, drwn_file, out_file
   CHARACTER(len=500)                       :: namelistf,stmp
   CHARACTER(len=500)                       :: argument, var_file
   CHARACTER(len= 20),DIMENSION(4)          :: dimnames
   CHARACTER(len= 20),DIMENSION(11)         :: varnames, outnames
   CHARACTER(len=500),DIMENSION(11)         :: filnames
   CHARACTER(6)                             :: mask_var           ! name of mask variable in file_m file            
   CHARACTER(6)                             :: var_name
   !!
   LOGICAL                                  :: ln_read_zsurf      ! read surface geopotential or not
   LOGICAL                                  :: ln_read_mask       ! read land-sea mask or not
   LOGICAL                                  :: ln_perio_latbc     ! use periodic BC along the domain latitudinal edges (for global data) or use zero-gradient BC (for regional data)
   LOGICAL                                  :: ln_c1d             ! output only a single column in output file
   LOGICAL                                  :: ln_hpg_frc         ! compute horizontal pressure gradient
   LOGICAL                                  :: ln_geo_wnd         ! compute goestrophic wind components 
   LOGICAL                                  :: ln_slp_smth        ! apply gibbs oscillation filetring on mean sea level pressure
   LOGICAL                                  :: ln_drw_smth        ! apply gibbs oscillation filetring on mean sea level pressure
   LOGICAL                                  :: ln_slp_log         ! log(sea-level pressure) or sea-level pressure
   LOGICAL                                  :: ln_lsm_land        ! if T mask is 1 over land and 0 over ocean if F it is the other way around
   LOGICAL                                  :: ln_impose_z1       ! impose the altitude of the first level in target grid
   INTEGER                                  :: ptemp_method       ! way to compute potential temperature 
                                                                  ! = 0  (absolute temperature)
                                                                  ! = 1  (potential temperature with local ref pressure)
                                                                  ! = 2  (potential temperature with global ref pressure on temperature perturbation)
                                                                  ! = 3  (potential temperature with global ref pressure)
   !!   
   REAL(8), PARAMETER     :: grav  =   9.80665

   !!---------------------------------------------------------------------
   !! List of variables read in the namelist file 
   NAMELIST/nml_dom/    jpka, hmax, theta_s, hc, ln_impose_z1, z1   
   NAMELIST/nml_opt/    ptemp_method, ln_slp_log, ln_slp_smth, ln_read_mask, ln_perio_latbc, &
      &                 ln_hpg_frc, ln_geo_wnd, ln_c1d, ln_read_zsurf, ln_lsm_land, ln_drw_smth
   NAMELIST/nml_fld/    cn_dir, file_u, file_v, file_t,               &
      &                 file_q, file_z, file_p, file_hpg, file_geos,  &
      &                 file_m, mask_var
   NAMELIST/nml_out/    grd_file, abl_file, drwn_file, var_name
   NAMELIST/nml_c1d/    iloc, jloc
   !!
   !! get the namelist file name
   CALL get_command_argument( 1, argument, ctrl, status)
   !
   SELECT CASE(status)
   CASE(0)
      namelistf = trim(argument)
   CASE(-1)
      WRITE(stdout,*) "### Error: file name too long"
      STOP
   CASE DEFAULT
      namelistf = 'namelist_abl_tools'
   END SELECT
   !!---------------------------------------------------------------------


   !!---------------------------------------------------------------------
   !! read namelist variables
   ctrl = 0
   OPEN(50, file=namelistf, status='old', form='formatted', access='sequential', iostat=ioerr)
   IF (ioerr /= 0) ctrl = ctrl + 1   
   READ(50,nml_dom, iostat=ioerr); IF (ioerr /= 0) ctrl = ctrl + 1   
   READ(50,nml_opt, iostat=ioerr); IF (ioerr /= 0) ctrl = ctrl + 1
   READ(50,nml_fld, iostat=ioerr); IF (ioerr /= 0) ctrl = ctrl + 1
   READ(50,nml_out, iostat=ioerr); IF (ioerr /= 0) ctrl = ctrl + 1  
   IF( ln_c1d ) THEN
      print*,'c1d is activated'
      READ(50,nml_c1d, iostat=ioerr); IF (ioerr /= 0) ctrl = ctrl + 1
   ENDIF

   IF (ctrl > 0) THEN
      WRITE(stdout,*) "### E R R O R while reading namelist file '",trim(namelistf),"'"
      WRITE(stdout,*) " ctrl = ",ctrl
      STOP
   ELSE 
      WRITE(stdout,*) " Namelist file ",trim(namelistf)," OK "    
   ENDIF
   IF( ln_hpg_frc .AND. ln_geo_wnd ) THEN
      WRITE(stdout,*) "### E R R O R conflicting options "
      WRITE(stdout,*) "ln_hpg_frc and ln_geo_wnd can not both be set to True"
      STOP   
   ENDIF
   
   SELECT CASE (ptemp_method) 
   CASE(0)
      WRITE(stdout,*) "Absolute temperature option is activated"
   CASE(1)   
      WRITE(stdout,*) "Potential temperature option with  local reference pressure is activated"   
   CASE(2)    
      WRITE(stdout,*) "Potential temperature option with global reference pressure on temperature perturbation is activated"   
   CASE(3)    
      WRITE(stdout,*) "Potential temperature option with global reference pressure is activated"  
   END SELECT
   
   IF(ln_slp_smth) WRITE(stdout,*) "MSLP smoothing option is activated" 
   IF(ln_hpg_frc ) WRITE(stdout,*) "Large-scale pressure gradient will be interpolated"      
   IF(ln_geo_wnd ) WRITE(stdout,*) "Geostrophic winds will be interpolated" 
   !!---------------------------------------------------------------------   
       
   !!-------------------------------------------------------------------------------------   
   !! list of variables to treat
   !!
   !! get the variable name
   CALL get_command_argument( 2, argument, ctrl, status)
   SELECT CASE(status)
   CASE(0)
      var_name = trim(argument)
   CASE(-1)
      WRITE(stdout,*) "### Error: file name too long"
      STOP
   END SELECT
   WRITE(stdout,*) "var_name: ", trim(var_name), "lenght: ", Len_Trim(var_name)
   !!
   IF( ln_hpg_frc ) THEN 
      file_in = trim(file_hpg )
   ELSE
      file_in = trim(file_geos)
   ENDIF
   varnames = [character(len=4  ) :: 'Z'    ,    'T',    'Q',    'U',    'V', 'MSL', 'LSM', 'uhpg', 'vhpg', 'ugeo', 'vgeo' ]
   outnames = [character(len=5  ) :: 'zsurf', 'tair', 'humi', 'uwnd', 'vwnd', 'slp',    '', 'uhpg', 'vhpg', 'ugeo', 'vgeo' ]
   filnames = [character(len=500) ::  file_z, file_t, file_q, file_u, file_v, file_p, file_m, file_in, file_in, file_in, file_in ]
   IF( ln_slp_log ) varnames(6) = 'LNSP'
 
   !!---------------------------------------------------------------------
   ! check files content
   ctrl = 0
   CALL Read_Ncdf_dim('lev',trim(cn_dir)//'/'//trim(file_t),jpka_in)   
   !
   IF (ln_read_zsurf) THEN
     IF( .not. VAR_EXISTENCE( trim(varnames(1)) , trim(cn_dir)//'/'//trim(file_z) )      &
        & .or. jpka_in == 1 ) ctrl = ctrl + 1 
   ENDIF
   WRITE(stdout,*) trim(varnames(1)), ctrl
   IF ( .not. VAR_EXISTENCE( trim(varnames(2)) , trim(cn_dir)//'/'//trim(file_t) ) ) ctrl = ctrl + 1 
   WRITE(stdout,*) trim(varnames(2)), ctrl
   IF ( .not. VAR_EXISTENCE( trim(varnames(3)) , trim(cn_dir)//'/'//trim(file_q) ) ) ctrl = ctrl + 1   
   WRITE(stdout,*) trim(varnames(3)), ctrl
   IF ( .not. VAR_EXISTENCE( trim(varnames(4)) , trim(cn_dir)//'/'//trim(file_u) ) ) ctrl = ctrl + 1
   WRITE(stdout,*) trim(varnames(4)), ctrl
   IF ( .not. VAR_EXISTENCE( trim(varnames(5)) , trim(cn_dir)//'/'//trim(file_v) ) ) ctrl = ctrl + 1 
   WRITE(stdout,*) trim(varnames(5)), ctrl
   IF ( .not. VAR_EXISTENCE( trim(varnames(6)) , trim(cn_dir)//'/'//trim(file_p) ) ) ctrl = ctrl + 1 
   WRITE(stdout,*) trim(varnames(6)), ctrl
   IF (ln_read_mask) THEN
     IF ( .not. VAR_EXISTENCE( trim(varnames(7)) , trim(cn_dir)//'/'//trim(file_m) ) ) ctrl = ctrl + 1 
     WRITE(stdout,*) trim(varnames(7)), ctrl
     varnames(7) = TRIM(mask_var)
   ENDIF
   IF (ln_hpg_frc) THEN
      IF ( .not. VAR_EXISTENCE( trim(varnames(8)) , trim(cn_dir)//'/'//trim(file_hpg) ) ) ctrl = ctrl + 1 
      WRITE(stdout,*) trim(varnames(8)), ctrl
      IF ( .not. VAR_EXISTENCE( trim(varnames(9)) , trim(cn_dir)//'/'//trim(file_hpg) ) ) ctrl = ctrl + 1    
      WRITE(stdout,*) trim(varnames(9)), ctrl
   ENDIF
   IF (ln_geo_wnd) THEN
      IF ( .not. VAR_EXISTENCE( trim(varnames(10)) , trim(cn_dir)//'/'//trim(file_geos) ) ) ctrl = ctrl + 1 
      WRITE(stdout,*) trim(varnames(10)), ctrl
      IF ( .not. VAR_EXISTENCE( trim(varnames(11)) , trim(cn_dir)//'/'//trim(file_geos) ) ) ctrl = ctrl + 1    
      WRITE(stdout,*) trim(varnames(11)), ctrl
   ENDIF
          
   IF ( ctrl > 0 ) THEN  
      WRITE(stdout,*) "### E R R O R while reading ECMWF atmospheric files "
      STOP
   ELSE
      WRITE(stdout,*) " ECMWF atmospheric files OK "        
   ENDIF
   !!---------------------------------------------------------------------


   !!---------------------------------------------------------------------
   !! read the dimensions for the input files
   CALL Read_Ncdf_dim ( 'time', trim(cn_dir)//'/'//trim(file_t), jptime  ) 
   CALL Read_Ncdf_dim ( 'lon' , trim(cn_dir)//'/'//trim(file_t), jpi     )     
   CALL Read_Ncdf_dim ( 'lat' , trim(cn_dir)//'/'//trim(file_t), jpj     )     
   CALL Read_Ncdf_dim ( 'nhym' , trim(cn_dir)//'/'//trim(file_t), nhym   )
   CALL Read_Ncdf_dim ( 'nhyi' , trim(cn_dir)//'/'//trim(file_t), nhyi   )
   !WRITE(stdout,*) "jpka_in, jptime, jpi, jpj, nhym, nhyi: ", jpka_in, jptime, jpi, jpj, nhym, nhyi
   !
   !!---------------------------------------------------------------------


   !!---------------------------------------------------------------------
   !! allocate arrays  
   ALLOCATE( A_w    (               0:jpka_in) )         
   ALLOCATE( B_w    (               0:jpka_in) )
   ALLOCATE( e3t_in ( 1:jpi, 1:jpj, 1:jpka_in) )
   ALLOCATE( ghw_in ( 1:jpi, 1:jpj, 0:jpka_in) ) 
   ALLOCATE( slp    ( 1:jpi, 1:jpj              ) )
   ALLOCATE( zsurf  ( 1:jpi, 1:jpj              ) )
   ALLOCATE( tair   ( 1:jpi, 1:jpj, 1:jpka_in, 1) )
   ALLOCATE( tpot   ( 1:jpi, 1:jpj, 1:jpka_in, 1) )
   ALLOCATE( humi   ( 1:jpi, 1:jpj, 1:jpka_in, 1) )       
   ALLOCATE( varout ( 1:jpi, 1:jpj, 1:jpka+1 , 1) )
   ALLOCATE( ind    ( 1:jpi, 1:jpj              ) )
   ALLOCATE( e3_bak ( 1:jpi, 1:jpj              ) )  
   ALLOCATE( ght    (               1:jpka+1    ) )
   ALLOCATE( ghw    (               1:jpka+1    ) )   
   ALLOCATE( e3t    (               1:jpka+1    ) )
   ALLOCATE( e3w    (               1:jpka+1    ) )      
   ALLOCATE( tmask  ( 1:jpi, 1:jpj              ) )
   ALLOCATE( tmask2 ( 1:jpi, 1:jpj              ) )
   IF( ln_c1d ) ALLOCATE( varc1d( 1:3, 1:3, 1:jpka+1 , 1 ) )
   IF( ln_c1d ) varc1d( 1:3, 1:3, 1:jpka+1 , 1 ) = 0.
   IF (jpka_in.NE.nhym) THEN
     ALLOCATE( tmp_fullw(1:nhyi) )
     ALLOCATE( tmp_fullm(1:nhym) )
   ENDIF
   !
   varout(:,:,:,1)    = 0.
      
   !!---------------------------------------------------------------------
   !! Read the mask and remove some closed seas 
   IF (ln_read_mask) THEN
     CALL init_atm_mask(jpi,jpj,trim(cn_dir)//'/'//trim(file_m),trim(mask_var),ln_lsm_land,tmask)
     IF (ln_geo_wnd) CALL Read_Ncdf_var ( 'tmask' , trim(cn_dir)//'/'//trim(file_geos), tmask2(:,:) )
     IF (ln_hpg_frc) CALL Read_Ncdf_var ( 'tmask' , trim(cn_dir)//'/'//trim(file_hpg) , tmask2(:,:) )
   ELSE
     tmask(:,:)  = 1.
     tmask2(:,:) = 1.
   ENDIF
   !!   
   
   !!---------------------------------------------------------------------
   !! Compute the altitude and layer thickness of the target grid
   CALL init_target_grid ( jpka, ght, ghw, e3t, e3w, hmax, hc, theta_s,   &
      &                                              ln_impose_z1, z1  )

   !! Write the grid file for the target grid
   CALL Write_Grid_File  ( jpka, ght, ghw, e3t, e3w, trim(cn_dir)//'/'//trim(grd_file) )

   
   !! Read the static A and B coefficients for the ECMWF vertical grid
   IF (jpka_in.EQ.nhym) THEN
     CALL Read_Ncdf_var    ( 'hyai', trim(cn_dir)//'/'//trim(file_t), A_w )
     CALL Read_Ncdf_var    ( 'hybi', trim(cn_dir)//'/'//trim(file_t), B_w )   
   ELSE
     CALL Read_Ncdf_var    ( 'hyai', trim(cn_dir)//'/'//trim(file_t), tmp_fullw )
     A_w(0:jpka_in) = tmp_fullw(nhyi-(jpka_in+1)+1:nhyi)
     CALL Read_Ncdf_var    ( 'hybi', trim(cn_dir)//'/'//trim(file_t), tmp_fullw )
     B_w(0:jpka_in) = tmp_fullw(nhyi-(jpka_in+1)+1:nhyi)
   ENDIF


   !!---------------------------------------------------------------------   
   !! create output file
   !!
   IF (Len_Trim(var_name) == 0) THEN
      out_file = trim(cn_dir)//'/'//trim(abl_file)
   ELSE
      out_file = trim(cn_dir)//'/'//trim(var_name)//'_'//trim(abl_file)
   ENDIF

   IF(ln_c1d) THEN
      CALL Init_output_File_c1d ( jpi, jpj, jpka, trim(cn_dir)//'/'//trim(file_t), out_file, tmask(:,:), iloc, jloc )
   ELSE 
      CALL Init_output_File ( jpi, jpj, jpka, trim(cn_dir)//'/'//trim(file_t), out_file, tmask(:,:) )
   ENDIF

   !!---------------------------------------------------------------------   
   !! Initialize the name of the dimensions for the result of the interpolation
   !!
   dimnames(1) = 'lon'
   dimnames(2) = 'lat'       
   dimnames(3) = 'jpka'
   dimnames(4) = 'time'
   CALL Write_Ncdf_var( 'tmask', dimnames(1:2), trim(out_file), tmask2, 'float' )

   !!---------------------------------------------------------------------
   ! Read time variable
   ALLOCATE(tmp1d (1:jptime))    
   CALL Read_Ncdf_var ( 'time', trim(cn_dir)//'/'//trim(file_t), tmp1d )    
   !!---------------------------------------------------------------------     
 
   DO kt=1,jptime
   !
      WRITE(stdout,*) '======================'
      WRITE(stdout,*) 'time = ',kt,'/',jptime   
      !
      CALL Write_Ncdf_var( 'time', dimnames(4:4), trim(out_file), tmp1d(kt:kt), kt, 'double' )
      !
      IF( kt == 1 ) THEN
         CALL Duplicate_lon_lat_time( trim(cn_dir)//'/'//trim(file_t), out_file ) 
         CALL add_globatt_real( out_file, "jpka"   , REAL(jpka) )
         CALL add_globatt_real( out_file, "hmax"   , hmax       )
         CALL add_globatt_real( out_file, "theta_s", theta_s    )
         CALL add_globatt_real( out_file, "hc"     , hc         )
         IF (ln_impose_z1) CALL add_globatt_real( out_file, "z1", z1 )
      ENDIF
      !
      ! read SLP
      !CALL Read_Ncdf_var ( varnames(6) , trim(cn_dir)//'/'//trim(file_p), slp,  kt )      !<-- (log of) surface pressure
      !IF (ln_slp_log) THEN
      !  DO jj = 1, jpj
      !     DO ji = 1, jpi      
      !       slp( ji ,jj )   = exp( slp(ji ,jj ) )
      !     END DO
      !  END DO
      !ENDIF
      CALL Read_Ncdf_var ( outnames(6) , trim(cn_dir)//'/'//trim(file_in), slp,  kt )      !<-- (log of) surface pressure
      !
      ! read ZSURF
      IF (ln_read_zsurf) THEN
        !CALL Read_Ncdf_var ( varnames(1) , trim(cn_dir)//'/'//trim(file_z), zsurf,  kt )      !<-- surface geopotential 
        CALL Read_Ncdf_var ( outnames(1) , trim(cn_dir)//'/'//trim(file_in), zsurf,  kt )      !<-- surface geopotential 
      ELSE
        zsurf(:,:) = 0.
      ENDIF
      !    
      ! Smoothing of SLP and ZSURF to remove gibbs oscillations (must be done on both fields or none of them)
      !IF( ln_slp_smth ) CALL smooth_field( jpi, jpj, slp(:,:), tmask(:,:), 3 )
      !IF (ln_read_zsurf.AND.ln_slp_smth) CALL smooth_field( jpi, jpj, zsurf(:,:), tmask(:,:), 3 )
      !IF( ln_slp_smth ) CALL DTV_Filter( jpi, jpj, slp(:,:), tmask(:,:), 25, kt )  !<-- not yet robust enough
      !
      ! read tair and HUMI
      CALL Read_Ncdf_var ( varnames(2) , trim(cn_dir)//'/'//trim(file_t), tair, kt    )      !<-- temperature   
      CALL Read_Ncdf_var ( varnames(3) , trim(cn_dir)//'/'//trim(file_q), humi, kt    )      !<-- humidity        
      WHERE(humi.LT.1.E-08) varout = 1.E-08 !<-- negative values in ECMWF
      !
      ! Reconstruct the ERA-Interim vertical grid in terms of altitude 
      ghw_in(:,:,1:jpka_in)  = 0.
      ghw_in(:,:,  jpka_in)  = zsurf(:,:) * (1. / grav)
      CALL get_atm_grid( jpi, jpj, jpka_in, slp, tair,  &
         &                          humi, A_w, B_w, e3t_in, ghw_in )
      !
      ! Compute potential temperature
      tpot = tair ! save tpot
      CALL get_pot_temp    ( jpi, jpj, jpka_in, slp, tpot, A_w, B_w,   &
         &                   tmask(:,:), ptemp_method, humi(:,:,jpka_in,1), 0.5*ghw_in(:,:,jpka_in-1) )

      ! Flip the vertical axis to go from k=0 at the bottom to k=N_in at the top of the atmosphere
      CALL flip_vert_dim ( 1, jpka_in, jpi, jpj, e3t_in          )       
      CALL flip_vert_dim ( 0, jpka_in, jpi, jpj, ghw_in          ) 
      CALL flip_vert_dim ( 1, jpka_in, jpi, jpj, tpot(:,:,:,1)   )       
      CALL flip_vert_dim ( 1, jpka_in, jpi, jpj, humi(:,:,:,1)   ) 
      
      ! Correct the layer thickness to match hmax  
      DO jj = 1, jpj
         DO ji = 1, jpi          
            cff = 0.
            DO jk=1,jpka_in
               cff = cff + e3t_in( ji, jj, jk )
               IF ( cff > hmax ) THEN
                  jk_in = jk
                  EXIT
               ENDIF 
            END DO  
            ind    ( ji, jj       ) = jk_in    
            e3_bak ( ji, jj       ) = e3t_in ( ji, jj, jk_in )  ! store the value of the original layer thickness 
            e3t_in ( ji, jj, jk_in) = e3t_in ( ji, jj, jk_in ) - ( cff - hmax )
         END DO  
      END DO 
      !
      IF (Len_Trim(var_name) == 0) THEN
        !/
        ! Interpolation of potential temperature TPOT
        CALL zinterp ( jpi, jpj, jpka, jpka_in, ind,  &
            &                         tpot, e3t_in, e3_bak, e3t, varout, jp_weno )  

        varout(:,:,1,1) = varout(:,:,2,1)

        IF (ln_c1d) THEN
           DO jj=1,3
              DO ji=1,3
                 varc1d( ji, jj , 1:jpka+1 , 1 ) = varout(iloc,jloc, 1:jpka+1 , 1 )
              END DO
           END DO
           CALL Write_Ncdf_var ('tair', dimnames(1:4), out_file, varc1d, kt, 'float' )
        ELSE
           CALL Write_Ncdf_var ('tair', dimnames(1:4), out_file, varout, kt, 'float' ) 
        ENDIF
        !
        ! Interpolation of HUMI
        CALL zinterp ( jpi, jpj, jpka, jpka_in, ind,  &
            &                        humi, e3t_in, e3_bak, e3t, varout, jp_weno  )

        !FL: dirty loop (possible issue in boundary conditions for WENO scheme)
        DO jj = 1, jpj
           DO ji = 1, jpi
              DO jk = 2, jpka+1
                 varout(ji,jj,jk,1) = MAX(varout(ji,jj,jk,1),1.E-08)                       !<-- negative values in ECMWF
              END DO
           END DO
        END DO

        varout(:,:,1,1) = varout(:,:,2,1) 

        IF (ln_c1d) THEN
           DO jj=1,3
              DO ji=1,3
                 varc1d( ji, jj , 1:jpka+1 , 1 ) = varout(iloc,jloc, 1:jpka+1 , 1 )
              END DO
           END DO
           CALL Write_Ncdf_var  ('humi', dimnames(1:4), out_file, varc1d, kt, 'float' )
        ELSE
           CALL Write_Ncdf_var ('humi', dimnames(1:4), out_file, varout, kt, 'float' )  
        ENDIF

        !
        ! Interpolate large-scale HPG or geostrophic wind
        !
        ! Read HPG
        IF (ln_hpg_frc) THEN
           CALL Read_Ncdf_var ( varnames(8) , trim(cn_dir)//'/'//trim(file_in), tair, kt )        
           CALL Read_Ncdf_var ( varnames(9) , trim(cn_dir)//'/'//trim(file_in), humi, kt ) 
        ENDIF
        ! Read geostrophic wind
        IF (ln_geo_wnd) THEN
           CALL Read_Ncdf_var ( varnames(10) , trim(cn_dir)//'/'//trim(file_in), tair, kt )        
           CALL Read_Ncdf_var ( varnames(11) , trim(cn_dir)//'/'//trim(file_in), humi, kt ) 
        ENDIF
        !
        CALL flip_vert_dim ( 1, jpka_in, jpi, jpj, tair( :,:,:,1 ) )       
        CALL flip_vert_dim ( 1, jpka_in, jpi, jpj, humi( :,:,:,1 ) ) 
        !      
        ! Interpolation of geostrophic U
        CALL zinterp ( jpi, jpj, jpka, jpka_in, ind,  &
            &                         tair, e3t_in, e3_bak, e3t, varout, jp_spln  )      
        varout(:,:,1,1) = varout(:,:,2,1)
           
           IF (ln_c1d) THEN
              DO jj=1,3
                 DO ji=1,3
                    varc1d( ji, jj , 1:jpka+1 , 1 ) = varout(iloc,jloc, 1:jpka+1 , 1 )
                 END DO
              END DO
              CALL Write_Ncdf_var ( 'uhpg', dimnames(1:4), out_file, varc1d, kt, 'float' )
           ELSE
              IF (ln_hpg_frc) CALL Write_Ncdf_var ( varnames( 8), dimnames(1:4), out_file, varout, kt, 'float' )
              IF (ln_geo_wnd) CALL Write_Ncdf_var ( varnames(10), dimnames(1:4), out_file, varout, kt, 'float' )
           ENDIF
           !
           ! Interpolation of geostrophic V
           CALL zinterp ( jpi, jpj, jpka, jpka_in, ind,  &
               &                         humi, e3t_in, e3_bak, e3t, varout, jp_spln  )       
           varout(:,:,1,1) = varout(:,:,2,1)

           IF (ln_c1d) THEN
              DO jj=1,3
                 DO ji=1,3
                    varc1d( ji, jj , 1:jpka+1 , 1 ) = varout(iloc,jloc, 1:jpka+1 , 1 )
                 END DO
              END DO
              CALL Write_Ncdf_var ( 'vhpg', dimnames(1:4), out_file, varc1d, kt, 'float' )
           ELSE
              IF (ln_hpg_frc) CALL Write_Ncdf_var ( varnames( 9), dimnames(1:4), out_file, varout, kt, 'float' )
              IF (ln_geo_wnd) CALL Write_Ncdf_var ( varnames(11), dimnames(1:4), out_file, varout, kt, 'float' )
           ENDIF
        !
        ! Interpolation of total winds
        !      
        ! Read wind
        CALL Read_Ncdf_var ( varnames(4) , trim(cn_dir)//'/'//trim(file_u), tair, kt )        
        CALL Read_Ncdf_var ( varnames(5) , trim(cn_dir)//'/'//trim(file_v), humi, kt ) 
        CALL flip_vert_dim ( 1, jpka_in, jpi, jpj, tair( :,:,:,1 ) )       
        CALL flip_vert_dim ( 1, jpka_in, jpi, jpj, humi( :,:,:,1 ) ) 
        !      
        ! Interpolation of total U
        CALL zinterp ( jpi, jpj, jpka, jpka_in, ind,  &
            &                         tair, e3t_in, e3_bak, e3t, varout, jp_spln  )      
        varout(:,:,1,1) = varout(:,:,2,1)

        IF(ln_c1d) THEN
           DO jj=1,3
              DO ji=1,3
                 varc1d( ji, jj , 1:jpka+1 , 1 ) = varout(iloc,jloc, 1:jpka+1 , 1 )
              END DO
           END DO
           CALL Write_Ncdf_var ( 'uwnd', dimnames(1:4), out_file, varc1d, kt, 'float' )
        ELSE
           CALL Write_Ncdf_var ( 'uwnd', dimnames(1:4), out_file, varout, kt, 'float' )
        ENDIF
        !
        ! Interpolation of total V
        CALL zinterp ( jpi, jpj, jpka, jpka_in, ind,  &
            &                         humi, e3t_in, e3_bak, e3t, varout, jp_spln  )       
        varout(:,:,1,1) = varout(:,:,2,1)

        IF(ln_c1d) THEN
           DO jj=1,3
              DO ji=1,3
                 varc1d( ji, jj , 1:jpka+1 , 1 ) = varout(iloc,jloc, 1:jpka+1 , 1 )
              END DO
           END DO
           CALL Write_Ncdf_var ( 'vwnd', dimnames(1:4), out_file, varc1d, kt, 'float' )
        ELSE
           CALL Write_Ncdf_var ( 'vwnd', dimnames(1:4), out_file, varout, kt, 'float' )
        ENDIF

      ELSE ! var_name
        !
        ctrl     =  minval(pack([(ji,ji=1,size(outnames))],outnames==var_name))
        var_file = filnames(ctrl)
        !
        !
        ! Interpolation of var_name
        !      
        IF ( (var_name.NE."tair").AND.(var_name.NE."humi") ) THEN
          ! Read var_name
          CALL Read_Ncdf_var ( varnames(ctrl) , trim(cn_dir)//'/'//trim(var_file), tair, kt )
          CALL flip_vert_dim ( 1, jpka_in, jpi, jpj, tair( :,:,:,1 ) )       
          ! Interpolation of var_name
          CALL zinterp ( jpi, jpj, jpka, jpka_in, ind,  &
              &                         tair, e3t_in, e3_bak, e3t, varout, jp_spln  )
        ELSE
          ! humi and tpot already read     
          IF (var_name.EQ."humi") tair = humi
          IF (var_name.EQ."tair") tair = tpot
          ! Interpolation of var_name
          CALL zinterp ( jpi, jpj, jpka, jpka_in, ind,  &
              &                         tair, e3t_in, e3_bak, e3t, varout, jp_weno )
        END IF

        varout(:,:,1,1) = varout(:,:,2,1)

        IF(ln_c1d) THEN
           DO jj=1,3
              DO ji=1,3
                 varc1d( ji, jj , 1:jpka+1 , 1 ) = varout(iloc,jloc, 1:jpka+1 , 1 )
              END DO
           END DO
           CALL Write_Ncdf_var ( outnames(ctrl), dimnames(1:4), out_file, varc1d, kt, 'float' )
        ELSE
           CALL Write_Ncdf_var ( outnames(ctrl), dimnames(1:4), out_file, varout, kt, 'float' )
        ENDIF
        !
      ENDIF ! var_name
   !
   END DO ! kt
   !
   DEALLOCATE(zsurf,slp,tair,humi,varout)     
   IF (jpka_in.NE.nhym) DEALLOCATE(tmp_fullw,tmp_fullm)
   !
   STOP
   !
END PROGRAM main
