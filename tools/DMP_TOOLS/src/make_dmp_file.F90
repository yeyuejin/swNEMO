PROGRAM make_dmp_file
  !================================================================================
  !               *** PROGRAM make_dmp_file ****
  !================================================================================
  !
  !  Purpose: Create a file containing a spacially varying
  !           restoration coefficient to be used by TRADMP
  !
  !  Method:  1) Read in tmask from mesh_mask file to use as a template
  !           2) Calculate restoration coefficients according to options
  !           specified in the namelist. The user may modify custom.F90 to
  !           specify specific damping options e.g. to mask certain regions only).
  !           3) Write the array to output file
  !
  !  History: Original code: Tim Graham (Jul 2014) - some code moved from
  !                            old tradmp.F90 module to this tool (as part of NEMO
  !                            simplification process).
  !-------------------------------------------------------------------------------

  ! Declare variables
  USE netcdf 
  USE utils
  USE coastdist
  USE med_red_seas
  USE zoom
  USE custom

  IMPLICIT NONE
  INTEGER  :: ji, jj, jk                         ! dummpy loop variables
  REAL(wp) :: zsdmp, zbdmp                     ! Surface and bottom damping coeff
  CHARACTER(LEN=200) :: meshfile = 'mesh_mask.nc'   ! mesh file
  CHARACTER(LEN=200) :: outfile = 'resto.nc'     ! output file
  REAL(wp) :: zlat, zlat2, zlat0

  ! Read namelist
  OPEN( UNIT=numnam, FILE='namelist', FORM='FORMATTED', STATUS='OLD' )
  READ( numnam, nam_dmp_create )
  CLOSE( numnam )
  
  IF ( ln_full_field .AND. lzoom ) THEN
    WRITE(numerr,*) 'Only one of ln_full_field and lzoom can be .true.'
    STOP
  ENDIF

  CALL grid_info(meshfile)
  WRITE(numout, *) 'jpi = ',jpi
  WRITE(numout, *) 'jpj = ',jpj
  WRITE(numout, *) 'jpk = ',jpk

  ALLOCATE( resto(jpi, jpj) )

  !Create output file
  CALL make_outfile( outfile )
  
  CALL check_nf90( nf90_get_var( ncin, gphit_id, gphit, (/ 1,1 /), (/ jpi, jpj /) ) )

  !Calculate surface and bottom damping coefficients
  zsdmp = 1._wp / ( pn_surf * rday )
  zbdmp = 1._wp / ( pn_bot  * rday )

  !Loop through levels and read in tmask for each level as starting point for
  !coefficient array
  DO jk = 1, jpk-1
     resto(:,:) = 0._wp
     
     IF (.NOT. (jk == 1 .AND. ln_zero_top_layer) ) THEN 
        !Read in tmask depth for this level
        CALL check_nf90( nf90_get_var( ncin, tmask_id, tmask, (/ 1,1,jk /), (/ jpi, jpj,1 /) ) )
        CALL check_nf90( nf90_get_var( ncin, gdept_id, gdept, (/ 1,1,jk /), (/ jpi, jpj,1 /) ) )
     

        IF ( ln_full_field ) THEN
           !Set basic value of resto
           DO jj = 1, jpj
              DO ji = 1, jpi
                 resto(ji,jj) = tmask(ji, jj) * (zbdmp + (zsdmp-zbdmp) * EXP(-gdept(ji,jj)/pn_dep))
              END DO
           END DO
           IF ((nn_hdmp > 0)) THEN
              zlat0 = 10. !width of latitude strip where resto decreases
              zlat2 = nn_hdmp + zlat0
              DO jj = 1, jpj
                 DO ji = 1, jpi
                    zlat = ABS(gphit(ji,jj))
                    IF ( nn_hdmp <= zlat .AND. zlat <= zlat2 ) THEN
                       resto(ji,jj) = resto(ji,jj) * 0.5_wp * (  1._wp - COS( rpi*(zlat-nn_hdmp)/zlat0 ) )
                    ELSE IF ( zlat < nn_hdmp ) THEN
                       resto(ji,jj) = 0._wp
                    ENDIF
                 END DO
              END DO
           ENDIF
    
           IF (ln_coast) THEN
              ! Reduce damping in vicinity of coastlines
              CALL coast_dist_weight(resto)
           ENDIF
        ENDIF

        ! Damping in Med/Red Seas (or local modifications if full field is set)
        IF (ln_med_red_seas .AND. (cp_cfg == 'orca') .AND. (.NOT. lzoom)) THEN
           CALL med_red_dmp(resto, jk, ln_old_31_lev_code)
        ENDIF

        IF ( lzoom ) THEN
              CALL dtacof_zoom(resto, tmask)
        ENDIF
        
        !Any user modifications can be added in the custom module
        IF ( ln_custom ) THEN
              CALL custom_resto( resto )
        ENDIF
     ENDIF

     ! Write out resto for this level
     CALL check_nf90( nf90_put_var( ncout, resto_id, resto, (/ 1,1,jk /), (/ jpi, jpj,1 /) ) )

  END DO
  
  ! Close the output file
  CALL check_nf90( nf90_close(ncout) )

END PROGRAM make_dmp_file
