PROGRAM create_data
  !
  USE io_netcdf
  USE bilinear_interp
  USE agrif_readwrite
  USE agrif_interpolation      
  !
  IMPLICIT NONE
  !
  !************************************************************************
  ! 									*
  ! PROGRAM  CREATE_DATA					        *
  !									*
  ! program to implement data interpolation to generate	 		*
  ! child grid forcing files						*
  !									*
  !Interpolation is carried out using bilinear interpolation		*
  !routine from SCRIP package						*		
  !									*
  !http://climate.lanl.gov/Software/SCRIP/				*
  !************************************************************************
  !
  INTEGER           :: narg, iargc, ji
  CHARACTER(len=80) :: namelistname

  narg = iargc()

  IF (narg == 0) THEN
     namelistname = 'namelist.input'
  ELSE
     CALL getarg(1,namelistname)
  ENDIF

  ! read input file (namelist.input)
  CALL read_namelist(namelistname)
  !
  ! Interpolate U grid  data
  ji = 1
  DO WHILE( TRIM(U_Files(ji)) /= '' )
     PRINT *,'Grid U forcing files = ',u_files(ji)
     !        
     CALL Interp_Extrap_var(U_FILES(ji), 'U') 
     ji = ji+1                
     !             
  END DO

  !
  ! Interpolate V grid  data
  ji = 1
  DO WHILE( TRIM(V_Files(ji)) /= '' )
     PRINT *,'Grid V forcing files = ',v_files(ji)
     !        
     CALL Interp_Extrap_var(V_FILES(ji), 'V') 
     ji = ji+1                
     !             
  END DO
  !
  ! Interpolate flux data
  ji = 1
  DO WHILE( TRIM(Flx_Files(ji)) /= '' )
     PRINT *,'flxfiles = ',flx_files(ji)
     !        
     CALL Interp_Extrap_var(FLX_FILES(ji), 'T') 
     ji = ji+1                
     !             
  END DO
  !
  WRITE(*,*) ' '
  WRITE(*,*) '******* forcing files successfully created *******' 
  WRITE(*,*) ' '  
  !
  STOP
END PROGRAM create_data
