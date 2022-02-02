MODULE module_io
   !!======================================================================
   !!                   ***  MODULE  module_io  ***
   !! ABL preprocessing tool netcdf I/O subroutines
   !!=====================================================================
   !! History : 2016-10  (F. Lemarié)  Original code
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!
   !!   FUNCTIONS   : Var_Existence    
   !!   SUBROUTINES : Read_Ncdf_dim, Write_Ncdf_dim, Read_Ncdf_var1d_Real,
   !!                 Read_Ncdf_var4d_Real_nt,       Read_Ncdf_var4d_Real_t,    
   !!                 Write_Ncdf_var1d_Real       , Write_Ncdf_var2d_Real  ,
   !!                 Write_Ncdf_var4d_Real_t     , Write_Ncdf_var1d_Real_t,
   !!                 Duplicate_lon_lat_time
   !!----------------------------------------------------------------------
   USE netcdf
   IMPLICIT NONE
    
   INTERFACE Read_Ncdf_var
          MODULE PROCEDURE Read_Ncdf_var1d_Real   ,       &
             &             Read_Ncdf_var2d_Real   ,       &
             &             Read_Ncdf_var2d_Real_t ,       &
             &             Read_Ncdf_var3d_Real_t ,       &
             &             Read_Ncdf_var4d_Real_t ,       &
             &             Read_Ncdf_var2d_Real_nt,       &
             &             Read_Ncdf_var4d_Real_nt
   END INTERFACE       
!
   INTERFACE Write_Ncdf_var
          MODULE PROCEDURE Write_Ncdf_var1d_Real  ,       &
             &             Write_Ncdf_var2d_Real  ,       &          
             &             Write_Ncdf_var1d_Real_t,       &
             &             Write_Ncdf_var2d_Real_t,       &
             &             Write_Ncdf_var4d_Real_t
   END INTERFACE      
!      
CONTAINS





   LOGICAL FUNCTION Var_Existence( varname , filename )
      !!---------------------------------------------------------------------
      !!                    ***  FUNCTION Var_Existence  ***
      !!                   
      !! ** Purpose : check if variables varname exists in filename file  
      !!
      !!----------------------------------------------------------------------
      CHARACTER(*),   intent(in) :: varname,filename
      INTEGER                    :: status,ncid,varid        
      
      status = nf90_open(TRIM(filename),NF90_NOWRITE,ncid)
      
      IF ( status/=nf90_noerr ) THEN    
            WRITE(*,*) "*** Var_Existence: unable to open netcdf file : ",TRIM(filename)
            stop
      END IF 
      
      status = nf90_inq_varid( ncid, varname, varid )
      
      IF ( status/=nf90_noerr ) THEN
         Var_Existence = .false.
      ELSE
         Var_Existence = .true.
      END IF   

   END FUNCTION Var_Existence
   
   
   
   SUBROUTINE Read_Ncdf_dim ( dimname, file, dimval )         
      !!---------------------------------------------------------------------
      !!                    ***  SUBROUTINE Read_Ncdf_dim  ***
      !!                   
      !! ** Purpose : read the integer dimension dimname in input file and   
      !!              store it in dimval
      !!
      !!----------------------------------------------------------------------

      CHARACTER(*),INTENT(in) :: dimname,file    
      INTEGER                 :: dimval
      INTEGER                 :: ncid,status,dimid
      !     
      status = nf90_open(file,NF90_NOWRITE,ncid)
      
      IF ( status/=nf90_noerr ) THEN    
         WRITE(*,*)"*** Read_Ncdf_dim: unable to open netcdf file : ",trim(file)
         STOP
      END IF     
      !     
      status = nf90_inq_dimid        ( ncid, dimname,     dimid  ) 
      status = nf90_inquire_dimension( ncid, dimid,   len=dimval )
      status = nf90_close( ncid )

   END SUBROUTINE Read_Ncdf_dim  







   SUBROUTINE Write_Ncdf_dim( dimname, file, dimval )   
      !!---------------------------------------------------------------------
      !!                    ***  SUBROUTINE Write_Ncdf_dim  ***
      !!                   
      !! ** Purpose : write the dimension dimname in the output file   
      !!
      !!----------------------------------------------------------------------               
      CHARACTER(*), INTENT(in) :: dimname,file    
      INTEGER                  :: dimval
      INTEGER                  :: ncid,status,dimid
      !     
      status = nf90_open(file,NF90_WRITE,ncid)
      IF ( status/=nf90_noerr ) THEN    
         WRITE(*,*)"*** Write_Ncdf_dim: to open netcdf file : ",file
         STOP
      END IF          
      
      status = nf90_redef(ncid)
      
      If( dimval.eq.0 ) THEN
         status = nf90_def_dim(ncid,dimname,nf90_unlimited,dimid)       
      ELSE
         status = nf90_def_dim(ncid,dimname,dimval,dimid)  
      END If    
      !!
      status = nf90_enddef(ncid)
      status = nf90_close(ncid)
      !
   END SUBROUTINE Write_Ncdf_dim   






   SUBROUTINE Read_Ncdf_var1d_Real( varname, file, tabvar )
      !!---------------------------------------------------------------------
      !!                    ***  SUBROUTINE Read_Ncdf_var1d_Real  ***
      !!                   
      !! ** Purpose : read the 1D variable varname in the input file   
      !!              and store it in tabvar
      !!
      !!----------------------------------------------------------------------  
      CHARACTER(*),              INTENT(in)    :: varname,file
      REAL(8), DIMENSION(:),ALLOCATABLE        :: tabvar
      INTEGER, DIMENSION(1)                    :: dimID
      INTEGER                                  :: dim1
      INTEGER                                  :: status,ncid
      INTEGER                                  :: varid             
      !
      status = nf90_open(file,NF90_NOWRITE,ncid)    
      IF ( status/=nf90_noerr ) THEN    
         WRITE(*,*) "*** Read_Ncdf_var1d_Real: unable to open netcdf file : ",file
         STOP
      END IF          
      status = nf90_inq_varid(ncid,varname,varid)
      status = nf90_inquire_variable(ncid,varid,dimids=dimID)
      status = nf90_inquire_dimension(ncid,dimID(1),len=dim1)                
      IF( .not. allocated(tabvar) ) THEN
         ALLOCATE( tabvar ( dim1 ) )  
      ELSE
         IF ( any(shape(tabvar)/=(/dim1/)) ) THEN
            DEALLOCATE ( tabvar          )   
            ALLOCATE   ( tabvar ( dim1 ) )   
            WRITE(*,*) 'Warning change shape of array for ',trim(varname)   
         END IF      
      END IF        
      status = nf90_get_var(ncid,varid,tabvar)     
      status = nf90_close(ncid)
      !
   END SUBROUTINE Read_Ncdf_var1d_Real


   SUBROUTINE Read_Ncdf_var2d_Real( varname, file, tabvar )    
      !!---------------------------------------------------------------------
      !!                    ***  SUBROUTINE Read_Ncdf_var2d_Real  ***
      !!                   
      !! ** Purpose : read the 2D variable varname in the input file   
      !!              and store it in 2D tabvar
      !!
      !!----------------------------------------------------------------------          
      CHARACTER(*),               INTENT(in)    :: varname, file
      REAL(8), DIMENSION(:,:),    INTENT(inout) :: tabvar
      INTEGER, DIMENSION(4)                     :: dimIDS
      INTEGER                                   :: dim1,dim2
      INTEGER                                   :: status,ncid
      INTEGER                                   :: varid             
      !
      status = nf90_open(file,NF90_NOWRITE,ncid)      
      IF (status/=nf90_noerr) THEN    
         WRITE(*,*)"*** Read_Ncdf_var2d_Real: unable to open netcdf file : ",file
         STOP
      END IF         
      status = nf90_inq_varid        (ncid , varname, varid)       
      status = nf90_inquire_variable (ncid , varid, dimids=dimIDS)
      status = nf90_inquire_dimension(ncid , dimIDS(1), len=dim1)
      status = nf90_inquire_dimension(ncid , dimIDS(2), len=dim2)
      status = nf90_get_var( ncid, varid, tabvar(:,:),      &
        &                start = (/1,1/), count=(/dim1,dim2/))                                            
      IF (status/=nf90_noerr) THEN    
         WRITE(*,*)"unable to retrieve netcdf variable : ",trim(varname)
         WRITE(*,*)"in file : ",trim(file)
         WRITE(*,*) "error code: ", status
         STOP
      END IF         
      status = nf90_close(ncid)
      !    
   END SUBROUTINE Read_Ncdf_var2d_Real

   
   SUBROUTINE Read_Ncdf_var2d_Real_nt( varname, file, tabvar, time, level )    
      !!---------------------------------------------------------------------
      !!                    ***  SUBROUTINE Read_Ncdf_var2d_Real_nt  ***
      !!                   
      !! ** Purpose : read the 4D variable varname in the input file   
      !!              for a given time and vertical level, and store it in 2D tabvar
      !!
      !!----------------------------------------------------------------------          
      CHARACTER(*),               INTENT(in)    :: varname, file
      INTEGER     ,               INTENT(in)    :: time, level
      !REAL(8), DIMENSION(:,:),    ALLOCATABLE   :: tabvar
      REAL(8), DIMENSION(:,:),    INTENT(inout) :: tabvar
      INTEGER, DIMENSION(4)                     :: dimIDS
      INTEGER                                   :: dim1,dim2
      INTEGER                                   :: status,ncid
      INTEGER                                   :: varid             
      !
      status = nf90_open(file,NF90_NOWRITE,ncid)      
      IF (status/=nf90_noerr) THEN    
         WRITE(*,*)"*** Read_Ncdf_var2d_Real_nt: unable to open netcdf file : ",file
         STOP
      END IF         
      status = nf90_inq_varid        (ncid , varname, varid)       
      status = nf90_inquire_variable (ncid , varid, dimids=dimIDS)
      status = nf90_inquire_dimension(ncid , dimIDS(1), len=dim1)
      status = nf90_inquire_dimension(ncid , dimIDS(2), len=dim2)
      !IF( .not. allocated( tabvar ) ) then
      !   ALLOCATE ( tabvar( dim1, dim2 ) )  
      !ELSE
      !   IF ( (size(tabvar,1) /= dim1) .OR. (size(tabvar,2) /= dim2) ) THEN
      !      DEALLOCATE( tabvar )
      !      ALLOCATE  ( tabvar (dim1, dim2 ) )
      !   END IF     
      !END IF       
      status = nf90_get_var( ncid, varid, tabvar(:,:),      &
        &                start = (/1,1,level,time/), count=(/dim1,dim2,1,1/))                                            
      IF (status/=nf90_noerr) THEN    
         WRITE(*,*)"unable to retrieve netcdf variable : ",trim(varname)
         WRITE(*,*)"in file : ",trim(file)
         WRITE(*,*) "error code: ", status
         STOP
      END IF         
      status = nf90_close(ncid)
      !    
   END SUBROUTINE Read_Ncdf_var2d_Real_nt


 
 
   SUBROUTINE Read_Ncdf_var4d_Real_nt( varname, file, tabvar, time, level )    
      !!---------------------------------------------------------------------
      !!                    ***  SUBROUTINE Read_Ncdf_var4d_Real_nt  ***
      !!                   
      !! ** Purpose : read the 4D variable varname in the input file   
      !!              for a given time and vertical level, and store it in 4D tabvar
      !!
      !!----------------------------------------------------------------------          
      CHARACTER(*),               INTENT(in)    :: varname,file
      INTEGER     ,               INTENT(in)    :: time,level
      !REAL(8), DIMENSION(:,:,:,:),ALLOCATABLE   :: tabvar
      REAL(8), DIMENSION(:,:,:,:),INTENT(inout) :: tabvar
      INTEGER, DIMENSION(4)                     :: dimIDS
      INTEGER                                   :: dim1,dim2
      INTEGER                                   :: status,ncid
      INTEGER                                   :: varid             
      !
      status = nf90_open(file,NF90_NOWRITE,ncid)      
      IF (status/=nf90_noerr) THEN    
         WRITE(*,*)"*** Read_Ncdf_var4d_Real_nt: unable to open netcdf file : ",file
         STOP
      END IF         
      status = nf90_inq_varid        (ncid , varname, varid)       
      status = nf90_inquire_variable (ncid , varid, dimids=dimIDS)
      status = nf90_inquire_dimension(ncid , dimIDS(1), len=dim1)
      status = nf90_inquire_dimension(ncid , dimIDS(2), len=dim2)
      !IF( .not. allocated( tabvar ) ) then
      !   ALLOCATE ( tabvar( dim1, dim2, 1, 1 ) )  
      !ELSE
      !   IF ( (size(tabvar,1) /= dim1) .OR. (size(tabvar,2) /= dim2) ) THEN
      !      DEALLOCATE( tabvar )
      !      ALLOCATE  ( tabvar (dim1, dim2, 1, 1 ) )
      !   END IF     
      !END IF       
      status = nf90_get_var( ncid, varid, tabvar(:,:,:,:),      &
        &                start = (/1,1,level,time/), count=(/dim1,dim2,1,1/))                                            
      IF (status/=nf90_noerr) THEN    
         WRITE(*,*)"unable to retrieve netcdf variable : ",trim(varname)
         WRITE(*,*)"in file : ",trim(file)
         WRITE(*,*) "error code: ", status
         STOP
      END IF         
      status = nf90_close(ncid)
      !    
   END SUBROUTINE Read_Ncdf_var4d_Real_nt
      
 
 
 
 
 
   SUBROUTINE Read_Ncdf_var3d_Real_t( varname, file, tabvar, time)   
      !!---------------------------------------------------------------------
      !!                    ***  SUBROUTINE Read_Ncdf_var3d_Real_t  ***
      !!                   
      !! ** Purpose : read the 3D variable varname in the input file   
      !!              for a given time level, and store it in tabvar
      !!
      !!----------------------------------------------------------------------           
      CHARACTER(*),               INTENT(in)    :: varname,file
      INTEGER     ,               INTENT(in)    :: time
      !REAL(8), DIMENSION(:,:,:),ALLOCATABLE     :: tabvar
      REAL(8), DIMENSION(:,:,:), INTENT(inout)  :: tabvar
      INTEGER, DIMENSION(3)                     :: dimIDS
      INTEGER                                   :: dim1,dim2
      INTEGER                                   :: status,ncid
      INTEGER                                   :: varid             
      !
      status = nf90_open(file,NF90_NOWRITE,ncid)      
      IF (status/=nf90_noerr) then    
         WRITE(*,*)"*** Read_Ncdf_var3d_Real_t: unable to open netcdf file : ",file
         STOP
      END IF

      status = nf90_inq_varid        (ncid, varname,varid)        
      status = nf90_inquire_variable (ncid, varid,dimids=dimIDS)
      status = nf90_inquire_dimension(ncid, dimIDS(1),len=dim1)
      status = nf90_inquire_dimension(ncid, dimIDS(2),len=dim2)

      !IF( .not. allocated(tabvar) ) ALLOCATE(tabvar(dim1,dim2,1))  

      status=nf90_get_var(ncid,varid,tabvar,start=(/1,1,time/))                                             
      IF ( status/=nf90_noerr ) THEN    
         WRITE(*,*)"unable to retrieve netcdf variable : ",trim(varname)
         WRITE(*,*)"in file : ",trim(file)
         WRITE(*,*) "error code: ", status
         STOP
      END IF           
      status = nf90_close(ncid)
      !
   END SUBROUTINE Read_Ncdf_var3d_Real_t        
 
      
   SUBROUTINE Read_Ncdf_var2d_Real_t(varname,file,tabvar,time)   
      !!---------------------------------------------------------------------
      !!                    ***  SUBROUTINE Read_Ncdf_var2d_Real_t  ***
      !!                   
      !! ** Purpose : read the 3D variable varname in the input file   
      !!              for a given time level, and store it in tabvar
      !!
      !!----------------------------------------------------------------------           
      CHARACTER(*),               INTENT(in)    :: varname,file
      INTEGER     ,               INTENT(in)    :: time
      !REAL(8), DIMENSION(:,:),ALLOCATABLE       :: tabvar
      REAL(8), DIMENSION(:,:), INTENT(inout)    :: tabvar
      INTEGER, DIMENSION(3)                     :: dimIDS
      INTEGER                                   :: dim1,dim2
      INTEGER                                   :: status,ncid
      INTEGER                                   :: varid             
      !
      status = nf90_open(file,NF90_NOWRITE,ncid)      
      IF (status/=nf90_noerr) then    
         WRITE(*,*)"*** Read_Ncdf_var2d_Real_t: unable to open netcdf file : ",file
         STOP
      END IF

      status = nf90_inq_varid        (ncid, varname,varid)        
      status = nf90_inquire_variable (ncid, varid,dimids=dimIDS)
      status = nf90_inquire_dimension(ncid, dimIDS(1),len=dim1)
      status = nf90_inquire_dimension(ncid, dimIDS(2),len=dim2)

      !IF( .not. allocated(tabvar) ) ALLOCATE(tabvar(dim1,dim2))  

      status=nf90_get_var(ncid,varid,tabvar,start=(/1,1,time/))                                             
      IF ( status/=nf90_noerr ) THEN    
         WRITE(*,*)"unable to retrieve netcdf variable : ",trim(varname)
         WRITE(*,*)"in file : ",trim(file)
         STOP
      END IF           
      status = nf90_close(ncid)
      !
   END SUBROUTINE Read_Ncdf_var2d_Real_t


   SUBROUTINE Read_Ncdf_var4d_Real_t(varname,file,tabvar,time)   
      !!---------------------------------------------------------------------
      !!                    ***  SUBROUTINE Read_Ncdf_var4d_Real_t  ***
      !!                   
      !! ** Purpose : read the 4D variable varname in the input file   
      !!              for a given time level, and store it in tabvar
      !!
      !!----------------------------------------------------------------------           
      CHARACTER(*),               INTENT(in)    :: varname,file
      INTEGER     ,               INTENT(in)    :: time
      !REAL(8), DIMENSION(:,:,:,:),ALLOCATABLE   :: tabvar
      REAL(8), DIMENSION(:,:,:,:), INTENT(inout):: tabvar
      INTEGER, DIMENSION(4)                     :: dimIDS
      INTEGER                                   :: dim1,dim2,dim3
      INTEGER                                   :: status,ncid
      INTEGER                                   :: varid             
      !
      status = nf90_open(file,NF90_NOWRITE,ncid)      
      IF (status/=nf90_noerr) then    
         WRITE(*,*)"*** Read_Ncdf_var4d_Real_t: unable to open netcdf file : ",file
         STOP
      END IF         

      status = nf90_inq_varid        (ncid, varname,varid)        
      status = nf90_inquire_variable (ncid, varid,dimids=dimIDS)
      status = nf90_inquire_dimension(ncid, dimIDS(1),len=dim1)
      status = nf90_inquire_dimension(ncid, dimIDS(2),len=dim2)
      status = nf90_inquire_dimension(ncid, dimIDS(3),len=dim3)

      !IF( .not. allocated(tabvar) ) ALLOCATE(tabvar(dim1,dim2,dim3,1))

      status=nf90_get_var(ncid,varid,tabvar,start=(/1,1,1,time/))                                             
      IF ( status/=nf90_noerr ) THEN    
         WRITE(*,*) "unable to retrieve netcdf variable : ",trim(varname)
         WRITE(*,*) "in file : ",trim(file)
         WRITE(*,*) "error code: ", status
         STOP
      END IF           
      status = nf90_close(ncid)
      !
   END SUBROUTINE Read_Ncdf_var4d_Real_t        
   





   SUBROUTINE Write_Ncdf_var1d_Real( varname, dimname, file, tabvar, typevar )
      !!---------------------------------------------------------------------
      !!                    ***  SUBROUTINE Write_Ncdf_var1d_Real  ***
      !!                   
      !! ** Purpose : write the 1D variable varname stored in tabvar 
      !!              in the output file using typevar type (float or double)             
      !!
      !!----------------------------------------------------------------------             
      CHARACTER(*),               INTENT(in)    :: varname,file,dimname,typevar
      REAL(8), DIMENSION(:), INTENT(in)         :: tabvar
      INTEGER                                   :: dimid
      INTEGER                                   :: status,ncid
      INTEGER                                   :: varid             
!
      status = nf90_open(file,NF90_WRITE,ncid)       
      IF (status/=nf90_noerr) then    
         WRITE(*,*)"*** Write_Ncdf_var1d_Real: unable to open netcdf file : ",file
         STOP
      END IF         
      status = nf90_inq_dimid(ncid,dimname, dimid)
      status = nf90_redef(ncid)
      SELECT CASE( TRIM(typevar) )
      CASE('double')
         status = nf90_def_var(ncid,varname,nf90_double,(/dimid/),varid)
      CASE('float')
         status = nf90_def_var(ncid,varname,nf90_float,(/dimid/),varid)      
      END SELECT
      status = nf90_enddef ( ncid )
      status = nf90_put_var( ncid ,varid,tabvar)        
      status = nf90_close  ( ncid )
      !
   END SUBROUTINE Write_Ncdf_var1d_Real 
      
      
      


   SUBROUTINE Write_Ncdf_var2d_Real( varname, dimname, file, tabvar, typevar )
      !!---------------------------------------------------------------------
      !!                    ***  SUBROUTINE Write_Ncdf_var2d_Real  ***
      !!                   
      !! ** Purpose : write the 2D variable varname stored in tabvar 
      !!              in the output file using typevar type (float or double)             
      !!
      !!----------------------------------------------------------------------          
      CHARACTER(*),               INTENT(in)    :: varname, file, typevar
      CHARACTER(*), DIMENSION(2), INTENT(in)    :: dimname      
      REAL(8), DIMENSION(:,:), INTENT(in)       :: tabvar
      INTEGER                                   :: dimid1, dimid2
      INTEGER                                   :: status, ncid
      INTEGER                                   :: varid             
!
      status = nf90_open(file,NF90_WRITE,ncid)       
      IF (status/=nf90_noerr) then    
         WRITE(*,*)"*** Write_Ncdf_var2d_Real: unable to open netcdf file : ",file
         WRITE(*,*)"*** Write_Ncdf_var2d_Real: variable : ", varname
         STOP
      END IF
         status = nf90_inq_dimid(ncid,dimname(1), dimid1)
         status = nf90_inq_dimid(ncid,dimname(2), dimid2)
         status = nf90_inq_varid(ncid,varname,varid)
         IF (status /= nf90_noerr) THEN
            status = nf90_redef(ncid)    
            SELECT CASE( TRIM(typevar) )
            CASE('double')
               status = nf90_def_var(ncid,varname,nf90_double,     &
               &                    (/dimid1,dimid2/),varid)
            CASE('float')
               status = nf90_def_var(ncid,varname,nf90_float,      &
               &                    (/dimid1,dimid2/),varid)     
            END SELECT
            status = nf90_enddef(ncid)
         END IF
      status = nf90_put_var(ncid,varid,tabvar)     
      status = nf90_close(ncid)
      ! 
   END SUBROUTINE Write_Ncdf_var2d_Real 






   SUBROUTINE Write_Ncdf_var4d_Real_t( varname, dimname, file, tabvar, time, typevar )
      !!---------------------------------------------------------------------
      !!                    ***  SUBROUTINE Write_Ncdf_var4d_Real_t  ***
      !!                   
      !! ** Purpose : write the 4D variable varname stored in tabvar 
      !!              in the output file at time level time using typevar type (float or double)             
      !!
      !!----------------------------------------------------------------------  
      CHARACTER(*),                INTENT(in)   :: varname,file,typevar
      CHARACTER(*), DIMENSION(4) , INTENT(in)   :: dimname
      INTEGER, INTENT(in)                       :: time
      REAL(8), DIMENSION(:,:,:,:), INTENT(in)   :: tabvar
      INTEGER                                   :: dimid1,dimid2,dimid3,dimid4
      INTEGER                                   :: status,ncid
      INTEGER                                   :: varid             
!
      status = nf90_open(file,NF90_WRITE,ncid)       
      IF ( status/=nf90_noerr ) THEN    
         WRITE(*,*)"*** Write_Ncdf_var4d_Real_t: unable to open netcdf file : ",file
         STOP
      END IF            
      IF ( time==1 .and. (TRIM(typevar)=='double'  .or. TRIM(typevar)=='float') ) THEN      
         status = nf90_inq_dimid(ncid,dimname(1), dimid1)
         status = nf90_inq_dimid(ncid,dimname(2), dimid2)
         status = nf90_inq_dimid(ncid,dimname(3), dimid3)
         status = nf90_inq_dimid(ncid,dimname(4), dimid4)
         status = nf90_redef(ncid)      
         SELECT CASE(TRIM(typevar))
         CASE('double')
            status = nf90_def_var(ncid,TRIM(varname),nf90_double,     &
            &                     (/dimid1,dimid2,dimid3,dimid4/),varid)  
         CASE('float')
            status = nf90_def_var(ncid,TRIM(varname),nf90_float,     &
            &                     (/dimid1,dimid2,dimid3,dimid4/),varid)    
         END SELECT
         status = nf90_enddef(ncid)
      ELSE
         status = nf90_inq_varid(ncid, varname, varid)
      END IF    
      status = nf90_put_var(ncid,varid,tabvar,start=(/1,1,1,time/))
      IF (status/=nf90_noerr) THEN    
         WRITE(*,*)"unable to store variable ",varname, &
	                                       " in file ",file
         WRITE(*,*)"erorr code: ", status
         STOP
      END IF        
      status = nf90_close(ncid)
      !
   END SUBROUTINE Write_Ncdf_var4d_Real_t           







    
   SUBROUTINE Write_Ncdf_var1d_Real_t(varname,dimname,file,tabvar,time,typevar)
      !!---------------------------------------------------------------------
      !!                    ***  SUBROUTINE Write_Ncdf_var1d_Real_t  ***
      !!                   
      !! ** Purpose : write the 1D variable varname stored in tabvar 
      !!              in the output file at time level time using typevar type (float or double)             
      !!
      !!----------------------------------------------------------------------  
      CHARACTER(*),               INTENT(in)    :: varname,file,typevar
      CHARACTER(*), DIMENSION(1) ,INTENT(in)    :: dimname
      INTEGER, INTENT(in)                       :: time
      REAL(8), DIMENSION(:), INTENT(in)         :: tabvar
      INTEGER                                   :: dimid1
      INTEGER                                   :: status,ncid
      INTEGER                                   :: varid             
      !
      status = nf90_open(file,NF90_WRITE,ncid)       
      IF (status/=nf90_noerr) THEN    
         WRITE(*,*) "*** Write_Ncdf_var1d_Real_t: unable to open netcdf file : ",file
         STOP
      END IF          
            
      IF( time==1 ) THEN
         status = nf90_inq_dimid(ncid,dimname(1), dimid1) 
         status = nf90_redef(ncid)
         SELECT CASE ( TRIM(typevar) )
         CASE ('double')
            status = nf90_def_var(ncid,varname,nf90_double,     &
             &                                (/dimid1/),varid) 
         CASE ('float')
            status = nf90_def_var(ncid,varname,nf90_float,     &
             &                                (/dimid1/),varid)    
         END SELECT      
         status = nf90_enddef(ncid)
      ELSE
         status = nf90_inq_varid(ncid, varname, varid)
      END IF  
         
      status = nf90_put_var(ncid,varid,tabvar,start=(/time/))
      
      IF (status/=nf90_noerr) THEN    
         WRITE(*,*) "unable to store variable ",varname, &
	                                     " in file ",file
         STOP
      END IF        
      status = nf90_close(ncid)
      !  
   END SUBROUTINE Write_Ncdf_var1d_Real_t 



   SUBROUTINE Write_Ncdf_var2d_Real_t( varname, dimname, file, tabvar, time, typevar)
      !!---------------------------------------------------------------------
      !!                    ***  SUBROUTINE Write_Ncdf_var2d_Real_t  ***
      !!                   
      !! ** Purpose : write the 1D variable varname stored in tabvar 
      !!              in the output file at time level time using typevar type (float or double)             
      !!
      !!----------------------------------------------------------------------  
      CHARACTER(*),               INTENT(in)    :: varname, file, typevar
      CHARACTER(*), DIMENSION(2), INTENT(in)    :: dimname
      INTEGER, INTENT(in)                       :: time
      REAL(8), DIMENSION(:,:), INTENT(in)       :: tabvar
      INTEGER                                   :: dimid1, dimid2, dimidt
      INTEGER                                   :: status,ncid
      INTEGER                                   :: varid
      INTEGER, DIMENSION(2)                     :: dim_size
      !
      status = nf90_open( file, NF90_WRITE, ncid)       
      IF (status/=nf90_noerr) THEN    
         WRITE(*,*) "*** Write_Ncdf_var2d_Real_t: unable to open netcdf file : ",file
         STOP
      END IF          

      IF( time==1 ) THEN
         status = nf90_inq_dimid( ncid, dimname(1), dimid1) 
         status = nf90_inq_dimid( ncid, dimname(2), dimid2)
         status = nf90_inq_dimid( ncid, "time"    , dimidt)
         !status = nf90_def_dim( ncid, "time", NF90_UNLIMITED, dimidt)
         status = nf90_redef(ncid)
         SELECT CASE ( TRIM(typevar) )
         CASE ('double')
            status = nf90_def_var( ncid, varname, nf90_double,     &
             &                                (/dimid1,dimid2,dimidt/), varid) 
         CASE ('float')
            status = nf90_def_var( ncid, varname, nf90_float,     &
             &                                (/dimid1,dimid2,dimidt/), varid)    
         END SELECT
         status = nf90_enddef(ncid)
      ELSE
         status = nf90_inq_varid( ncid, varname, varid )
      END IF  

      dim_size = SHAPE(tabvar)
      status = nf90_put_var( ncid, varid, tabvar(:,:), start=(/1,1,time/), count=(/dim_size(1),dim_size(2),1/) )
      
      IF (status/=nf90_noerr) THEN    
         WRITE(*,*) "unable to store variable ",varname, &
	                                     " in file ",file
         WRITE(*,*) "error code: ", status
         STOP
      END IF        
      status = nf90_close(ncid)
      !  
   END SUBROUTINE Write_Ncdf_var2d_Real_t 



   SUBROUTINE add_globatt_real( file, att_name, att_value )

      INTEGER      :: ncid, status
      CHARACTER(*) :: file, att_name
      REAL(8)      :: att_value

      status = nf90_open( file, nf90_write, ncid)

      ! Enter define mode so we can add the attribute
      status = nf90_redef( ncid )

      ! ...  put the range attribute, setting it to eight byte reals...
      status = nf90_put_att( ncid, NF90_GLOBAL, att_name, att_value )

      ! Leave define mode
      status = nf90_enddef(ncid)

   END SUBROUTINE add_globatt_real



   SUBROUTINE Duplicate_lon_lat_time( file_in, file_out )
      !!---------------------------------------------------------------------
      !!                    ***  SUBROUTINE Duplicate_lon_lat_time  ***
      !!                   
      !! ** Purpose : duplicate the attribute of lon, lat and time variables            
      !!              from file_in in file_out
      !!----------------------------------------------------------------------  
      CHARACTER(*),               INTENT(in)    ::  file_in, file_out
      INTEGER                                   ::  status
      INTEGER                                   ::  ncid_in, ncid_out    
      INTEGER                                   :: varid_in,varid_out 
      
      status = nf90_open(file_in ,NF90_NOWRITE,ncid_in )
      status = nf90_open(file_out,NF90_WRITE  ,ncid_out)
        
      status = nf90_inq_varid(ncid_in,'lon',varid_in)
      status = nf90_inq_varid(ncid_out,'lon',varid_out)
      status = nf90_redef(ncid_out)     
      status = nf90_copy_att(ncid_in,varid_in,'standard_name',ncid_out,varid_out)
      status = nf90_copy_att(ncid_in,varid_in,'long_name',ncid_out,varid_out)
      status = nf90_copy_att(ncid_in,varid_in,'units',ncid_out,varid_out) 
      status = nf90_copy_att(ncid_in,varid_in,'axis',ncid_out,varid_out)     
      status = nf90_enddef(ncid_out)  
      
      status = nf90_inq_varid(ncid_in,'lat',varid_in)
      status = nf90_inq_varid(ncid_out,'lat',varid_out)
      status = nf90_redef(ncid_out)     
      status = nf90_copy_att(ncid_in,varid_in,'standard_name',ncid_out,varid_out)
      status = nf90_copy_att(ncid_in,varid_in,'long_name',ncid_out,varid_out)
      status = nf90_copy_att(ncid_in,varid_in,'units',ncid_out,varid_out) 
      status = nf90_copy_att(ncid_in,varid_in,'axis',ncid_out,varid_out)     
      status = nf90_enddef(ncid_out)        
      
      status = nf90_inq_varid(ncid_in,'time',varid_in)
      status = nf90_inq_varid(ncid_out,'time',varid_out)
      status = nf90_redef(ncid_out)     
      status = nf90_copy_att(ncid_in,varid_in,'standard_name',ncid_out,varid_out)
      status = nf90_copy_att(ncid_in,varid_in,'units',ncid_out,varid_out)
      status = nf90_copy_att(ncid_in,varid_in,'calendar',ncid_out,varid_out)    
      status = nf90_copy_att(ncid_in,varid_in,'axis',ncid_out,varid_out)  
      status = nf90_enddef(ncid_out)        
      
      status = nf90_close(ncid_in)
      status = nf90_close(ncid_out)
      !   
   END SUBROUTINE Duplicate_lon_lat_time 



   SUBROUTINE Duplicate_lev_hyb( file_in, file_out )
      !!---------------------------------------------------------------------
      !!                    ***  SUBROUTINE Duplicate_lon_lat_time  ***
      !!                   
      !! ** Purpose : duplicate the attribute of lon, lat and time variables            
      !!              from file_in in file_out
      !!----------------------------------------------------------------------  
      CHARACTER(*),               INTENT(in)    ::  file_in, file_out
      INTEGER                                   ::  status
      INTEGER                                   ::  ncid_in, ncid_out    
      INTEGER                                   :: varid_in,varid_out 
      
      status = nf90_open(file_in ,NF90_NOWRITE,ncid_in )
      status = nf90_open(file_out,NF90_WRITE  ,ncid_out)
        
      status = nf90_inq_varid(ncid_in,'hyai',varid_in)
      status = nf90_inq_varid(ncid_out,'hyai',varid_out)
      status = nf90_redef(ncid_out)     
      status = nf90_copy_att(ncid_in,varid_in,'long_name',ncid_out,varid_out)
      status = nf90_copy_att(ncid_in,varid_in,'units',ncid_out,varid_out)     
      status = nf90_enddef(ncid_out)  
      
      status = nf90_inq_varid(ncid_in,'hybi',varid_in)
      status = nf90_inq_varid(ncid_out,'hybi',varid_out)
      status = nf90_redef(ncid_out)     
      status = nf90_copy_att(ncid_in,varid_in,'long_name',ncid_out,varid_out)
      status = nf90_copy_att(ncid_in,varid_in,'units',ncid_out,varid_out)   
      status = nf90_enddef(ncid_out)        

      status = nf90_inq_varid(ncid_in,'hyam',varid_in)
      status = nf90_inq_varid(ncid_out,'hyam',varid_out)
      status = nf90_redef(ncid_out)     
      status = nf90_copy_att(ncid_in,varid_in,'long_name',ncid_out,varid_out)
      status = nf90_copy_att(ncid_in,varid_in,'units',ncid_out,varid_out)     
      status = nf90_enddef(ncid_out)  
      
      status = nf90_inq_varid(ncid_in,'hybm',varid_in)
      status = nf90_inq_varid(ncid_out,'hybm',varid_out)
      status = nf90_redef(ncid_out)     
      status = nf90_copy_att(ncid_in,varid_in,'long_name',ncid_out,varid_out)
      status = nf90_copy_att(ncid_in,varid_in,'units',ncid_out,varid_out)   
      status = nf90_enddef(ncid_out)  
      
      status = nf90_inq_varid(ncid_in,'lev',varid_in)
      status = nf90_inq_varid(ncid_out,'lev',varid_out)
      status = nf90_redef(ncid_out)     
      status = nf90_copy_att(ncid_in,varid_in,'standard_name',ncid_out,varid_out)
      status = nf90_copy_att(ncid_in,varid_in,'long_name',ncid_out,varid_out)      
      status = nf90_copy_att(ncid_in,varid_in,'formula',ncid_out,varid_out)    
      status = nf90_copy_att(ncid_in,varid_in,'formula_terms',ncid_out,varid_out)                
      status = nf90_copy_att(ncid_in,varid_in,'units',ncid_out,varid_out)
      status = nf90_copy_att(ncid_in,varid_in,'positive',ncid_out,varid_out)    
      status = nf90_enddef(ncid_out)        
      
      status = nf90_close(ncid_in)
      status = nf90_close(ncid_out)
      !   
   END SUBROUTINE Duplicate_lev_hyb

END MODULE module_io
