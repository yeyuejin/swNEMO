MODULE iom
   !!=====================================================================
   !!                    ***  MODULE  iom ***
   !! Input/Output manager :  Library to read input files
   !!====================================================================
   !! History :  2.0  ! 2005-12  (J. Belier) Original code
   !!            2.0  ! 2006-02  (S. Masson) Adaptation to NEMO
   !!            3.0  ! 2007-07  (D. Storkey) Changes to iom_gettime
   !!            3.4  ! 2012-12  (R. Bourdalle-Badie and G. Reffray)  add C1D case  
   !!--------------------------------------------------------------------

   !!--------------------------------------------------------------------
   !!   iom_open       : open a file read only
   !!   iom_close      : close a file or all files opened by iom
   !!   iom_get        : read a field (interfaced to several routines)
   !!   iom_gettime    : read the time axis cdvar in the file
   !!   iom_varid      : get the id of a variable in a file
   !!   iom_rstput     : write a field in a restart file (interfaced to several routines)
   !!--------------------------------------------------------------------
   USE dom_oce         ! ocean space and time domain
   USE lbclnk          ! lateal boundary condition / mpp exchanges
   USE iom_def         ! iom variables definitions
   USE iom_nf90        ! NetCDF format with native NetCDF library
   USE in_out_manager  ! I/O manager
   USE lib_mpp           ! MPP library

   IMPLICIT NONE
   PUBLIC   !   must be public to be able to access iom_def through iom
   
   LOGICAL, PUBLIC, PARAMETER ::   lk_iomput = .FALSE.       !: iom_put flag
   PUBLIC iom_init, iom_swap, iom_open, iom_close, iom_setkt, iom_varid, iom_get, iom_gettime, iom_rstput, iom_put
   PUBLIC iom_getatt, iom_use, iom_context_finalize

   PRIVATE iom_rp0d, iom_rp1d, iom_rp2d, iom_rp3d
   PRIVATE iom_g0d, iom_g1d, iom_g2d, iom_g3d, iom_get_123d
   PRIVATE iom_p1d, iom_p2d, iom_p3d

   INTERFACE iom_get
      MODULE PROCEDURE iom_g0d, iom_g1d, iom_g2d, iom_g3d
   END INTERFACE
   INTERFACE iom_getatt
      MODULE PROCEDURE iom_g0d_intatt
   END INTERFACE
   INTERFACE iom_rstput
      MODULE PROCEDURE iom_rp0d, iom_rp1d, iom_rp2d, iom_rp3d
   END INTERFACE
  INTERFACE iom_put
     MODULE PROCEDURE iom_p0d, iom_p1d, iom_p2d, iom_p3d
  END INTERFACE

   !!----------------------------------------------------------------------
   !! NEMO/OCE 4.0 , NEMO Consortium (2018)
   !! $Id: iom.F90 8572 2017-09-28 08:27:06Z cbricaud $
   !! Software governed by the CeCILL licence (./LICENSE)
   !!----------------------------------------------------------------------

CONTAINS

   SUBROUTINE iom_init( cdname ) 
      !!----------------------------------------------------------------------
      !!                     ***  ROUTINE   ***
      !!
      !! ** Purpose :   
      !!
      !!----------------------------------------------------------------------
      CHARACTER(len=*), INTENT(in)  :: cdname
      
   END SUBROUTINE iom_init


   SUBROUTINE iom_swap( cdname )
      !!---------------------------------------------------------------------
      !!                   ***  SUBROUTINE  iom_swap  ***
      !!
      !! ** Purpose :  swap context between different agrif grid for xmlio_server
      !!---------------------------------------------------------------------
      CHARACTER(len=*), INTENT(in) :: cdname
      !
   END SUBROUTINE iom_swap


   SUBROUTINE iom_open( cdname, kiomid, ldwrt, kdom, kiolib, ldstop, ldiof )
      !!---------------------------------------------------------------------
      !!                   ***  SUBROUTINE  iom_open  ***
      !!
      !! ** Purpose :  open an input file (return 0 if not found)
      !!---------------------------------------------------------------------
      CHARACTER(len=*), INTENT(in   )           ::   cdname   ! File name
      INTEGER         , INTENT(  out)           ::   kiomid   ! iom identifier of the opened file
      LOGICAL         , INTENT(in   ), OPTIONAL ::   ldwrt    ! open in write modeb          (default = .FALSE.)
      INTEGER         , INTENT(in   ), OPTIONAL ::   kdom     ! Type of domain to be written (default = jpdom_local_noovlap)
      INTEGER         , INTENT(in   ), OPTIONAL ::   kiolib   ! library used to open the file (default = jpnf90) 
      LOGICAL         , INTENT(in   ), OPTIONAL ::   ldstop   ! stop if open to read a non-existing file (default = .TRUE.)
      LOGICAL         , INTENT(in   ), OPTIONAL ::   ldiof    ! Interp On the Fly, needed for AGRIF (default = .FALSE.)

      CHARACTER(LEN=256)    ::   clname    ! the name of the file based on cdname [[+clcpu]+clcpu]
      CHARACTER(LEN=256)    ::   cltmpn    ! tempory name to store clname (in writting mode)
      CHARACTER(LEN=10)     ::   clsuffix  ! ".nc" or ".dimg"
      CHARACTER(LEN=15)     ::   clcpu     ! the cpu number (max jpmax_digits digits)
      CHARACTER(LEN=256)    ::   clinfo    ! info character
      LOGICAL               ::   llok      ! check the existence 
      LOGICAL               ::   llwrt     ! local definition of ldwrt
      LOGICAL               ::   llnoov    ! local definition to read overlap
      LOGICAL               ::   llstop    ! local definition of ldstop
      LOGICAL               ::   lliof     ! local definition of ldiof
      INTEGER               ::   iolib     ! library do we use to open the file
      INTEGER               ::   icnt      ! counter for digits in clcpu (max = jpmax_digits)
      INTEGER               ::   iln, ils  ! lengths of character
      INTEGER               ::   idom      ! type of domain
      INTEGER               ::   istop     ! 
      INTEGER, DIMENSION(2,5) ::   idompar ! domain parameters: 
      ! local number of points for x,y dimensions
      ! position of first local point for x,y dimensions
      ! position of last local point for x,y dimensions
      ! start halo size for x,y dimensions
      ! end halo size for x,y dimensions
      !---------------------------------------------------------------------
      ! Initializations and control
      ! =============
      kiomid = -1
      clinfo = '                    iom_open ~~~  '
      istop = nstop
      ! if iom_open is called for the first time: initialize iom_file(:)%nfid to 0
      ! (could be done when defining iom_file in f95 but not in f90)
      IF( Agrif_Root() ) THEN
         IF( iom_open_init == 0 ) THEN
            iom_file(:)%nfid = 0
            iom_open_init = 1
         ENDIF
      ENDIF
      ! do we read or write the file?
      IF( PRESENT(ldwrt) ) THEN   ;   llwrt = ldwrt
      ELSE                        ;   llwrt = .FALSE.
      ENDIF
      ! do we call ctl_stop if we try to open a non-existing file in read mode?
      IF( PRESENT(ldstop) ) THEN   ;   llstop = ldstop
      ELSE                         ;   llstop = .TRUE.
      ENDIF
      ! what library do we use to open the file?
      IF( PRESENT(kiolib) ) THEN   ;   iolib = kiolib
      ELSE                         ;   iolib = jpnf90
      ENDIF
      ! are we using interpolation on the fly?
      IF( PRESENT(ldiof) ) THEN   ;   lliof = ldiof
      ELSE                        ;   lliof = .FALSE.
      ENDIF
      ! do we read the overlap 
      ! ugly patch SM+JMM+RB to overwrite global definition in some cases
      llnoov = (jpni * jpnj ) == jpnij .AND. .NOT. lk_agrif 
      ! create the file name by added, if needed, TRIM(Agrif_CFixed()) and TRIM(clsuffix)
      ! =============
      clname   = trim(cdname)
      IF ( .NOT. Agrif_Root() .AND. .NOT. lliof ) THEN
         iln    = INDEX(clname,'/') 
         cltmpn = clname(1:iln)
         clname = clname(iln+1:LEN_TRIM(clname))
         clname=TRIM(cltmpn)//TRIM(Agrif_CFixed())//'_'//TRIM(clname)
      ENDIF
      ! which suffix should we use?
      SELECT CASE (iolib)
      CASE (jpnf90   ) ;   clsuffix = '.nc'
      CASE DEFAULT     ;   clsuffix = ''
      END SELECT
      ! Add the suffix if needed
      iln = LEN_TRIM(clname)
      ils = LEN_TRIM(clsuffix)
      IF( iln <= ils .OR. INDEX( TRIM(clname), TRIM(clsuffix), back = .TRUE. ) /= iln - ils + 1 )   &
         &   clname = TRIM(clname)//TRIM(clsuffix)
      cltmpn = clname   ! store this name
      ! try to find if the file to be opened already exist
      ! =============
      INQUIRE( FILE = clname, EXIST = llok )
      IF( .NOT.llok ) THEN
         ! we try to add the cpu number to the name
            WRITE(clcpu,*) narea-1
         clcpu  = TRIM(ADJUSTL(clcpu))
         iln = INDEX(clname,TRIM(clsuffix), back = .TRUE.)
         clname = clname(1:iln-1)//'_'//TRIM(clcpu)//TRIM(clsuffix)
         icnt = 0
         INQUIRE( FILE = clname, EXIST = llok ) 
         ! we try different formats for the cpu number by adding 0
         DO WHILE( .NOT.llok .AND. icnt < jpmax_digits )
            clcpu  = "0"//trim(clcpu)
            clname = clname(1:iln-1)//'_'//TRIM(clcpu)//TRIM(clsuffix)
            INQUIRE( FILE = clname, EXIST = llok )
            icnt = icnt + 1
         END DO
      ENDIF
      IF( llwrt ) THEN
         ! check the domain definition
! JMM + SM: ugly patch before getting the new version of lib_mpp)
!         idom = jpdom_local_noovlap   ! default definition
         IF( llnoov ) THEN   ;   idom = jpdom_local_noovlap   ! default definition
         ELSE                ;   idom = jpdom_local_full      ! default definition
         ENDIF
         IF( PRESENT(kdom) )   idom = kdom
         ! create the domain informations
         ! =============
         SELECT CASE (idom)
         CASE (jpdom_local_full)
            idompar(:,1) = (/ jpi             , jpj              /)
            idompar(:,2) = (/ nimpp           , njmpp            /)
            idompar(:,3) = (/ nimpp + jpi - 1 , njmpp + jpj - 1  /)
            idompar(:,4) = (/ nldi - 1        , nldj - 1         /)
            idompar(:,5) = (/ jpi - nlei      , jpj - nlej       /)
         CASE (jpdom_local_noextra)
            idompar(:,1) = (/ nlci            , nlcj             /)
            idompar(:,2) = (/ nimpp           , njmpp            /)
            idompar(:,3) = (/ nimpp + nlci - 1, njmpp + nlcj - 1 /)
            idompar(:,4) = (/ nldi - 1        , nldj - 1         /)
            idompar(:,5) = (/ nlci - nlei     , nlcj - nlej      /)
         CASE (jpdom_local_noovlap)
            idompar(:,1) = (/ nlei  - nldi + 1, nlej  - nldj + 1 /)
            idompar(:,2) = (/ nimpp + nldi - 1, njmpp + nldj - 1 /)
            idompar(:,3) = (/ nimpp + nlei - 1, njmpp + nlej - 1 /)
            idompar(:,4) = (/ 0               , 0                /)
            idompar(:,5) = (/ 0               , 0                /)
         CASE DEFAULT
            CALL ctl_stop( TRIM(clinfo), 'wrong value of kdom, only jpdom_local* cases are accepted' )
         END SELECT
      ENDIF
      ! Open the NetCDF or RSTDIMG file
      ! =============
      ! do we have some free file identifier?
      IF( MINVAL(iom_file(:)%nfid) /= 0 )   &
         &   CALL ctl_stop( TRIM(clinfo), 'No more free file identifier', 'increase jpmax_files in iom_def' )
      ! if no file was found...
      IF( .NOT. llok ) THEN
         IF( .NOT. llwrt ) THEN   ! we are in read mode 
            IF( llstop ) THEN   ;   CALL ctl_stop( TRIM(clinfo), 'File '//TRIM(cltmpn)//'* not found' )
            ELSE                ;   istop = nstop + 1   ! make sure that istop /= nstop so we don't open the file
            ENDIF
         ELSE                     ! we are in write mode so we 
            clname = cltmpn       ! get back the file name without the cpu number
         ENDIF
      ELSE
         IF( llwrt .AND. .NOT. ln_clobber ) THEN   ! we stop as we want to write in a new file 
            CALL ctl_stop( TRIM(clinfo), 'We want to write in a new file but '//TRIM(clname)//' already exists...' )
            istop = nstop + 1                      ! make sure that istop /= nstop so we don't open the file
         ELSEIF( llwrt ) THEN     ! the file exists and we are in write mode with permission to 
            clname = cltmpn       ! overwrite so get back the file name without the cpu number
         ENDIF
      ENDIF
      IF( istop == nstop ) THEN   ! no error within this routine
         SELECT CASE (iolib)
         CASE (jpnf90   )   ;   CALL iom_nf90_open(    clname, kiomid, llwrt, llok, idompar )
         CASE DEFAULT
         END SELECT
      ENDIF
      !
   END SUBROUTINE iom_open


   SUBROUTINE iom_close( kiomid )
      !!--------------------------------------------------------------------
      !!                   ***  SUBROUTINE  iom_close  ***
      !!
      !! ** Purpose : close an input file, or all files opened by iom
      !!--------------------------------------------------------------------
      INTEGER, INTENT(inout), OPTIONAL ::   kiomid   ! iom identifier of the file to be closed
      !                                              ! return 0 when file is properly closed
      !                                              ! No argument: all files opened by iom are closed

      INTEGER ::   jf         ! dummy loop indices
      INTEGER ::   i_s, i_e   ! temporary integer
      CHARACTER(LEN=100)    ::   clinfo    ! info character
      !---------------------------------------------------------------------
      !
      clinfo = '                    iom_close ~~~  '
      IF( PRESENT(kiomid) ) THEN
         i_s = kiomid
         i_e = kiomid
      ELSE
         i_s = 1
         i_e = jpmax_files
      ENDIF

      IF( i_s > 0 ) THEN
         DO jf = i_s, i_e
            IF( iom_file(jf)%nfid > 0 ) THEN
               SELECT CASE (iom_file(jf)%iolib)
               CASE (jpnf90   )   ;   CALL iom_nf90_close(    jf )
               CASE DEFAULT
               END SELECT
               iom_file(jf)%nfid       = 0          ! free the id 
               IF( PRESENT(kiomid) )   kiomid = 0   ! return 0 as id to specify that the file was closed
               IF(lwp) WRITE(numout,*) TRIM(clinfo)//' close file: '//TRIM(iom_file(jf)%name)//' ok'
            ELSEIF( PRESENT(kiomid) ) THEN
               WRITE(ctmp1,*) '--->',  kiomid
               CALL ctl_stop( TRIM(clinfo)//' Invalid file identifier', ctmp1 )
            ENDIF
         END DO
      ENDIF
      !    
   END SUBROUTINE iom_close


   FUNCTION iom_varid ( kiomid, cdvar, kdimsz, kndims, ldstop )  
      !!-----------------------------------------------------------------------
      !!                  ***  FUNCTION  iom_varid  ***
      !!
      !! ** Purpose : get the id of a variable in a file (return 0 if not found)
      !!-----------------------------------------------------------------------
      INTEGER              , INTENT(in   )           ::   kiomid   ! file Identifier
      CHARACTER(len=*)     , INTENT(in   )           ::   cdvar    ! name of the variable
      INTEGER, DIMENSION(:), INTENT(  out), OPTIONAL ::   kdimsz   ! size of the dimensions
      INTEGER,               INTENT(  out), OPTIONAL ::   kndims   ! size of the dimensions
      LOGICAL              , INTENT(in   ), OPTIONAL ::   ldstop   ! stop if looking for non-existing variable (default = .TRUE.)
      !
      INTEGER                        ::   iom_varid, iiv, i_nvd
      LOGICAL                        ::   ll_fnd
      CHARACTER(LEN=100)             ::   clinfo                   ! info character
      LOGICAL                        ::   llstop                   ! local definition of ldstop
      !!-----------------------------------------------------------------------
      iom_varid = 0                         ! default definition
      ! do we call ctl_stop if we look for non-existing variable?
      IF( PRESENT(ldstop) ) THEN   ;   llstop = ldstop
      ELSE                         ;   llstop = .TRUE.
      ENDIF
      !
      IF( kiomid > 0 ) THEN
         clinfo = 'iom_varid, file: '//trim(iom_file(kiomid)%name)//', var: '//trim(cdvar)
         IF( iom_file(kiomid)%nfid == 0 ) THEN 
            CALL ctl_stop( trim(clinfo), 'the file is not open' )
         ELSE
            ll_fnd  = .FALSE.
            iiv = 0
            !
            DO WHILE ( .NOT.ll_fnd .AND. iiv < iom_file(kiomid)%nvars )
               iiv = iiv + 1
               ll_fnd  = ( TRIM(cdvar) == TRIM(iom_file(kiomid)%cn_var(iiv)) )
            END DO
            !
            IF( .NOT.ll_fnd ) THEN
               iiv = iiv + 1
               IF( iiv <= jpmax_vars ) THEN
                  SELECT CASE (iom_file(kiomid)%iolib)
                  CASE (jpnf90   )   ;   iom_varid = iom_nf90_varid  ( kiomid, cdvar, iiv, kdimsz, kndims )
                  CASE DEFAULT   
                  END SELECT
               ELSE
                  CALL ctl_stop( trim(clinfo), 'Too many variables in the file '//iom_file(kiomid)%name,   &
                        &                         'increase the parameter jpmax_vars')
               ENDIF
               IF( llstop .AND. iom_varid == -1 )   CALL ctl_stop( TRIM(clinfo)//' not found' ) 
            ELSE
               iom_varid = iiv
               IF( PRESENT(kdimsz) ) THEN 
                  i_nvd = iom_file(kiomid)%ndims(iiv)
                  IF( i_nvd == size(kdimsz) ) THEN
                     kdimsz(:) = iom_file(kiomid)%dimsz(1:i_nvd,iiv)
                  ELSE
                     WRITE(ctmp1,*) i_nvd, size(kdimsz)
                     CALL ctl_stop( trim(clinfo), 'error in kdimsz size'//trim(ctmp1) )
                  ENDIF
               ENDIF
               IF( PRESENT(kndims) )  kndims = iom_file(kiomid)%ndims(iiv)
            ENDIF
         ENDIF
      ENDIF
      !
   END FUNCTION iom_varid


   !!----------------------------------------------------------------------
   !!                   INTERFACE iom_get
   !!----------------------------------------------------------------------
   SUBROUTINE iom_g0d( kiomid, cdvar, pvar, ktime )
      INTEGER         , INTENT(in   )                 ::   kiomid    ! Identifier of the file
      CHARACTER(len=*), INTENT(in   )                 ::   cdvar     ! Name of the variable
      REAL(wp)        , INTENT(  out)                 ::   pvar      ! read field
      INTEGER         , INTENT(in   ),     OPTIONAL   ::   ktime     ! record number
      !
      INTEGER                                         ::   idvar     ! variable id
      INTEGER                                         ::   idmspc    ! number of spatial dimensions
      INTEGER         , DIMENSION(1)                  ::   itime     ! record number
      CHARACTER(LEN=100)                              ::   clinfo    ! info character
      CHARACTER(LEN=100)                              ::   clname    ! file name
      CHARACTER(LEN=1)                                ::   cldmspc   !
      !
      itime = 1
      IF( PRESENT(ktime) ) itime = ktime
      !
      clname = iom_file(kiomid)%name
      clinfo = '          iom_g0d, file: '//trim(clname)//', var: '//trim(cdvar)
      !
      IF( kiomid > 0 ) THEN
         idvar = iom_varid( kiomid, cdvar )
         IF( iom_file(kiomid)%nfid > 0 .AND. idvar > 0 ) THEN
            idmspc = iom_file ( kiomid )%ndims( idvar )
            IF( iom_file(kiomid)%luld(idvar) )  idmspc = idmspc - 1
            WRITE(cldmspc , fmt='(i1)') idmspc
            IF( idmspc > 0 )  CALL ctl_stop( TRIM(clinfo), 'When reading to a 0D array, we do not accept data', &
                                 &                         'with 1 or more spatial dimensions: '//cldmspc//' were found.' , &
                                 &                         'Use ncwa -a to suppress the unnecessary dimensions' )
            SELECT CASE (iom_file(kiomid)%iolib)
            CASE (jpnf90   )   ;   CALL iom_nf90_get(    kiomid, idvar, pvar, itime )
            CASE DEFAULT    
            END SELECT
         ENDIF
      ENDIF
   END SUBROUTINE iom_g0d

   SUBROUTINE iom_g1d( kiomid, kdom, cdvar, pvar, ktime, kstart, kcount )
      INTEGER         , INTENT(in   )                         ::   kiomid    ! Identifier of the file
      INTEGER         , INTENT(in   )                         ::   kdom      ! Type of domain to be read
      CHARACTER(len=*), INTENT(in   )                         ::   cdvar     ! Name of the variable
      REAL(wp)        , INTENT(  out), DIMENSION(:)           ::   pvar      ! read field
      INTEGER         , INTENT(in   )              , OPTIONAL ::   ktime     ! record number
      INTEGER         , INTENT(in   ), DIMENSION(1), OPTIONAL ::   kstart    ! start axis position of the reading 
      INTEGER         , INTENT(in   ), DIMENSION(1), OPTIONAL ::   kcount    ! number of points in each axis
      !
      IF( kiomid > 0 ) THEN
         IF( iom_file(kiomid)%nfid > 0 ) CALL iom_get_123d( kiomid, kdom       , cdvar        , pv_r1d=pvar,   &
              &                                                     ktime=ktime, kstart=kstart, kcount=kcount )
      ENDIF
   END SUBROUTINE iom_g1d

   SUBROUTINE iom_g2d( kiomid, kdom, cdvar, pvar, ktime, kstart, kcount, lrowattr )
      INTEGER         , INTENT(in   )                           ::   kiomid    ! Identifier of the file
      INTEGER         , INTENT(in   )                           ::   kdom      ! Type of domain to be read
      CHARACTER(len=*), INTENT(in   )                           ::   cdvar     ! Name of the variable
      REAL(wp)        , INTENT(  out), DIMENSION(:,:)           ::   pvar      ! read field
      INTEGER         , INTENT(in   )                , OPTIONAL ::   ktime     ! record number
      INTEGER         , INTENT(in   ), DIMENSION(2)  , OPTIONAL ::   kstart    ! start axis position of the reading 
      INTEGER         , INTENT(in   ), DIMENSION(2)  , OPTIONAL ::   kcount    ! number of points in each axis
      LOGICAL         , INTENT(in   )                , OPTIONAL ::   lrowattr  ! logical flag telling iom_get to
                                                                               ! look for and use a file attribute
                                                                               ! called open_ocean_jstart to set the start
                                                                               ! value for the 2nd dimension (netcdf only)
      !
      IF( kiomid > 0 ) THEN
         IF( iom_file(kiomid)%nfid > 0 ) CALL iom_get_123d( kiomid, kdom       , cdvar        , pv_r2d=pvar,   &
              &                                                     ktime=ktime, kstart=kstart, kcount=kcount, &
              &                                                     lrowattr=lrowattr )
      ENDIF
   END SUBROUTINE iom_g2d

   SUBROUTINE iom_g3d( kiomid, kdom, cdvar, pvar, ktime, kstart, kcount, lrowattr )
      INTEGER         , INTENT(in   )                             ::   kiomid    ! Identifier of the file
      INTEGER         , INTENT(in   )                             ::   kdom      ! Type of domain to be read
      CHARACTER(len=*), INTENT(in   )                             ::   cdvar     ! Name of the variable
      REAL(wp)        , INTENT(  out), DIMENSION(:,:,:)           ::   pvar      ! read field
      INTEGER         , INTENT(in   )                  , OPTIONAL ::   ktime     ! record number
      INTEGER         , INTENT(in   ), DIMENSION(3)    , OPTIONAL ::   kstart    ! start axis position of the reading 
      INTEGER         , INTENT(in   ), DIMENSION(3)    , OPTIONAL ::   kcount    ! number of points in each axis
      LOGICAL         , INTENT(in   )                  , OPTIONAL ::   lrowattr  ! logical flag telling iom_get to
                                                                                 ! look for and use a file attribute
                                                                                 ! called open_ocean_jstart to set the start
                                                                                 ! value for the 2nd dimension (netcdf only)
      !
      IF( kiomid > 0 ) THEN
         IF( iom_file(kiomid)%nfid > 0 ) CALL iom_get_123d( kiomid, kdom       , cdvar        , pv_r3d=pvar,   &
              &                                                     ktime=ktime, kstart=kstart, kcount=kcount, &
              &                                                     lrowattr=lrowattr )
      ENDIF
   END SUBROUTINE iom_g3d
   !!----------------------------------------------------------------------

   SUBROUTINE iom_get_123d( kiomid, kdom  , cdvar ,   &
         &                  pv_r1d, pv_r2d, pv_r3d,   &
         &                  ktime , kstart, kcount,   &
         &                  lrowattr                )
      !!-----------------------------------------------------------------------
      !!                  ***  ROUTINE  iom_get_123d  ***
      !!
      !! ** Purpose : read a 1D/2D/3D variable
      !!
      !! ** Method : read ONE record at each CALL
      !!-----------------------------------------------------------------------
      INTEGER                    , INTENT(in   )           ::   kiomid     ! Identifier of the file
      INTEGER                    , INTENT(in   )           ::   kdom       ! Type of domain to be read
      CHARACTER(len=*)           , INTENT(in   )           ::   cdvar      ! Name of the variable
      REAL(wp), DIMENSION(:)     , INTENT(  out), OPTIONAL ::   pv_r1d     ! read field (1D case)
      REAL(wp), DIMENSION(:,:)   , INTENT(  out), OPTIONAL ::   pv_r2d     ! read field (2D case)
      REAL(wp), DIMENSION(:,:,:) , INTENT(  out), OPTIONAL ::   pv_r3d     ! read field (3D case)
      INTEGER                    , INTENT(in   ), OPTIONAL ::   ktime      ! record number
      INTEGER , DIMENSION(:)     , INTENT(in   ), OPTIONAL ::   kstart     ! start position of the reading in each axis 
      INTEGER , DIMENSION(:)     , INTENT(in   ), OPTIONAL ::   kcount     ! number of points to be read in each axis
      LOGICAL                    , INTENT(in   ), OPTIONAL ::   lrowattr   ! logical flag telling iom_get to
                                                                           ! look for and use a file attribute
                                                                           ! called open_ocean_jstart to set the start
                                                                           ! value for the 2nd dimension (netcdf only)
      !
      LOGICAL                        ::   llnoov      ! local definition to read overlap
      LOGICAL                        ::   luse_jattr  ! local definition to read open_ocean_jstart file attribute
      INTEGER                        ::   jstartrow   ! start point for 2nd dimension optionally set by file attribute
      INTEGER                        ::   jl          ! loop on number of dimension 
      INTEGER                        ::   idom        ! type of domain
      INTEGER                        ::   idvar       ! id of the variable
      INTEGER                        ::   inbdim      ! number of dimensions of the variable
      INTEGER                        ::   idmspc      ! number of spatial dimensions 
      INTEGER                        ::   itime       ! record number
      INTEGER                        ::   istop       ! temporary value of nstop
      INTEGER                        ::   ix1, ix2, iy1, iy2   ! subdomain indexes
      INTEGER                        ::   ji, jj      ! loop counters
      INTEGER                        ::   irankpv     ! 
      INTEGER                        ::   ind1, ind2  ! substring index
      INTEGER, DIMENSION(jpmax_dims) ::   istart      ! starting point to read for each axis
      INTEGER, DIMENSION(jpmax_dims) ::   icnt        ! number of value to read along each axis 
      INTEGER, DIMENSION(jpmax_dims) ::   idimsz      ! size of the dimensions of the variable
      INTEGER, DIMENSION(jpmax_dims) ::   ishape      ! size of the dimensions of the variable
      REAL(wp)                       ::   zscf, zofs  ! sacle_factor and add_offset
      INTEGER                        ::   itmp        ! temporary integer
      CHARACTER(LEN=256)             ::   clinfo      ! info character
      CHARACTER(LEN=256)             ::   clname      ! file name
      CHARACTER(LEN=1)               ::   clrankpv, cldmspc      ! 
      !---------------------------------------------------------------------
      !
      clname = iom_file(kiomid)%name   !   esier to read
      clinfo = '          iom_get_123d, file: '//trim(clname)//', var: '//trim(cdvar)
      ! local definition of the domain ?
      idom = kdom
      ! do we read the overlap 
      ! ugly patch SM+JMM+RB to overwrite global definition in some cases
      llnoov = (jpni * jpnj ) == jpnij .AND. .NOT. lk_agrif 
      ! check kcount and kstart optionals parameters...
      IF( PRESENT(kcount) .AND. (.NOT. PRESENT(kstart)) ) CALL ctl_stop(trim(clinfo), 'kcount present needs kstart present')
      IF( PRESENT(kstart) .AND. (.NOT. PRESENT(kcount)) ) CALL ctl_stop(trim(clinfo), 'kstart present needs kcount present')
      IF( PRESENT(kstart) .AND. idom /= jpdom_unknown   ) CALL ctl_stop(trim(clinfo), 'kstart present needs kdom = jpdom_unknown')

      luse_jattr = .false.
      IF( PRESENT(lrowattr) ) THEN
         IF( lrowattr .AND. idom /= jpdom_data   ) CALL ctl_stop(trim(clinfo), 'lrowattr present and true needs kdom = jpdom_data')
         IF( lrowattr .AND. idom == jpdom_data   ) luse_jattr = .true.
      ENDIF
      IF( luse_jattr ) THEN
         SELECT CASE (iom_file(kiomid)%iolib)
         CASE (jpnf90   )   
             ! Ok
         CASE DEFAULT    
         END SELECT
      ENDIF

      ! Search for the variable in the data base (eventually actualize data)
      istop = nstop
      idvar = iom_varid( kiomid, cdvar )
      !
      IF( idvar > 0 ) THEN
         ! to write iom_file(kiomid)%dimsz in a shorter way !
         idimsz(:) = iom_file(kiomid)%dimsz(:, idvar) 
         inbdim = iom_file(kiomid)%ndims(idvar)            ! number of dimensions in the file
         idmspc = inbdim                                   ! number of spatial dimensions in the file
         IF( iom_file(kiomid)%luld(idvar) )   idmspc = inbdim - 1
         IF( idmspc > 3 )   CALL ctl_stop(trim(clinfo), 'the file has more than 3 spatial dimensions this case is not coded...') 
         !
         ! update idom definition...
         ! Identify the domain in case of jpdom_auto(glo/dta) definition
         IF( idom == jpdom_autoglo .OR. idom == jpdom_autodta ) THEN            
            IF( idom == jpdom_autoglo ) THEN   ;   idom = jpdom_global 
            ELSE                               ;   idom = jpdom_data
            ENDIF
            ind1 = INDEX( clname, '_', back = .TRUE. ) + 1
            ind2 = INDEX( clname, '.', back = .TRUE. ) - 1
            IF( ind2 > ind1 ) THEN   ;   IF( VERIFY( clname(ind1:ind2), '0123456789' ) == 0 )   idom = jpdom_local   ;   ENDIF
         ENDIF
         ! Identify the domain in case of jpdom_local definition
         IF( idom == jpdom_local ) THEN
            IF(     idimsz(1) == jpi               .AND. idimsz(2) == jpj               ) THEN   ;   idom = jpdom_local_full
            ELSEIF( idimsz(1) == nlci              .AND. idimsz(2) == nlcj              ) THEN   ;   idom = jpdom_local_noextra
            ELSEIF( idimsz(1) == (nlei - nldi + 1) .AND. idimsz(2) == (nlej - nldj + 1) ) THEN   ;   idom = jpdom_local_noovlap
            ELSE   ;   CALL ctl_stop( trim(clinfo), 'impossible to identify the local domain' )
            ENDIF
         ENDIF
         !
         ! check the consistency between input array and data rank in the file
         !
         ! initializations
         itime = 1
         IF( PRESENT(ktime) ) itime = ktime

         irankpv = 1 * COUNT( (/PRESENT(pv_r1d)/) ) + 2 * COUNT( (/PRESENT(pv_r2d)/) ) + 3 * COUNT( (/PRESENT(pv_r3d)/) )
         WRITE(clrankpv, fmt='(i1)') irankpv
         WRITE(cldmspc , fmt='(i1)') idmspc
         !
         IF(     idmspc <  irankpv ) THEN 
            CALL ctl_stop( TRIM(clinfo), 'The file has only '//cldmspc//' spatial dimension',   &
               &                         'it is impossible to read a '//clrankpv//'D array from this file...' )
         ELSEIF( idmspc == irankpv ) THEN
            IF( PRESENT(pv_r1d) .AND. idom /= jpdom_unknown )   &
               &   CALL ctl_stop( TRIM(clinfo), 'case not coded...You must use jpdom_unknown' )
         ELSEIF( idmspc >  irankpv ) THEN
               IF( PRESENT(pv_r2d) .AND. itime == 1 .AND. idimsz(3) == 1 .AND. idmspc == 3 ) THEN
                  CALL ctl_warn( trim(clinfo), '2D array but 3 spatial dimensions for the data...'              ,   &
                        &         'As the size of the z dimension is 1 and as we try to read the first record, ',   &
                        &         'we accept this case, even if there is a possible mix-up between z and time dimension' )   
                  idmspc = idmspc - 1
               ELSE
                  CALL ctl_stop( TRIM(clinfo), 'To keep iom lisibility, when reading a '//clrankpv//'D array,'         ,   &
                     &                         'we do not accept data with '//cldmspc//' spatial dimensions',   &
                     &                         'Use ncwa -a to suppress the unnecessary dimensions' )
               ENDIF
         ENDIF

         !
         ! definition of istart and icnt
         !
         icnt  (:) = 1
         istart(:) = 1
         istart(idmspc+1) = itime

         IF(              PRESENT(kstart)       ) THEN ; istart(1:idmspc) = kstart(1:idmspc) ; icnt(1:idmspc) = kcount(1:idmspc)
         ELSE
            IF(           idom == jpdom_unknown ) THEN                                       ; icnt(1:idmspc) = idimsz(1:idmspc)
            ELSE 
               IF( .NOT. PRESENT(pv_r1d) ) THEN   !   not a 1D array
                  IF(     idom == jpdom_data    ) THEN
                     jstartrow = 1
                     IF( luse_jattr ) THEN
                        CALL iom_getatt(kiomid, 'open_ocean_jstart', jstartrow ) ! -999 is returned if the attribute is not found
                        jstartrow = MAX(1,jstartrow)
                     ENDIF
                     istart(1:2) = (/ mig(1), mjg(1) + jstartrow - 1 /)  ! icnt(1:2) done below
                  ELSEIF( idom == jpdom_global  ) THEN ; istart(1:2) = (/ nimpp , njmpp  /)  ! icnt(1:2) done below
                  ENDIF
                  ! we do not read the overlap                     -> we start to read at nldi, nldj
! JMM + SM: ugly patch before getting the new version of lib_mpp)
!                  IF( idom /= jpdom_local_noovlap )   istart(1:2) = istart(1:2) + (/ nldi - 1, nldj - 1 /)
                  IF( llnoov .AND. idom /= jpdom_local_noovlap ) istart(1:2) = istart(1:2) + (/ nldi - 1, nldj - 1 /)
                  ! we do not read the overlap and the extra-halos -> from nldi to nlei and from nldj to nlej 
! JMM + SM: ugly patch before getting the new version of lib_mpp)
!                  icnt(1:2) = (/ nlei - nldi + 1, nlej - nldj + 1 /)
                  IF( llnoov ) THEN   ;   icnt(1:2) = (/ nlei - nldi + 1, nlej - nldj + 1 /)
                  ELSE                ;   icnt(1:2) = (/ nlci           , nlcj            /)
                  ENDIF
                  IF( PRESENT(pv_r3d) ) THEN
                     IF( idom == jpdom_data ) THEN   ; icnt(3) = jpkdta
                     ELSE                            ; icnt(3) = jpk
                     ENDIF
                  ENDIF
               ENDIF
            ENDIF
         ENDIF

         ! check that istart and icnt can be used with this file
         !-
         DO jl = 1, jpmax_dims
            itmp = istart(jl)+icnt(jl)-1
            IF( itmp > idimsz(jl) .AND. idimsz(jl) /= 0 ) THEN
               WRITE( ctmp1, FMT="('(istart(', i1, ') + icnt(', i1, ') - 1) = ', i5)" ) jl, jl, itmp
               WRITE( ctmp2, FMT="(' is larger than idimsz(', i1,') = ', i5)"         ) jl, idimsz(jl)
               CALL ctl_stop( trim(clinfo), 'start and count too big regarding to the size of the data, ', ctmp1, ctmp2 )     
            ENDIF
         END DO

         ! check that icnt matches the input array
         !-     
         IF( idom == jpdom_unknown ) THEN
            IF( irankpv == 1 )        ishape(1:1) = SHAPE(pv_r1d)
            IF( irankpv == 2 )        ishape(1:2) = SHAPE(pv_r2d)
            IF( irankpv == 3 )        ishape(1:3) = SHAPE(pv_r3d)
            ctmp1 = 'd'
         ELSE
            IF( irankpv == 2 ) THEN
! JMM + SM: ugly patch before getting the new version of lib_mpp)
!               ishape(1:2) = SHAPE(pv_r2d(nldi:nlei,nldj:nlej  ))   ;   ctmp1 = 'd(nldi:nlei,nldj:nlej)'
               IF( llnoov ) THEN ; ishape(1:2)=SHAPE(pv_r2d(nldi:nlei,nldj:nlej  )) ; ctmp1='d(nldi:nlei,nldj:nlej)'
               ELSE              ; ishape(1:2)=SHAPE(pv_r2d(1   :nlci,1   :nlcj  )) ; ctmp1='d(1:nlci,1:nlcj)'
               ENDIF
            ENDIF
            IF( irankpv == 3 ) THEN 
! JMM + SM: ugly patch before getting the new version of lib_mpp)
!               ishape(1:3) = SHAPE(pv_r3d(nldi:nlei,nldj:nlej,:))   ;   ctmp1 = 'd(nldi:nlei,nldj:nlej,:)'
               IF( llnoov ) THEN ; ishape(1:3)=SHAPE(pv_r3d(nldi:nlei,nldj:nlej,:)) ; ctmp1='d(nldi:nlei,nldj:nlej,:)'
               ELSE              ; ishape(1:3)=SHAPE(pv_r3d(1   :nlci,1   :nlcj,:)) ; ctmp1='d(1:nlci,1:nlcj,:)'
               ENDIF
            ENDIF
         ENDIF
         
         DO jl = 1, irankpv
            WRITE( ctmp2, FMT="(', ', i1,'): ', i5,' /= icnt(', i1,'):', i5)" ) jl, ishape(jl), jl, icnt(jl)
            IF( ishape(jl) /= icnt(jl) )   CALL ctl_stop( TRIM(clinfo), 'size(pv_r'//clrankpv//TRIM(ctmp1)//TRIM(ctmp2) )
         END DO

      ENDIF

      ! read the data
      !-     
      IF( idvar > 0 .AND. istop == nstop ) THEN   ! no additional errors until this point...
         !
         ! find the right index of the array to be read
! JMM + SM: ugly patch before getting the new version of lib_mpp)
!         IF( idom /= jpdom_unknown ) THEN   ;   ix1 = nldi   ;   ix2 = nlei      ;   iy1 = nldj   ;   iy2 = nlej
!         ELSE                               ;   ix1 = 1      ;   ix2 = icnt(1)   ;   iy1 = 1      ;   iy2 = icnt(2)
!         ENDIF
         IF( llnoov ) THEN
            IF( idom /= jpdom_unknown ) THEN   ;   ix1 = nldi   ;   ix2 = nlei      ;   iy1 = nldj   ;   iy2 = nlej
            ELSE                               ;   ix1 = 1      ;   ix2 = icnt(1)   ;   iy1 = 1      ;   iy2 = icnt(2)
            ENDIF
         ELSE
            IF( idom /= jpdom_unknown ) THEN   ;   ix1 = 1      ;   ix2 = nlci      ;   iy1 = 1      ;   iy2 = nlcj
            ELSE                               ;   ix1 = 1      ;   ix2 = icnt(1)   ;   iy1 = 1      ;   iy2 = icnt(2)
            ENDIF
         ENDIF
      
         SELECT CASE (iom_file(kiomid)%iolib)
         CASE (jpnf90   )   ;   CALL iom_nf90_get(    kiomid, idvar, inbdim, istart, icnt, ix1, ix2, iy1, iy2,   &
            &                                         pv_r1d, pv_r2d, pv_r3d )
         CASE DEFAULT    
         END SELECT

         IF( istop == nstop ) THEN   ! no additional errors until this point...
            IF(lwp) WRITE(numout,"(10x,' read ',a,' (rec: ',i6,') in ',a,' ok')") TRIM(cdvar), itime, TRIM(iom_file(kiomid)%name)
          
            !--- overlap areas and extra hallows (mpp)
            IF(     PRESENT(pv_r2d) .AND. idom /= jpdom_unknown ) THEN
               CALL lbc_lnk( pv_r2d,'Z',-999.,'no0' )
            ELSEIF( PRESENT(pv_r3d) .AND. idom /= jpdom_unknown ) THEN
               ! this if could be simplified with the new lbc_lnk that works with any size of the 3rd dimension
               IF( icnt(3) == jpk ) THEN
                  CALL lbc_lnk( pv_r3d,'Z',-999.,'no0' )
               ELSE   ! put some arbitrary value (a call to lbc_lnk will be done later...)
                  DO jj = nlcj+1, jpj   ;   pv_r3d(1:nlci, jj, :) = pv_r3d(1:nlci, nlej, :)   ;   END DO
                  DO ji = nlci+1, jpi   ;   pv_r3d(ji    , : , :) = pv_r3d(nlei  , :   , :)   ;   END DO
               ENDIF
            ENDIF
            
            !--- Apply scale_factor and offset
            zscf = iom_file(kiomid)%scf(idvar)      ! scale factor
            zofs = iom_file(kiomid)%ofs(idvar)      ! offset
            IF(     PRESENT(pv_r1d) ) THEN
               IF( zscf /= 1. )   pv_r1d(:) = pv_r1d(:) * zscf 
               IF( zofs /= 0. )   pv_r1d(:) = pv_r1d(:) + zofs
            ELSEIF( PRESENT(pv_r2d) ) THEN
!CDIR COLLAPSE
               IF( zscf /= 1.)   pv_r2d(:,:) = pv_r2d(:,:) * zscf
!CDIR COLLAPSE
               IF( zofs /= 0.)   pv_r2d(:,:) = pv_r2d(:,:) + zofs
            ELSEIF( PRESENT(pv_r3d) ) THEN
!CDIR COLLAPSE
               IF( zscf /= 1.)   pv_r3d(:,:,:) = pv_r3d(:,:,:) * zscf
!CDIR COLLAPSE
               IF( zofs /= 0.)   pv_r3d(:,:,:) = pv_r3d(:,:,:) + zofs
            ENDIF
            !
         ENDIF
         !
      ENDIF
      !
   END SUBROUTINE iom_get_123d


   SUBROUTINE iom_gettime( kiomid, ptime, cdvar, kntime, cdunits, cdcalendar )
      !!--------------------------------------------------------------------
      !!                   ***  SUBROUTINE iom_gettime  ***
      !!
      !! ** Purpose : read the time axis cdvar in the file 
      !!--------------------------------------------------------------------
      INTEGER                    , INTENT(in   ) ::   kiomid     ! file Identifier
      REAL(wp), DIMENSION(:)     , INTENT(  out) ::   ptime      ! the time axis
      CHARACTER(len=*), OPTIONAL , INTENT(in   ) ::   cdvar      ! time axis name
      INTEGER         , OPTIONAL , INTENT(  out) ::   kntime     ! number of times in file
      CHARACTER(len=*), OPTIONAL , INTENT(  out) ::   cdunits    ! units attribute of time coordinate
      CHARACTER(len=*), OPTIONAL , INTENT(  out) ::   cdcalendar ! calendar attribute of 
      !
      INTEGER, DIMENSION(1) :: kdimsz
      INTEGER            ::   idvar    ! id of the variable
      CHARACTER(LEN=32)  ::   tname    ! local name of time coordinate
      CHARACTER(LEN=100) ::   clinfo   ! info character
      !---------------------------------------------------------------------
      !
      IF ( PRESENT(cdvar) ) THEN
         tname = cdvar
      ELSE
         tname = iom_file(kiomid)%uldname
      ENDIF
      IF( kiomid > 0 ) THEN
         clinfo = 'iom_gettime, file: '//trim(iom_file(kiomid)%name)//', var: '//trim(tname)
         IF ( PRESENT(kntime) ) THEN
            idvar  = iom_varid( kiomid, tname, kdimsz = kdimsz )
            kntime = kdimsz(1)
         ELSE
            idvar = iom_varid( kiomid, tname )
         ENDIF
         !
         ptime(:) = 0. ! default definition
         IF( idvar > 0 ) THEN
            IF( iom_file(kiomid)%ndims(idvar) == 1 ) THEN
               IF( iom_file(kiomid)%luld(idvar) ) THEN
                  IF( iom_file(kiomid)%dimsz(1,idvar) <= size(ptime) ) THEN
                     SELECT CASE (iom_file(kiomid)%iolib)
                     CASE (jpnf90   )   ;   CALL iom_nf90_gettime(   kiomid, idvar, ptime, cdunits, cdcalendar )
                     CASE DEFAULT    
                     END SELECT
                  ELSE
                     WRITE(ctmp1,*) 'error with the size of ptime ',size(ptime),iom_file(kiomid)%dimsz(1,idvar)
                     CALL ctl_stop( trim(clinfo), trim(ctmp1) )
                  ENDIF
               ELSE
                  CALL ctl_stop( trim(clinfo), 'variable dimension is not unlimited... use iom_get' )
               ENDIF
            ELSE
               CALL ctl_stop( trim(clinfo), 'the variable has more than 1 dimension' )
            ENDIF
         ELSE
            CALL ctl_stop( trim(clinfo), 'variable not found in '//iom_file(kiomid)%name )
         ENDIF
      ENDIF
      !
   END SUBROUTINE iom_gettime


   !!----------------------------------------------------------------------
   !!                   INTERFACE iom_getatt
   !!----------------------------------------------------------------------
   SUBROUTINE iom_g0d_intatt( kiomid, cdatt, pvar )
      INTEGER         , INTENT(in   )                 ::   kiomid    ! Identifier of the file
      CHARACTER(len=*), INTENT(in   )                 ::   cdatt     ! Name of the attribute
      INTEGER         , INTENT(  out)                 ::   pvar      ! read field
      !
      IF( kiomid > 0 ) THEN
         IF( iom_file(kiomid)%nfid > 0 ) THEN
            SELECT CASE (iom_file(kiomid)%iolib)
            CASE (jpnf90   )   ;   CALL iom_nf90_getatt( kiomid, cdatt, pvar )
            CASE DEFAULT    
            END SELECT
         ENDIF
      ENDIF
   END SUBROUTINE iom_g0d_intatt


   !!----------------------------------------------------------------------
   !!                   INTERFACE iom_rstput
   !!----------------------------------------------------------------------
   SUBROUTINE iom_rp0d( kt, kwrite, kiomid, cdvar, pvar, ktype )
      INTEGER         , INTENT(in)                         ::   kt       ! ocean time-step
      INTEGER         , INTENT(in)                         ::   kwrite   ! writing time-step
      INTEGER         , INTENT(in)                         ::   kiomid   ! Identifier of the file 
      CHARACTER(len=*), INTENT(in)                         ::   cdvar    ! time axis name
      REAL(wp)        , INTENT(in)                         ::   pvar     ! written field
      INTEGER         , INTENT(in), OPTIONAL               ::   ktype    ! variable external type
      INTEGER :: ivid   ! variable id
      IF( kiomid > 0 ) THEN
         IF( iom_file(kiomid)%nfid > 0 ) THEN
            ivid = iom_varid( kiomid, cdvar, ldstop = .FALSE. )
            SELECT CASE (iom_file(kiomid)%iolib)
            CASE (jpnf90   )   ;   CALL iom_nf90_rstput(   kt, kwrite, kiomid, cdvar, ivid, ktype, pv_r0d = pvar )
            CASE DEFAULT     
            END SELECT
         ENDIF
      ENDIF
   END SUBROUTINE iom_rp0d

   SUBROUTINE iom_rp1d( kt, kwrite, kiomid, cdvar, pvar, ktype )
      INTEGER         , INTENT(in)                         ::   kt       ! ocean time-step
      INTEGER         , INTENT(in)                         ::   kwrite   ! writing time-step
      INTEGER         , INTENT(in)                         ::   kiomid   ! Identifier of the file 
      CHARACTER(len=*), INTENT(in)                         ::   cdvar    ! time axis name
      REAL(wp)        , INTENT(in), DIMENSION(          :) ::   pvar     ! written field
      INTEGER         , INTENT(in), OPTIONAL               ::   ktype    ! variable external type
      INTEGER :: ivid   ! variable id
      IF( kiomid > 0 ) THEN
         IF( iom_file(kiomid)%nfid > 0 ) THEN
            ivid = iom_varid( kiomid, cdvar, ldstop = .FALSE. )
            SELECT CASE (iom_file(kiomid)%iolib)
            CASE (jpnf90   )   ;   CALL iom_nf90_rstput(   kt, kwrite, kiomid, cdvar, ivid, ktype, pv_r1d = pvar )
            CASE DEFAULT     
            END SELECT
         ENDIF
      ENDIF
   END SUBROUTINE iom_rp1d

   SUBROUTINE iom_rp2d( kt, kwrite, kiomid, cdvar, pvar, ktype )
      INTEGER         , INTENT(in)                         ::   kt       ! ocean time-step
      INTEGER         , INTENT(in)                         ::   kwrite   ! writing time-step
      INTEGER         , INTENT(in)                         ::   kiomid   ! Identifier of the file 
      CHARACTER(len=*), INTENT(in)                         ::   cdvar    ! time axis name
      REAL(wp)        , INTENT(in), DIMENSION(:,    :    ) ::   pvar     ! written field
      INTEGER         , INTENT(in), OPTIONAL               ::   ktype    ! variable external type
      INTEGER :: ivid   ! variable id
      IF( kiomid > 0 ) THEN
         IF( iom_file(kiomid)%nfid > 0 ) THEN
            ivid = iom_varid( kiomid, cdvar, ldstop = .FALSE. )
            SELECT CASE (iom_file(kiomid)%iolib)
            CASE (jpnf90   )   ;   CALL iom_nf90_rstput(   kt, kwrite, kiomid, cdvar, ivid, ktype, pv_r2d = pvar )
            CASE DEFAULT     
            END SELECT
         ENDIF
      ENDIF
   END SUBROUTINE iom_rp2d

   SUBROUTINE iom_rp3d( kt, kwrite, kiomid, cdvar, pvar, ktype )
      INTEGER         , INTENT(in)                         ::   kt       ! ocean time-step
      INTEGER         , INTENT(in)                         ::   kwrite   ! writing time-step
      INTEGER         , INTENT(in)                         ::   kiomid   ! Identifier of the file 
      CHARACTER(len=*), INTENT(in)                         ::   cdvar    ! time axis name
      REAL(wp)        , INTENT(in),       DIMENSION(:,:,:) ::   pvar     ! written field
      INTEGER         , INTENT(in), OPTIONAL               ::   ktype    ! variable external type
      INTEGER :: ivid   ! variable id
      IF( kiomid > 0 ) THEN
         IF( iom_file(kiomid)%nfid > 0 ) THEN
            ivid = iom_varid( kiomid, cdvar, ldstop = .FALSE. )
            SELECT CASE (iom_file(kiomid)%iolib)
            CASE (jpnf90   )   ;   CALL iom_nf90_rstput(   kt, kwrite, kiomid, cdvar, ivid, ktype, pv_r3d = pvar )
            CASE DEFAULT     
            END SELECT
         ENDIF
      ENDIF
   END SUBROUTINE iom_rp3d


   !!----------------------------------------------------------------------
   !!                   INTERFACE iom_put
   !!----------------------------------------------------------------------
   SUBROUTINE iom_p0d( cdname, pfield0d )
      CHARACTER(LEN=*), INTENT(in) ::   cdname
      REAL(wp)        , INTENT(in) ::   pfield0d
      REAL(wp)        , DIMENSION(jpi,jpj) ::   zz     ! masson
      IF( .FALSE. )   WRITE(numout,*) cdname, pfield0d   ! useless test to avoid compilation warnings
   END SUBROUTINE iom_p0d

   SUBROUTINE iom_p1d( cdname, pfield1d )
      CHARACTER(LEN=*)          , INTENT(in) ::   cdname
      REAL(wp),     DIMENSION(:), INTENT(in) ::   pfield1d
      IF( .FALSE. )   WRITE(numout,*) cdname, pfield1d   ! useless test to avoid compilation warnings
   END SUBROUTINE iom_p1d

   SUBROUTINE iom_p2d( cdname, pfield2d )
      CHARACTER(LEN=*)            , INTENT(in) ::   cdname
      REAL(wp),     DIMENSION(:,:), INTENT(in) ::   pfield2d
      IF( .FALSE. )   WRITE(numout,*) cdname, pfield2d   ! useless test to avoid compilation warnings
   END SUBROUTINE iom_p2d

   SUBROUTINE iom_p3d( cdname, pfield3d )
      CHARACTER(LEN=*)                , INTENT(in) ::   cdname
      REAL(wp),       DIMENSION(:,:,:), INTENT(in) ::   pfield3d
      IF( .FALSE. )   WRITE(numout,*) cdname, pfield3d   ! useless test to avoid compilation warnings
   END SUBROUTINE iom_p3d
   !!----------------------------------------------------------------------



   SUBROUTINE iom_setkt( kt, cdname )
      INTEGER         , INTENT(in)::   kt 
      CHARACTER(LEN=*), INTENT(in) ::   cdname
      IF( .FALSE. )   WRITE(numout,*) kt, cdname   ! useless test to avoid compilation warnings
   END SUBROUTINE iom_setkt

   SUBROUTINE iom_context_finalize( cdname )
      CHARACTER(LEN=*), INTENT(in) ::   cdname
      IF( .FALSE. )   WRITE(numout,*)  cdname   ! useless test to avoid compilation warnings
   END SUBROUTINE iom_context_finalize


   LOGICAL FUNCTION iom_use( cdname )
      CHARACTER(LEN=*), INTENT(in) ::   cdname
      iom_use = .FALSE.
   END FUNCTION iom_use
   
   !!======================================================================
END MODULE iom
