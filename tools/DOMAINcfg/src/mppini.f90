MODULE mppini
   !!==============================================================================
   !!                       ***  MODULE mppini   ***
   !! Ocean initialization : distributed memory computing initialization
   !!==============================================================================

   !!----------------------------------------------------------------------
   !!   mpp_init       : Lay out the global domain over processors
   !!   mpp_init2      : Lay out the global domain over processors 
   !!                    with land processor elimination
   !!   mpp_init_ioispl: IOIPSL initialization in mpp
   !!----------------------------------------------------------------------
   USE dom_oce         ! ocean space and time domain 
   USE in_out_manager  ! I/O Manager
   USE lib_mpp         ! distribued memory computing library
   USE ioipsl

   IMPLICIT NONE
   PRIVATE

   PUBLIC mpp_init       ! called by opa.F90
   PUBLIC mpp_init2      ! called by opa.F90

   !!----------------------------------------------------------------------
   !! NEMO/OCE 4.0 , NEMO Consortium (2018)
   !! $Id: mppini.F90 6412 2016-03-31 16:22:32Z lovato $ 
   !! Software governed by the CeCILL licence     (./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   !!----------------------------------------------------------------------
   !!   'key_mpp_mpi'          OR         MPI massively parallel processing
   !!----------------------------------------------------------------------

   SUBROUTINE mpp_init
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE mpp_init  ***
      !!                    
      !! ** Purpose :   Lay out the global domain over processors.
      !!
      !! ** Method  :   Global domain is distributed in smaller local domains.
      !!      Periodic condition is a function of the local domain position
      !!      (global boundary or neighbouring domain) and of the global
      !!      periodic
      !!      Type :         jperio global periodic condition
      !!                     nperio local  periodic condition
      !!
      !! ** Action  : - set domain parameters
      !!                    nimpp     : longitudinal index 
      !!                    njmpp     : latitudinal  index
      !!                    nperio    : lateral condition type 
      !!                    narea     : number for local area
      !!                    nlci      : first dimension
      !!                    nlcj      : second dimension
      !!                    nbondi    : mark for "east-west local boundary"
      !!                    nbondj    : mark for "north-south local boundary"
      !!                    nproc     : number for local processor
      !!                    noea      : number for local neighboring processor
      !!                    nowe      : number for local neighboring processor
      !!                    noso      : number for local neighboring processor
      !!                    nono      : number for local neighboring processor
      !!
      !! History :
      !!        !  94-11  (M. Guyon)  Original code
      !!        !  95-04  (J. Escobar, M. Imbard)
      !!        !  98-02  (M. Guyon)  FETI method
      !!        !  98-05  (M. Imbard, J. Escobar, L. Colombet )  SHMEM and MPI versions
      !!   8.5  !  02-08  (G. Madec)  F90 : free form
      !!   3.4  !  11-11  (C. Harris) decomposition changes for running with CICE
      !!----------------------------------------------------------------------
      INTEGER  ::   ji, jj, jn   ! dummy loop indices
      INTEGER  ::   ii, ij, ifreq, il1, il2            ! local integers
      INTEGER  ::   iresti, irestj, ijm1, imil, inum   !   -      -
      REAL(wp) ::   zidom, zjdom                       ! local scalars
      INTEGER, DIMENSION(jpni,jpnj) ::   iimppt, ijmppt, ilcit, ilcjt   ! local workspace
      !!----------------------------------------------------------------------

      IF(lwp) WRITE(numout,*)
      IF(lwp) WRITE(numout,*) 'mpp_init : Message Passing MPI'
      IF(lwp) WRITE(numout,*) '~~~~~~~~'


      !  1. Dimension arrays for subdomains
      ! -----------------------------------
      !  Computation of local domain sizes ilcit() ilcjt()
      !  These dimensions depend on global sizes jpni,jpnj and jpiglo,jpjglo
      !  The subdomains are squares leeser than or equal to the global
      !  dimensions divided by the number of processors minus the overlap
      !  array (cf. par_oce.F90).
      
      nreci  = 2 * jpreci
      nrecj  = 2 * jprecj
      iresti = MOD( jpiglo - nreci , jpni )
      irestj = MOD( jpjglo - nrecj , jpnj )

      IF(  iresti == 0 )   iresti = jpni


      DO jj = 1, jpnj
         DO ji = 1, iresti
            ilcit(ji,jj) = jpi
         END DO
         DO ji = iresti+1, jpni
            ilcit(ji,jj) = jpi -1
         END DO
      END DO
      
      nfilcit(:,:) = ilcit(:,:)
      IF( irestj == 0 )   irestj = jpnj


      DO ji = 1, jpni
         DO jj = 1, irestj
            ilcjt(ji,jj) = jpj
         END DO
         DO jj = irestj+1, jpnj
            ilcjt(ji,jj) = jpj -1
         END DO
      END DO
      

      !  2. Index arrays for subdomains
      ! -------------------------------
      
      iimppt(:,:) = 1
      ijmppt(:,:) = 1
      
      IF( jpni > 1 ) THEN
         DO jj = 1, jpnj
            DO ji = 2, jpni
               iimppt(ji,jj) = iimppt(ji-1,jj) + ilcit(ji-1,jj) - nreci
            END DO
         END DO
      ENDIF
      nfiimpp(:,:)=iimppt(:,:)

      IF( jpnj > 1 ) THEN
         DO jj = 2, jpnj
            DO ji = 1, jpni
               ijmppt(ji,jj) = ijmppt(ji,jj-1)+ilcjt(ji,jj-1)-nrecj
            END DO
         END DO
      ENDIF
      
      ! 3. Subdomain description
      ! ------------------------

      DO jn = 1, jpnij
         ii = 1 + MOD( jn-1, jpni )
         ij = 1 + (jn-1) / jpni
         nfipproc(ii,ij) = jn - 1
         nimppt(jn) = iimppt(ii,ij)
         njmppt(jn) = ijmppt(ii,ij)
         nlcit (jn) = ilcit (ii,ij)     
         nlci       = nlcit (jn)     
         nlcjt (jn) = ilcjt (ii,ij)     
         nlcj       = nlcjt (jn)
         nbondj = -1                                   ! general case
         IF( jn   >  jpni          )   nbondj = 0      ! first row of processor
         IF( jn   >  (jpnj-1)*jpni )   nbondj = 1      ! last  row of processor
         IF( jpnj == 1             )   nbondj = 2      ! one processor only in j-direction
         ibonjt(jn) = nbondj
         
         nbondi = 0                                    ! 
         IF( MOD( jn, jpni ) == 1 )   nbondi = -1      !
         IF( MOD( jn, jpni ) == 0 )   nbondi =  1      !
         IF( jpni            == 1 )   nbondi =  2      ! one processor only in i-direction
         ibonit(jn) = nbondi
         
         nldi =  1   + jpreci
         nlei = nlci - jpreci
         IF( nbondi == -1 .OR. nbondi == 2 )   nldi = 1
         IF( nbondi ==  1 .OR. nbondi == 2 )   nlei = nlci
         nldj =  1   + jprecj
         nlej = nlcj - jprecj
         IF( nbondj == -1 .OR. nbondj == 2 )   nldj = 1
         IF( nbondj ==  1 .OR. nbondj == 2 )   nlej = nlcj
         nldit(jn) = nldi
         nleit(jn) = nlei
         nldjt(jn) = nldj
         nlejt(jn) = nlej
      END DO

      ! 4. Subdomain print
      ! ------------------
      
      IF(lwp) WRITE(numout,*)
      IF(lwp) WRITE(numout,*) ' mpp_init: defines mpp subdomains'
      IF(lwp) WRITE(numout,*) ' ~~~~~~  ----------------------'
      IF(lwp) WRITE(numout,*)
      IF(lwp) WRITE(numout,*) 'iresti=',iresti,' irestj=',irestj
      IF(lwp) WRITE(numout,*)
      IF(lwp) WRITE(numout,*) 'jpni=',jpni,' jpnj=',jpnj
      zidom = nreci
      DO ji = 1, jpni
         zidom = zidom + ilcit(ji,1) - nreci
      END DO
      IF(lwp) WRITE(numout,*)
      IF(lwp) WRITE(numout,*)' sum ilcit(i,1)=', zidom, ' jpiglo=', jpiglo

      zjdom = nrecj
      DO jj = 1, jpnj
         zjdom = zjdom + ilcjt(1,jj) - nrecj
      END DO
      IF(lwp) WRITE(numout,*)' sum ilcit(1,j)=', zjdom, ' jpjglo=', jpjglo
      IF(lwp) WRITE(numout,*)

      IF(lwp) THEN
         ifreq = 4
         il1   = 1
         DO jn = 1, (jpni-1)/ifreq+1
            il2 = MIN( jpni, il1+ifreq-1 )
            WRITE(numout,*)
            WRITE(numout,9200) ('***',ji = il1,il2-1)
            DO jj = jpnj, 1, -1
               WRITE(numout,9203) ('   ',ji = il1,il2-1)
               WRITE(numout,9202) jj, ( ilcit(ji,jj),ilcjt(ji,jj),ji = il1,il2 )
               WRITE(numout,9204) (nfipproc(ji,jj),ji=il1,il2)
               WRITE(numout,9203) ('   ',ji = il1,il2-1)
               WRITE(numout,9200) ('***',ji = il1,il2-1)
            END DO
            WRITE(numout,9201) (ji,ji = il1,il2)
            il1 = il1+ifreq
         END DO
 9200     FORMAT('     ***',20('*************',a3))
 9203     FORMAT('     *     ',20('         *   ',a3))
 9201     FORMAT('        ',20('   ',i3,'          '))
 9202     FORMAT(' ',i3,' *  ',20(i3,'  x',i3,'   *   '))
 9204     FORMAT('     *  ',20('      ',i3,'   *   '))
      ENDIF

      ! 5. From global to local
      ! -----------------------

      nperio = 0
      IF( jperio == 2 .AND. nbondj == -1 )   nperio = 2


      ! 6. Subdomain neighbours
      ! ----------------------

      nproc = narea - 1
      noso  = nproc - jpni
      nowe  = nproc - 1
      noea  = nproc + 1
      nono  = nproc + jpni
      ! great neighbours
      npnw = nono - 1
      npne = nono + 1
      npsw = noso - 1
      npse = noso + 1
      nbsw = 1
      nbnw = 1
      IF( MOD( nproc, jpni ) == 0 ) THEN
         nbsw = 0
         nbnw = 0
      ENDIF
      nbse = 1
      nbne = 1
      IF( MOD( nproc, jpni ) == jpni-1 ) THEN
         nbse = 0
         nbne = 0
      ENDIF
      IF(nproc < jpni) THEN
         nbsw = 0
         nbse = 0
      ENDIF
      IF( nproc >= (jpnj-1)*jpni ) THEN
         nbnw = 0
         nbne = 0
      ENDIF
      nlcj = nlcjt(narea)  
      nlci = nlcit(narea)  
      nldi = nldit(narea)
      nlei = nleit(narea)
      nldj = nldjt(narea)
      nlej = nlejt(narea)
      nbondi = ibonit(narea)
      nbondj = ibonjt(narea)
      nimpp  = nimppt(narea)  
      njmpp  = njmppt(narea)  

     ! Save processor layout in layout.dat file 
       IF (lwp) THEN
        CALL ctl_opn( inum, 'layout.dat', 'REPLACE', 'FORMATTED', 'SEQUENTIAL', -1, numout, .FALSE., narea )
        WRITE(inum,'(a)') '   jpnij     jpi     jpj     jpk  jpiglo  jpjglo'
        WRITE(inum,'(6i8)') jpnij,jpi,jpj,jpk,jpiglo,jpjglo
        WRITE(inum,'(a)') 'NAREA nlci nlcj nldi nldj nlei nlej nimpp njmpp'

        DO  jn = 1, jpnij
         WRITE(inum,'(9i5)') jn, nlcit(jn), nlcjt(jn), &
                                      nldit(jn), nldjt(jn), &
                                      nleit(jn), nlejt(jn), &
                                      nimppt(jn), njmppt(jn)
        END DO
        CLOSE(inum)   
      END IF


      ! w a r n i n g  narea (zone) /= nproc (processors)!

      IF( jperio == 1 .OR. jperio == 4 .OR. jperio == 6 ) THEN
         IF( jpni == 1 )THEN
            nbondi = 2
            nperio = 1
         ELSE
            nbondi = 0
         ENDIF
         IF( MOD( narea, jpni ) == 0 ) THEN
            noea = nproc-(jpni-1)
            npne = npne-jpni
            npse = npse-jpni
         ENDIF
         IF( MOD( narea, jpni ) == 1 ) THEN
            nowe = nproc+(jpni-1)
            npnw = npnw+jpni
            npsw = npsw+jpni
         ENDIF
         nbsw = 1
         nbnw = 1
         nbse = 1
         nbne = 1
         IF( nproc < jpni ) THEN
            nbsw = 0
            nbse = 0
         ENDIF
         IF( nproc >= (jpnj-1)*jpni ) THEN
            nbnw = 0
            nbne = 0
         ENDIF
      ENDIF
      npolj = 0
      IF( jperio == 3 .OR. jperio == 4 ) THEN
         ijm1 = jpni*(jpnj-1)
         imil = ijm1+(jpni+1)/2
         IF( narea > ijm1 ) npolj = 3
         IF( MOD(jpni,2) == 1 .AND. narea == imil ) npolj = 4
         IF( npolj == 3 ) nono = jpni*jpnj-narea+ijm1
      ENDIF
      IF( jperio == 5 .OR. jperio == 6 ) THEN
          ijm1 = jpni*(jpnj-1)
          imil = ijm1+(jpni+1)/2
          IF( narea > ijm1) npolj = 5
          IF( MOD(jpni,2) == 1 .AND. narea == imil ) npolj = 6
          IF( npolj == 5 ) nono = jpni*jpnj-narea+ijm1
      ENDIF

      ! Periodicity : no corner if nbondi = 2 and nperio != 1

      IF(lwp) THEN
         WRITE(numout,*) ' nproc  = ', nproc
         WRITE(numout,*) ' nowe   = ', nowe  , ' noea   =  ', noea
         WRITE(numout,*) ' nono   = ', nono  , ' noso   =  ', noso
         WRITE(numout,*) ' nbondi = ', nbondi
         WRITE(numout,*) ' nbondj = ', nbondj
         WRITE(numout,*) ' npolj  = ', npolj
         WRITE(numout,*) ' nperio = ', nperio
         WRITE(numout,*) ' nlci   = ', nlci
         WRITE(numout,*) ' nlcj   = ', nlcj
         WRITE(numout,*) ' nimpp  = ', nimpp
         WRITE(numout,*) ' njmpp  = ', njmpp
         WRITE(numout,*) ' nreci  = ', nreci  , ' npse   = ', npse
         WRITE(numout,*) ' nrecj  = ', nrecj  , ' npsw   = ', npsw
         WRITE(numout,*) ' jpreci = ', jpreci , ' npne   = ', npne
         WRITE(numout,*) ' jprecj = ', jprecj , ' npnw   = ', npnw
         WRITE(numout,*)
      ENDIF

      IF( nperio == 1 .AND. jpni /= 1 ) CALL ctl_stop( ' mpp_init: error on cyclicity' )

      ! Prepare mpp north fold

      IF( jperio >= 3 .AND. jperio <= 6 .AND. jpni > 1 ) THEN
         CALL mpp_ini_north
         IF(lwp) WRITE(numout,*) ' mpp_init : North fold boundary prepared for jpni >1'
      ENDIF

      ! Prepare NetCDF output file (if necessary)
      CALL mpp_init_ioipsl

   END SUBROUTINE mpp_init

   SUBROUTINE mpp_init2
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE mpp_init2  ***
      !!
      !! * Purpose :   Lay out the global domain over processors.
      !!     FOR USING THIS VERSION, A PREPROCESSING TRAITMENT IS RECOMMENDED
      !!     FOR DEFINING BETTER CUTTING OUT.
      !!       This routine is used with a the bathymetry file.
      !!       In this version, the land processors are avoided and the adress
      !!     processor (nproc, narea,noea, ...) are calculated again.
      !!     The jpnij parameter can be lesser than jpni x jpnj
      !!     and this jpnij parameter must be calculated before with an
      !!     algoritmic preprocessing program.
      !!
      !! ** Method  :   Global domain is distributed in smaller local domains.
      !!      Periodic condition is a function of the local domain position
      !!      (global boundary or neighbouring domain) and of the global
      !!      periodic
      !!      Type :         jperio global periodic condition
      !!                     nperio local  periodic condition
      !!
      !! ** Action :        nimpp     : longitudinal index 
      !!                    njmpp     : latitudinal  index
      !!                    nperio    : lateral condition type 
      !!                    narea     : number for local area
      !!                    nlci      : first dimension
      !!                    nlcj      : second dimension
      !!                    nproc     : number for local processor
      !!                    noea      : number for local neighboring processor
      !!                    nowe      : number for local neighboring processor
      !!                    noso      : number for local neighboring processor
      !!                    nono      : number for local neighboring processor
      !!
      !! History :
      !!        !  94-11  (M. Guyon)  Original code
      !!        !  95-04  (J. Escobar, M. Imbard)
      !!        !  98-02  (M. Guyon)  FETI method
      !!        !  98-05  (M. Imbard, J. Escobar, L. Colombet )  SHMEM and MPI versions
      !!   9.0  !  04-01  (G. Madec, J.M Molines)  F90 : free form , north fold jpni > 1
      !!----------------------------------------------------------------------
      USE in_out_manager  ! I/O Manager
      USE iom
      !! 
      INTEGER :: ji, jj, jn, jproc, jarea     ! dummy loop indices
      INTEGER ::  inum                        ! temporary logical unit
      INTEGER ::  idir                        ! temporary integers
      INTEGER ::  jstartrow                   ! temporary integers
      INTEGER ::   ios                        ! Local integer output status for namelist read
      INTEGER ::   &
         ii, ij, ifreq, il1, il2,          &  ! temporary integers
         icont, ili, ilj,                  &  !    "          "
         isurf, ijm1, imil,                &  !    "          "
         iino, ijno, iiso, ijso,           &  !    "          " 
         iiea, ijea, iiwe, ijwe,           &  !    "          "
         iinw, ijnw, iine, ijne,           &  !    "          "
         iisw, ijsw, iise, ijse,           &  !    "          "
         iresti, irestj, iproc                !    "          "
      INTEGER, DIMENSION(jpnij) ::   &
         iin, ijn          
      INTEGER, DIMENSION(jpni,jpnj) ::   &
         iimppt, ijmppt, ilci  , ilcj  ,   &  ! temporary workspace
         ipproc, ibondj, ibondi, ipolj ,   &  !    "           "
         ilei  , ilej  , ildi  , ildj  ,   &  !    "           "
         ioea  , iowe  , ioso  , iono  ,   &  !    "           "
         ione  , ionw  , iose  , iosw  ,   &  !    "           "
         ibne  , ibnw  , ibse  , ibsw         !    "           "
      INTEGER,  DIMENSION(jpiglo,jpjglo) ::   &
         imask                                ! temporary global workspace
      REAL(wp), DIMENSION(jpiglo,jpjglo) ::   &
         zdta, zdtaisf                     ! temporary data workspace
      REAL(wp) ::   zidom , zjdom          ! temporary scalars

      ! read namelist for ln_zco
      NAMELIST/namzgr/ ln_zco, ln_zps, ln_sco, ln_isfcav, ln_linssh

      !!----------------------------------------------------------------------
      !!  OPA 9.0 , LOCEAN-IPSL (2005) 
      !! $Id: mppini_2.h90 6412 2016-03-31 16:22:32Z lovato $
      !! This software is governed by the CeCILL licence see modipsl/doc/NEMO_CeCILL.txt
      !!----------------------------------------------------------------------

      REWIND( numnam_ref )              ! Namelist namzgr in reference namelist : Vertical coordinate
      READ  ( numnam_ref, namzgr, IOSTAT = ios, ERR = 901)
901   IF( ios /= 0 ) CALL ctl_nam ( ios , 'namzgr in reference namelist', lwp )

      REWIND( numnam_cfg )              ! Namelist namzgr in configuration namelist : Vertical coordinate
      READ  ( numnam_cfg, namzgr, IOSTAT = ios, ERR = 902 )
902   IF( ios /= 0 ) CALL ctl_nam ( ios , 'namzgr in configuration namelist', lwp )
      IF(lwm) WRITE ( numond, namzgr )

      IF(lwp)WRITE(numout,*)
      IF(lwp)WRITE(numout,*) 'mpp_init : Message Passing MPI'
      IF(lwp)WRITE(numout,*) '~~~~~~~~'
      IF(lwp)WRITE(numout,*) ' '

      IF( jpni*jpnj < jpnij ) CALL ctl_stop( ' jpnij > jpni x jpnj impossible' )

      ! 0. initialisation
      ! -----------------

      ! open the file
      ! Remember that at this level in the code, mpp is not yet initialized, so
      ! the file must be open with jpdom_unknown, and kstart and kcount forced 
      jstartrow = 1
      IF ( ln_zco ) THEN 
         CALL iom_open ( 'bathy_level.nc', inum )   ! Level bathymetry
          ! Optionally use a file attribute (open_ocean_jstart) to set a start row for reading from the global file
          ! This allows the unextended grid bathymetry to be stored in the same file as the under ice-shelf extended bathymetry
         CALL iom_getatt(inum, 'open_ocean_jstart', jstartrow ) ! -999 is returned if the attribute is not found
         jstartrow = MAX(1,jstartrow)
         CALL iom_get ( inum, jpdom_unknown, 'Bathy_level', zdta, kstart=(/jpizoom,jpjzoom+jstartrow-1/), kcount=(/jpiglo,jpjglo/) )
      ELSE
         CALL iom_open ( 'bathy_meter.nc', inum )   ! Meter bathy in case of partial steps
         IF ( ln_isfcav ) THEN
             CALL iom_get ( inum, jpdom_unknown, 'Bathymetry_isf' , zdta, kstart=(/jpizoom,jpjzoom/), kcount=(/jpiglo,jpjglo/) )
         ELSE
             ! Optionally use a file attribute (open_ocean_jstart) to set a start row for reading from the global file
             ! This allows the unextended grid bathymetry to be stored in the same file as the under ice-shelf extended bathymetry
             CALL iom_getatt(inum, 'open_ocean_jstart', jstartrow ) ! -999 is returned if the attribute is not found
             jstartrow = MAX(1,jstartrow)
             CALL iom_get ( inum, jpdom_unknown, 'Bathymetry' , zdta, kstart=(/jpizoom,jpjzoom+jstartrow-1/)   &
                &                                                   , kcount=(/jpiglo,jpjglo/) )
         ENDIF
      ENDIF
      CALL iom_close (inum)
      
      ! used to compute the land processor in case of not masked bathy file.
      zdtaisf(:,:) = 0.0_wp
      IF ( ln_isfcav ) THEN
         CALL iom_open ( 'bathy_meter.nc', inum )   ! Meter bathy in case of partial steps
         CALL iom_get ( inum, jpdom_unknown, 'isf_draft' , zdtaisf, kstart=(/jpizoom,jpjzoom/), kcount=(/jpiglo,jpjglo/) )
      END IF
      CALL iom_close (inum)

      ! land/sea mask over the global/zoom domain

      imask(:,:)=1
      WHERE ( zdta(:,:) - zdtaisf(:,:) <= rn_isfhmin ) imask = 0

      !  1. Dimension arrays for subdomains
      ! -----------------------------------

      !  Computation of local domain sizes ilci() ilcj()
      !  These dimensions depend on global sizes jpni,jpnj and jpiglo,jpjglo
      !  The subdomains are squares leeser than or equal to the global
      !  dimensions divided by the number of processors minus the overlap
      !  array.

      nreci=2*jpreci
      nrecj=2*jprecj
      iresti = 1 + MOD( jpiglo - nreci -1 , jpni )
      irestj = 1 + MOD( jpjglo - nrecj -1 , jpnj )

      ilci(1:iresti      ,:) = jpi
      ilci(iresti+1:jpni ,:) = jpi-1

      ilcj(:,      1:irestj) = jpj
      ilcj(:, irestj+1:jpnj) = jpj-1

      nfilcit(:,:) = ilci(:,:)

      IF(lwp) WRITE(numout,*)
      IF(lwp) WRITE(numout,*) ' mpp_init2: defines mpp subdomains'
      IF(lwp) WRITE(numout,*) ' ~~~~~~  ----------------------'
      IF(lwp) WRITE(numout,*)
      IF(lwp) WRITE(numout,*) 'iresti=',iresti,' irestj=',irestj
      IF(lwp) WRITE(numout,*)
      IF(lwp) WRITE(numout,*) 'jpni=',jpni,' jpnj=',jpnj

      zidom = nreci + sum(ilci(:,1) - nreci ) 
      IF(lwp) WRITE(numout,*)
      IF(lwp) WRITE(numout,*)' sum ilci(i,1)=',zidom,' jpiglo=',jpiglo

      zjdom = nrecj + sum(ilcj(1,:) - nrecj ) 
      IF(lwp) WRITE(numout,*) ' sum ilcj(1,j)=',zjdom,' jpjglo=',jpjglo
      IF(lwp) WRITE(numout,*)


      !  2. Index arrays for subdomains
      ! -------------------------------

      iimppt(:,:) = 1
      ijmppt(:,:) = 1
      ipproc(:,:) = -1

      IF( jpni > 1 )THEN
         DO jj = 1, jpnj
            DO ji = 2, jpni
               iimppt(ji,jj) = iimppt(ji-1,jj) + ilci(ji-1,jj) - nreci
            END DO
         END DO
      ENDIF
      nfiimpp(:,:) = iimppt(:,:)

      IF( jpnj > 1 )THEN
         DO jj = 2, jpnj
            DO ji = 1, jpni
               ijmppt(ji,jj) = ijmppt(ji,jj-1) + ilcj(ji,jj-1) - nrecj
            END DO
         END DO
      ENDIF


      ! 3. Subdomain description in the Regular Case
      ! --------------------------------------------

      nperio = 0
      icont = -1
      DO jarea = 1, jpni*jpnj
         ii = 1 + MOD(jarea-1,jpni)
         ij = 1 +    (jarea-1)/jpni
         ili = ilci(ii,ij)
         ilj = ilcj(ii,ij)
         ibondj(ii,ij) = -1
         IF( jarea >  jpni          )   ibondj(ii,ij) = 0
         IF( jarea >  (jpnj-1)*jpni )   ibondj(ii,ij) = 1
         IF( jpnj  == 1             )   ibondj(ii,ij) = 2
         ibondi(ii,ij) = 0
         IF( MOD(jarea,jpni) == 1 )   ibondi(ii,ij) = -1
         IF( MOD(jarea,jpni) == 0 )   ibondi(ii,ij) =  1
         IF( jpni            == 1 )   ibondi(ii,ij) =  2

         ! 2.4 Subdomain neighbors

         iproc = jarea - 1
         ioso(ii,ij) = iproc - jpni
         iowe(ii,ij) = iproc - 1
         ioea(ii,ij) = iproc + 1
         iono(ii,ij) = iproc + jpni
         ildi(ii,ij) = 1 + jpreci
         ilei(ii,ij) = ili -jpreci
         ionw(ii,ij) = iono(ii,ij) - 1
         ione(ii,ij) = iono(ii,ij) + 1
         iosw(ii,ij) = ioso(ii,ij) - 1
         iose(ii,ij) = ioso(ii,ij) + 1
         ibsw(ii,ij) = 1
         ibnw(ii,ij) = 1
         IF( MOD(iproc,jpni) == 0 ) THEN
            ibsw(ii,ij) = 0
            ibnw(ii,ij) = 0
         ENDIF
         ibse(ii,ij) = 1
         ibne(ii,ij) = 1
         IF( MOD(iproc,jpni) == jpni-1 ) THEN
            ibse(ii,ij) = 0
            ibne(ii,ij) = 0
         ENDIF
         IF( iproc < jpni ) THEN
            ibsw(ii,ij) = 0
            ibse(ii,ij) = 0
         ENDIF
         IF( iproc >= (jpnj-1)*jpni ) THEN
            ibnw(ii,ij) = 0
            ibne(ii,ij) = 0
         ENDIF
         IF( ibondi(ii,ij) == -1 .OR. ibondi(ii,ij) == 2 ) ildi(ii,ij) = 1
         IF( ibondi(ii,ij) ==  1 .OR. ibondi(ii,ij) == 2 ) ilei(ii,ij) = ili
         ildj(ii,ij) =  1  + jprecj
         ilej(ii,ij) = ilj - jprecj
         IF( ibondj(ii,ij) == -1 .OR. ibondj(ii,ij) == 2 ) ildj(ii,ij) = 1
         IF( ibondj(ii,ij) ==  1 .OR. ibondj(ii,ij) == 2 ) ilej(ii,ij) = ilj

         ! warning ii*ij (zone) /= nproc (processors)!

         IF( jperio == 1 .OR. jperio == 4 .OR. jperio == 6 ) THEN
            IF( jpni == 1 )THEN
               ibondi(ii,ij) = 2
               nperio = 1
            ELSE
               ibondi(ii,ij) = 0
            ENDIF
            IF( MOD(jarea,jpni) == 0 ) THEN
               ioea(ii,ij) = iproc - (jpni-1)
               ione(ii,ij) = ione(ii,ij) - jpni
               iose(ii,ij) = iose(ii,ij) - jpni
            ENDIF
            IF( MOD(jarea,jpni) == 1 ) THEN
               iowe(ii,ij) = iproc + jpni - 1
               ionw(ii,ij) = ionw(ii,ij) + jpni
               iosw(ii,ij) = iosw(ii,ij) + jpni 
            ENDIF
            ibsw(ii,ij) = 1
            ibnw(ii,ij) = 1
            ibse(ii,ij) = 1
            ibne(ii,ij) = 1
            IF( iproc < jpni ) THEN
               ibsw(ii,ij) = 0
               ibse(ii,ij) = 0
            ENDIF
            IF( iproc >= (jpnj-1)*jpni ) THEN
               ibnw(ii,ij) = 0
               ibne(ii,ij) = 0
            ENDIF
         ENDIF
         ipolj(ii,ij) = 0
         IF( jperio == 3 .OR. jperio == 4 ) THEN
            ijm1 = jpni*(jpnj-1)
            imil = ijm1+(jpni+1)/2
            IF( jarea > ijm1 ) ipolj(ii,ij) = 3
            IF( MOD(jpni,2) == 1 .AND. jarea == imil ) ipolj(ii,ij) = 4
            IF( ipolj(ii,ij) == 3 ) iono(ii,ij) = jpni*jpnj-jarea+ijm1   ! MPI rank of northern neighbour
         ENDIF
         IF( jperio == 5 .OR. jperio == 6 ) THEN
            ijm1 = jpni*(jpnj-1)
            imil = ijm1+(jpni+1)/2
            IF( jarea > ijm1) ipolj(ii,ij) = 5
            IF( MOD(jpni,2) == 1 .AND. jarea == imil ) ipolj(ii,ij) = 6
            IF( ipolj(ii,ij) == 5) iono(ii,ij) = jpni*jpnj-jarea+ijm1    ! MPI rank of northern neighbour
         ENDIF

         ! Check wet points over the entire domain to preserve the MPI communication stencil
         isurf = 0
         DO jj = 1, ilj
            DO  ji = 1, ili
               IF( imask(ji+iimppt(ii,ij)-1, jj+ijmppt(ii,ij)-1) == 1) isurf = isurf+1
            END DO
         END DO

         IF(isurf /= 0) THEN
            icont = icont + 1
            ipproc(ii,ij) = icont
            iin(icont+1) = ii
            ijn(icont+1) = ij
         ENDIF
      END DO

      nfipproc(:,:) = ipproc(:,:)

      ! Control
      IF(icont+1 /= jpnij) THEN
         WRITE(ctmp1,*) ' jpni =',jpni,' jpnj =',jpnj
         WRITE(ctmp2,*) ' jpnij =',jpnij, '< jpni x jpnj' 
         WRITE(ctmp3,*) ' ***********, mpp_init2 finds jpnij=',icont+1
         CALL ctl_stop( ' Eliminate land processors algorithm', '', ctmp1, ctmp2, '', ctmp3 )
      ENDIF

      ! 4. Subdomain print
      ! ------------------

      IF(lwp) THEN
         ifreq = 4
         il1 = 1
         DO jn = 1,(jpni-1)/ifreq+1
            il2 = MIN(jpni,il1+ifreq-1)
            WRITE(numout,*)
            WRITE(numout,9400) ('***',ji=il1,il2-1)
            DO jj = jpnj, 1, -1
               WRITE(numout,9403) ('   ',ji=il1,il2-1)
               WRITE(numout,9402) jj, (ilci(ji,jj),ilcj(ji,jj),ji=il1,il2)
               WRITE(numout,9404) (ipproc(ji,jj),ji=il1,il2)
               WRITE(numout,9403) ('   ',ji=il1,il2-1)
               WRITE(numout,9400) ('***',ji=il1,il2-1)
            END DO
            WRITE(numout,9401) (ji,ji=il1,il2)
            il1 = il1+ifreq
         END DO
 9400     FORMAT('     ***',20('*************',a3))
 9403     FORMAT('     *     ',20('         *   ',a3))
 9401     FORMAT('        ',20('   ',i3,'          '))
 9402     FORMAT(' ',i3,' *  ',20(i3,'  x',i3,'   *   '))
 9404     FORMAT('     *  ',20('      ',i3,'   *   '))
      ENDIF


      ! 5. neighbour treatment
      ! ----------------------

      DO jarea = 1, jpni*jpnj
         iproc = jarea-1
         ii = 1 + MOD(jarea-1,jpni)
         ij = 1 +    (jarea-1)/jpni
         IF( ipproc(ii,ij) == -1 .AND. iono(ii,ij) >= 0   &
            .AND. iono(ii,ij) <= jpni*jpnj-1 ) THEN
            iino = 1 + MOD(iono(ii,ij),jpni)
            ijno = 1 +    (iono(ii,ij))/jpni
              ! Need to reverse the logical direction of communication 
              ! for northern neighbours of northern row processors (north-fold)
              ! i.e. need to check that the northern neighbour only communicates
              ! to the SOUTH (or not at all) if this area is land-only (#1057)
            idir = 1
            IF( ij .eq. jpnj .AND. ijno .eq. jpnj ) idir = -1    
            IF( ibondj(iino,ijno) == idir ) ibondj(iino,ijno)=2
            IF( ibondj(iino,ijno) == 0 ) ibondj(iino,ijno) = -idir
         ENDIF
         IF( ipproc(ii,ij) == -1 .AND. ioso(ii,ij) >= 0   &
            .AND. ioso(ii,ij) <= jpni*jpnj-1 ) THEN
            iiso = 1 + MOD(ioso(ii,ij),jpni)
            ijso = 1 +    (ioso(ii,ij))/jpni
            IF( ibondj(iiso,ijso) == -1 ) ibondj(iiso,ijso) = 2
            IF( ibondj(iiso,ijso) ==  0 ) ibondj(iiso,ijso) = 1
         ENDIF
         IF( ipproc(ii,ij) == -1 .AND. ioea(ii,ij) >= 0   &
            .AND. ioea(ii,ij) <= jpni*jpnj-1) THEN
            iiea = 1 + MOD(ioea(ii,ij),jpni)
            ijea = 1 +    (ioea(ii,ij))/jpni
            IF( ibondi(iiea,ijea) == 1 ) ibondi(iiea,ijea) = 2
            IF( ibondi(iiea,ijea) == 0 ) ibondi(iiea,ijea) = -1
         ENDIF
         IF( ipproc(ii,ij) == -1 .AND. iowe(ii,ij) >= 0   &
            .AND. iowe(ii,ij) <= jpni*jpnj-1) THEN
            iiwe = 1 + MOD(iowe(ii,ij),jpni)
            ijwe = 1 +    (iowe(ii,ij))/jpni
            IF( ibondi(iiwe,ijwe) == -1 ) ibondi(iiwe,ijwe) = 2
            IF( ibondi(iiwe,ijwe) ==  0 ) ibondi(iiwe,ijwe) = 1
         ENDIF
         IF( ipproc(ii,ij) == -1 .AND. ibne(ii,ij) == 1 ) THEN
            iine = 1 + MOD(ione(ii,ij),jpni)
            ijne = 1 +    (ione(ii,ij))/jpni
            IF( ibsw(iine,ijne) == 1 ) ibsw(iine,ijne) = 0
         ENDIF
         IF( ipproc(ii,ij) == -1 .AND. ibsw(ii,ij) == 1 ) THEN
            iisw = 1 + MOD(iosw(ii,ij),jpni)
            ijsw = 1 +    (iosw(ii,ij))/jpni
            IF( ibne(iisw,ijsw) == 1 ) ibne(iisw,ijsw) = 0
         ENDIF
         IF( ipproc(ii,ij) == -1 .AND. ibnw(ii,ij) == 1 ) THEN
            iinw = 1 + MOD(ionw(ii,ij),jpni)
            ijnw = 1 +    (ionw(ii,ij))/jpni
            IF( ibse(iinw,ijnw) == 1 ) ibse(iinw,ijnw)=0
         ENDIF
         IF( ipproc(ii,ij) == -1 .AND. ibse(ii,ij) == 1 ) THEN
            iise = 1 + MOD(iose(ii,ij),jpni)
            ijse = 1 +    (iose(ii,ij))/jpni
            IF( ibnw(iise,ijse) == 1 ) ibnw(iise,ijse) = 0
         ENDIF
      END DO


      ! 6. Change processor name
      ! ------------------------

      nproc = narea-1
      ii = iin(narea)
      ij = ijn(narea)

      ! set default neighbours
      noso = ioso(ii,ij)
      nowe = iowe(ii,ij)
      noea = ioea(ii,ij)
      nono = iono(ii,ij) 
      npse = iose(ii,ij)
      npsw = iosw(ii,ij)
      npne = ione(ii,ij)
      npnw = ionw(ii,ij)

      ! check neighbours location
      IF( ioso(ii,ij) >= 0 .AND. ioso(ii,ij) <= (jpni*jpnj-1) ) THEN 
         iiso = 1 + MOD(ioso(ii,ij),jpni)
         ijso = 1 +    (ioso(ii,ij))/jpni
         noso = ipproc(iiso,ijso)
      ENDIF
      IF( iowe(ii,ij) >= 0 .AND. iowe(ii,ij) <= (jpni*jpnj-1) ) THEN 
         iiwe = 1 + MOD(iowe(ii,ij),jpni)
         ijwe = 1 +    (iowe(ii,ij))/jpni
         nowe = ipproc(iiwe,ijwe)
      ENDIF
      IF( ioea(ii,ij) >= 0 .AND. ioea(ii,ij) <= (jpni*jpnj-1) ) THEN 
         iiea = 1 + MOD(ioea(ii,ij),jpni)
         ijea = 1 +    (ioea(ii,ij))/jpni
         noea = ipproc(iiea,ijea)
      ENDIF
      IF( iono(ii,ij) >= 0 .AND. iono(ii,ij) <= (jpni*jpnj-1) ) THEN 
         iino = 1 + MOD(iono(ii,ij),jpni)
         ijno = 1 +    (iono(ii,ij))/jpni
         nono = ipproc(iino,ijno)
      ENDIF
      IF( iose(ii,ij) >= 0 .AND. iose(ii,ij) <= (jpni*jpnj-1) ) THEN 
         iise = 1 + MOD(iose(ii,ij),jpni)
         ijse = 1 +    (iose(ii,ij))/jpni
         npse = ipproc(iise,ijse)
      ENDIF
      IF( iosw(ii,ij) >= 0 .AND. iosw(ii,ij) <= (jpni*jpnj-1) ) THEN 
         iisw = 1 + MOD(iosw(ii,ij),jpni)
         ijsw = 1 +    (iosw(ii,ij))/jpni
         npsw = ipproc(iisw,ijsw)
      ENDIF
      IF( ione(ii,ij) >= 0 .AND. ione(ii,ij) <= (jpni*jpnj-1) ) THEN 
         iine = 1 + MOD(ione(ii,ij),jpni)
         ijne = 1 +    (ione(ii,ij))/jpni
         npne = ipproc(iine,ijne)
      ENDIF
      IF( ionw(ii,ij) >= 0 .AND. ionw(ii,ij) <= (jpni*jpnj-1) ) THEN 
         iinw = 1 + MOD(ionw(ii,ij),jpni)
         ijnw = 1 +    (ionw(ii,ij))/jpni
         npnw = ipproc(iinw,ijnw)
      ENDIF
      nbnw = ibnw(ii,ij)
      nbne = ibne(ii,ij)
      nbsw = ibsw(ii,ij)
      nbse = ibse(ii,ij)
      nlcj = ilcj(ii,ij)  
      nlci = ilci(ii,ij)  
      nldi = ildi(ii,ij)
      nlei = ilei(ii,ij)
      nldj = ildj(ii,ij)
      nlej = ilej(ii,ij)
      nbondi = ibondi(ii,ij)
      nbondj = ibondj(ii,ij)
      nimpp = iimppt(ii,ij)  
      njmpp = ijmppt(ii,ij)  
      DO jproc = 1, jpnij
         ii = iin(jproc)
         ij = ijn(jproc)
         nimppt(jproc) = iimppt(ii,ij)  
         njmppt(jproc) = ijmppt(ii,ij)  
         nlcjt(jproc) = ilcj(ii,ij)
         nlcit(jproc) = ilci(ii,ij)
         nldit(jproc) = ildi(ii,ij)
         nleit(jproc) = ilei(ii,ij)
         nldjt(jproc) = ildj(ii,ij)
         nlejt(jproc) = ilej(ii,ij)
      END DO

      ! Save processor layout in ascii file
      IF (lwp) THEN
         CALL ctl_opn( inum, 'layout.dat', 'REPLACE', 'FORMATTED', 'SEQUENTIAL', -1, numout, .FALSE., narea )
         WRITE(inum,'(a)') '   jpnij     jpi     jpj     jpk  jpiglo  jpjglo'
         WRITE(inum,'(6i8)') jpnij,jpi,jpj,jpk,jpiglo,jpjglo
         WRITE(inum,'(a)') 'NAREA nlci nlcj nldi nldj nlei nlej nimpp njmpp'

        DO  jproc = 1, jpnij
         WRITE(inum,'(9i5)') jproc, nlcit(jproc), nlcjt(jproc), &
                                      nldit(jproc), nldjt(jproc), &
                                      nleit(jproc), nlejt(jproc), &
                                      nimppt(jproc), njmppt(jproc)
        END DO
        CLOSE(inum)   
      END IF

      ! Defined npolj, either 0, 3 , 4 , 5 , 6
      ! In this case the important thing is that npolj /= 0
      ! Because if we go through these line it is because jpni >1 and thus
      ! we must use lbcnorthmpp, which tests only npolj =0 or npolj /= 0

      npolj = 0
      ij = ijn(narea)

      IF( jperio == 3 .OR. jperio == 4 ) THEN
         IF( ij == jpnj ) npolj = 3
      ENDIF

      IF( jperio == 5 .OR. jperio == 6 ) THEN
         IF( ij == jpnj ) npolj = 5
      ENDIF

      ! Periodicity : no corner if nbondi = 2 and nperio != 1

      IF(lwp) THEN
         WRITE(numout,*) ' nproc  = ', nproc
         WRITE(numout,*) ' nowe   = ', nowe  , ' noea   =  ', noea
         WRITE(numout,*) ' nono   = ', nono  , ' noso   =  ', noso
         WRITE(numout,*) ' nbondi = ', nbondi
         WRITE(numout,*) ' nbondj = ', nbondj
         WRITE(numout,*) ' npolj  = ', npolj
         WRITE(numout,*) ' nperio = ', nperio
         WRITE(numout,*) ' nlci   = ', nlci
         WRITE(numout,*) ' nlcj   = ', nlcj
         WRITE(numout,*) ' nimpp  = ', nimpp
         WRITE(numout,*) ' njmpp  = ', njmpp
         WRITE(numout,*) ' nreci  = ', nreci  , ' npse   = ', npse
         WRITE(numout,*) ' nrecj  = ', nrecj  , ' npsw   = ', npsw
         WRITE(numout,*) ' jpreci = ', jpreci , ' npne   = ', npne
         WRITE(numout,*) ' jprecj = ', jprecj , ' npnw   = ', npnw
         WRITE(numout,*)
      ENDIF

      IF( nperio == 1 .AND. jpni /= 1 ) CALL ctl_stop( ' mpp_init2: error on cyclicity' )

      ! Prepare mpp north fold

      IF( jperio >= 3 .AND. jperio <= 6 .AND. jpni > 1 ) THEN
         CALL mpp_ini_north
         IF(lwp) WRITE(numout,*) ' mpp_init2 : North fold boundary prepared for jpni >1'
      ENDIF

      ! Prepare NetCDF output file (if necessary)
      CALL mpp_init_ioipsl


   END SUBROUTINE mpp_init2

   SUBROUTINE mpp_init_ioipsl
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE mpp_init_ioipsl  ***
      !!
      !! ** Purpose :   
      !!
      !! ** Method  :   
      !!
      !! History :
      !!   9.0  !  04-03  (G. Madec )  MPP-IOIPSL 
      !!   " "  !  08-12  (A. Coward)  addition in case of jpni*jpnj < jpnij
      !!----------------------------------------------------------------------
      INTEGER, DIMENSION(2) ::   iglo, iloc, iabsf, iabsl, ihals, ihale, idid
      !!----------------------------------------------------------------------

      ! The domain is split only horizontally along i- or/and j- direction
      ! So we need at the most only 1D arrays with 2 elements.
      ! Set idompar values equivalent to the jpdom_local_noextra definition
      ! used in IOM. This works even if jpnij .ne. jpni*jpnj.
      iglo(1) = jpiglo
      iglo(2) = jpjglo
      iloc(1) = nlci
      iloc(2) = nlcj
      iabsf(1) = nimppt(narea)
      iabsf(2) = njmppt(narea)
      iabsl(:) = iabsf(:) + iloc(:) - 1
      ihals(1) = nldi - 1
      ihals(2) = nldj - 1
      ihale(1) = nlci - nlei
      ihale(2) = nlcj - nlej
      idid(1) = 1
      idid(2) = 2

      IF(lwp) THEN
          WRITE(numout,*)
          WRITE(numout,*) 'mpp_init_ioipsl :   iloc  = ', iloc (1), iloc (2)
          WRITE(numout,*) '~~~~~~~~~~~~~~~     iabsf = ', iabsf(1), iabsf(2)
          WRITE(numout,*) '                    ihals = ', ihals(1), ihals(2)
          WRITE(numout,*) '                    ihale = ', ihale(1), ihale(2)
      ENDIF
      !
      CALL flio_dom_set ( jpnij, nproc, idid, iglo, iloc, iabsf, iabsl, ihals, ihale, 'BOX', nidom)
      !
   END SUBROUTINE mpp_init_ioipsl  


   !!======================================================================
END MODULE mppini
