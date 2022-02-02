MODULE lbcnfd
   !!======================================================================
   !!                       ***  MODULE  lbcnfd  ***
   !! Ocean        : north fold  boundary conditions
   !!======================================================================
   !! History :  3.2  ! 2009-03  (R. Benshila)  Original code 
   !!            3.5  ! 2013-07 (I. Epicoco, S. Mocavero - CMCC) MPP optimization 
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   lbc_nfd       : generic interface for lbc_nfd_3d and lbc_nfd_2d routines
   !!   lbc_nfd_3d    : lateral boundary condition: North fold treatment for a 3D arrays   (lbc_nfd)
   !!   lbc_nfd_2d    : lateral boundary condition: North fold treatment for a 2D arrays   (lbc_nfd)
   !!   mpp_lbc_nfd_3d    : North fold treatment for a 3D arrays optimized for MPP
   !!   mpp_lbc_nfd_2d    : North fold treatment for a 2D arrays optimized for MPP
   !!----------------------------------------------------------------------
   USE dom_oce        ! ocean space and time domain 
   USE in_out_manager ! I/O manager

   IMPLICIT NONE
   PRIVATE

   INTERFACE lbc_nfd
      MODULE PROCEDURE   lbc_nfd_3d, lbc_nfd_2d
   END INTERFACE
   !
   INTERFACE mpp_lbc_nfd
      MODULE PROCEDURE   mpp_lbc_nfd_3d, mpp_lbc_nfd_2d
   END INTERFACE

   PUBLIC   lbc_nfd       ! north fold conditions
   PUBLIC   mpp_lbc_nfd   ! north fold conditions (parallel case)

   INTEGER, PUBLIC, PARAMETER            ::   jpmaxngh = 3               !:
   INTEGER, PUBLIC                       ::   nsndto, nfsloop, nfeloop   !:
   INTEGER, PUBLIC, DIMENSION (jpmaxngh) ::   isendto                    !: processes to which communicate

   !!----------------------------------------------------------------------
   !! NEMO/OCE 4.0 , NEMO Consortium (2018)
   !! $Id: lbcnfd.F90 6140 2015-12-21 11:35:23Z timgraham $
   !! Software governed by the CeCILL licence     (./LICENSE)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE lbc_nfd_3d( pt3d, cd_type, psgn )
      !!----------------------------------------------------------------------
      !!                  ***  routine lbc_nfd_3d  ***
      !!
      !! ** Purpose :   3D lateral boundary condition : North fold treatment
      !!              without processor exchanges. 
      !!
      !! ** Method  :   
      !!
      !! ** Action  :   pt3d with updated values along the north fold
      !!----------------------------------------------------------------------
      CHARACTER(len=1)          , INTENT(in   ) ::   cd_type   ! define the nature of ptab array grid-points
      !                                                        !   = T , U , V , F , W points
      REAL(wp)                  , INTENT(in   ) ::   psgn      ! control of the sign change
      !                                                        !   = -1. , the sign is changed if north fold boundary
      !                                                        !   =  1. , the sign is kept  if north fold boundary
      REAL(wp), DIMENSION(:,:,:), INTENT(inout) ::   pt3d      ! 3D array on which the boundary condition is applied
      !
      INTEGER  ::   ji, jk
      INTEGER  ::   ijt, iju, ijpj, ijpjm1
      !!----------------------------------------------------------------------

      SELECT CASE ( jpni )
      CASE ( 1 )     ;   ijpj = nlcj      ! 1 proc only  along the i-direction
      CASE DEFAULT   ;   ijpj = 4         ! several proc along the i-direction
      END SELECT
      ijpjm1 = ijpj-1

      DO jk = 1, jpk
         !
         SELECT CASE ( npolj )
         !
         CASE ( 3 , 4 )                        ! *  North fold  T-point pivot
            !
            SELECT CASE ( cd_type )
            CASE ( 'T' , 'W' )                         ! T-, W-point
               DO ji = 2, jpiglo
                  ijt = jpiglo-ji+2
                  pt3d(ji,ijpj,jk) = psgn * pt3d(ijt,ijpj-2,jk)
               END DO
               pt3d(1,ijpj,jk) = psgn * pt3d(3,ijpj-2,jk)
               DO ji = jpiglo/2+1, jpiglo
                  ijt = jpiglo-ji+2
                  pt3d(ji,ijpjm1,jk) = psgn * pt3d(ijt,ijpjm1,jk)
               END DO
            CASE ( 'U' )                               ! U-point
               DO ji = 1, jpiglo-1
                  iju = jpiglo-ji+1
                  pt3d(ji,ijpj,jk) = psgn * pt3d(iju,ijpj-2,jk)
               END DO
               pt3d(   1  ,ijpj,jk) = psgn * pt3d(    2   ,ijpj-2,jk)
               pt3d(jpiglo,ijpj,jk) = psgn * pt3d(jpiglo-1,ijpj-2,jk) 
               DO ji = jpiglo/2, jpiglo-1
                  iju = jpiglo-ji+1
                  pt3d(ji,ijpjm1,jk) = psgn * pt3d(iju,ijpjm1,jk)
               END DO
            CASE ( 'V' )                               ! V-point
               DO ji = 2, jpiglo
                  ijt = jpiglo-ji+2
                  pt3d(ji,ijpj-1,jk) = psgn * pt3d(ijt,ijpj-2,jk)
                  pt3d(ji,ijpj  ,jk) = psgn * pt3d(ijt,ijpj-3,jk)
               END DO
               pt3d(1,ijpj,jk) = psgn * pt3d(3,ijpj-3,jk) 
            CASE ( 'F' )                               ! F-point
               DO ji = 1, jpiglo-1
                  iju = jpiglo-ji+1
                  pt3d(ji,ijpj-1,jk) = psgn * pt3d(iju,ijpj-2,jk)
                  pt3d(ji,ijpj  ,jk) = psgn * pt3d(iju,ijpj-3,jk)
               END DO
               pt3d(   1  ,ijpj,jk) = psgn * pt3d(    2   ,ijpj-3,jk)
               pt3d(jpiglo,ijpj,jk) = psgn * pt3d(jpiglo-1,ijpj-3,jk) 
            END SELECT
            !
         CASE ( 5 , 6 )                        ! *  North fold  F-point pivot
            !
            SELECT CASE ( cd_type )
            CASE ( 'T' , 'W' )                         ! T-, W-point
               DO ji = 1, jpiglo
                  ijt = jpiglo-ji+1
                  pt3d(ji,ijpj,jk) = psgn * pt3d(ijt,ijpj-1,jk)
               END DO
            CASE ( 'U' )                               ! U-point
               DO ji = 1, jpiglo-1
                  iju = jpiglo-ji
                  pt3d(ji,ijpj,jk) = psgn * pt3d(iju,ijpj-1,jk)
               END DO
               pt3d(jpiglo,ijpj,jk) = psgn * pt3d(1,ijpj-1,jk)
            CASE ( 'V' )                               ! V-point
               DO ji = 1, jpiglo
                  ijt = jpiglo-ji+1
                  pt3d(ji,ijpj,jk) = psgn * pt3d(ijt,ijpj-2,jk)
               END DO
               DO ji = jpiglo/2+1, jpiglo
                  ijt = jpiglo-ji+1
                  pt3d(ji,ijpjm1,jk) = psgn * pt3d(ijt,ijpjm1,jk)
               END DO
            CASE ( 'F' )                               ! F-point
               DO ji = 1, jpiglo-1
                  iju = jpiglo-ji
                  pt3d(ji,ijpj  ,jk) = psgn * pt3d(iju,ijpj-2,jk)
               END DO
               pt3d(jpiglo,ijpj,jk) = psgn * pt3d(1,ijpj-2,jk)
               DO ji = jpiglo/2+1, jpiglo-1
                  iju = jpiglo-ji
                  pt3d(ji,ijpjm1,jk) = psgn * pt3d(iju,ijpjm1,jk)
               END DO
            END SELECT
            !
         CASE DEFAULT                           ! *  closed : the code probably never go through
            !
            SELECT CASE ( cd_type)
            CASE ( 'T' , 'U' , 'V' , 'W' )             ! T-, U-, V-, W-points
               pt3d(:, 1  ,jk) = 0.e0
               pt3d(:,ijpj,jk) = 0.e0
            CASE ( 'F' )                               ! F-point
               pt3d(:,ijpj,jk) = 0.e0
            END SELECT
            !
         END SELECT     !  npolj
         !
      END DO
      !
   END SUBROUTINE lbc_nfd_3d


   SUBROUTINE lbc_nfd_2d( pt2d, cd_type, psgn, pr2dj )
      !!----------------------------------------------------------------------
      !!                  ***  routine lbc_nfd_2d  ***
      !!
      !! ** Purpose :   2D lateral boundary condition : North fold treatment
      !!       without processor exchanges. 
      !!
      !! ** Method  :   
      !!
      !! ** Action  :   pt2d with updated values along the north fold
      !!----------------------------------------------------------------------
      CHARACTER(len=1)        , INTENT(in   ) ::   cd_type   ! define the nature of ptab array grid-points
      !                                                      ! = T , U , V , F , W points
      REAL(wp)                , INTENT(in   ) ::   psgn      ! control of the sign change
      !                                                      !   = -1. , the sign is changed if north fold boundary
      !                                                      !   =  1. , the sign is kept  if north fold boundary
      REAL(wp), DIMENSION(:,:), INTENT(inout) ::   pt2d      ! 2D array on which the boundary condition is applied
      INTEGER , OPTIONAL      , INTENT(in   ) ::   pr2dj     ! number of additional halos
      !
      INTEGER  ::   ji, jl, ipr2dj
      INTEGER  ::   ijt, iju, ijpj, ijpjm1
      !!----------------------------------------------------------------------

      SELECT CASE ( jpni )
      CASE ( 1 )     ;   ijpj = nlcj      ! 1 proc only  along the i-direction
      CASE DEFAULT   ;   ijpj = 4         ! several proc along the i-direction
      END SELECT
      !
      IF( PRESENT(pr2dj) ) THEN           ! use of additional halos
         ipr2dj = pr2dj
         IF( jpni > 1 )   ijpj = ijpj + ipr2dj
      ELSE
         ipr2dj = 0 
      ENDIF
      !
      ijpjm1 = ijpj-1


      SELECT CASE ( npolj )
      !
      CASE ( 3, 4 )                       ! *  North fold  T-point pivot
         !
         SELECT CASE ( cd_type )
         !
         CASE ( 'T' , 'W' )                               ! T- , W-points
            DO jl = 0, ipr2dj
               DO ji = 2, jpiglo
                  ijt=jpiglo-ji+2
                  pt2d(ji,ijpj+jl) = psgn * pt2d(ijt,ijpj-2-jl)
               END DO
            END DO
            pt2d(1,ijpj)   = psgn * pt2d(3,ijpj-2)
            DO ji = jpiglo/2+1, jpiglo
               ijt=jpiglo-ji+2
               pt2d(ji,ijpj-1) = psgn * pt2d(ijt,ijpj-1)
            END DO
         CASE ( 'U' )                                     ! U-point
            DO jl = 0, ipr2dj
               DO ji = 1, jpiglo-1
                  iju = jpiglo-ji+1
                  pt2d(ji,ijpj+jl) = psgn * pt2d(iju,ijpj-2-jl)
               END DO
            END DO
            pt2d(   1  ,ijpj  ) = psgn * pt2d(    2   ,ijpj-2)
            pt2d(jpiglo,ijpj  ) = psgn * pt2d(jpiglo-1,ijpj-2)
            pt2d(1     ,ijpj-1) = psgn * pt2d(jpiglo  ,ijpj-1)   
            DO ji = jpiglo/2, jpiglo-1
               iju = jpiglo-ji+1
               pt2d(ji,ijpjm1) = psgn * pt2d(iju,ijpjm1)
            END DO
         CASE ( 'V' )                                     ! V-point
            DO jl = -1, ipr2dj
               DO ji = 2, jpiglo
                  ijt = jpiglo-ji+2
                  pt2d(ji,ijpj+jl) = psgn * pt2d(ijt,ijpj-3-jl)
               END DO
            END DO
            pt2d( 1 ,ijpj)   = psgn * pt2d( 3 ,ijpj-3) 
         CASE ( 'F' )                                     ! F-point
            DO jl = -1, ipr2dj
               DO ji = 1, jpiglo-1
                  iju = jpiglo-ji+1
                  pt2d(ji,ijpj+jl) = psgn * pt2d(iju,ijpj-3-jl)
               END DO
            END DO
            pt2d(   1  ,ijpj)   = psgn * pt2d(    2   ,ijpj-3)
            pt2d(jpiglo,ijpj)   = psgn * pt2d(jpiglo-1,ijpj-3)
            pt2d(jpiglo,ijpj-1) = psgn * pt2d(jpiglo-1,ijpj-2)      
            pt2d(   1  ,ijpj-1) = psgn * pt2d(    2   ,ijpj-2)      
         CASE ( 'I' )                                     ! ice U-V point (I-point)
            DO jl = 0, ipr2dj
               pt2d(2,ijpj+jl) = psgn * pt2d(3,ijpj-1+jl)
               DO ji = 3, jpiglo
                  iju = jpiglo - ji + 3
                  pt2d(ji,ijpj+jl) = psgn * pt2d(iju,ijpj-1-jl)
               END DO
            END DO
         CASE ( 'J' )                                     ! first ice U-V point
            DO jl =0, ipr2dj
               pt2d(2,ijpj+jl) = psgn * pt2d(3,ijpj-1+jl)
               DO ji = 3, jpiglo
                  iju = jpiglo - ji + 3
                  pt2d(ji,ijpj+jl) = psgn * pt2d(iju,ijpj-1-jl)
               END DO
            END DO
         CASE ( 'K' )                                     ! second ice U-V point
            DO jl =0, ipr2dj
               pt2d(2,ijpj+jl) = psgn * pt2d(3,ijpj-1+jl)
               DO ji = 3, jpiglo
                  iju = jpiglo - ji + 3
                  pt2d(ji,ijpj+jl) = psgn * pt2d(iju,ijpj-1-jl)
               END DO
            END DO
         END SELECT
         !
      CASE ( 5, 6 )                        ! *  North fold  F-point pivot
         !
         SELECT CASE ( cd_type )
         CASE ( 'T' , 'W' )                               ! T-, W-point
            DO jl = 0, ipr2dj
               DO ji = 1, jpiglo
                  ijt = jpiglo-ji+1
                  pt2d(ji,ijpj+jl) = psgn * pt2d(ijt,ijpj-1-jl)
               END DO
            END DO
         CASE ( 'U' )                                     ! U-point
            DO jl = 0, ipr2dj
               DO ji = 1, jpiglo-1
                  iju = jpiglo-ji
                  pt2d(ji,ijpj+jl) = psgn * pt2d(iju,ijpj-1-jl)
               END DO
            END DO
            pt2d(jpiglo,ijpj) = psgn * pt2d(1,ijpj-1)
         CASE ( 'V' )                                     ! V-point
            DO jl = 0, ipr2dj
               DO ji = 1, jpiglo
                  ijt = jpiglo-ji+1
                  pt2d(ji,ijpj+jl) = psgn * pt2d(ijt,ijpj-2-jl)
               END DO
            END DO
            DO ji = jpiglo/2+1, jpiglo
               ijt = jpiglo-ji+1
               pt2d(ji,ijpjm1) = psgn * pt2d(ijt,ijpjm1)
            END DO
         CASE ( 'F' )                               ! F-point
            DO jl = 0, ipr2dj
               DO ji = 1, jpiglo-1
                  iju = jpiglo-ji
                  pt2d(ji,ijpj+jl) = psgn * pt2d(iju,ijpj-2-jl)
               END DO
            END DO
            pt2d(jpiglo,ijpj) = psgn * pt2d(1,ijpj-2)
            DO ji = jpiglo/2+1, jpiglo-1
               iju = jpiglo-ji
               pt2d(ji,ijpjm1) = psgn * pt2d(iju,ijpjm1)
            END DO
         CASE ( 'I' )                                  ! ice U-V point (I-point)
            pt2d( 2 ,ijpj:ijpj+ipr2dj) = 0.e0
            DO jl = 0, ipr2dj
               DO ji = 2 , jpiglo-1
                  ijt = jpiglo - ji + 2
                  pt2d(ji,ijpj+jl)= 0.5 * ( pt2d(ji,ijpj-1-jl) + psgn * pt2d(ijt,ijpj-1-jl) )
               END DO
            END DO
         CASE ( 'J' )                                  ! first ice U-V point
            pt2d( 2 ,ijpj:ijpj+ipr2dj) = 0.e0
            DO jl = 0, ipr2dj
               DO ji = 2 , jpiglo-1
                  ijt = jpiglo - ji + 2
                  pt2d(ji,ijpj+jl)= pt2d(ji,ijpj-1-jl)
               END DO
            END DO
         CASE ( 'K' )                                  ! second ice U-V point
            pt2d( 2 ,ijpj:ijpj+ipr2dj) = 0.e0
            DO jl = 0, ipr2dj
               DO ji = 2 , jpiglo-1
                  ijt = jpiglo - ji + 2
                  pt2d(ji,ijpj+jl)= pt2d(ijt,ijpj-1-jl)
               END DO
            END DO
         END SELECT
         !
      CASE DEFAULT                           ! *  closed : the code probably never go through
         !
         SELECT CASE ( cd_type)
         CASE ( 'T' , 'U' , 'V' , 'W' )                 ! T-, U-, V-, W-points
            pt2d(:, 1:1-ipr2dj     ) = 0.e0
            pt2d(:,ijpj:ijpj+ipr2dj) = 0.e0
         CASE ( 'F' )                                   ! F-point
            pt2d(:,ijpj:ijpj+ipr2dj) = 0.e0
         CASE ( 'I' )                                   ! ice U-V point
            pt2d(:, 1:1-ipr2dj     ) = 0.e0
            pt2d(:,ijpj:ijpj+ipr2dj) = 0.e0
         CASE ( 'J' )                                   ! first ice U-V point
            pt2d(:, 1:1-ipr2dj     ) = 0.e0
            pt2d(:,ijpj:ijpj+ipr2dj) = 0.e0
         CASE ( 'K' )                                   ! second ice U-V point
            pt2d(:, 1:1-ipr2dj     ) = 0.e0
            pt2d(:,ijpj:ijpj+ipr2dj) = 0.e0
         END SELECT
         !
      END SELECT
      !
   END SUBROUTINE lbc_nfd_2d


   SUBROUTINE mpp_lbc_nfd_3d( pt3dl, pt3dr, cd_type, psgn )
      !!----------------------------------------------------------------------
      !!                  ***  routine mpp_lbc_nfd_3d  ***
      !!
      !! ** Purpose :   3D lateral boundary condition : North fold treatment
      !!              without processor exchanges. 
      !!
      !! ** Method  :   
      !!
      !! ** Action  :   pt3d with updated values along the north fold
      !!----------------------------------------------------------------------
      CHARACTER(len=1)          , INTENT(in   ) ::   cd_type   ! define the nature of ptab array grid-points
      !                                                        !   = T , U , V , F , W points
      REAL(wp)                  , INTENT(in   ) ::   psgn      ! control of the sign change
      !                                                        !   = -1. , the sign is changed if north fold boundary
      !                                                        !   =  1. , the sign is kept    if north fold boundary
      REAL(wp), DIMENSION(:,:,:), INTENT(inout) ::   pt3dl     ! 3D array on which the boundary condition is applied
      REAL(wp), DIMENSION(:,:,:), INTENT(in   ) ::   pt3dr     ! 3D array on which the boundary condition is applied
      !
      INTEGER  ::   ji, jk
      INTEGER  ::   ijt, iju, ijpj, ijpjm1, ijta, ijua, jia, startloop, endloop
      !!----------------------------------------------------------------------
      !
      SELECT CASE ( jpni )
      CASE ( 1 )     ;   ijpj = nlcj      ! 1 proc only  along the i-direction
      CASE DEFAULT   ;   ijpj = 4         ! several proc along the i-direction
      END SELECT
      ijpjm1 = ijpj-1

         !
         SELECT CASE ( npolj )
         !
         CASE ( 3 , 4 )                        ! *  North fold  T-point pivot
            !
            SELECT CASE ( cd_type )
            CASE ( 'T' , 'W' )                         ! T-, W-point
               IF (nimpp .ne. 1) THEN
                 startloop = 1
               ELSE
                 startloop = 2
               ENDIF

               DO jk = 1, jpk
                  DO ji = startloop, nlci
                     ijt = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 4
                     pt3dl(ji,ijpj,jk) = psgn * pt3dr(ijt,ijpj-2,jk)
                  END DO
                  IF(nimpp .eq. 1) THEN
                     pt3dl(1,ijpj,jk) = psgn * pt3dl(3,ijpj-2,jk)
                  ENDIF
               END DO

               IF(nimpp .ge. (jpiglo/2+1)) THEN
                 startloop = 1
               ELSEIF(((nimpp+nlci-1) .ge. (jpiglo/2+1)) .AND. (nimpp .lt. (jpiglo/2+1))) THEN
                 startloop = jpiglo/2+1 - nimpp + 1
               ELSE
                 startloop = nlci + 1
               ENDIF
               IF(startloop .le. nlci) THEN
                 DO jk = 1, jpk
                    DO ji = startloop, nlci
                       ijt = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 4
                       jia = ji + nimpp - 1
                       ijta = jpiglo - jia + 2
                       IF((ijta .ge. (startloop + nimpp - 1)) .and. (ijta .lt. jia)) THEN
                          pt3dl(ji,ijpjm1,jk) = psgn * pt3dl(ijta-nimpp+1,ijpjm1,jk)
                       ELSE
                          pt3dl(ji,ijpjm1,jk) = psgn * pt3dr(ijt,ijpjm1,jk)
                       ENDIF
                    END DO
                 END DO
               ENDIF


            CASE ( 'U' )                               ! U-point
               IF ((nimpp + nlci - 1) .ne. jpiglo) THEN
                  endloop = nlci
               ELSE
                  endloop = nlci - 1
               ENDIF
               DO jk = 1, jpk
                  DO ji = 1, endloop
                     iju = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 3
                     pt3dl(ji,ijpj,jk) = psgn * pt3dr(iju,ijpj-2,jk)
                  END DO
                  IF(nimpp .eq. 1) THEN
                     pt3dl(   1  ,ijpj,jk) = psgn * pt3dl(    2   ,ijpj-2,jk)
                  ENDIF
                  IF((nimpp + nlci - 1) .eq. jpiglo) THEN
                     pt3dl(nlci,ijpj,jk) = psgn * pt3dl(nlci-1,ijpj-2,jk)
                  ENDIF
               END DO

               IF ((nimpp + nlci - 1) .ne. jpiglo) THEN
                  endloop = nlci
               ELSE
                  endloop = nlci - 1
               ENDIF
               IF(nimpp .ge. (jpiglo/2)) THEN
                  startloop = 1
               ELSEIF(((nimpp+nlci-1) .ge. (jpiglo/2)) .AND. (nimpp .lt. (jpiglo/2))) THEN
                  startloop = jpiglo/2 - nimpp + 1
               ELSE
                  startloop = endloop + 1
               ENDIF
               IF (startloop .le. endloop) THEN
                 DO jk = 1, jpk
                    DO ji = startloop, endloop
                      iju = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 3
                      jia = ji + nimpp - 1
                      ijua = jpiglo - jia + 1
                      IF((ijua .ge. (startloop + nimpp - 1)) .and. (ijua .lt. jia)) THEN
                        pt3dl(ji,ijpjm1,jk) = psgn * pt3dl(ijua-nimpp+1,ijpjm1,jk)
                      ELSE
                        pt3dl(ji,ijpjm1,jk) = psgn * pt3dr(iju,ijpjm1,jk)
                      ENDIF
                    END DO
                 END DO
               ENDIF

            CASE ( 'V' )                               ! V-point
               IF (nimpp .ne. 1) THEN
                  startloop = 1
               ELSE
                  startloop = 2
               ENDIF
               DO jk = 1, jpk
                  DO ji = startloop, nlci
                     ijt = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 4
                     pt3dl(ji,ijpj-1,jk) = psgn * pt3dr(ijt,ijpj-2,jk)
                     pt3dl(ji,ijpj  ,jk) = psgn * pt3dr(ijt,ijpj-3,jk)
                  END DO
                  IF(nimpp .eq. 1) THEN
                     pt3dl(1,ijpj,jk) = psgn * pt3dl(3,ijpj-3,jk)
                  ENDIF
               END DO
            CASE ( 'F' )                               ! F-point
               IF ((nimpp + nlci - 1) .ne. jpiglo) THEN
                  endloop = nlci
               ELSE
                  endloop = nlci - 1
               ENDIF
               DO jk = 1, jpk
                  DO ji = 1, endloop
                     iju = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 3
                     pt3dl(ji,ijpj-1,jk) = psgn * pt3dr(iju,ijpj-2,jk)
                     pt3dl(ji,ijpj  ,jk) = psgn * pt3dr(iju,ijpj-3,jk)
                  END DO
                  IF(nimpp .eq. 1) THEN
                     pt3dl(   1  ,ijpj,jk) = psgn * pt3dl(    2   ,ijpj-3,jk)
                  ENDIF
                  IF((nimpp + nlci - 1) .eq. jpiglo) THEN
                     pt3dl(nlci,ijpj,jk) = psgn * pt3dl(nlci-1,ijpj-3,jk)
                  ENDIF
               END DO
            END SELECT
            !

         CASE ( 5 , 6 )                        ! *  North fold  F-point pivot
            !
            SELECT CASE ( cd_type )
            CASE ( 'T' , 'W' )                         ! T-, W-point
               DO jk = 1, jpk
                  DO ji = 1, nlci
                     ijt = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 3
                     pt3dl(ji,ijpj,jk) = psgn * pt3dr(ijt,ijpj-1,jk)
                  END DO
               END DO

            CASE ( 'U' )                               ! U-point
               IF ((nimpp + nlci - 1) .ne. jpiglo) THEN
                  endloop = nlci
               ELSE
                  endloop = nlci - 1
               ENDIF
               DO jk = 1, jpk
                  DO ji = 1, endloop
                     iju = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 2
                     pt3dl(ji,ijpj,jk) = psgn * pt3dr(iju,ijpj-1,jk)
                  END DO
                  IF((nimpp + nlci - 1) .eq. jpiglo) THEN
                     pt3dl(nlci,ijpj,jk) = psgn * pt3dr(1,ijpj-1,jk)
                  ENDIF
               END DO

            CASE ( 'V' )                               ! V-point
               DO jk = 1, jpk
                  DO ji = 1, nlci
                     ijt = jpiglo - ji- nimpp - nfiimpp(isendto(1),jpnj) + 3
                     pt3dl(ji,ijpj,jk) = psgn * pt3dr(ijt,ijpj-2,jk)
                  END DO
               END DO

               IF(nimpp .ge. (jpiglo/2+1)) THEN
                  startloop = 1
               ELSEIF(((nimpp+nlci-1) .ge. (jpiglo/2+1)) .AND. (nimpp .lt. (jpiglo/2+1))) THEN
                  startloop = jpiglo/2+1 - nimpp + 1
               ELSE
                  startloop = nlci + 1
               ENDIF
               IF(startloop .le. nlci) THEN
                 DO jk = 1, jpk
                    DO ji = startloop, nlci
                       ijt = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 3
                       pt3dl(ji,ijpjm1,jk) = psgn * pt3dr(ijt,ijpjm1,jk)
                    END DO
                 END DO
               ENDIF

            CASE ( 'F' )                               ! F-point
               IF ((nimpp + nlci - 1) .ne. jpiglo) THEN
                  endloop = nlci
               ELSE
                  endloop = nlci - 1
               ENDIF
               DO jk = 1, jpk
                  DO ji = 1, endloop
                     iju = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 2
                     pt3dl(ji,ijpj ,jk) = psgn * pt3dr(iju,ijpj-2,jk)
                  END DO
                  IF((nimpp + nlci - 1) .eq. jpiglo) THEN
                     pt3dl(nlci,ijpj,jk) = psgn * pt3dr(1,ijpj-2,jk)
                  ENDIF
               END DO

               IF ((nimpp + nlci - 1) .ne. jpiglo) THEN
                  endloop = nlci
               ELSE
                  endloop = nlci - 1
               ENDIF
               IF(nimpp .ge. (jpiglo/2+1)) THEN
                  startloop = 1
               ELSEIF(((nimpp+nlci-1) .ge. (jpiglo/2+1)) .AND. (nimpp .lt. (jpiglo/2+1))) THEN
                  startloop = jpiglo/2+1 - nimpp + 1
               ELSE
                  startloop = endloop + 1
               ENDIF
               IF (startloop .le. endloop) THEN
                  DO jk = 1, jpk
                     DO ji = startloop, endloop
                        iju = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 2
                        pt3dl(ji,ijpjm1,jk) = psgn * pt3dr(iju,ijpjm1,jk)
                     END DO
                  END DO
               ENDIF

            END SELECT

         CASE DEFAULT                           ! *  closed : the code probably never go through
            !
            SELECT CASE ( cd_type)
            CASE ( 'T' , 'U' , 'V' , 'W' )             ! T-, U-, V-, W-points
               pt3dl(:, 1  ,jk) = 0.e0
               pt3dl(:,ijpj,jk) = 0.e0
            CASE ( 'F' )                               ! F-point
               pt3dl(:,ijpj,jk) = 0.e0
            END SELECT
            !
         END SELECT     !  npolj
         !
      !
   END SUBROUTINE mpp_lbc_nfd_3d


   SUBROUTINE mpp_lbc_nfd_2d( pt2dl, pt2dr, cd_type, psgn )
      !!----------------------------------------------------------------------
      !!                  ***  routine mpp_lbc_nfd_2d  ***
      !!
      !! ** Purpose :   2D lateral boundary condition : North fold treatment
      !!       without processor exchanges. 
      !!
      !! ** Method  :   
      !!
      !! ** Action  :   pt2d with updated values along the north fold
      !!----------------------------------------------------------------------
      CHARACTER(len=1)        , INTENT(in   ) ::   cd_type   ! define the nature of ptab array grid-points
      !                                                      ! = T , U , V , F , W points
      REAL(wp)                , INTENT(in   ) ::   psgn      ! control of the sign change
      !                                                      !   = -1. , the sign is changed if north fold boundary
      !                                                      !   =  1. , the sign is kept  if north fold boundary
      REAL(wp), DIMENSION(:,:), INTENT(inout) ::   pt2dl     ! 2D array on which the boundary condition is applied
      REAL(wp), DIMENSION(:,:), INTENT(in   ) ::   pt2dr     ! 2D array on which the boundary condition is applied
      !
      INTEGER  ::   ji
      INTEGER  ::   ijt, iju, ijpj, ijpjm1, ijta, ijua, jia, startloop, endloop
      !!----------------------------------------------------------------------

      SELECT CASE ( jpni )
      CASE ( 1 )     ;   ijpj = nlcj      ! 1 proc only  along the i-direction
      CASE DEFAULT   ;   ijpj = 4         ! several proc along the i-direction
      END SELECT
      !
      ijpjm1 = ijpj-1


      SELECT CASE ( npolj )
      !
      CASE ( 3, 4 )                       ! *  North fold  T-point pivot
         !
         SELECT CASE ( cd_type )
         !
         CASE ( 'T' , 'W' )                               ! T- , W-points
            IF (nimpp .ne. 1) THEN
              startloop = 1
            ELSE
              startloop = 2
            ENDIF
            DO ji = startloop, nlci
              ijt=jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 4
              pt2dl(ji,ijpj) = psgn * pt2dr(ijt,ijpjm1-1)
            END DO
            IF (nimpp .eq. 1) THEN
              pt2dl(1,ijpj)   = psgn * pt2dl(3,ijpj-2)
            ENDIF

            IF(nimpp .ge. (jpiglo/2+1)) THEN
               startloop = 1
            ELSEIF(((nimpp+nlci-1) .ge. (jpiglo/2+1)) .AND. (nimpp .lt. (jpiglo/2+1))) THEN
               startloop = jpiglo/2+1 - nimpp + 1
            ELSE
               startloop = nlci + 1
            ENDIF
            DO ji = startloop, nlci
               ijt=jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 4
               jia = ji + nimpp - 1
               ijta = jpiglo - jia + 2
               IF((ijta .ge. (startloop + nimpp - 1)) .and. (ijta .lt. jia)) THEN
                  pt2dl(ji,ijpjm1) = psgn * pt2dl(ijta-nimpp+1,ijpjm1)
               ELSE
                  pt2dl(ji,ijpjm1) = psgn * pt2dr(ijt,ijpjm1)
               ENDIF
            END DO

         CASE ( 'U' )                                     ! U-point
            IF ((nimpp + nlci - 1) .ne. jpiglo) THEN
               endloop = nlci
            ELSE
               endloop = nlci - 1
            ENDIF
            DO ji = 1, endloop
               iju = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 3
               pt2dl(ji,ijpj) = psgn * pt2dr(iju,ijpjm1-1)
            END DO

            IF (nimpp .eq. 1) THEN
              pt2dl(   1  ,ijpj  ) = psgn * pt2dl(    2   ,ijpj-2)
              pt2dl(1     ,ijpj-1) = psgn * pt2dr(jpiglo - nfiimpp(isendto(1), jpnj) + 1, ijpj-1)
            ENDIF
            IF((nimpp + nlci - 1) .eq. jpiglo) THEN
              pt2dl(nlci,ijpj  ) = psgn * pt2dl(nlci-1,ijpj-2)
            ENDIF

            IF ((nimpp + nlci - 1) .ne. jpiglo) THEN
               endloop = nlci
            ELSE
               endloop = nlci - 1
            ENDIF
            IF(nimpp .ge. (jpiglo/2)) THEN
               startloop = 1
            ELSEIF(((nimpp+nlci-1) .ge. (jpiglo/2)) .AND. (nimpp .lt. (jpiglo/2))) THEN
               startloop = jpiglo/2 - nimpp + 1
            ELSE
               startloop = endloop + 1
            ENDIF
            DO ji = startloop, endloop
               iju = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 3
               jia = ji + nimpp - 1
               ijua = jpiglo - jia + 1
               IF((ijua .ge. (startloop + nimpp - 1)) .and. (ijua .lt. jia)) THEN
                  pt2dl(ji,ijpjm1) = psgn * pt2dl(ijua-nimpp+1,ijpjm1)
               ELSE
                  pt2dl(ji,ijpjm1) = psgn * pt2dr(iju,ijpjm1)
               ENDIF
            END DO

         CASE ( 'V' )                                     ! V-point
            IF (nimpp .ne. 1) THEN
              startloop = 1
            ELSE
              startloop = 2
            ENDIF
            DO ji = startloop, nlci
              ijt=jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 4
              pt2dl(ji,ijpjm1) = psgn * pt2dr(ijt,ijpjm1-1)
              pt2dl(ji,ijpj) = psgn * pt2dr(ijt,ijpjm1-2)
            END DO
            IF (nimpp .eq. 1) THEN
              pt2dl( 1 ,ijpj)   = psgn * pt2dl( 3 ,ijpj-3) 
            ENDIF

         CASE ( 'F' )                                     ! F-point
            IF ((nimpp + nlci - 1) .ne. jpiglo) THEN
               endloop = nlci
            ELSE
               endloop = nlci - 1
            ENDIF
            DO ji = 1, endloop
               iju = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 3
               pt2dl(ji,ijpjm1) = psgn * pt2dr(iju,ijpjm1-1)
               pt2dl(ji,ijpj) = psgn * pt2dr(iju,ijpjm1-2)
            END DO
            IF (nimpp .eq. 1) THEN
              pt2dl(   1  ,ijpj)   = psgn * pt2dl(    2   ,ijpj-3)
              pt2dl(   1  ,ijpj-1) = psgn * pt2dl(    2   ,ijpj-2)
            ENDIF
            IF((nimpp + nlci - 1) .eq. jpiglo) THEN
              pt2dl(nlci,ijpj)   = psgn * pt2dl(nlci-1,ijpj-3)
              pt2dl(nlci,ijpj-1) = psgn * pt2dl(nlci-1,ijpj-2) 
            ENDIF

         CASE ( 'I' )                                     ! ice U-V point (I-point)
            IF (nimpp .ne. 1) THEN
               startloop = 1
            ELSE
               startloop = 3
               pt2dl(2,ijpj) = psgn * pt2dr(3,ijpjm1)
            ENDIF
            DO ji = startloop, nlci
               iju = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 5
               pt2dl(ji,ijpj) = psgn * pt2dr(iju,ijpjm1)
            END DO

         CASE ( 'J' )                                     ! first ice U-V point
            IF (nimpp .ne. 1) THEN
               startloop = 1
            ELSE
               startloop = 3
               pt2dl(2,ijpj) = psgn * pt2dr(3,ijpjm1)
            ENDIF
            DO ji = startloop, nlci
               iju = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 5
               pt2dl(ji,ijpj) = psgn * pt2dr(iju,ijpjm1)
            END DO

         CASE ( 'K' )                                     ! second ice U-V point
            IF (nimpp .ne. 1) THEN
               startloop = 1
            ELSE
               startloop = 3
               pt2dl(2,ijpj) = psgn * pt2dr(3,ijpjm1)
            ENDIF
            DO ji = startloop, nlci
               iju = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 5
               pt2dl(ji,ijpj) = psgn * pt2dr(iju,ijpjm1)
            END DO

         END SELECT
         !
      CASE ( 5, 6 )                        ! *  North fold  F-point pivot
         !
         SELECT CASE ( cd_type )
         CASE ( 'T' , 'W' )                               ! T-, W-point
            DO ji = 1, nlci
               ijt = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 3
               pt2dl(ji,ijpj) = psgn * pt2dr(ijt,ijpjm1)
            END DO

         CASE ( 'U' )                                     ! U-point
            IF ((nimpp + nlci - 1) .ne. jpiglo) THEN
               endloop = nlci
            ELSE
               endloop = nlci - 1
            ENDIF
            DO ji = 1, endloop
               iju = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 2
               pt2dl(ji,ijpj) = psgn * pt2dr(iju,ijpjm1)
            END DO
            IF((nimpp + nlci - 1) .eq. jpiglo) THEN
               pt2dl(nlci,ijpj) = psgn * pt2dr(1,ijpj-1)
            ENDIF

         CASE ( 'V' )                                     ! V-point
            DO ji = 1, nlci
               ijt = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 3
               pt2dl(ji,ijpj) = psgn * pt2dr(ijt,ijpjm1-1)
            END DO
            IF(nimpp .ge. (jpiglo/2+1)) THEN
               startloop = 1
            ELSEIF(((nimpp+nlci-1) .ge. (jpiglo/2+1)) .AND. (nimpp .lt. (jpiglo/2+1))) THEN
               startloop = jpiglo/2+1 - nimpp + 1
            ELSE
               startloop = nlci + 1
            ENDIF
            DO ji = startloop, nlci
               ijt = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 3
               pt2dl(ji,ijpjm1) = psgn * pt2dr(ijt,ijpjm1)
            END DO

         CASE ( 'F' )                               ! F-point
            IF ((nimpp + nlci - 1) .ne. jpiglo) THEN
               endloop = nlci
            ELSE
               endloop = nlci - 1
            ENDIF
            DO ji = 1, endloop
               iju = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 2
               pt2dl(ji,ijpj) = psgn * pt2dr(iju,ijpjm1-1)
            END DO
            IF((nimpp + nlci - 1) .eq. jpiglo) THEN
                pt2dl(nlci,ijpj) = psgn * pt2dr(1,ijpj-2)
            ENDIF

            IF ((nimpp + nlci - 1) .ne. jpiglo) THEN
               endloop = nlci
            ELSE
               endloop = nlci - 1
            ENDIF
            IF(nimpp .ge. (jpiglo/2+1)) THEN
               startloop = 1
            ELSEIF(((nimpp+nlci-1) .ge. (jpiglo/2+1)) .AND. (nimpp .lt. (jpiglo/2+1))) THEN
               startloop = jpiglo/2+1 - nimpp + 1
            ELSE
               startloop = endloop + 1
            ENDIF

            DO ji = startloop, endloop
               iju = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 2
               pt2dl(ji,ijpjm1) = psgn * pt2dr(iju,ijpjm1)
            END DO

         CASE ( 'I' )                                  ! ice U-V point (I-point)
               IF (nimpp .ne. 1) THEN
                  startloop = 1
               ELSE
                  startloop = 2
               ENDIF
               IF ((nimpp + nlci - 1) .ne. jpiglo) THEN
                  endloop = nlci
               ELSE
                  endloop = nlci - 1
               ENDIF
               DO ji = startloop , endloop
                  ijt = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 4
                  pt2dl(ji,ijpj)= 0.5 * (pt2dr(ji,ijpjm1) + psgn * pt2dr(ijt,ijpjm1))
               END DO

         CASE ( 'J' )                                  ! first ice U-V point
               IF (nimpp .ne. 1) THEN
                  startloop = 1
               ELSE
                  startloop = 2
               ENDIF
               IF ((nimpp + nlci - 1) .ne. jpiglo) THEN
                  endloop = nlci
               ELSE
                  endloop = nlci - 1
               ENDIF
               DO ji = startloop , endloop
                  ijt = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 4
                  pt2dl(ji,ijpj) = pt2dr(ji,ijpjm1)
               END DO

         CASE ( 'K' )                                  ! second ice U-V point
               IF (nimpp .ne. 1) THEN
                  startloop = 1
               ELSE
                  startloop = 2
               ENDIF
               IF ((nimpp + nlci - 1) .ne. jpiglo) THEN
                  endloop = nlci
               ELSE
                  endloop = nlci - 1
               ENDIF
               DO ji = startloop, endloop
                  ijt = jpiglo - ji - nimpp - nfiimpp(isendto(1),jpnj) + 4
                  pt2dl(ji,ijpj) = pt2dr(ijt,ijpjm1)
               END DO

         END SELECT
         !
      CASE DEFAULT                           ! *  closed : the code probably never go through
         !
         SELECT CASE ( cd_type)
         CASE ( 'T' , 'U' , 'V' , 'W' )                 ! T-, U-, V-, W-points
            pt2dl(:, 1     ) = 0.e0
            pt2dl(:,ijpj) = 0.e0
         CASE ( 'F' )                                   ! F-point
            pt2dl(:,ijpj) = 0.e0
         CASE ( 'I' )                                   ! ice U-V point
            pt2dl(:, 1     ) = 0.e0
            pt2dl(:,ijpj) = 0.e0
         CASE ( 'J' )                                   ! first ice U-V point
            pt2dl(:, 1     ) = 0.e0
            pt2dl(:,ijpj) = 0.e0
         CASE ( 'K' )                                   ! second ice U-V point
            pt2dl(:, 1     ) = 0.e0
            pt2dl(:,ijpj) = 0.e0
         END SELECT
         !
      END SELECT
      !
   END SUBROUTINE mpp_lbc_nfd_2d

   !!======================================================================
END MODULE lbcnfd
