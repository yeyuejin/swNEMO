!----------------------------------------------------------------------
! NEMO system team, System and Interface for oceanic RElocable Nesting
!----------------------------------------------------------------------
!
! DESCRIPTION:
!> @brief
!> This module groups lateral boundary conditions subroutine.
!>
!> @details
!>
!> @warning keep only non mpp case
!>
!> @author
!> G. Madec
!>
!>  @date June, 1997 - Original code
!> @date September, 2002
!> - F90: Free form and module
!>  @date Marsh, 2009
!> - R. Benshila : External north fold treatment  
!>  @date December, 2012
!> - S.Mocavero, I. Epicoco : Add 'lbc_bdy_lnk' and lbc_obc_lnk' routine to optimize the BDY/OBC communications
!> @date December, 2012
!> - R. Bourdalle-Badie and G. Reffray : add a C1D case 
!> @date January, 2015 
!> - J.Paul : rewrite with SIREN coding rules
!> @date Marsh, 2015 
!> - J.Paul : add hide subroutine
!>
!> @note Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
!----------------------------------------------------------------------
MODULE lbc

   USE kind                            ! F90 kind parameter
   ! NOTE_avoid_public_variables_if_possible

   ! function and subroutine
   PUBLIC :: lbc_lnk 
   PUBLIC :: lbc_nfd  
   PUBLIC :: lbc_hide  

   PRIVATE :: lbc__lnk_3d
   PRIVATE :: lbc__lnk_2d
   PRIVATE :: lbc__nfd_3d
   PRIVATE :: lbc__nfd_2d
   PRIVATE :: lbc__hide_lnk_2d
   PRIVATE :: lbc__hide_nfd
   PRIVATE :: lbc__hide_nfd_2d

   INTERFACE lbc_lnk
      MODULE PROCEDURE   lbc__lnk_3d
      MODULE PROCEDURE   lbc__lnk_2d
   END INTERFACE

   INTERFACE lbc_nfd
      MODULE PROCEDURE   lbc__nfd_3d
      MODULE PROCEDURE   lbc__nfd_2d
   END INTERFACE

   INTERFACE lbc_hide
      MODULE PROCEDURE   lbc__hide_lnk_2d
   END INTERFACE

   INTERFACE lbc__hide_nfd
      MODULE PROCEDURE   lbc__hide_nfd_2d
   END INTERFACE
    
CONTAINS
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE lbc__lnk_3d(dd_array, cd_type, id_perio, dd_psgn, dd_fill)
   !------------------------------------------------------------------- 
   !> @brief This subroutine set lateral boundary conditions on a 3D array (non mpp case) 
   !>
   !> @details
   !>             dd_psign = -1 :    change the sign across the north fold
   !>                      =  1 : no change of the sign across the north fold
   !>                      =  0 : no change of the sign across the north fold and
   !>                             strict positivity preserved: use inner row/column
   !>                             for closed boundaries.
   !> @author J.Paul 
   !> - January, 2015- rewrite with SIREN coding rules 
   !> 
   !> @param[inout] dd_array  3D array 
   !> @param[in] cd_type point grid
   !> @param[in] id_perio NEMO periodicity of the grid
   !> @param[in] dd_psgn 
   !> @param[in] dd_fill   fillValue
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp), DIMENSION(:,:,:), INTENT(INOUT) :: dd_array
      CHARACTER(LEN=*)          , INTENT(IN   ) :: cd_type
      INTEGER(i4)               , INTENT(IN   ) :: id_perio
      REAL(dp),                   INTENT(IN   ) :: dd_psgn
      REAL(dp),                   INTENT(IN   ), OPTIONAL :: dd_fill

      ! local variable
      REAL(dp)    :: dl_fill

      INTEGER(i4) :: il_jpi
      INTEGER(i4) :: il_jpj
      INTEGER(i4) :: il_jpim1
      !----------------------------------------------------------------
      IF( PRESENT( dd_fill ) ) THEN   ;   dl_fill = dd_fill      ! set FillValue (zero by default)
      ELSE                            ;   dl_fill = 0._dp
      ENDIF

      il_jpi=SIZE(dd_array(:,:,:),DIM=1)
      il_jpj=SIZE(dd_array(:,:,:),DIM=2)

      il_jpim1=il_jpi-1
      !
      !                                     ! East-West boundaries
      !                                     ! ====================
      SELECT CASE ( id_perio )
      !
      CASE ( 1 , 4 , 6 )                       !**  cyclic east-west
         dd_array(    1 ,:,:) = dd_array(il_jpim1,:,:)            ! all points
         dd_array(il_jpi,:,:) = dd_array(     2  ,:,:)
         !
      CASE DEFAULT                             !**  East closed  --  West closed
         SELECT CASE ( TRIM(cd_type) )
         CASE ( 'T' , 'U' , 'V' , 'W' )             ! T-, U-, V-, W-points
            dd_array(    1 ,:,:) = dl_fill
            dd_array(il_jpi,:,:) = dl_fill
         CASE ( 'F' )                               ! F-point
            dd_array(il_jpi,:,:) = dl_fill
         END SELECT
         !
      END SELECT
      !
      !                                     ! North-South boundaries
      !                                     ! ======================
      SELECT CASE ( id_perio )
      !
      CASE ( 2 )                               !**  South symmetric  --  North closed
         SELECT CASE ( TRIM(cd_type) )
         CASE ( 'T' , 'U' , 'W' )                   ! T-, U-, W-points
            dd_array(:,    1 ,:) = dd_array(:,3,:)
            dd_array(:,il_jpj,:) = dl_fill
         CASE ( 'V' , 'F' )                         ! V-, F-points
            dd_array(:,    1 ,:) = dd_psgn * dd_array(:,2,:)
            dd_array(:,il_jpj,:) = dl_fill
         END SELECT
         !
      CASE ( 3 , 4 , 5 , 6 )                   !**  North fold  T or F-point pivot  --  South closed
         SELECT CASE ( TRIM(cd_type) )                    ! South : closed
         CASE ( 'T' , 'U' , 'V' , 'W' , 'I' )             ! all points except F-point
            dd_array(:, 1 ,:) = dl_fill
         END SELECT
         !                                          ! North fold
         CALL lbc_nfd( dd_array(:,:,:), cd_type, id_perio, dd_psgn )
         !
      CASE DEFAULT                             !**  North closed  --  South closed
         SELECT CASE ( cd_type )
         CASE ( 'T' , 'U' , 'V' , 'W' )             ! T-, U-, V-, W-points
            dd_array(:,    1 ,:) = dl_fill
            dd_array(:,il_jpj,:) = dl_fill
         CASE ( 'F' )                               ! F-point
            dd_array(:,il_jpj,:) = dl_fill
         END SELECT
         !
      END SELECT

   END SUBROUTINE lbc__lnk_3d
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE lbc__lnk_2d(dd_array, cd_type, id_perio, dd_psgn, dd_fill)
   !------------------------------------------------------------------- 
   !> @brief This subroutine set lateral boundary conditions on a 2D array (non mpp case) 
   !>
   !> @details
   !>             dd_psign = -1 :    change the sign across the north fold
   !>                      =  1 : no change of the sign across the north fold
   !>                      =  0 : no change of the sign across the north fold and
   !>                             strict positivity preserved: use inner row/column
   !>                             for closed boundaries.
   !> @author J.Paul 
   !> - January, 2015- rewrite with SIREN coding rules
   !> 
   !> @param[inout] dd_array  2D array 
   !> @param[in] cd_type point grid
   !> @param[in] id_perio NEMO periodicity of the grid
   !> @param[in] dd_psgn 
   !> @param[in] dd_fill   fillValue
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp), DIMENSION(:,:), INTENT(INOUT) :: dd_array
      CHARACTER(LEN=*)        , INTENT(IN   ) :: cd_type
      INTEGER(i4)             , INTENT(IN   ) :: id_perio
      REAL(dp)                , INTENT(IN   ) :: dd_psgn
      REAL(dp)                , INTENT(IN   ), OPTIONAL :: dd_fill

      ! local variable
      REAL(dp)    :: dl_fill

      INTEGER(i4) :: il_jpi
      INTEGER(i4) :: il_jpj
      INTEGER(i4) :: il_jpim1
      !----------------------------------------------------------------
      IF( PRESENT( dd_fill ) ) THEN   ;   dl_fill = dd_fill      ! set FillValue (zero by default)
      ELSE                            ;   dl_fill = 0._dp
      ENDIF

      il_jpi=SIZE(dd_array(:,:),DIM=1)
      il_jpj=SIZE(dd_array(:,:),DIM=2)

      il_jpim1=il_jpi-1

      !
      !                                     ! East-West boundaries
      !                                     ! ====================
      SELECT CASE ( id_perio )
      !
      CASE ( 1 , 4 , 6 )                       !** cyclic east-west
         dd_array(    1 ,:) = dd_array(il_jpim1,:)               ! all points
         dd_array(il_jpi,:) = dd_array(     2  ,:)
         !
      CASE DEFAULT                             !** East closed  --  West closed
         SELECT CASE ( TRIM(cd_type) )
         CASE ( 'T' , 'U' , 'V' , 'W' )            ! T-, U-, V-, W-points
            dd_array(    1 ,:) = dl_fill
            dd_array(il_jpi,:) = dl_fill
         CASE ( 'F' )                              ! F-point
            dd_array(il_jpi,:) = dl_fill
         END SELECT
         !
      END SELECT
      !
      !                                     ! North-South boundaries
      !                                     ! ======================
      SELECT CASE ( id_perio )
      !
      CASE ( 2 )                               !**  South symmetric  --  North closed
         SELECT CASE ( TRIM(cd_type) )
         CASE ( 'T' , 'U' , 'W' )                   ! T-, U-, W-points
            dd_array(:,   1  ) = dd_array(:,3)
            dd_array(:,il_jpj) = dl_fill
         CASE ( 'V' , 'F' )                         ! V-, F-points
            dd_array(:,   1  ) = dd_psgn * dd_array(:,2)
            dd_array(:,il_jpj) = dl_fill
         END SELECT
         !
      CASE ( 3 , 4 , 5 , 6 )                   !**  North fold  T or F-point pivot  --  South closed
         SELECT CASE ( TRIM(cd_type) )                    ! South : closed
         CASE ( 'T' , 'U' , 'V' , 'W' , 'I' )             ! all points except F-point
            dd_array(:, 1 ) = dl_fill
         END SELECT
         !                                          ! North fold
         CALL lbc_nfd( dd_array(:,:), cd_type, id_perio, dd_psgn )
         !
      CASE DEFAULT                             !**  North closed  --  South closed
         SELECT CASE ( TRIM(cd_type) )
         CASE ( 'T' , 'U' , 'V' , 'W' )             ! T-, U-, V-, W-points
            dd_array(:,   1  ) = dl_fill
            dd_array(:,il_jpj) = dl_fill
         CASE ( 'F' )                               ! F-point
            dd_array(:,il_jpj) = dl_fill
         END SELECT
         !
      END SELECT

   END SUBROUTINE lbc__lnk_2d
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE lbc__nfd_3d(dd_array, cd_type, id_perio, dd_psgn)
   !------------------------------------------------------------------- 
   !> @brief This subroutine manage 3D lateral boundary condition : 
   !> North fold treatment without processor exchanges. 
   !>
   !> @warning keep only non mpp case
   !>
   !> @author J.Paul 
   !> - January, 2015- rewrite with SIREN coding rules
   !> 
   !> @param[inout] dd_array  3D array 
   !> @param[in] cd_type point grid
   !> @param[in] id_perio NEMO periodicity of the grid
   !> @param[in] dd_psgn 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp), DIMENSION(:,:,:), INTENT(INOUT) :: dd_array
      CHARACTER(LEN=*)          , INTENT(IN   ) :: cd_type
      INTEGER(i4)               , INTENT(IN   ) :: id_perio
      REAL(dp)                  , INTENT(IN   ) :: dd_psgn

      ! local variable
      INTEGER(i4) :: il_jpi
      INTEGER(i4) :: il_jpj
      INTEGER(i4) :: il_jpk
      INTEGER(i4) :: il_jpim1
      INTEGER(i4) :: il_jpjm1

      INTEGER(i4) :: ijt
      INTEGER(i4) :: iju

      ! loop indices
      INTEGER(i4) :: ji
      INTEGER(i4) :: jk
      !----------------------------------------------------------------
      
      il_jpi=SIZE(dd_array(:,:,:),DIM=1)
      il_jpj=SIZE(dd_array(:,:,:),DIM=2)
      il_jpk=SIZE(dd_array(:,:,:),DIM=3)

      il_jpim1=il_jpi-1
      il_jpjm1=il_jpj-1

      DO jk = 1, il_jpk
         !
         SELECT CASE ( id_perio )
         !
         CASE ( 3 , 4 )                        ! *  North fold  T-point pivot
            !
            SELECT CASE ( TRIM(cd_type) )
            CASE ( 'T' , 'W' )                         ! T-, W-point
               DO ji = 2, il_jpi
                  ijt = il_jpi-ji+2
                  dd_array(ji,il_jpj,jk) = dd_psgn * dd_array(ijt,il_jpj-2,jk)
               END DO
               dd_array(1,il_jpj,jk) = dd_psgn * dd_array(3,il_jpj-2,jk)
               DO ji = il_jpi/2+1, il_jpi
                  ijt = il_jpi-ji+2
                  dd_array(ji,il_jpjm1,jk) = dd_psgn * dd_array(ijt,il_jpjm1,jk)
               END DO
            CASE ( 'U' )                               ! U-point
               DO ji = 1, il_jpi-1
                  iju = il_jpi-ji+1
                  dd_array(ji,il_jpj,jk) = dd_psgn * dd_array(iju,il_jpj-2,jk)
               END DO
               dd_array(   1  ,il_jpj,jk) = dd_psgn * dd_array(    2   ,il_jpj-2,jk)
               dd_array(il_jpi,il_jpj,jk) = dd_psgn * dd_array(il_jpi-1,il_jpj-2,jk) 
               DO ji = il_jpi/2, il_jpi-1
                  iju = il_jpi-ji+1
                  dd_array(ji,il_jpjm1,jk) = dd_psgn * dd_array(iju,il_jpjm1,jk)
               END DO
            CASE ( 'V' )                               ! V-point
               DO ji = 2, il_jpi
                  ijt = il_jpi-ji+2
                  dd_array(ji,il_jpj-1,jk) = dd_psgn * dd_array(ijt,il_jpj-2,jk)
                  dd_array(ji,il_jpj  ,jk) = dd_psgn * dd_array(ijt,il_jpj-3,jk)
               END DO
               dd_array(1,il_jpj,jk) = dd_psgn * dd_array(3,il_jpj-3,jk) 
            CASE ( 'F' )                               ! F-point
               DO ji = 1, il_jpi-1
                  iju = il_jpi-ji+1
                  dd_array(ji,il_jpj-1,jk) = dd_psgn * dd_array(iju,il_jpj-2,jk)
                  dd_array(ji,il_jpj  ,jk) = dd_psgn * dd_array(iju,il_jpj-3,jk)
               END DO
               dd_array(   1  ,il_jpj,jk) = dd_psgn * dd_array(    2   ,il_jpj-3,jk)
               dd_array(il_jpi,il_jpj,jk) = dd_psgn * dd_array(il_jpi-1,il_jpj-3,jk) 
            END SELECT
            !
         CASE ( 5 , 6 )                        ! *  North fold  F-point pivot
            !
            SELECT CASE ( TRIM(cd_type) )
            CASE ( 'T' , 'W' )                         ! T-, W-point
               DO ji = 1, il_jpi
                  ijt = il_jpi-ji+1
                  dd_array(ji,il_jpj,jk) = dd_psgn * dd_array(ijt,il_jpj-1,jk)
               END DO
            CASE ( 'U' )                               ! U-point
               DO ji = 1, il_jpi-1
                  iju = il_jpi-ji
                  dd_array(ji,il_jpj,jk) = dd_psgn * dd_array(iju,il_jpj-1,jk)
               END DO
               dd_array(il_jpi,il_jpj,jk) = dd_psgn * dd_array(1,il_jpj-1,jk)
            CASE ( 'V' )                               ! V-point
               DO ji = 1, il_jpi
                  ijt = il_jpi-ji+1
                  dd_array(ji,il_jpj,jk) = dd_psgn * dd_array(ijt,il_jpj-2,jk)
               END DO
               DO ji = il_jpi/2+1, il_jpi
                  ijt = il_jpi-ji+1
                  dd_array(ji,il_jpjm1,jk) = dd_psgn * dd_array(ijt,il_jpjm1,jk)
               END DO
            CASE ( 'F' )                               ! F-point
               DO ji = 1, il_jpi-1
                  iju = il_jpi-ji
                  dd_array(ji,il_jpj  ,jk) = dd_psgn * dd_array(iju,il_jpj-2,jk)
               END DO
               dd_array(il_jpi,il_jpj,jk) = dd_psgn * dd_array(1,il_jpj-2,jk)
               DO ji = il_jpi/2+1, il_jpi-1
                  iju = il_jpi-ji
                  dd_array(ji,il_jpjm1,jk) = dd_psgn * dd_array(iju,il_jpjm1,jk)
               END DO
            END SELECT
            !
         CASE DEFAULT                           ! *  closed : the code probably never go through
            !
            SELECT CASE ( TRIM(cd_type))
            CASE ( 'T' , 'U' , 'V' , 'W' )             ! T-, U-, V-, W-points
               dd_array(:, 1  ,jk) = 0.e0
               dd_array(:,il_jpj,jk) = 0.e0
            CASE ( 'F' )                               ! F-point
               dd_array(:,il_jpj,jk) = 0.e0
            END SELECT
            !
         END SELECT     !  id_perio
         !
      END DO

   END SUBROUTINE lbc__nfd_3d
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE lbc__nfd_2d(dd_array, cd_type, id_perio, dd_psgn)
   !------------------------------------------------------------------- 
   !> @brief This subroutine manage 2D lateral boundary condition : 
   !> North fold treatment without processor exchanges. 
   !>
   !> @warning keep only non mpp case
   !> @warning do not use additional halos
   !>
   !> @author J.Paul 
   !> - January, 2015- rewrite with SIREN coding rules
   !> 
   !> @param[inout] dd_array  2D array 
   !> @param[in] cd_type point grid
   !> @param[in] id_perio NEMO periodicity of the grid
   !> @param[in] dd_psgn 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp), DIMENSION(:,:), INTENT(INOUT) :: dd_array
      CHARACTER(LEN=*)        , INTENT(IN   ) :: cd_type
      INTEGER(i4)             , INTENT(IN   ) :: id_perio
      REAL(dp)                , INTENT(IN   ) :: dd_psgn

      ! local variable
      INTEGER(i4) :: il_jpi
      INTEGER(i4) :: il_jpj
      INTEGER(i4) :: il_jpim1
      INTEGER(i4) :: il_jpjm1

      INTEGER(i4) :: ijt
      INTEGER(i4) :: iju

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      il_jpi=SIZE(dd_array(:,:),DIM=1)
      il_jpj=SIZE(dd_array(:,:),DIM=2)

      il_jpim1=il_jpi-1
      il_jpjm1=il_jpj-1

      SELECT CASE ( id_perio )
      !
      CASE ( 3, 4 )                       ! *  North fold  T-point pivot
         !
         SELECT CASE ( TRIM(cd_type) )
         !
         CASE ( 'T' , 'W' )                               ! T- , W-points
            DO ji = 2, il_jpi
               ijt=il_jpi-ji+2
               dd_array(ji,il_jpj) = dd_psgn * dd_array(ijt,il_jpj-2)
            END DO
            dd_array(1,il_jpj)   = dd_psgn * dd_array(3,il_jpj-2)
            dd_array(1,il_jpj-1) = dd_psgn * dd_array(3,il_jpj-1)
            DO ji = il_jpi/2+1, il_jpi
               ijt=il_jpi-ji+2
               dd_array(ji,il_jpj-1) = dd_psgn * dd_array(ijt,il_jpj-1)
            END DO
         CASE ( 'U' )                                     ! U-point
            DO ji = 1, il_jpi-1
               iju = il_jpi-ji+1
               dd_array(ji,il_jpj) = dd_psgn * dd_array(iju,il_jpj-2)
            END DO
            dd_array(   1  ,il_jpj  ) = dd_psgn * dd_array(    2   ,il_jpj-2)
            dd_array(il_jpi,il_jpj  ) = dd_psgn * dd_array(il_jpi-1,il_jpj-2)
            dd_array(1     ,il_jpj-1) = dd_psgn * dd_array(il_jpi  ,il_jpj-1)   
            DO ji = il_jpi/2, il_jpi-1
               iju = il_jpi-ji+1
               dd_array(ji,il_jpjm1) = dd_psgn * dd_array(iju,il_jpjm1)
            END DO
         CASE ( 'V' )                                     ! V-point
            DO ji = 2, il_jpi
               ijt = il_jpi-ji+2
               dd_array(ji,il_jpj) = dd_psgn * dd_array(ijt,il_jpj-3)
            END DO
            dd_array( 1 ,il_jpj)   = dd_psgn * dd_array( 3 ,il_jpj-3) 
         CASE ( 'F' )                                     ! F-point
            DO ji = 1, il_jpi-1
               iju = il_jpi-ji+1
               dd_array(ji,il_jpj) = dd_psgn * dd_array(iju,il_jpj-3)
            END DO
            dd_array(   1  ,il_jpj)   = dd_psgn * dd_array(    2   ,il_jpj-3)
            dd_array(il_jpi,il_jpj)   = dd_psgn * dd_array(il_jpi-1,il_jpj-3)
            dd_array(il_jpi,il_jpj-1) = dd_psgn * dd_array(il_jpi-1,il_jpj-2)      
            dd_array(   1  ,il_jpj-1) = dd_psgn * dd_array(    2   ,il_jpj-2)      
         CASE ( 'I' )                                     ! ice U-V point (I-point)
            dd_array(2,il_jpj) = dd_psgn * dd_array(3,il_jpj-1)
            DO ji = 3, il_jpi
               iju = il_jpi - ji + 3
               dd_array(ji,il_jpj) = dd_psgn * dd_array(iju,il_jpj-1)
            END DO
         CASE ( 'J' )                                     ! first ice U-V point
            dd_array(2,il_jpj) = dd_psgn * dd_array(3,il_jpj-1)
            DO ji = 3, il_jpi
               iju = il_jpi - ji + 3
               dd_array(ji,il_jpj) = dd_psgn * dd_array(iju,il_jpj-1)
            END DO
         CASE ( 'K' )                                     ! second ice U-V point
            dd_array(2,il_jpj) = dd_psgn * dd_array(3,il_jpj-1)
            DO ji = 3, il_jpi
               iju = il_jpi - ji + 3
               dd_array(ji,il_jpj) = dd_psgn * dd_array(iju,il_jpj-1)
            END DO
         END SELECT
         !
      CASE ( 5, 6 )                        ! *  North fold  F-point pivot
         !
         SELECT CASE ( TRIM(cd_type) )
         CASE ( 'T' , 'W' )                               ! T-, W-point
            DO ji = 1, il_jpi
               ijt = il_jpi-ji+1
               dd_array(ji,il_jpj) = dd_psgn * dd_array(ijt,il_jpj-1)
            END DO
         CASE ( 'U' )                                     ! U-point
            DO ji = 1, il_jpi-1
               iju = il_jpi-ji
               dd_array(ji,il_jpj) = dd_psgn * dd_array(iju,il_jpj-1)
            END DO
            dd_array(il_jpi,il_jpj) = dd_psgn * dd_array(1,il_jpj-1)
         CASE ( 'V' )                                     ! V-point
            DO ji = 1, il_jpi
               ijt = il_jpi-ji+1
               dd_array(ji,il_jpj) = dd_psgn * dd_array(ijt,il_jpj-2)
            END DO
            DO ji = il_jpi/2+1, il_jpi
               ijt = il_jpi-ji+1
               dd_array(ji,il_jpjm1) = dd_psgn * dd_array(ijt,il_jpjm1)
            END DO
         CASE ( 'F' )                               ! F-point
            DO ji = 1, il_jpi-1
               iju = il_jpi-ji
               dd_array(ji,il_jpj) = dd_psgn * dd_array(iju,il_jpj-2)
            END DO
            dd_array(il_jpi,il_jpj) = dd_psgn * dd_array(1,il_jpj-2)
            DO ji = il_jpi/2+1, il_jpi-1
               iju = il_jpi-ji
               dd_array(ji,il_jpjm1) = dd_psgn * dd_array(iju,il_jpjm1)
            END DO
         CASE ( 'I' )                                  ! ice U-V point (I-point)
            dd_array( 2 ,il_jpj) = 0.e0
            DO ji = 2 , il_jpi-1
               ijt = il_jpi - ji + 2
               dd_array(ji,il_jpj)= 0.5 * ( dd_array(ji,il_jpj-1) + dd_psgn * dd_array(ijt,il_jpj-1) )
            END DO
         CASE ( 'J' )                                  ! first ice U-V point
            dd_array( 2 ,il_jpj) = 0.e0
            DO ji = 2 , il_jpi-1
               ijt = il_jpi - ji + 2
               dd_array(ji,il_jpj)= dd_array(ji,il_jpj-1)
            END DO
         CASE ( 'K' )                                  ! second ice U-V point
            dd_array( 2 ,il_jpj) = 0.e0
            DO ji = 2 , il_jpi-1
               ijt = il_jpi - ji + 2
               dd_array(ji,il_jpj)= dd_array(ijt,il_jpj-1)
            END DO
         END SELECT
         !
      CASE DEFAULT                           ! *  closed : the code probably never go through
         !
         SELECT CASE ( TRIM(cd_type) )
         CASE ( 'T' , 'U' , 'V' , 'W' )                 ! T-, U-, V-, W-points
            dd_array(:, 1    ) = 0.e0
            dd_array(:,il_jpj) = 0.e0
         CASE ( 'F' )                                   ! F-point
            dd_array(:,il_jpj) = 0.e0
         CASE ( 'I' )                                   ! ice U-V point
            dd_array(:, 1    ) = 0.e0
            dd_array(:,il_jpj) = 0.e0
         CASE ( 'J' )                                   ! first ice U-V point
            dd_array(:, 1    ) = 0.e0
            dd_array(:,il_jpj) = 0.e0
         CASE ( 'K' )                                   ! second ice U-V point
            dd_array(:, 1    ) = 0.e0
            dd_array(:,il_jpj) = 0.e0
         END SELECT
         !
      END SELECT

   END SUBROUTINE lbc__nfd_2d
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE lbc__hide_lnk_2d(dd_array, cd_type, id_perio, dd_psgn, dd_fill)
   !------------------------------------------------------------------- 
   !> @brief This subroutine hide lateral boundary conditions on a 2D array (non mpp case) 
   !>
   !> @details
   !>             dd_psign = -1 :    change the sign across the north fold
   !>                      =  1 : no change of the sign across the north fold
   !>                      =  0 : no change of the sign across the north fold and
   !>                             strict positivity preserved: use inner row/column
   !>                             for closed boundaries.
   !> @author J.Paul 
   !> - Marsh, 2015- initial version
   !> 
   !> @param[inout] dd_array  2D array 
   !> @param[in] cd_type point grid
   !> @param[in] id_perio NEMO periodicity of the grid
   !> @param[in] dd_psgn 
   !> @param[in] dd_fill   fillValue
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp), DIMENSION(:,:), INTENT(INOUT) :: dd_array
      CHARACTER(LEN=*)        , INTENT(IN   ) :: cd_type
      INTEGER(i4)             , INTENT(IN   ) :: id_perio
      REAL(dp)                , INTENT(IN   ) :: dd_psgn
      REAL(dp)                , INTENT(IN   ), OPTIONAL :: dd_fill

      ! local variable
      REAL(dp)    :: dl_fill

      INTEGER(i4) :: il_jpi
      INTEGER(i4) :: il_jpj
      INTEGER(i4) :: il_jpim1
      !----------------------------------------------------------------
      IF( PRESENT( dd_fill ) ) THEN   ;   dl_fill = dd_fill      ! set FillValue (zero by default)
      ELSE                            ;   dl_fill = 0._dp
      ENDIF

      il_jpi=SIZE(dd_array(:,:),DIM=1)
      il_jpj=SIZE(dd_array(:,:),DIM=2)

      il_jpim1=il_jpi-1

      !
      !                                     ! East-West boundaries
      !                                     ! ====================
      SELECT CASE ( id_perio )
      !
      CASE ( 1 , 4 , 6 )                       !** cyclic east-west
         dd_array(    1 ,:) = dl_fill               ! all points
         dd_array(il_jpi,:) = dl_fill
         !
      CASE DEFAULT                             !** East closed  --  West closed
         SELECT CASE ( TRIM(cd_type) )
         CASE ( 'T' , 'U' , 'V' , 'W' )            ! T-, U-, V-, W-points
            dd_array(    1 ,:) = dl_fill
            dd_array(il_jpi,:) = dl_fill
         CASE ( 'F' )                              ! F-point
            dd_array(il_jpi,:) = dl_fill
         END SELECT
         !
      END SELECT
      !
      !                                     ! North-South boundaries
      !                                     ! ======================
      SELECT CASE ( id_perio )
      !
      CASE ( 2 )                               !**  South symmetric  --  North closed
         SELECT CASE ( TRIM(cd_type) )
         CASE ( 'T' , 'U' , 'W' )                   ! T-, U-, W-points
            dd_array(:,   1  ) = dl_fill
            dd_array(:,il_jpj) = dl_fill
         CASE ( 'V' , 'F' )                         ! V-, F-points
            dd_array(:,   1  ) = dl_fill
            dd_array(:,il_jpj) = dl_fill
         END SELECT
         !
      CASE ( 3 , 4 , 5 , 6 )                   !**  North fold  T or F-point pivot  --  South closed
         SELECT CASE ( TRIM(cd_type) )                    ! South : closed
         CASE ( 'T' , 'U' , 'V' , 'W' , 'I' )             ! all points except F-point
            dd_array(:, 1 ) = dl_fill
         END SELECT
         !                                          ! North fold
         CALL lbc__hide_nfd( dd_array(:,:), cd_type, id_perio, dd_psgn, &
         &                   dd_fill=dl_fill )
         !
      CASE DEFAULT                             !**  North closed  --  South closed
         SELECT CASE ( TRIM(cd_type) )
         CASE ( 'T' , 'U' , 'V' , 'W' )             ! T-, U-, V-, W-points
            dd_array(:,   1  ) = dl_fill
            dd_array(:,il_jpj) = dl_fill
         CASE ( 'F' )                               ! F-point
            dd_array(:,il_jpj) = dl_fill
         END SELECT
         !
      END SELECT

   END SUBROUTINE lbc__hide_lnk_2d
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   SUBROUTINE lbc__hide_nfd_2d(dd_array, cd_type, id_perio, dd_psgn, dd_fill)
   !------------------------------------------------------------------- 
   !> @brief This subroutine manage 2D lateral boundary condition : 
   !> hide North fold treatment without processor exchanges. 
   !>
   !> @warning keep only non mpp case
   !> @warning do not use additional halos
   !>
   !> @author J.Paul 
   !> - Marsh, 2015- initial version
   !> 
   !> @param[inout] dd_array  2D array 
   !> @param[in] cd_type point grid
   !> @param[in] id_perio NEMO periodicity of the grid
   !> @param[in] dd_psgn 
   !> @param[in] dd_fill 
   !-------------------------------------------------------------------

      IMPLICIT NONE

      ! Argument
      REAL(dp), DIMENSION(:,:), INTENT(INOUT) :: dd_array
      CHARACTER(LEN=*)        , INTENT(IN   ) :: cd_type
      INTEGER(i4)             , INTENT(IN   ) :: id_perio
      REAL(dp)                , INTENT(IN   ) :: dd_psgn
      REAL(dp)                , INTENT(IN   ), OPTIONAL :: dd_fill

      ! local variable
      REAL(dp)    :: dl_fill

      INTEGER(i4) :: il_jpi
      INTEGER(i4) :: il_jpj
      INTEGER(i4) :: il_jpim1
      INTEGER(i4) :: il_jpjm1

      ! loop indices
      INTEGER(i4) :: ji
      !----------------------------------------------------------------
      IF( PRESENT( dd_fill ) ) THEN   ;   dl_fill = dd_fill      ! set FillValue (zero by default)
      ELSE                            ;   dl_fill = 0._dp
      ENDIF

      il_jpi=SIZE(dd_array(:,:),DIM=1)
      il_jpj=SIZE(dd_array(:,:),DIM=2)

      il_jpim1=il_jpi-1
      il_jpjm1=il_jpj-1

      SELECT CASE ( id_perio )
      !
      CASE ( 3, 4 )                       ! *  North fold  T-point pivot
         !
         SELECT CASE ( TRIM(cd_type) )
         !
         CASE ( 'T' , 'W' )                               ! T- , W-points
            DO ji = 2, il_jpi
               dd_array(ji,il_jpj) = dl_fill
            END DO
            dd_array(1,il_jpj)   = dl_fill
            DO ji = il_jpi/2+2, il_jpi
               dd_array(ji,il_jpj-1) = dl_fill
            END DO
         CASE ( 'U' )                                     ! U-point
            DO ji = 1, il_jpi-1
               dd_array(ji,il_jpj) = dl_fill
            END DO
            dd_array(   1  ,il_jpj  ) = dl_fill 
            dd_array(il_jpi,il_jpj  ) = dl_fill
            dd_array(1     ,il_jpj-1) = dl_fill
            DO ji = il_jpi/2+1, il_jpi-1
               dd_array(ji,il_jpjm1) = dl_fill
            END DO
         CASE ( 'V' )                                     ! V-point
            DO ji = 2, il_jpi
               dd_array(ji,il_jpj) = dl_fill
            END DO
            dd_array( 1 ,il_jpj)   = dl_fill 
         CASE ( 'F' )                                     ! F-point
            DO ji = 1, il_jpi-1
               dd_array(ji,il_jpj) = dl_fill
            END DO
            dd_array(   1  ,il_jpj)   = dl_fill
            dd_array(il_jpi,il_jpj)   = dl_fill
            dd_array(il_jpi,il_jpj-1) = dl_fill 
            dd_array(   1  ,il_jpj-1) = dl_fill
         END SELECT
         !
      CASE ( 5, 6 )                        ! *  North fold  F-point pivot
         !
         SELECT CASE ( TRIM(cd_type) )
         CASE ( 'T' , 'W' )                               ! T-, W-point
            DO ji = 1, il_jpi
               dd_array(ji,il_jpj) = dl_fill
            END DO
         CASE ( 'U' )                                     ! U-point
            DO ji = 1, il_jpi-1
               dd_array(ji,il_jpj) = dl_fill
            END DO
            dd_array(il_jpi,il_jpj) = dl_fill
         CASE ( 'V' )                                     ! V-point
            DO ji = 1, il_jpi
               dd_array(ji,il_jpj) = dl_fill
            END DO
            DO ji = il_jpi/2+2, il_jpi
               dd_array(ji,il_jpjm1) = dl_fill
            END DO
         CASE ( 'F' )                               ! F-point
            DO ji = 1, il_jpi-1
               dd_array(ji,il_jpj) = dl_fill
            END DO
            dd_array(il_jpi,il_jpj) = dl_fill
            DO ji = il_jpi/2+2, il_jpi-1
               dd_array(ji,il_jpjm1) = dl_fill
            END DO
         END SELECT
         !
      CASE DEFAULT                           ! *  closed : the code probably never go through
         !
         SELECT CASE ( TRIM(cd_type) )
         CASE ( 'T' , 'U' , 'V' , 'W' )                 ! T-, U-, V-, W-points
            dd_array(:, 1    ) = dl_fill
            dd_array(:,il_jpj) = dl_fill
         CASE ( 'F' )                                   ! F-point
            dd_array(:,il_jpj) = dl_fill
         END SELECT
         !
      END SELECT

   END SUBROUTINE lbc__hide_nfd_2d
   !~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END MODULE lbc
