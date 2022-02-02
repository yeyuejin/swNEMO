MODULE module_interp
   !!======================================================================
   !!                   ***  MODULE  module_interp  ***
   !! Ocean forcing:  bulk thermohaline forcing of the ocean (or ice)
   !!=====================================================================
   !! History : 2016-10  (F. Lemarié)  Original code
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   zinterp                : 
   !!   reconstructandremap    : 
   !!   reconstructandremap_ps :
   !! 
   !!----------------------------------------------------------------------
   IMPLICIT NONE

CONTAINS

   SUBROUTINE zinterp( jpi, jpj, jpka, jpka_in, ind, tab_in, e3t_in, e3_bak, &   
      &                              e3t_out, tab_out, interp_type           )   
      
      !!---------------------------------------------------------------------
      !!                    ***  ROUTINE zinterp  ***
      !!                   
      !! ** Purpose :   
      !!
      !! ** Method  :
      !!
      !! ** Action  :  
      !!----------------------------------------------------------------------
      INTEGER, INTENT(in   ) ::   jpi, jpj
      INTEGER, INTENT(in   ) ::   jpka, jpka_in, interp_type
      INTEGER, INTENT(in   ) ::   ind     ( 1:jpi, 1:jpj            )
      REAL(8), INTENT(inout) ::   tab_in  ( 1:jpi, 1:jpj, 1:jpka_in )
      REAL(8), INTENT(in   ) ::   e3t_in  ( 1:jpi, 1:jpj, 1:jpka_in )
      REAL(8), INTENT(in   ) ::   e3_bak  ( 1:jpi, 1:jpj            )            
      REAL(8), INTENT(in   ) ::   e3t_out (               1:jpka+1  )                           
      REAL(8), INTENT(  out) ::   tab_out ( 1:jpi, 1:jpj, 1:jpka+1  )  
      !!
      INTEGER                :: ji,jj,k_in
      REAL(8)                :: val1,val2,cff
      
      SELECT CASE(interp_type)
      CASE(1)      ! WENO
         DO jj = 1,jpj
            DO ji = 1,jpi
               k_in = ind( ji, jj )
               val1 = tab_in ( ji, jj, k_in-1 )
               val2 = tab_in ( ji, jj, k_in   )
               cff  =        val1 * e3_bak( ji, jj         )     &
                  & +        val2 * e3t_in( ji, jj, k_in-1 )     &
                  & + (val2-val1) * e3t_in( ji, jj, k_in   )             
               tab_in( ji, jj, k_in ) = cff / ( e3_bak( ji, jj ) + e3t_in ( ji, jj, k_in-1 ) )
               !
               CALL reconstructandremap( tab_in ( ji, jj, 1:k_in   ), e3t_in( ji, jj, 1:k_in ),   &
                  &                      tab_out( ji, jj, 2:jpka+1 ), e3t_out (    2:jpka+1  ),   &
                  &                      k_in, jpka )  
               !
            END DO
         END DO   
      CASE(2)      ! SPLINES
         DO jj = 1,jpj
            DO ji = 1,jpi
               k_in = ind( ji, jj )
               val1 = tab_in ( ji, jj, k_in-1 )
               val2 = tab_in ( ji, jj, k_in   )
               cff  =        val1 * e3_bak( ji, jj         )     &
                  & +        val2 * e3t_in( ji, jj, k_in-1 )     &
                  & + (val2-val1) * e3t_in( ji, jj, k_in   )             
               tab_in( ji, jj, k_in ) = cff / ( e3_bak( ji, jj ) + e3t_in ( ji, jj, k_in-1 ) )
               !
               CALL reconstructandremap_ps( tab_in ( ji, jj, 1:k_in   ), e3t_in( ji, jj, 1:k_in ),   &
                  &                         tab_out( ji, jj, 2:jpka+1 ), e3t_out (    2:jpka+1  ),   &
                  &                         k_in, jpka )  
               !
            END DO
         END DO  
      CASE DEFAULT
         WRITE(*,*) "### Error: problem in zinterp, interp_type not set properly"
         STOP
      END SELECT      
      !
   END SUBROUTINE zinterp     





!
!===================================================================================================
subroutine reconstructandremap(tabin,hin,tabout,hout,N,Nout)
!---------------------------------------------------------------------------------------------------
      implicit none
      integer            :: N, Nout
      real(8)            :: tabin(N), tabout(Nout)
      real(8)            :: hin(N), hout(Nout)
      real(8)            :: coeffremap(N,3),zwork(N,3)
      real(8)            :: zwork2(N+1,3)
      integer            :: k
      real(8), parameter :: dsmll=1.0d-8  
      real(8)            :: q,q01,q02,q001,q002,q0
      real(8)            :: z_win(1:N+1), z_wout(1:Nout+1)
      real(8),parameter  :: dpthin = 1.D-3
      integer            :: k1, kbox, ktop, ka, kbot
      real(8)            :: tsum, qbot, rpsum, zbox, ztop, zthk, zbot, offset, qtop
!-----

!----------------------
      z_win(1)=0.; z_wout(1)= 0.
      do k=1,N
       z_win(k+1)=z_win(k)+hin(k)
      enddo 
      
      do k=1,Nout
       z_wout(k+1)=z_wout(k)+hout(k)       
      enddo       

        do k=2,N
          zwork(k,1)=1./(hin(k-1)+hin(k))
        enddo
        
        do k=2,N-1
          q0 = 1./(hin(k-1)+hin(k)+hin(k+1))
          zwork(k,2)=hin(k-1)+2.*hin(k)+hin(k+1)
          zwork(k,3)=q0
        enddo       
      
        do k= 2,N
        zwork2(k,1)=zwork(k,1)*(tabin(k)-tabin(k-1))
        enddo
        
        coeffremap(:,1) = tabin(:)
 
         do k=2,N-1
        q001 = hin(k)*zwork2(k+1,1)
        q002 = hin(k)*zwork2(k,1)        
        if (q001*q002 < 0) then
          q001 = 0.
          q002 = 0.
        endif
        q=zwork(k,2)
        q01=q*zwork2(k+1,1)
        q02=q*zwork2(k,1)
        if (abs(q001) > abs(q02)) q001 = q02
        if (abs(q002) > abs(q01)) q002 = q01
        
        q=(q001-q002)*zwork(k,3)
        q001=q001-q*hin(k+1)
        q002=q002+q*hin(k-1)
        
        coeffremap(k,3)=coeffremap(k,1)+q001
        coeffremap(k,2)=coeffremap(k,1)-q002
        
        zwork2(k,1)=(2.*q001-q002)**2
        zwork2(k,2)=(2.*q002-q001)**2
        enddo
        
        do k=1,N
        if     (k.eq.1 .or. k.eq.N .or.   hin(k).le.dpthin) then
        coeffremap(k,3) = coeffremap(k,1)
        coeffremap(k,2) = coeffremap(k,1)
        zwork2(k,1) = 0.
        zwork2(k,2) = 0.
        endif
        enddo
        
        do k=2,N
        q002=max(zwork2(k-1,2),dsmll)
        q001=max(zwork2(k,1),dsmll)
        zwork2(k,3)=(q001*coeffremap(k-1,3)+q002*coeffremap(k,2))/(q001+q002)
        enddo
        
        zwork2(1,3) = 2*coeffremap(1,1)-zwork2(2,3)
        zwork2(N+1,3)=2*coeffremap(N,1)-zwork2(N,3)
 
        do k=1,N
        q01=zwork2(k+1,3)-coeffremap(k,1)
        q02=coeffremap(k,1)-zwork2(k,3)
        q001=2.*q01
        q002=2.*q02
        if (q01*q02<0) then
          q01=0.
          q02=0.
        elseif (abs(q01)>abs(q002)) then
          q01=q002
        elseif (abs(q02)>abs(q001)) then
          q02=q001
        endif
        coeffremap(k,2)=coeffremap(k,1)-q02
        coeffremap(k,3)=coeffremap(k,1)+q01
        enddo

      zbot=0.0
      kbot=1
      do k=1,Nout
        ztop=zbot  !top is bottom of previous layer
        ktop=kbot
        if     (ztop.ge.z_win(ktop+1)) then
          ktop=ktop+1
        endif
        
        zbot=z_wout(k+1)
        zthk=zbot-ztop

        if     (zthk.gt.dpthin .and. ztop.lt.z_wout(Nout+1)) then

          kbot=ktop
          do while (z_win(kbot+1).lt.zbot.and.kbot.lt.N)
            kbot=kbot+1
          enddo
          zbox=zbot
          do k1= k+1,Nout
            if     (z_wout(k1+1)-z_wout(k1).gt.dpthin) then
              exit !thick layer
            else
              zbox=z_wout(k1+1)  !include thin adjacent layers
              if     (zbox.eq.z_wout(Nout+1)) then
                exit !at bottom
              endif
            endif
          enddo
          zthk=zbox-ztop

          kbox=ktop
          do while (z_win(kbox+1).lt.zbox.and.kbox.lt.N)
            kbox=kbox+1
          enddo
          
          if     (ktop.eq.kbox) then


            if     (z_wout(k)  .ne.z_win(kbox)   .or.z_wout(k+1).ne.z_win(kbox+1)     ) then

              if     (hin(kbox).gt.dpthin) then
                q001 = (zbox-z_win(kbox))/hin(kbox)
                q002 = (ztop-z_win(kbox))/hin(kbox)
                q01=q001**2+q002**2+q001*q002+1.-2.*(q001+q002)
                q02=q01-1.+(q001+q002)
                q0=1.-q01-q02
              else
                q0 = 1.0
                q01 = 0.
                q02 = 0.
              endif
          tabout(k)=q0*coeffremap(kbox,1)+q01*coeffremap(kbox,2)+q02*coeffremap(kbox,3)
              
            else
            tabout(k) = tabin(kbox)
              
            endif 

          else

            if     (ktop.le.k .and. kbox.ge.k) then
              ka = k
            elseif (kbox-ktop.ge.3) then
              ka = (kbox+ktop)/2
            elseif (hin(ktop).ge.hin(kbox)) then
              ka = ktop
            else
              ka = kbox
            endif !choose ka

            offset=coeffremap(ka,1)

            qtop = z_win(ktop+1)-ztop !partial layer thickness
            if     (hin(ktop).gt.dpthin) then
              q=(ztop-z_win(ktop))/hin(ktop)
              q01=q*(q-1.)
              q02=q01+q
              q0=1-q01-q02            
            else
              q0 = 1.
              q01 = 0.
              q02 = 0.
            endif
            
            tsum =((q0*coeffremap(ktop,1)+q01*coeffremap(ktop,2)+q02*coeffremap(ktop,3))-offset)*qtop

            do k1= ktop+1,kbox-1
              tsum =tsum +(coeffremap(k1,1)-offset)*hin(k1)
            enddo !k1

            
            qbot = zbox-z_win(kbox) !partial layer thickness
            if     (hin(kbox).gt.dpthin) then
              q=qbot/hin(kbox)
              q01=(q-1.)**2
              q02=q01-1.+q
              q0=1-q01-q02                            
            else
              q0 = 1.0
              q01 = 0.
              q02 = 0.
            endif
           
            tsum = tsum +((q0*coeffremap(kbox,1)+q01*coeffremap(kbox,2)+q02*coeffremap(kbox,3))-offset)*qbot
            
            rpsum=1.0d0/zthk
              tabout(k)=offset+tsum*rpsum
              
          endif !single or multiple layers
        else
        if (k==1) then
        print *,'problem = ',zthk,z_wout(k+1),hout(1)
        endif
         tabout(k) = tabout(k-1)
          
        endif !normal:thin layer
      enddo !k
            
      return
      
!---------------------------------------------------------------------------------------------------
end subroutine reconstructandremap
!===================================================================================================
!









!
!===================================================================================================
      subroutine reconstructandremap_ps(tabin,hin,tabout,hout,N,Nout)      ! parabloc spline
!---------------------------------------------------------------------------------------------------      
      implicit none
      integer N, Nout
      real(8) tabin(N), tabout(Nout)
      real(8) hin(N), hout(Nout)
      real(8) coeffremap(N,3),zwork(N,3)
      real(8) zwork2(N+1,3)
      
      real(8) my_zwork(0:N,3)
      real(8) my_zwork2(0:N,3)
      
      integer k
      double precision, parameter :: dsmll=1.0d-8  
      real(8) q,q01,q02,q001,q002,q0
      real(8) z_win(1:N+1), z_wout(1:Nout+1)
      real(8),parameter :: dpthin = 1.D-3
      integer :: k1, kbox, ktop, ka, kbot
      real(8) :: tsum, qbot, rpsum, zbox, ztop, zthk, zbot, offset, qtop
      real(8) :: p
      real(8) :: qtri(0:N)

      z_win(1)=0.; z_wout(1)= 0.
      do k=1,N
       z_win(k+1)=z_win(k)+hin(k)
      enddo 
      
      do k=1,Nout
       z_wout(k+1)=z_wout(k)+hout(k)       
      enddo       

        do k=2,N
          zwork(k,1)=1./(hin(k-1)+hin(k))
        enddo
        
        do k=2,N-1
          q0 = 1./(hin(k-1)+hin(k)+hin(k+1))
          zwork(k,2)=hin(k-1)+2.*hin(k)+hin(k+1)
          zwork(k,3)=q0
        enddo       
      
        do k= 2,N
        zwork2(k,1)=zwork(k,1)*(tabin(k)-tabin(k-1))
        enddo
        
        coeffremap(:,1) = tabin(:)
 
         do k=2,N-1
        q001 = hin(k)*zwork2(k+1,1)
        q002 = hin(k)*zwork2(k,1)        
 !       if (q001*q002 < 0) then
 !         q001 = 0.
 !         q002 = 0.
 !       endif
        q=zwork(k,2)
        q01=q*zwork2(k+1,1)
        q02=q*zwork2(k,1)
 !       if (abs(q001) > abs(q02)) q001 = q02
 !       if (abs(q002) > abs(q01)) q002 = q01
        
        q=(q001-q002)*zwork(k,3)
        q001=q001-q*hin(k+1)
        q002=q002+q*hin(k-1)
        
        coeffremap(k,3)=coeffremap(k,1)+q001
        coeffremap(k,2)=coeffremap(k,1)-q002
        
        zwork2(k,1)=(2.*q001-q002)**2
        zwork2(k,2)=(2.*q002-q001)**2
        enddo
        
        do k=1,N
        if     (k.eq.1 .or. k.eq.N .or.   hin(k).le.dpthin) then
        coeffremap(k,3) = coeffremap(k,1)
        coeffremap(k,2) = coeffremap(k,1)
        zwork2(k,1) = 0.
        zwork2(k,2) = 0.
        endif
        enddo
        
        do k=2,N
        q002=max(zwork2(k-1,2),dsmll)
        q001=max(zwork2(k,1),dsmll)
        zwork2(k,3)=(q001*coeffremap(k-1,3)+q002*coeffremap(k,2))/(q001+q002)
        enddo
        
        zwork2(1,3) = 2*coeffremap(1,1)-zwork2(2,3)
        zwork2(N+1,3)=2*coeffremap(N,1)-zwork2(N,3)
 
        do k=1,N
        q01=zwork2(k+1,3)-coeffremap(k,1)
        q02=coeffremap(k,1)-zwork2(k,3)
!        q001=2.*q01
!        q002=2.*q02
!        if (q01*q02<0) then
!          q01=0.
!          q02=0.
!        elseif (abs(q01)>abs(q002)) then
!          q01=q002
!        elseif (abs(q02)>abs(q001)) then
!          q02=q001
!        endif
        coeffremap(k,2)=coeffremap(k,1)-q02
        coeffremap(k,3)=coeffremap(k,1)+q01
        enddo
        
        
        do k=0,N
        if (k==0) then
        my_zwork(k,1)=0.
        my_zwork(k,2)=1.
        my_zwork(k,3)=0.5
        my_zwork2(k,1)=1.5*tabin(1)
        elseif (k==N) then
        my_zwork(k,1)=0.5
        my_zwork(k,2)=1.
        my_zwork(k,3)=0.
        my_zwork2(k,1)=1.5*tabin(k)        
        else
        my_zwork(k,1)=hin(k+1)
        my_zwork(k,2)=2.*(hin(k)+hin(k+1))
        my_zwork(k,3)=hin(k)
        my_zwork2(k,1)=3.*(hin(k+1)*tabin(k)+hin(k)*tabin(k+1))
        my_zwork2(k,2)=my_zwork2(k,1)
        endif
        enddo
        
        qtri(0)=-my_zwork(0,3)/my_zwork(0,2)
        my_zwork2(0,1)=my_zwork2(0,1)/my_zwork(0,2)
        
        do k=1,N
        p=1.0/(my_zwork(k,2)+my_zwork(k,1)*qtri(k-1))
        qtri(k)=-my_zwork(k,3)*p
        my_zwork2(k,1)=(my_zwork2(k,1)-my_zwork(k,1)*my_zwork2(k-1,1))*p
        enddo
        
        do k=N-1,0,-1
        my_zwork2(k,1)=my_zwork2(k,1)+qtri(k)*my_zwork2(k+1,1)
        enddo
        
        do k=1,N
        coeffremap(k,2)=my_zwork2(k-1,1)
        coeffremap(k,3)=my_zwork2(k,1)
        enddo
        
        do k=2,N-1
!        print *,'VAL22 = ',my_zwork(k,1)*my_zwork2(k-1,1)
!     &+my_zwork(k,2)*my_zwork2(k,1)
!     &  +my_zwork(k,3)*my_zwork2(k+1,1),my_zwork2(k,2)
        enddo

      zbot=0.0
      kbot=1
      do k=1,Nout
        ztop=zbot  !top is bottom of previous layer
        ktop=kbot
        if     (ztop.ge.z_win(ktop+1)) then
          ktop=ktop+1
        endif
        
        zbot=z_wout(k+1)
        zthk=zbot-ztop

        if     (zthk.gt.dpthin .and. ztop.lt.z_wout(Nout+1)) then

          kbot=ktop
          do while (z_win(kbot+1).lt.zbot.and.kbot.lt.N)
            kbot=kbot+1
          enddo
          zbox=zbot
          do k1= k+1,Nout
            if     (z_wout(k1+1)-z_wout(k1).gt.dpthin) then
              exit !thick layer
            else
              zbox=z_wout(k1+1)  !include thin adjacent layers
              if     (zbox.eq.z_wout(Nout+1)) then
                exit !at bottom
              endif
            endif
          enddo
          zthk=zbox-ztop

          kbox=ktop
          do while (z_win(kbox+1).lt.zbox.and.kbox.lt.N)
            kbox=kbox+1
          enddo
          
          if     (ktop.eq.kbox) then


            if     (z_wout(k)  .ne.z_win(kbox).or.z_wout(k+1).ne.z_win(kbox+1)     ) then

              if     (hin(kbox).gt.dpthin) then
                q001 = (zbox-z_win(kbox))/hin(kbox)
                q002 = (ztop-z_win(kbox))/hin(kbox)
                q01=q001**2+q002**2+q001*q002+1.-2.*(q001+q002)
                q02=q01-1.+(q001+q002)
                q0=1.-q01-q02
              else
                q0 = 1.0
                q01 = 0.
                q02 = 0.
              endif
          tabout(k)=q0*coeffremap(kbox,1)+q01*coeffremap(kbox,2) &
       +q02*coeffremap(kbox,3)
            else
            tabout(k) = tabin(kbox)
              
            endif 

          else

            if     (ktop.le.k .and. kbox.ge.k) then
              ka = k
            elseif (kbox-ktop.ge.3) then
              ka = (kbox+ktop)/2
            elseif (hin(ktop).ge.hin(kbox)) then
              ka = ktop
            else
              ka = kbox
            endif !choose ka

            offset=coeffremap(ka,1)

            qtop = z_win(ktop+1)-ztop !partial layer thickness
            if     (hin(ktop).gt.dpthin) then
              q=(ztop-z_win(ktop))/hin(ktop)
              q01=q*(q-1.)
              q02=q01+q
              q0=1-q01-q02            
            else
              q0 = 1.
              q01 = 0.
              q02 = 0.
            endif
            
            tsum =((q0*coeffremap(ktop,1)+q01*coeffremap(ktop,2)+ &
       q02*coeffremap(ktop,3))-offset)*qtop

            do k1= ktop+1,kbox-1
              tsum =tsum +(coeffremap(k1,1)-offset)*hin(k1)
            enddo !k1

            
            qbot = zbox-z_win(kbox) !partial layer thickness
            if     (hin(kbox).gt.dpthin) then
              q=qbot/hin(kbox)
              q01=(q-1.)**2
              q02=q01-1.+q
              q0=1-q01-q02                            
            else
              q0 = 1.0
              q01 = 0.
              q02 = 0.
            endif
           
        tsum = tsum +((q0*coeffremap(kbox,1)+q01*coeffremap(kbox,2)+ &
       q02*coeffremap(kbox,3))-offset)*qbot
            
            rpsum=1.0d0/zthk
              tabout(k)=offset+tsum*rpsum
              
          endif !single or multiple layers
        else
        if (k==1) then
        print *,'problem = ',zthk,z_wout(k+1),hout(1),hin(1),hin(2)
        stop
        endif
         tabout(k) = tabout(k-1)
          
        endif !normal:thin layer
      enddo !k
            
      return
!---------------------------------------------------------------------------------------------------
      end subroutine reconstructandremap_ps
!===================================================================================================
!

end module module_interp
