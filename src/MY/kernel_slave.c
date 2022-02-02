/*************************************************************************
	> File Name: src/OPT/SLAVE/kernel_slave.c
	> Author: zhousc
	> Created Time: 2020年09月22日 星期二 15时43分23秒
 ************************************************************************/

# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include "simd.h"
# include "crts.h"
# include "struct.h"
# include "math_cxl.h"

# define dsimd
//# define dsimd_p2z2
# define RMA_BUFFER_LEN 64
# define remote_ldm_addr(coreid,ldm_var) ((unsigned long)&ldm_var | ((unsigned long)coreid << 20 ) |(1ULL << 45))
# define lock CRTS_smutex_lock_array()
# define unlock CRTS_smutex_unlock_array()
# define tid CRTS_tid
# define dmax(x,y) ((x)>(y) ? (x):(y))
# define dmin(x,y) ((x)<(y) ? (x):(y))

# define ALLSYN athread_syn(ARRAY_SCOPE, 0xffff)

__thread_local crts_rply_t dma_rply = 0;
__thread_local crts_rply_t get_rply[2], put_rply[2];
__thread_local unsigned int D_COUNT = 0;
__thread_local volatile int rma_rply_l = 0, rma_rply_r = 0, rma_signal = 0, zdfsh2_rply = 0;
__thread_local int j_len, k_idx, j_idx, j_od, bais, n_bais, dma_len;
__thread_local int bias_l, bias_c, bias_n, dma_len;
__thread_local int k_id, j_id, k_st, j_st, k_block, j_block, k_ed;
__thread_local double stbu[64], stbv[64];

// op == 1,2
void devide_sid(int op, int *k_id, int *j_id){
   int mod = 1 << op; //2^op
   mod--;
   *k_id = _PEN >> op; // _PEN / 2^op
   *j_id = _PEN & mod;
}

// (myid + 1) ==  (1<<op);
void area(int op, int dim, int myid, int *st, int *block){
  int val = 1 << op;
  int mod = dim >> op;
  int res = dim & (val -1);

  if(res == 0){
  	*block = mod;  
    *st = myid * mod;
  } else {
	  if(myid < res){
	    *block = mod +1;
      *st = myid * (*block);
	  } else {
		  *block = mod;  
	    *st = myid * mod + res;
	  }	
  }
}

inline double stencil_max(double x1, double x2, double x3, double x4, double x5, double x6, double x7) {
  double a1 = dmax(x1,x2);
  double a2 = dmax(a1,x3);
  double a3 = dmax(a2,x4);
  double a4 = dmax(a3,x5);
  double a5 = dmax(a4,x6);
  return dmax(a5,x7);
}

inline double stencil_min(double x1, double x2, double x3, double x4, double x5, double x6, double x7) {
  double a1 = dmin(x1,x2);
  double a2 = dmin(a1,x3);
  double a3 = dmin(a2,x4);
  double a4 = dmin(a3,x5);
  double a5 = dmin(a4,x6);
  return dmin(a5,x7);
}

inline double min3(double x1, double x2, double x3) {
  double a1 = dmin(x1,x2);
  return dmin(a1,x3);
}

inline double sign(double x1, double x2) {
  int s = (x2<0) ? -1 : 1;
  return fabs(x1) * s;
}

void slave_fct1_(var1 *v1) {
  var1 v;
  CRTS_dma_iget(&v, v1, sizeof(var1), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  double *pun = v.loc_pun, *pvn = v.loc_pvn, *ptb = v.loc_ptb, *zwx = v.loc_zwx, *zwy = v.loc_zwy, *ptn = v.loc_ptn, *zwz = v.loc_zwz, *pwn = v.loc_pwn, *wmask = v.loc_wmask, *e3t_n=v.loc_e3t_n, *e3t_a=v.loc_e3t_a, *e3t_b = v.loc_e3t_b, *r1_e1e2t = v.loc_r1_e1e2t, *tmask = v.loc_tmask, *pta = v.loc_pta, *zwi = v.loc_zwi;
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1, jn = v.jn-1;
  double p2dt = v.p2dt;
  // local
  double zfp_ui, zfm_ui, zfp_vj, zfm_vj, zfp_wk, zfm_wk, ztra;
  int ji, jj, i;
  int zwz_j_len, zwz_dma_len;

  for(i = tid; i < 2*jpkm1; i+=64) {
    j_od = i % 2;
    k_idx = i / 2; // get时k维度索引
    j_len = (jpjm1%2 == 1 && j_od == 0) ? jpjm1/2+1 : jpjm1/2; // 每一块j维度的大小
    j_idx = j_od == 0 ? 0 : jpjm1%2+j_len;
    bais = k_idx*jpi*jpj + j_idx*jpi;
    n_bais = jn*jpi*jpj*jpk;
    dma_len = jpi*j_len*8;

    // 计算zwx zwy
    double local_pun[j_len][jpi], local_pvn[j_len][jpi], local_zwx[j_len][jpi], local_zwy[j_len][jpi], local_ptb[j_len+1][jpi];

    CRTS_dma_iget(&local_pun[0][0], pun+bais, dma_len, &dma_rply);
    CRTS_dma_iget(&local_pvn[0][0], pvn+bais, dma_len, &dma_rply);
    CRTS_dma_iget(&local_ptb[0][0], ptb+n_bais+bais, jpi*(j_len+1)*8, &dma_rply);
    CRTS_dma_iget(&local_zwx[0][0], zwx+bais, dma_len, &dma_rply);
    CRTS_dma_iget(&local_zwy[0][0], zwy+bais, dma_len, &dma_rply);
    D_COUNT+=5;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jj = 0; jj < j_len; jj++) {
      for(ji = 0; ji < fs_jpim1; ji++){
        zfp_ui = local_pun[jj][ji] + fabs(local_pun[jj][ji]);
        zfm_ui = local_pun[jj][ji] - fabs(local_pun[jj][ji]);
        zfp_vj = local_pvn[jj][ji] + fabs(local_pvn[jj][ji]);
        zfm_vj = local_pvn[jj][ji] - fabs(local_pvn[jj][ji]);
        local_zwx[jj][ji] = 0.5 * (zfp_ui * local_ptb[jj][ji] + zfm_ui * local_ptb[jj][ji+1]);
        local_zwy[jj][ji] = 0.5 * (zfp_vj * local_ptb[jj][ji] + zfm_vj * local_ptb[jj+1][ji]);
      }
    }

    CRTS_dma_iput(zwx+bais, &local_zwx[0][0], dma_len, &dma_rply);
    CRTS_dma_iput(zwy+bais, &local_zwy[0][0], dma_len, &dma_rply);
    D_COUNT+=2;
    //CRTS_dma_wait_value(&dma_rply, D_COUNT);

    // 计算zwz
    zwz_j_len = (j_od == 0 ? j_len : j_len+1);
    zwz_dma_len = (j_od == 0 ? dma_len : jpi*zwz_j_len*8);
    //zwz_j_len = jpj;
    //zwz_dma_len = jpi*j_len*8;
    double local_ptb_up[zwz_j_len][jpi];
    double local_pwn[zwz_j_len][jpi], local_wmask[zwz_j_len][jpi], local_zwz[zwz_j_len][jpi];

    CRTS_dma_iget(&local_pwn[0][0], pwn+bais, zwz_dma_len, &dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
    if(k_idx == 0) {
      for(jj = 0; jj < zwz_j_len; jj++) {
        for(ji = 0; ji < jpi; ji++) {
          local_zwz[jj][ji] = local_pwn[jj][ji] * local_ptb[jj][ji];
        }
      }
    } else {
      CRTS_dma_iget(&local_ptb_up[0][0], ptb+n_bais+(k_idx-1)*jpi*jpj+j_idx*jpi, zwz_dma_len, &dma_rply);
      CRTS_dma_iget(&local_wmask[0][0], wmask+bais, zwz_dma_len, &dma_rply);
      D_COUNT+=2;
      CRTS_dma_wait_value(&dma_rply, D_COUNT);
      
      for(jj = 0; jj < zwz_j_len; jj++) {
        for(ji = 0; ji < jpi; ji++) {
          zfp_wk = local_pwn[jj][ji] + fabs(local_pwn[jj][ji]);
          zfm_wk = local_pwn[jj][ji] - fabs(local_pwn[jj][ji]);
          local_zwz[jj][ji] = 0.5 * (zfp_wk * local_ptb[jj][ji] + zfm_wk * local_ptb_up[jj][ji]) * local_wmask[jj][ji];
        }
      }
    }

    CRTS_dma_iput(zwz+bais, &local_zwz[0][0], zwz_dma_len, &dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  } 
}

void slave_fct2_(var1 *v1) {
  var1 v;
  CRTS_dma_iget(&v, v1, sizeof(var1), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  double *ptb = v.loc_ptb, *zwx = v.loc_zwx, *zwy = v.loc_zwy, *zwz = v.loc_zwz, \
         *e3t_n=v.loc_e3t_n, *e3t_a=v.loc_e3t_a, *e3t_b = v.loc_e3t_b, *r1_e1e2t = v.loc_r1_e1e2t, \
         *tmask = v.loc_tmask, *pta = v.loc_pta, *zwi = v.loc_zwi;
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1, jn = v.jn-1;
  double p2dt = v.p2dt;
  // local
  double ztra;
  int ji, jj, i;

  for(i = tid; i < 2*jpkm1; i+=64) {
    // 计算分块相关变量 
    j_od = i & 1;
    k_idx = i >> 1; // get时k维度索引
    j_len = (jpjm1%2 == 1 && j_od == 0) ? jpjm1/2+1 : jpjm1/2; // 每一块j维度的大小
    j_idx = j_od == 0 ? 0 : jpjm1%2+j_len;
    if(j_od==1) {
      j_len += 1;
      j_idx -= 1;
    }
    bais = k_idx*jpi*jpj + j_idx*jpi;
    n_bais = jn*jpi*jpj*jpk;
    dma_len = jpi*j_len*8;

    double local_zwx[j_len][jpi], local_zwy[j_len][jpi], local_zwz[j_len][jpi], local_zwz_nt[j_len][jpi], \
           local_e3t_n[j_len][jpi], local_e3t_a[j_len][jpi], local_e3t_b[j_len][jpi], local_r1_e1e2t[j_len][jpi], \
           local_tmask[j_len][jpi], local_pta[j_len][jpi], local_ptb[j_len][jpi], local_zwi[j_len][jpi];			 

    double ztra[j_len][jpi];
		double tmp;

    CRTS_dma_iget(&local_e3t_n[0][0],    e3t_n+bais,         dma_len, &dma_rply);
    CRTS_dma_iget(&local_r1_e1e2t[0][0], r1_e1e2t+j_idx*jpi, dma_len, &dma_rply);
    CRTS_dma_iget(&local_tmask[0][0],    tmask+bais,         dma_len, &dma_rply);
    CRTS_dma_iget(&local_pta[0][0],      pta+n_bais+bais,    dma_len, &dma_rply);
    CRTS_dma_iget(&local_zwx[0][0],      zwx+bais,           dma_len, &dma_rply);
    CRTS_dma_iget(&local_zwy[0][0],      zwy+bais,           dma_len, &dma_rply);
    CRTS_dma_iget(&local_zwz[0][0],      zwz+bais,           dma_len, &dma_rply);
    CRTS_dma_iget(&local_zwz_nt[0][0], zwz+(k_idx+1)*jpi*jpj+j_idx*jpi, dma_len, &dma_rply);
    D_COUNT+=8;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
    
    CRTS_dma_iget(&local_e3t_b[0][0],    e3t_b+bais,         dma_len, &dma_rply);
    CRTS_dma_iget(&local_ptb[0][0],      ptb+n_bais+bais,    dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3t_a[0][0],    e3t_a+bais,         dma_len, &dma_rply);
    D_COUNT+=3;

    for(jj = 1; jj < j_len; jj++) {
      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
        tmp = -(local_zwx[jj][ji] - local_zwx[jj][ji-1] \
               + local_zwy[jj][ji] - local_zwy[jj-1][ji] \
               + local_zwz[jj][ji] - local_zwz_nt[jj][ji]) * local_r1_e1e2t[jj][ji];
        local_pta[jj][ji] += tmp / local_e3t_n[jj][ji] * local_tmask[jj][ji];

				ztra[jj][ji]  = tmp;
      }
    }

    CRTS_dma_wait_value(&dma_rply, D_COUNT);
    CRTS_dma_iput(pta+n_bais+k_idx*jpi*jpj+(j_idx+1)*jpi, &local_pta[1][0], jpi*(j_len-1)*8, &dma_rply);

    for(jj = 1; jj < j_len; jj++) {
      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
        local_zwi[jj][ji] = (local_e3t_b[jj][ji] * local_ptb[jj][ji] + p2dt * ztra[jj][ji]) / local_e3t_a[jj][ji] * local_tmask[jj][ji];
      }
    }

    CRTS_dma_iput(zwi+k_idx*jpi*jpj+(j_idx+1)*jpi, &local_zwi[1][0], jpi*(j_len-1)*8, &dma_rply);
    D_COUNT+=2;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

void slave_fct3_(var1 *v1) {
  var1 v;
  CRTS_dma_iget(&v, v1, sizeof(var1), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  double *pun = v.loc_pun, *pvn = v.loc_pvn, *ptb = v.loc_ptb, *zwx = v.loc_zwx, *zwy = v.loc_zwy, \
         *zwz = v.loc_zwz, *pwn = v.loc_pwn, *wmask = v.loc_wmask, *e3t_n=v.loc_e3t_n, *e3t_a=v.loc_e3t_a, \
         *e3t_b = v.loc_e3t_b, *r1_e1e2t = v.loc_r1_e1e2t, *tmask = v.loc_tmask, *pta = v.loc_pta, *zwi = v.loc_zwi, \
         *ptn = v.loc_ptn;
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1, jn = v.jn-1;
  // local
  int ji, jj, i;

  for(i = tid; i < 2*jpkm1; i+=64) {
    // 计算分块相关变量 
    j_od = i & 1;
    k_idx = i >> 1; // get时k维度索引
    j_len = (jpjm1%2 == 1 && j_od == 0) ? jpjm1/2+1 : jpjm1/2; // 每一块j维度的大小
    j_idx = j_od == 0 ? 0 : jpjm1%2+j_len;
    bais = k_idx*jpi*jpj + j_idx*jpi;
    n_bais = jn*jpi*jpj*jpk;
    dma_len = jpi*j_len*8;

    // 定义local计算数组
    double local_pun[j_len][jpi], local_pvn[j_len][jpi], local_ptn[j_len+1][jpi], local_zwx[j_len][jpi], local_zwy[j_len][jpi];
    int bais = k_idx*jpi*jpj + j_idx*jpi;
    int n_bais = jn*jpi*jpj*jpk;
    int dma_len = jpi*j_len*8;

    CRTS_dma_iget(&local_pun[0][0], pun+bais, dma_len, &dma_rply);
    CRTS_dma_iget(&local_pvn[0][0], pvn+bais, dma_len, &dma_rply);
    CRTS_dma_iget(&local_ptn[0][0], ptn+n_bais+bais, jpi*(j_len+1)*8, &dma_rply);
    CRTS_dma_iget(&local_zwx[0][0], zwx+bais, dma_len, &dma_rply);
    CRTS_dma_iget(&local_zwy[0][0], zwy+bais, dma_len, &dma_rply);
    D_COUNT+=5;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jj = 0; jj < j_len; jj++) {
      for(ji = 0; ji < fs_jpim1; ji++){
        local_zwx[jj][ji] = (double)0.5 * local_pun[jj][ji] * (local_ptn[jj][ji] + local_ptn[jj][ji+1]) - local_zwx[jj][ji];
        local_zwy[jj][ji] = (double)0.5 * local_pvn[jj][ji] * (local_ptn[jj][ji] + local_ptn[jj+1][ji]) - local_zwy[jj][ji];
      }
    }

    CRTS_dma_iput(zwx+bais, &local_zwx[0][0], dma_len, &dma_rply);
    CRTS_dma_iput(zwy+bais, &local_zwy[0][0], dma_len, &dma_rply);
    D_COUNT+=2;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    double local_zwz[j_len][jpi], local_pwn[j_len][jpi], local_wmask[j_len][jpi], local_ptn_up[j_len][jpi];
    if(k_idx == 0) {
      for(jj = 0; jj < j_len; jj++) {
        for(ji = 0; ji < jpi; ji++) {
          local_zwz[jj][ji] = (double)0;
        }
      }
    } else {
      CRTS_dma_iget(&local_zwz[0][0],    zwz+bais,                               dma_len, &dma_rply);
      CRTS_dma_iget(&local_pwn[0][0],    pwn+bais,                               dma_len, &dma_rply);
      CRTS_dma_iget(&local_ptn[0][0],    ptn+n_bais+bais,                        dma_len, &dma_rply);
      CRTS_dma_iget(&local_ptn_up[0][0], ptn+n_bais+(k_idx-1)*jpi*jpj+j_idx*jpi, dma_len, &dma_rply);
      CRTS_dma_iget(&local_wmask[0][0],  wmask+bais,                             dma_len, &dma_rply);
      D_COUNT+=5;
      CRTS_dma_wait_value(&dma_rply, D_COUNT);

      for(jj = 1; jj < j_len; jj++) {
        for(ji = fs_2-1; ji < fs_jpim1; ji++) {
          local_zwz[jj][ji] = (local_pwn[jj][ji] * (double)0.5 * (local_ptn[jj][ji] + local_ptn_up[jj][ji]) - local_zwz[jj][ji]) * local_wmask[jj][ji];
        }
      }
      if(j_od == 1) {
        for(ji = fs_2-1; ji < fs_jpim1; ji++) {
          local_zwz[0][ji] = (local_pwn[0][ji] * (double)0.5 * (local_ptn[0][ji] + local_ptn_up[0][ji]) - local_zwz[0][ji]) * local_wmask[0][ji];
        }
      }
    }
    CRTS_dma_iput(zwz+bais, &local_zwz[0][0], dma_len, &dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }  
}

void slave_fct4_(var1 *v1) {
  var1 v;
  CRTS_dma_iget(&v, v1, sizeof(var1), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  double *zwx = v.loc_zwx, *zwy = v.loc_zwy, *zwz = v.loc_zwz, *e3t_n=v.loc_e3t_n, *e3t_a=v.loc_e3t_a, *r1_e1e2t = v.loc_r1_e1e2t, \
         *tmask = v.loc_tmask, *pta = v.loc_pta, *zwi = v.loc_zwi;
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1, jn = v.jn-1;
  double p2dt = v.p2dt;
  // local
  double ztra;
  int ji, jj, i;

  for(i = tid; i < 2*jpkm1; i+=64) {
    // 计算分块相关变量 
    j_od = i & 1;
    k_idx = i >> 1; // get时k维度索引
    j_len = (jpjm1%2 == 1 && j_od == 0) ? jpjm1/2+1 : jpjm1/2; // 每一块j维度的大小
    j_idx = j_od == 0 ? 0 : jpjm1%2+j_len;
    if(j_od==1) {
      j_len += 1;
      j_idx -= 1;
    }
    bais = k_idx*jpi*jpj + j_idx*jpi;
    n_bais = jn*jpi*jpj*jpk;
    dma_len = jpi*j_len*8;

    double local_zwx[j_len][jpi], local_zwy[j_len][jpi], local_zwz[j_len][jpi], local_zwz_nt[j_len][jpi], \
           local_e3t_n[j_len][jpi], local_e3t_a[j_len][jpi], local_r1_e1e2t[j_len][jpi], \
           local_tmask[j_len][jpi], local_pta[j_len][jpi], local_zwi[j_len][jpi];

    CRTS_dma_iget(&local_e3t_n[0][0], e3t_n+bais, dma_len, &dma_rply);
    CRTS_dma_iget(&local_r1_e1e2t[0][0], r1_e1e2t+j_idx*jpi, dma_len, &dma_rply);
    CRTS_dma_iget(&local_pta[0][0], pta+n_bais+bais, dma_len, &dma_rply);
    CRTS_dma_iget(&local_zwx[0][0], zwx+bais, dma_len, &dma_rply);
    CRTS_dma_iget(&local_zwy[0][0], zwy+bais, dma_len, &dma_rply);
    CRTS_dma_iget(&local_zwz[0][0], zwz+bais, dma_len, &dma_rply);
    CRTS_dma_iget(&local_zwz_nt[0][0], zwz+(k_idx+1)*jpi*jpj+j_idx*jpi, dma_len, &dma_rply);
    D_COUNT+=7;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
    //athread_get(PE_MODE, tmask+bais, &local_tmask[0][0], dma_len, &get_reply, 0, 0, 0);
    //athread_get(PE_MODE, e3t_a+bais, &local_e3t_a[0][0], dma_len, &get_reply, 0, 0, 0);
    
    for(jj = 1; jj < j_len; jj++) {
      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
        ztra = -(local_zwx[jj][ji] - local_zwx[jj][ji-1] \
               + local_zwy[jj][ji] - local_zwy[jj-1][ji] \
               + local_zwz[jj][ji] - local_zwz_nt[jj][ji]) * local_r1_e1e2t[jj][ji];
        local_pta[jj][ji] += ztra / local_e3t_n[jj][ji];
        //local_zwi[jj][ji] += p2dt * ztra / local_e3t_a[jj][ji] * local_tmask[jj][ji];
      }
    }
    CRTS_dma_iput(pta+n_bais+k_idx*jpi*jpj+(j_idx+1)*jpi, &local_pta[1][0], jpi*(j_len-1)*8, &dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  } 
}


void slave_nono1_(var2 *v2) {
  // get var
  var2 v;
  CRTS_dma_iget(&v, v2, sizeof(var2), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");
  double *paft=v.loc_paft, *pbef=v.loc_pbef, *tmask=v.loc_tmask, *zbup=v.loc_zbup, *zbdo=v.loc_zbdo;
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  double zbig = v.zbig;

  // local var
  int ji, jj, k;
  int k_id, j_id, k_st, j_st, k_block, j_block, k_ed, kcnt;

  // devide
  devide_sid(0, &k_id, &j_id);
  area(6, jpk, k_id, &k_st, &k_block);
  area(0, jpj, j_id, &j_st, &j_block);
  //j_id = 0; j_st = 0; j_block = jpjm1;

  k_ed = k_st + k_block;
  dma_len = jpi * j_block * 8;

  double local_paft[j_block][jpi], local_pbef[j_block][jpi], local_tmask[j_block][jpi], local_zbup[j_block][jpi], local_zbdo[j_block][jpi];

  for(k = k_st; k < k_ed; k++) {
    bias_c = k*jpi*jpj + j_st*jpi;

    CRTS_dma_iget(&local_pbef[0][0],  pbef+bias_c,  dma_len, &dma_rply);
    CRTS_dma_iget(&local_paft[0][0],  paft+bias_c,  dma_len, &dma_rply);
    CRTS_dma_iget(&local_tmask[0][0], tmask+bias_c, dma_len, &dma_rply);
    D_COUNT+=3;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
    
    for(jj = 0; jj < j_block; jj++) {
      for(ji = 0; ji < jpi; ji++){
        local_zbup[jj][ji] = dmax(local_pbef[jj][ji] * local_tmask[jj][ji] - zbig * ((double)1.0 - local_tmask[jj][ji]), local_paft[jj][ji] * local_tmask[jj][ji] - zbig * ((double)1.0 - local_tmask[jj][ji]));
        local_zbdo[jj][ji] = dmin(local_pbef[jj][ji] * local_tmask[jj][ji] + zbig * ((double)1.0 - local_tmask[jj][ji]), local_paft[jj][ji] * local_tmask[jj][ji] + zbig * ((double)1.0 - local_tmask[jj][ji]));
      }
    }

    CRTS_dma_iput(zbup+bias_c, &local_zbup[0][0], dma_len, &dma_rply);
    CRTS_dma_iput(zbdo+bias_c, &local_zbdo[0][0], dma_len, &dma_rply);
    D_COUNT+=2;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

void slave_nono2_(var2 *v2) {
  // get var
  var2 v;
  CRTS_dma_iget(&v, v2, sizeof(var2), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");
  double *paa=v.loc_paa, *pbb=v.loc_pbb, *pcc=v.loc_pcc, *paft=v.loc_paft, *zbup=v.loc_zbup, *zbdo=v.loc_zbdo, \
         *zbetup=v.loc_zbetup, *zbetdo=v.loc_zbetdo, *e1e2t=v.loc_e1e2t, *e3t_n=v.loc_e3t_n;
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1, jn = v.jn-1;
  double p2dt = v.p2dt, zrtrn = v.zrtrn;

  // local var
  int ji, jj, k;
  int k_id, j_id, k_st, j_st, k_block, j_block, k_ed, kcnt = 0;
  double zpos, zneg, zbt;
  int off_l, off_c, off_n; // n-1, n, n+1
  int bias_l, bias_c, bias_n;

  // devide
  devide_sid(2, &k_id, &j_id);
  area(4, jpkm1, k_id, &k_st, &k_block);
  area(2, jpjm1, j_id, &j_st, &j_block);
  if(j_st != 0) {
    j_block += 1;
    j_st -= 1;
  }

  // cal var
  double local_zbup[3][j_block+1][jpi], local_zbdo[3][j_block+1][jpi], local_paa[j_block][jpi], local_pbb[j_block][jpi], local_pcc[3][j_block][jpi], \
         local_zbetup[j_block][jpi], local_zbetdo[j_block][jpi], local_paft[j_block][jpi], local_e1e2t[j_block][jpi], local_e3t_n[j_block][jpi];
  double zup[j_block][jpi], zdo[j_block][jpi];

  k_ed = k_st + k_block;
  dma_len = jpi*j_block*8;
  int dma_len_add1 = jpi*(j_block+1)*8;

  CRTS_dma_iget(&local_e1e2t[0][0],        e1e2t+j_st*jpi, dma_len, &dma_rply);
  D_COUNT++;
  for(k = k_st; k < k_ed; k++) {
    int k_l = k == 0 ? k : (k-1);
	  off_l = k_l % 3; off_c = k % 3; off_n = (k+1) % 3;
    bias_l = k_l*jpi*jpj + j_st*jpi; bias_c = k*jpi*jpj + j_st*jpi; bias_n = (k+1)*jpi*jpj + j_st*jpi;

    if(kcnt == 0) { // first get (n-1 n)
      CRTS_dma_iget(&local_zbup[off_l][0][0], zbup+bias_l, dma_len_add1, &dma_rply);
      CRTS_dma_iget(&local_zbdo[off_l][0][0], zbdo+bias_l, dma_len_add1, &dma_rply);
      CRTS_dma_iget(&local_zbup[off_c][0][0], zbup+bias_c, dma_len_add1, &dma_rply);
      CRTS_dma_iget(&local_zbdo[off_c][0][0], zbdo+bias_c, dma_len_add1, &dma_rply);
      CRTS_dma_iget(&local_pcc[off_c][0][0],  pcc+bias_c,  dma_len,      &dma_rply);
      D_COUNT+=5;
      kcnt = 1;
    }
    CRTS_dma_iget(&local_zbup[off_n][0][0], zbup+bias_n,   dma_len_add1, &dma_rply);
    CRTS_dma_iget(&local_zbdo[off_n][0][0], zbdo+bias_n,   dma_len_add1, &dma_rply);
    D_COUNT+=2;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    // async get
    CRTS_dma_iget(&local_pcc[off_n][0][0],  pcc+bias_n,    dma_len,      &dma_rply);
    CRTS_dma_iget(&local_paa[0][0],         paa+bias_c,    dma_len,      &dma_rply);
    CRTS_dma_iget(&local_pbb[0][0],         pbb+bias_c,    dma_len,      &dma_rply);
    CRTS_dma_iget(&local_e3t_n[0][0],       e3t_n+bias_c,  dma_len,      &dma_rply);
    CRTS_dma_iget(&local_paft[0][0],        paft+bias_c,   dma_len,      &dma_rply);
    D_COUNT+=5;
    
    for(jj = 1; jj < j_block; jj++) {
      for(ji = fs_2-1; ji < fs_jpim1; ji++){
        zup[jj][ji] = stencil_max(local_zbup[off_c][jj][ji], \
                          local_zbup[off_c][jj][ji-1], local_zbup[off_c][jj][ji+1], \
                          local_zbup[off_c][jj-1][ji], local_zbup[off_c][jj+1][ji], \
                          local_zbup[off_l][jj][ji], local_zbup[off_n][jj][ji]);
        zdo[jj][ji] = stencil_min(local_zbdo[off_c][jj][ji], \
                          local_zbdo[off_c][jj][ji-1], local_zbdo[off_c][jj][ji+1], \
                          local_zbdo[off_c][jj-1][ji], local_zbdo[off_c][jj+1][ji], \
                          local_zbdo[off_l][jj][ji], local_zbdo[off_n][jj][ji]);
      }
    }
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
    for(jj = 1; jj < j_block; jj++) {
      for(ji = fs_2-1; ji < fs_jpim1; ji++){
        zpos = dmax(0.0, local_paa[jj][ji-1]) - dmin(0.0, local_paa[jj][ji]) \
             + dmax(0.0, local_pbb[jj-1][ji]) - dmin(0.0, local_pbb[jj][ji]) \
             + dmax(0.0, local_pcc[off_n][jj][ji]) - dmin(0.0, local_pcc[off_c][jj][ji]);
        zneg = dmax(0.0, local_paa[jj][ji]) - dmin(0.0, local_paa[jj][ji-1]) \
             + dmax(0.0, local_pbb[jj][ji]) - dmin(0.0, local_pbb[jj-1][ji]) \
             + dmax(0.0, local_pcc[off_c][jj][ji]) - dmin(0.0, local_pcc[off_n][jj][ji]);
        zbt = local_e1e2t[jj][ji] * local_e3t_n[jj][ji] / p2dt;
        local_zbetup[jj][ji] = (zup[jj][ji] - local_paft[jj][ji]) / (zpos + zrtrn) * zbt;
        local_zbetdo[jj][ji] = (local_paft[jj][ji] - zdo[jj][ji]) / (zneg + zrtrn) * zbt;
      }
    }
    CRTS_dma_iput(zbetup+k*jpi*jpj+(j_st+1)*jpi, &local_zbetup[1][0], jpi*(j_block-1)*8, &dma_rply);
    CRTS_dma_iput(zbetdo+k*jpi*jpj+(j_st+1)*jpi, &local_zbetdo[1][0], jpi*(j_block-1)*8, &dma_rply);
    D_COUNT+=2;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

void slave_nono3_(var2 *v2) {
  // get var
  var2 v;
  CRTS_dma_iget(&v, v2, sizeof(var2), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");
  double *zbetdo=v.loc_zbetdo, *zbetup=v.loc_zbetup, *paa=v.loc_paa, *pbb=v.loc_pbb, *pcc=v.loc_pcc;
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;

  // local var
  int ji, jj, k;
  int k_id, j_id, k_st, j_st, k_block, j_block, k_ed, kcnt = 0;
  double zau, zbu, zcu, zav, zbv, zcv, za, zb, zc;
  int off_c, off_n; // n, n+1
  int bias_l, bias_c, bias_n;

  // devide
  devide_sid(0, &k_id, &j_id);
  area(6, jpkm1, k_id, &k_st, &k_block);
  area(0, jpjm1, j_id, &j_st, &j_block);
  if(j_st != 0) {
    j_block += 1;
    j_st -= 1;
  }

  k_ed = k_st + k_block;
  dma_len = jpi*j_block*8;
  int dma_len_add1 = jpi*(j_block+1)*8;

  double local_zbetdo[2][j_block+1][jpi], local_zbetup[2][j_block+1][jpi], local_paa[j_block][jpi], local_pbb[j_block][jpi], local_pcc_nt[j_block][jpi];

  k = k_st;
  off_c = k & 1; off_n = (k+1) & 1;
  bias_c = k*jpi*jpj + j_st*jpi; bias_n = (k+1)*jpi*jpj + j_st*jpi;
  CRTS_dma_iget(&local_zbetdo[off_c][0][0], zbetdo+bias_c, dma_len_add1, &dma_rply);
  CRTS_dma_iget(&local_zbetup[off_c][0][0], zbetup+bias_c, dma_len_add1, &dma_rply);
  D_COUNT+=2;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);

  for(k = k_st; k < k_ed; k++) {
	off_c = k & 1; off_n = (k+1) & 1;
    bias_c = k*jpi*jpj + j_st*jpi; bias_n = (k+1)*jpi*jpj + j_st*jpi;

    CRTS_dma_iget(&local_zbetup[off_n][0][0], zbetup+bias_n, dma_len_add1, &dma_rply);
    CRTS_dma_iget(&local_zbetdo[off_n][0][0], zbetdo+bias_n, dma_len_add1, &dma_rply);
    CRTS_dma_iget(&local_pcc_nt[0][0],        pcc+bias_n,    dma_len,      &dma_rply);
    CRTS_dma_iget(&local_paa[0][0],           paa+bias_c,    dma_len,      &dma_rply);
    CRTS_dma_iget(&local_pbb[0][0],           pbb+bias_c,    dma_len,      &dma_rply);
    D_COUNT+=5;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jj = 1; jj < jpjm1; jj++) {
      
      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
        zau = min3((double)1, local_zbetdo[off_c][jj][ji], local_zbetup[off_c][jj][ji+1]);
        zbu = min3((double)1, local_zbetup[off_c][jj][ji], local_zbetdo[off_c][jj][ji+1]);
        zcu = 0.5 + sign(0.5, local_paa[jj][ji]);
        local_paa[jj][ji] *= zcu * zau + ((double)1 - zcu) * zbu;

        zav = min3((double)1, local_zbetdo[off_c][jj][ji], local_zbetup[off_c][jj+1][ji]);
        zbv = min3((double)1, local_zbetup[off_c][jj][ji], local_zbetdo[off_c][jj+1][ji]);
        zcv = 0.5 + sign(0.5, local_pbb[jj][ji]);
        local_pbb[jj][ji] *= zcv * zav + ((double)1 - zcv) * zbv;

        za = min3(1.0, local_zbetdo[off_n][jj][ji], local_zbetup[off_c][jj][ji]);
        zb = min3(1.0, local_zbetup[off_n][jj][ji], local_zbetdo[off_c][jj][ji]);
        zc = 0.5 + sign(0.5, local_pcc_nt[jj][ji]);
        local_pcc_nt[jj][ji] *= zc * za + ((double)1 - zc) * zb;
      }
    }
    CRTS_dma_iput(paa+k*jpi*jpj+(j_st+1)*jpi, &local_paa[1][0], jpi*(j_block-1)*8, &dma_rply);
    CRTS_dma_iput(pbb+k*jpi*jpj+(j_st+1)*jpi, &local_pbb[1][0], jpi*(j_block-1)*8, &dma_rply);
    CRTS_dma_iput(pcc+(k+1)*jpi*jpj+(j_st+1)*jpi, &local_pcc_nt[1][0], jpi*(j_block-1)*8, &dma_rply);
    D_COUNT+=3;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

void slave_iso1_(var3 *v3) {
  var3 v;
  CRTS_dma_iget(&v, v3, sizeof(var3), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  double *wmask=v.loc_wmask, *umask=v.loc_umask, *vmask=v.loc_vmask, *pahu=v.loc_pahu, *pahv=v.loc_pahv, *wslpi=v.loc_wslpi, *wslpj=v.loc_wslpj, *ah_wslp2=v.loc_ah_wslp2;
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  // local
  double zmsku, zmskv ; // zahu_w, zahv_w; 
  int ji, jj, i;
  int db[2], off_0, off_1, last_bais, kcnt;
  int k_id, j_id, k_st, k_ed, j_st, j_ed, k_block, j_block;

  devide_sid(1, &k_id, &j_id);
  area(5, jpkm1, k_id, &k_st, &k_block);
  area(1, jpjm1, j_id, &j_st, &j_block);
  j_len = j_block;
  j_idx = j_st;
  if(j_st != 0){
    j_len += 1;
    j_idx -= 1;
  }

  dma_len = jpi*j_len*8;
//  if(_PEN == 0) printf("jpkm1: %d\n", jpkm1);
  double local_wmask[j_len][jpi], local_umask[2][j_len][jpi], local_vmask[2][j_len][jpi], local_pahu[2][j_len][jpi], local_pahv[2][j_len][jpi], local_wslpi[j_len][jpi], local_wslpj[j_len][jpi], local_ah_wslp2[j_len][jpi], zahu_w[j_len][jpi], zahv_w[j_len][jpi] ;
  kcnt = 0; k_ed = k_st+k_block;

  for(i = k_st; i < k_ed ; i++){

	if(i == 0) continue;
    bais = i*jpi*jpj + j_idx*jpi;

	db[0] = i & 1;
	db[1] = (i+1) & 1;
	off_0 = db[0]; off_1 = db[1];

	if(kcnt == 0){
	  last_bais = (i-1)*jpj*jpi+j_idx*jpi;
      CRTS_dma_iget(&local_umask[off_0][0][0],    umask+last_bais,    dma_len, &dma_rply);
      CRTS_dma_iget(&local_vmask[off_0][0][0],    vmask+last_bais,    dma_len, &dma_rply);
      CRTS_dma_iget(&local_pahu[off_0][0][0],     pahu+last_bais,     dma_len, &dma_rply);
      CRTS_dma_iget(&local_pahv[off_0][0][0],     pahv+last_bais,     dma_len, &dma_rply);
	  D_COUNT += 4;
	  kcnt = 1;
	}

    CRTS_dma_iget(&local_wmask[0][0],    wmask+bais,    dma_len, &dma_rply);
    CRTS_dma_iget(&local_umask[off_1][0][0],    umask+bais,    dma_len, &dma_rply);
    CRTS_dma_iget(&local_vmask[off_1][0][0],    vmask+bais,    dma_len, &dma_rply);
    CRTS_dma_iget(&local_pahu[off_1][0][0],     pahu+bais,     dma_len, &dma_rply);
    CRTS_dma_iget(&local_pahv[off_1][0][0],     pahv+bais,     dma_len, &dma_rply);
    D_COUNT += 5;

    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    CRTS_dma_iget(&local_wslpi[0][0],    wslpi+bais,    dma_len, &dma_rply);
    CRTS_dma_iget(&local_wslpj[0][0],    wslpj+bais,    dma_len, &dma_rply);
    D_COUNT += 2;

    for(jj = 1; jj < j_len; jj++) {
      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
        zmsku = local_wmask[jj][ji] / dmax(local_umask[off_0][jj][ji]   + local_umask[off_1][jj][ji-1] \
                                         + local_umask[off_0][jj][ji-1] + local_umask[off_1][jj][ji], (double)1.0);
        zmskv = local_wmask[jj][ji] / dmax(local_vmask[off_0][jj][ji]   + local_vmask[off_1][jj-1][ji] \
                                         + local_vmask[off_0][jj-1][ji] + local_vmask[off_1][jj][ji], (double)1.0);
        zahu_w[jj][ji] = (local_pahu[off_0][jj][ji]   + local_pahu[off_1][jj][ji-1] \
                + local_pahu[off_0][jj][ji-1] + local_pahu[off_1][jj][ji]) * zmsku;
        zahv_w[jj][ji] = (local_pahv[off_0][jj][ji]   + local_pahv[off_1][jj-1][ji] \
                + local_pahv[off_0][jj-1][ji] + local_pahv[off_1][jj][ji]) * zmskv;
      }
    }

	CRTS_dma_wait_value(&dma_rply, D_COUNT);

	doublev8 zv, wsi, zu, wsj, vr;
    for(jj = 1; jj < j_len; jj++) {
      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
        local_ah_wslp2[jj][ji] = zahu_w[jj][ji] * local_wslpi[jj][ji] * local_wslpi[jj][ji] \
                               + zahv_w[jj][ji] * local_wslpj[jj][ji] * local_wslpj[jj][ji];
      }
    }

    CRTS_dma_iput(ah_wslp2+ i*jpi*jpj + (j_idx+1)*jpi, &local_ah_wslp2[1][0], jpi*(j_len-1)*8, &dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

void slave_iso2_(var3 *v3){
  var3 v;
  CRTS_dma_iget(&v, v3, sizeof(var3), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  double *zdit=v.loc_zdit, *zdjt=v.loc_zdjt, *ptb=v.loc_ptb, *umask=v.loc_umask, *vmask=v.loc_vmask, *wmask=v.loc_wmask, *pahu=v.loc_pahu, *pahv=v.loc_pahv, *e2_e1u=v.loc_e2_e1u, *e3u_n=v.loc_e3u_n, *e1_e2v=v.loc_e1_e2v, *e3v_n=v.loc_e3v_n, *e2u=v.loc_e2u, *uslp=v.loc_uslp, *e1v=v.loc_e1v, *vslp=v.loc_vslp, *pta=v.loc_pta, *r1_e1e2t=v.loc_r1_e1e2t, *e3t_n=v.loc_e3t_n, *zzz=v.loc_zzz;
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1, jn = v.jn-1;
  double zsign = v.zsign;
  // local
  int ji, jj, i;
  int k_id, j_id, k_st, j_st, k_block, j_block, k_ed;
  int last_bais, next_bais, off_l, off_n, off_cur, off1_l, off1_cur, dma_len1;
  int db1[3], db[2];
  int cnt=0;

  devide_sid(2, &k_id, &j_id);
  area(4, jpkm1, k_id, &k_st, &k_block);
  area(2, jpjm1, j_id, &j_st, &j_block);
  j_len = j_block;
  j_idx = j_st;
 
  volatile int *p;
  double local_ptb[3][j_len+1][jpi], local_umask[j_len][jpi], local_vmask[j_len][jpi], local_zdit[j_len][jpi], local_zdjt[j_len][jpi];
  double local_zdk1t[j_len+1][jpi], local_zdkt[j_len+1][jpi], local_wmask[2][j_len+1][jpi];
  double local_pahu[j_len][jpi], local_pahv[j_len][jpi], local_e2_e1u[j_len][jpi], local_e1_e2v[j_len][jpi], local_e3u_n[j_len][jpi], local_e3v_n[j_len][jpi], local_e2u[j_len][jpi], local_e1v[j_len][jpi], local_uslp[j_len][jpi], local_vslp[j_len][jpi];
  double zabe1, zabe2, zmsku, zmskv, zcof1, zcof2;
  double local_zftu[j_len][jpi], local_zftv[j_len][jpi], local_pta[j_len][jpi];
  double local_r1_e1e2t[j_len][jpi], local_e3t_n[j_len][jpi];

  k_ed = k_st + k_block;
  dma_len = jpi*j_len*8;
  dma_len1 = jpi*(j_len+1)*8;
  n_bais = jn*jpj*jpi*jpk;
   
  CRTS_dma_iget(&local_e2u[0][0],    e2u+j_idx*jpi,    dma_len, &dma_rply);
  CRTS_dma_iget(&local_e1v[0][0],    e1v+j_idx*jpi,    dma_len, &dma_rply);
  CRTS_dma_iget(&local_e2_e1u[0][0], e2_e1u+j_idx*jpi, dma_len, &dma_rply);
  CRTS_dma_iget(&local_e1_e2v[0][0], e1_e2v+j_idx*jpi, dma_len, &dma_rply);
  D_COUNT += 4;

  for(i = k_st; i < k_ed; i++){
	  db1[0] = i % 3;  db1[1] = (i+1) % 3; db1[2] = (i+2) % 3;
	  off_l = db1[0];  off_cur = db1[1]; off_n = db1[2];

	  db[0] = i & 1;   db[1] = (i+1) & 1;
	  off1_l = db[0]; off1_cur = db[1];

    bais = i*jpi*jpj + j_idx*jpi;

    if(i == k_st){	
   	  CRTS_dma_iget(&local_ptb[off_l][0][0], ptb+n_bais+ (i-1)*jpi*jpj + j_idx*jpi, dma_len1, &dma_rply);
	    CRTS_dma_iget(&local_ptb[off_cur][0][0], ptb+n_bais+ bais, dma_len1, &dma_rply);
	    D_COUNT += 2;
    }

	  CRTS_dma_iget(&local_ptb[off_n][0][0], ptb+n_bais+ (i+1)*jpi*jpj + j_idx*jpi, dma_len1, &dma_rply);
	  CRTS_dma_iget(&local_umask[0][0], umask+bais, dma_len, &dma_rply);
	  CRTS_dma_iget(&local_vmask[0][0], vmask+bais, dma_len, &dma_rply);
    D_COUNT +=3;

	  CRTS_dma_wait_value(&dma_rply, D_COUNT);
	  if(i == k_st){
	    CRTS_dma_iget(&local_wmask[off1_l][0][0], wmask+ i*jpi*jpj + j_idx*jpi, dma_len1, &dma_rply);
	    D_COUNT++;
	  } 

	  CRTS_dma_iget(&local_wmask[off1_cur][0][0], wmask+ (i+1)*jpi*jpj + j_idx*jpi, dma_len1, &dma_rply);
	  D_COUNT += 1;
    
    for(jj = 0; jj < j_len; jj++) {
      for(ji = 0; ji < fs_jpim1; ji++) {
        local_zdit[jj][ji] = (local_ptb[off_cur][jj][ji+1] - local_ptb[off_cur][jj][ji]) * local_umask[jj][ji];
        local_zdjt[jj][ji] = (local_ptb[off_cur][jj+1][ji] - local_ptb[off_cur][jj][ji]) * local_vmask[jj][ji];
      }
    }

	  CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jj = 0; jj < j_len+1; jj++) {
      for(ji = 0; ji < jpi; ji++) {
        local_zdk1t[jj][ji] = (local_ptb[off_cur][jj][ji] - local_ptb[off_n][jj][ji]) * local_wmask[off1_cur][jj][ji];
      }
    }
	
	  if(i == 0){
      for(jj = 0; jj < j_len+1; jj++) {
        for(ji = 0; ji < jpi; ji++) {
	        local_zdkt[jj][ji] = local_zdk1t[jj][ji];
		    }
	    }	

	  }else{

      for(jj = 0; jj < j_len+1; jj++) {
        for(ji = 0; ji < jpi; ji++) {
          local_zdkt[jj][ji] = (local_ptb[off_l][jj][ji] - local_ptb[off_cur][jj][ji]) * local_wmask[off1_l][jj][ji];
        }
      }
	  }

    CRTS_dma_iget(&local_pahu[0][0],   pahu+bais,        dma_len, &dma_rply);
    CRTS_dma_iget(&local_pahv[0][0],   pahv+bais,        dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3u_n[0][0],  e3u_n+bais,       dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3v_n[0][0],  e3v_n+bais,       dma_len, &dma_rply);
    CRTS_dma_iget(&local_uslp[0][0],   uslp+bais,        dma_len, &dma_rply);
    CRTS_dma_iget(&local_vslp[0][0],   vslp+bais,        dma_len, &dma_rply);
    D_COUNT+=6;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    CRTS_dma_iget(&local_r1_e1e2t[0][0], r1_e1e2t+j_idx*jpi, dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3t_n[0][0],    e3t_n+bais,         dma_len, &dma_rply);
    CRTS_dma_iget(&local_pta[0][0],      pta+n_bais+bais,    dma_len, &dma_rply);
    D_COUNT+=3;

    for(jj = 0; jj < j_len; jj++) {
      for(ji = 0; ji < fs_jpim1; ji++) {
        zabe1 = local_pahu[jj][ji] * local_e2_e1u[jj][ji] * local_e3u_n[jj][ji];
        zabe2 = local_pahv[jj][ji] * local_e1_e2v[jj][ji] * local_e3v_n[jj][ji];
        zmsku = 1.0 / dmax(local_wmask[off1_l][jj][ji+1] + local_wmask[off1_cur][jj][ji] \
                         + local_wmask[off1_cur][jj][ji+1] + local_wmask[off1_l][jj][ji], 1.0);
        zmskv = 1.0 / dmax(local_wmask[off1_l][jj+1][ji] + local_wmask[off1_cur][jj][ji] \
                         + local_wmask[off1_cur][jj+1][ji] + local_wmask[off1_l][jj][ji], 1.0);
        zcof1 = - local_pahu[jj][ji] * local_e2u[jj][ji] * local_uslp[jj][ji] * zmsku;
        zcof2 = - local_pahv[jj][ji] * local_e1v[jj][ji] * local_vslp[jj][ji] * zmskv;
        local_zftu[jj][ji] = (zabe1 * local_zdit[jj][ji] \
                            + zcof1 * (local_zdkt[jj][ji+1] + local_zdk1t[jj][ji] \
                                     + local_zdk1t[jj][ji+1] + local_zdkt[jj][ji])) * local_umask[jj][ji];
        local_zftv[jj][ji] = (zabe2 * local_zdjt[jj][ji] \
                            + zcof2 * (local_zdkt[jj+1][ji] + local_zdk1t[jj][ji] \
                                     + local_zdk1t[jj+1][ji] + local_zdkt[jj][ji])) * local_vmask[jj][ji];
      }
    }
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    CRTS_dma_iput(zdit+bais,       &local_zdit[0][0], dma_len, &dma_rply);
    CRTS_dma_iput(zdjt+bais,       &local_zdjt[0][0], dma_len, &dma_rply);
   	D_COUNT += 2;

	  if(j_id != 3){	
      p = remote_ldm_addr(_PEN+1, rma_signal);
	    while(*p != 0);
      CRTS_rma_put(&local_zftv[j_len-1][0], jpi*8, _PEN+1, &stbv[0], &rma_rply_r);
	    *p = 1;
	  }

	  if(j_id != 0){
 //         jj =0;          
          while(rma_signal != 1);
		      for(ji = 0; ji < fs_jpim1; ji++){
             local_pta[0][ji] += zsign * (local_zftu[0][ji] - local_zftu[0][ji-1] \
                                    + local_zftv[0][ji] - stbv[ji]) \
                                    * local_r1_e1e2t[0][ji] / local_e3t_n[0][ji];
          }
		      rma_signal = 0;
	   }

	   for(jj = 1; jj < j_len; jj++){
		     for(ji = fs_2-1 ; ji < fs_jpim1; ji++){
             local_pta[jj][ji] += zsign * (local_zftu[jj][ji] - local_zftu[jj][ji-1] \
                                    + local_zftv[jj][ji] - local_zftv[jj-1][ji]) \
                                    * local_r1_e1e2t[jj][ji] / local_e3t_n[jj][ji];
          }
     }		  

 
    if(j_st == 0)
       CRTS_dma_iput(pta+n_bais+ i*jpi*jpj+(j_idx+1)*jpi, &local_pta[1][0], jpi*(j_len-1)*8, &dma_rply);
	  else
       CRTS_dma_iput(pta+n_bais+ i*jpi*jpj+j_idx*jpi, &local_pta[0][0], dma_len, &dma_rply);

	  D_COUNT += 1;

    CRTS_dma_wait_value(&dma_rply, D_COUNT);

  }
   
}

void slave_iso3_(var3 *v3) {
  var3 v;
  CRTS_dma_iget(&v, v3, sizeof(var3), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  double *wmask=v.loc_wmask, *umask=v.loc_umask, *vmask=v.loc_vmask, *pahu=v.loc_pahu, *pahv=v.loc_pahv, *e2t=v.loc_e2t, *e1t=v.loc_e1t, *wslpi=v.loc_wslpi, *wslpj=v.loc_wslpj, *zdit=v.loc_zdit, *zdjt=v.loc_zdjt, *e1e2t=v.loc_e1e2t, *e3w_n=v.loc_e3w_n, *ah_wslp2=v.loc_ah_wslp2, *akz=v.loc_akz, *ptb=v.loc_ptb, *ztfw=v.loc_ztfw;
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1, jn = v.jn-1;
  // local
  int ji, jj, i;
  int k_id, j_id, k_st, j_st, k_block, j_block, k_ed;
  int last_bais, next_bais, off_l, off_n, off_cur, off1_l, off1_cur, dma_len1;
  int db1[3], db[2];
  int cnt=0;

  devide_sid(2, &k_id, &j_id);
  area(4, jpkm1, k_id, &k_st, &k_block);
  area(2, jpjm1, j_id, &j_st, &j_block);
  j_len = j_block;
  j_idx = j_st;

  if(j_st != 0){
    j_len += 1;
    j_idx -= 1;
  }

  double local_wmask[j_len][jpi], local_umask[2][j_len][jpi], local_vmask[2][j_len][jpi], local_pahu[2][j_len][jpi], \
	     local_pahv[2][j_len][jpi], local_zdit[2][j_len][jpi], local_zdjt[2][j_len][jpi];
  double local_wslpi[j_len][jpi], local_wslpj[j_len][jpi], local_e1e2t[j_len][jpi], \
	     local_e3w_n[j_len][jpi], local_ah_wslp2[j_len][jpi], local_akz[j_len][jpi], local_ptb[2][j_len][jpi], local_ztfw[j_len][jpi];
  double local_e2t[j_len][jpi], local_e1t[j_len][jpi];

  k_ed = k_st + k_block;
  dma_len = jpi*j_len*8;
//  dma_len1 = jpi*(j_len+1)*8;
  n_bais = jn*jpj*jpi*jpk;

  CRTS_dma_iget(&local_e2t[0][0],      e2t+j_idx*jpi,      dma_len, &dma_rply);
  CRTS_dma_iget(&local_e1t[0][0],      e1t+j_idx*jpi,      dma_len, &dma_rply);
  CRTS_dma_iget(&local_e1e2t[0][0],    e1e2t+j_idx*jpi,    dma_len, &dma_rply);
  D_COUNT += 3;

  for(i = k_st; i < k_ed; i++){
		if(i == 0) continue;
	  db[0] = i & 1;   db[1] = (i+1) & 1;
	  off_l = db[0]; off_cur = db[1];

    bais = i*jpi*jpj + j_idx*jpi;
    int up_bais = (i-1)*jpi*jpj + j_idx*jpi;
	  if(cnt == 0){
      CRTS_dma_iget(&local_umask[off_l][0][0], umask+up_bais,      dma_len, &dma_rply);
      CRTS_dma_iget(&local_vmask[off_l][0][0], vmask+up_bais,      dma_len, &dma_rply);
      CRTS_dma_iget(&local_pahu[off_l][0][0],  pahu+up_bais,       dma_len, &dma_rply);
      CRTS_dma_iget(&local_pahv[off_l][0][0],  pahv+up_bais,       dma_len, &dma_rply);
      CRTS_dma_iget(&local_zdit[off_l][0][0],  zdit+up_bais,       dma_len, &dma_rply);
      CRTS_dma_iget(&local_zdjt[off_l][0][0],  zdjt+up_bais,       dma_len, &dma_rply);
      CRTS_dma_iget(&local_ptb[off_l][0][0],   ptb+n_bais+up_bais, dma_len, &dma_rply);
	    D_COUNT += 7;
			cnt = 1;
	  }

    CRTS_dma_iget(&local_wmask[0][0],    wmask+bais,         dma_len, &dma_rply);
    CRTS_dma_iget(&local_umask[off_cur][0][0],    umask+bais,         dma_len, &dma_rply);
    CRTS_dma_iget(&local_vmask[off_cur][0][0],    vmask+bais,         dma_len, &dma_rply);
    CRTS_dma_iget(&local_pahu[off_cur][0][0],     pahu+bais,          dma_len, &dma_rply);
    CRTS_dma_iget(&local_pahv[off_cur][0][0],     pahv+bais,          dma_len, &dma_rply);
    CRTS_dma_iget(&local_wslpi[0][0],    wslpi+bais,         dma_len, &dma_rply);
    CRTS_dma_iget(&local_wslpj[0][0],    wslpj+bais,         dma_len, &dma_rply);
  	D_COUNT += 7;

    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    CRTS_dma_iget(&local_zdit[off_cur][0][0],     zdit+bais,          dma_len, &dma_rply);
    CRTS_dma_iget(&local_zdjt[off_cur][0][0],     zdjt+bais,          dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3w_n[0][0],    e3w_n+bais,         dma_len, &dma_rply);
    CRTS_dma_iget(&local_ah_wslp2[0][0], ah_wslp2+bais,      dma_len, &dma_rply);
    CRTS_dma_iget(&local_akz[0][0],      akz+bais,           dma_len, &dma_rply);
    CRTS_dma_iget(&local_ptb[off_cur][0][0],      ptb+n_bais+bais,    dma_len, &dma_rply);
    D_COUNT += 6;

    double zmsku, zmskv, zahu_w, zahv_w; 
	  double zcoef3[j_len][jpi], zcoef4[j_len][jpi];
    double abc;
    for(jj = 1; jj < j_len; jj++) {
      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
          zmsku = local_wmask[jj][ji] / dmax(local_umask[off_l][jj][ji] + local_umask[off_cur][jj][ji-1] \
                                           + local_umask[off_l][jj][ji-1] + local_umask[off_cur][jj][ji] , (double)1);
          zmskv = local_wmask[jj][ji] / dmax(local_vmask[off_l][jj][ji] + local_vmask[off_cur][jj-1][ji] \
                                           + local_vmask[off_l][jj-1][ji] + local_vmask[off_cur][jj][ji] , (double)1);
          zahu_w = (local_pahu[off_l][jj][ji] + local_pahu[off_cur][jj][ji-1] \
                  + local_pahu[off_l][jj][ji-1] + local_pahu[off_cur][jj][ji]) * zmsku;
          zahv_w = (local_pahv[off_l][jj][ji] + local_pahv[off_cur][jj-1][ji] \
                  + local_pahv[off_l][jj-1][ji] + local_pahv[off_cur][jj][ji]) * zmskv;
          zcoef3[jj][ji] = - zahu_w * local_e2t[jj][ji] * zmsku * local_wslpi[jj][ji];
          zcoef4[jj][ji] = - zahv_w * local_e1t[jj][ji] * zmskv * local_wslpj[jj][ji];
      }
    }

    CRTS_dma_wait_value(&dma_rply, D_COUNT);
    for(jj = 1; jj < j_len; jj++) {
      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
          local_ztfw[jj][ji] = zcoef3[jj][ji] * (local_zdit[off_l][jj][ji]   + local_zdit[off_cur][jj][ji-1] \
                                       + local_zdit[off_l][jj][ji-1] + local_zdit[off_cur][jj][ji]) \
                             + zcoef4[jj][ji] * (local_zdjt[off_l][jj][ji]   + local_zdjt[off_cur][jj-1][ji] \
                                       + local_zdjt[off_l][jj-1][ji] + local_zdjt[off_cur][jj][ji]);
          local_ztfw[jj][ji] = local_ztfw[jj][ji] + local_e1e2t[jj][ji] / local_e3w_n[jj][ji] * local_wmask[jj][ji] \
                                         * (local_ah_wslp2[jj][ji] - local_akz[jj][ji]) \
                                         * (local_ptb[off_l][jj][ji] - local_ptb[off_cur][jj][ji]);
      }
    }

    CRTS_dma_iput(ztfw+i*jpi*jpj+(j_idx+1)*jpi, &local_ztfw[1][0], jpi*(j_len-1)*8, &dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }

}

void slave_iso4_(var3 *v3) {
  var3 v;
  CRTS_dma_iget(&v, v3, sizeof(var3), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  double *pta=v.loc_pta, *r1_e1e2t=v.loc_r1_e1e2t, *e3t_n=v.loc_e3t_n, *ztfw=v.loc_ztfw;
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1, jn = v.jn-1;
  double zsign = v.zsign;
  // local
	int ji, jj, i;
  int db[2], off_0, off_1, last_bais, kcnt;
  int k_id, j_id, k_st, k_ed, j_st, j_ed, k_block, j_block;

	k_id = _PEN;
  area(6, jpkm1, k_id, &k_st, &k_block);
  j_st = 0;
  j_idx = j_st;
	j_len = jpjm1;
  dma_len = jpi*j_len*8;
  n_bais = jn*jpj*jpi*jpk;
  int nt_bais;// = (k_idx+1)*jpi*jpj + j_idx*jpi;
  
  double local_pta[j_len][jpi], local_ztfw[2][j_len][jpi], local_r1_e1e2t[j_len][jpi], local_e3t_n[j_len][jpi];

  CRTS_dma_iget(&local_r1_e1e2t[0][0], r1_e1e2t, dma_len, &dma_rply);
	D_COUNT++; 

  k_ed = k_st + k_block;
  for(i = k_st; i < k_ed; i++){
		off_0 = i & 1;
		off_1 = (i+1) & 1;
    bais = i*jpi*jpj;
		nt_bais = (i+1)*jpi*jpj;
		if(i==k_st){
      CRTS_dma_iget(&local_ztfw[off_0][0][0],     ztfw+bais,          dma_len, &dma_rply);
			D_COUNT++;
	  }

    CRTS_dma_iget(&local_ztfw[off_1][0][0],  ztfw+nt_bais,       dma_len, &dma_rply);
    CRTS_dma_iget(&local_pta[0][0],      pta+n_bais+bais,    dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3t_n[0][0],    e3t_n+bais,         dma_len, &dma_rply);
    D_COUNT+=3;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jj = 1; jj < j_len; jj++) {
      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
          local_pta[jj][ji] += zsign * (local_ztfw[off_0][jj][ji] - local_ztfw[off_1][jj][ji]) \
                                      * local_r1_e1e2t[jj][ji] / local_e3t_n[jj][ji];
      }
    }
    CRTS_dma_iput(pta+n_bais+i*jpi*jpj+jpi, &local_pta[1][0], jpi*(j_len-1)*8, &dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
	  
	}

}	

void slave_zdf1_(var4 *v4){
  unsigned long t1, t2;
	CRTS_ssync_array();
  var4 v;
  CRTS_dma_iget(&v, v4, sizeof(var4), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  double *zwi=v.loc_zwi, *zws=v.loc_zws, *zwd=v.loc_zwd, *zwt=v.loc_zwt, *e3w_n=v.loc_e3w_n, *e3t_a=v.loc_e3t_a;
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1, jn = v.jn-1;
  double p2dt = v.p2dt;
  // local
  int ji, jj, i, jk, k_st, k_ed, k_block;
	int off_0, off_1, bais_put, dma_put_len, bais_next;
  	
  area(6, jpkm1, _PEN, &k_st, &k_block);
	k_ed = k_st + k_block;
	dma_len = jpjm1 * jpi * 8;
	dma_put_len = (jpjm1 - 1) * jpi * 8;
  double local_zwi[jpjm1][jpi], local_zws[jpjm1][jpi], local_zwd[jpjm1][jpi], local_zwt[2][jpjm1][jpi], 
				 local_e3w_n[2][jpjm1][jpi], local_e3t_a[jpjm1][jpi] __attribute__ ((aligned(64)));

  jk = k_st; 
  bais = jk * jpj * jpi; bais_next = (jk+1) * jpj * jpi;
	off_0 = jk & 1; off_1 = (jk+1) & 1;
  CRTS_dma_iget(&local_zwt[off_0][0][0], zwt + bais, dma_len, &dma_rply);
  CRTS_dma_iget(&local_e3w_n[off_0][0][0], e3w_n + bais, dma_len, &dma_rply);
  D_COUNT += 2;

	for(jk = k_st; jk < k_ed; jk++){
	   off_0 = jk & 1; off_1 = (jk+1) & 1;
		 bais = jk * jpj * jpi; bais_next = (jk+1) * jpj * jpi;


	   CRTS_dma_iget(&local_zwt[off_1][0][0], zwt + bais_next, dma_len, &dma_rply);
		 CRTS_dma_iget(&local_e3w_n[off_1][0][0], e3w_n + bais_next, dma_len, &dma_rply);
		 D_COUNT += 2;
     bais_put = jk * jpj *jpi + jpi;
		 CRTS_dma_wait_value(&dma_rply, D_COUNT);

		 CRTS_dma_iget(&local_e3t_a[0][0], e3t_a + bais, dma_len, &dma_rply);
		 D_COUNT++;

#ifdef dsimd

		 doublev8 vzwt, vzwt_n, ve3w_n, ve3w_nn, vzwi, vzws;
		 doublev8 vp2dt = p2dt;
		 for(jj = 1; jj < jpjm1; jj++){
		   for(ji = fs_2 - 1; ji < fs_jpim1 - 8; ji+=8){
			   simd_loadu(vzwt, &local_zwt[off_0][jj][ji]);
			   simd_loadu(vzwt_n, &local_zwt[off_1][jj][ji]);
				 simd_loadu(ve3w_n, &local_e3w_n[off_0][jj][ji]);
				 simd_loadu(ve3w_nn, &local_e3w_n[off_1][jj][ji]);
				 vzwi = - vp2dt * vzwt / ve3w_n;
				 vzws = - vp2dt * vzwt_n / ve3w_nn;

				 simd_storeu(vzwi, &local_zwi[jj][ji]);
				 simd_storeu(vzws, &local_zws[jj][ji]);
			 }
			 for(; ji < fs_jpim1; ji++){
         local_zwi[jj][ji] = - p2dt * local_zwt[off_0][jj][ji] / local_e3w_n[off_0][jj][ji];
         local_zws[jj][ji] = - p2dt * local_zwt[off_1][jj][ji] / local_e3w_n[off_1][jj][ji];
			 }
		 }
#else		 
		 for(jj = 1; jj < jpjm1; jj++){
			 for(ji = fs_2 - 1; ji < fs_jpim1; ji++){
         local_zwi[jj][ji] = - p2dt * local_zwt[off_0][jj][ji] / local_e3w_n[off_0][jj][ji];
         local_zws[jj][ji] = - p2dt * local_zwt[off_1][jj][ji] / local_e3w_n[off_1][jj][ji];
			 }
		 }
#endif
		 CRTS_dma_wait_value(&dma_rply, D_COUNT);

     CRTS_dma_iput(zwi+bais_put, &local_zwi[1][0], dma_put_len , &dma_rply);
     CRTS_dma_iput(zws+bais_put, &local_zws[1][0], dma_put_len , &dma_rply);

		 for(jj = 1; jj < jpjm1; jj++){
		   for(ji = fs_2 - 1; ji < fs_jpim1; ji++){
         local_zwd[jj][ji] = local_e3t_a[jj][ji] - local_zwi[jj][ji] - local_zws[jj][ji];
			 }
		 }

     CRTS_dma_iput(zwd+bais_put, &local_zwd[1][0], dma_put_len , &dma_rply);
     D_COUNT+=3;
     CRTS_dma_wait_value(&dma_rply, D_COUNT);
	}

}

void slave_zdf2_(var4 *v4) {
//	CRTS_ssync_array();
  var4 v;
  CRTS_dma_iget(&v, v4, sizeof(var4), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  double *zwi=v.loc_zwi, *zwt=v.loc_zwt, *zws=v.loc_zws, *zwd=v.loc_zwd;
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1, jn = v.jn-1;
  // local
  int ji, jj, jk, off_0, off_1;
	int j_st, j_block;

  dma_len = jpi*8;
  int curr, up, bais0;

	int slave_tasks  = 13;
  int mod  = jpjm1 / slave_tasks;
  int res = jpjm1 % slave_tasks;

  if(res == 0){
	  j_block = mod;  
    j_st = _PEN * mod;
  }else {
	  if(_PEN < res){
	    j_block = mod +1;
      j_st = _PEN * j_block;
	  }else{
		  j_block = mod;  
	    j_st = _PEN * mod + res;
	  }	
  }

  int ldm_size = j_block * jpi;
  double local_zwi[ldm_size], local_zwd[ldm_size], local_zws_up[ldm_size], local_zwt[2][ldm_size]  __attribute__ ((aligned(64)));

  if(_PEN < slave_tasks){	

    for(jk = 0; jk < jpkm1; jk++) {
      bais = jk*jpi*jpj + j_st*jpi;
      dma_len = j_block*jpi*8;
      off_0 = jk & 1; off_1 = (jk+1) & 1;

      CRTS_dma_iget(&local_zwd[0], zwd+bais, dma_len, &dma_rply);
      D_COUNT++;

      if(jk == 0) {
        CRTS_dma_wait_value(&dma_rply, D_COUNT);
        for(ji = 0; ji < ldm_size; ji++){
          local_zwt[off_0][ji] = local_zwd[ji];
				}

      } else {
        CRTS_dma_iget(&local_zwi[0], zwi+bais, dma_len, &dma_rply);
        CRTS_dma_iget(&local_zws_up[0], zws+(jk-1)*jpi*jpj + j_st*jpi, dma_len, &dma_rply);
        D_COUNT+=2;
        CRTS_dma_wait_value(&dma_rply, D_COUNT);
        
#ifdef dsimd
				doublev8 vzwi, vzws, vzwt, vzwt_last, vzwd;
        for(ji = 0; ji < ldm_size-8; ji+=8) {

				  simd_loadu(vzwt, &local_zwt[off_0][ji]);
				  simd_loadu(vzwt_last, &local_zwt[off_1][ji]);
					simd_load(vzwi, &local_zwi[ji]);
					simd_load(vzwd, &local_zwd[ji]);
					simd_load(vzws, &local_zws_up[ji]);

				  vzwt = vzwd	- vzwi * vzws / vzwt_last;

          simd_storeu(vzwt, &local_zwt[off_0][ji]);     
        }
        for(; ji < ldm_size; ji++){
          local_zwt[off_0][ji] = local_zwd[ji] - local_zwi[ji] * local_zws_up[ji] / local_zwt[off_1][ji];
				}	
#else
        for(ji = 0; ji < ldm_size; ji++){
          local_zwt[off_0][ji] = local_zwd[ji] - local_zwi[ji] * local_zws_up[ji] / local_zwt[off_1][ji];
				}	

#endif

      }

      CRTS_dma_iput(zwt+bais, &local_zwt[off_0][0], dma_len, &dma_rply);
      D_COUNT++;
      CRTS_dma_wait_value(&dma_rply, D_COUNT);
    }
  }

}	

void slave_zdf3_(var4 *v4) {
  var4 v;
  CRTS_dma_iget(&v, v4, sizeof(var4), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  double *pta=v.loc_pta, *e3t_b=v.loc_e3t_b, *ptb=v.loc_ptb, *e3t_n = v.loc_e3t_n, *zwi=v.loc_zwi, *zwt=v.loc_zwt;
  double p2dt = v.p2dt;
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1, jn = v.jn-1;
  // local
  int ji, jj, jk, off_0, off_1;
  double zrhs;
	int row_id, col_id, j_st, j_block;
   
	int slave_tasks  = 13;
  int mod  = jpjm1 / slave_tasks;
  int res = jpjm1 % slave_tasks;

  if(res == 0){
	  j_block = mod;  
    j_st = _PEN * mod;
  }else {
	  if(_PEN < res){
	    j_block = mod +1;
      j_st = _PEN * j_block;
	  }else{
		  j_block = mod;  
	    j_st = _PEN * mod + res;
	  }	
  }

  int ldm_size = j_block * jpi;
  double local_pta[2][ldm_size], local_e3t_b[ldm_size], local_ptb[ldm_size], \
		     local_e3t_n[ldm_size], local_zwi[ldm_size], local_zwt_up[ldm_size] __attribute__ ((aligned(64)));

  if(_PEN < slave_tasks){	

    for(jk = 0; jk < jpkm1; jk++) {
      bais = jk*jpi*jpj + j_st*jpi;
      dma_len = j_block*jpi*8;
      n_bais = jn*jpj*jpi*jpk;
      off_0 = jk & 1; off_1 = (jk+1) & 1;

      CRTS_dma_iget(&local_pta[off_0][0], pta+n_bais+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_ptb[0], ptb+n_bais+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_e3t_b[0], e3t_b+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_e3t_n[0], e3t_n+bais, dma_len, &dma_rply);
      D_COUNT+=4;
      CRTS_dma_wait_value(&dma_rply, D_COUNT);

      if(jk == 0) {
#ifdef dsimd				
				doublev8 ve3t_b, vptb, ve3t_n, vpta;
				doublev8 vp2dt = p2dt;
        for(ji = 0; ji < ldm_size-8; ji+=8){
				  simd_loadu(vpta, &local_pta[off_0][ji]);
					simd_load(vptb, &local_ptb[ji]);
					simd_load(ve3t_b, &local_e3t_b[ji]);
					simd_load(ve3t_n, &local_e3t_n[ji]);
          vpta = ve3t_b * vptb + vp2dt * ve3t_n * vpta;
          simd_storeu(vpta, &local_pta[off_0][ji]);     
				}
        for(; ji < ldm_size; ji++){
					local_pta[off_0][ji] = local_e3t_b[ji] * local_ptb[ji] + p2dt * local_e3t_n[ji] * local_pta[off_0][ji];
				}	
#else

        for(ji = 0; ji < ldm_size; ji++){
					local_pta[off_0][ji] = local_e3t_b[ji] * local_ptb[ji] + p2dt * local_e3t_n[ji] * local_pta[off_0][ji];
				}	
#endif				
      } else {
        CRTS_dma_iget(&local_zwi[0], zwi+bais, dma_len, &dma_rply);
        CRTS_dma_iget(&local_zwt_up[0], zwt+(jk-1)*jpi*jpj + j_st*jpi, dma_len, &dma_rply);
        D_COUNT+=2;
        CRTS_dma_wait_value(&dma_rply, D_COUNT);
        
				doublev8 ve3t_b, vptb, ve3t_n, vpta, vzwi, vzwt, vpta_last;
				doublev8 vp2dt = p2dt;
#ifdef dsimd				
        for(ji = 0; ji < ldm_size-8; ji+=8) {

				  simd_loadu(vpta, &local_pta[off_0][ji]);
				  simd_loadu(vpta_last, &local_pta[off_1][ji]);
					simd_load(ve3t_b, &local_e3t_b[ji]);
					simd_load(ve3t_n, &local_e3t_n[ji]);
					simd_load(vptb, &local_ptb[ji]);
					simd_load(vzwi, &local_zwi[ji]);
					simd_load(vzwt, &local_zwt_up[ji]);

          vpta = ve3t_b * vptb + vp2dt * ve3t_n * vpta;
					vpta = vpta - vzwi/vzwt * vpta_last; 
					
          simd_storeu(vpta, &local_pta[off_0][ji]);     
        }
				
        for(; ji < ldm_size; ji++){
          zrhs = local_e3t_b[ji] * local_ptb[ji] + p2dt * local_e3t_n[ji] * local_pta[off_0][ji];
          local_pta[off_0][ji] = zrhs - local_zwi[ji] / local_zwt_up[ji] * local_pta[off_1][ji];
				}	
#else

        for(ji = 0; ji < ldm_size; ji++){
          zrhs = local_e3t_b[ji] * local_ptb[ji] + p2dt * local_e3t_n[ji] * local_pta[off_0][ji];
          local_pta[off_0][ji] = zrhs - local_zwi[ji] / local_zwt_up[ji] * local_pta[off_1][ji];
				}	
#endif				
      }
      CRTS_dma_iput(pta+n_bais+bais, &local_pta[off_0][0], dma_len, &dma_rply);
      D_COUNT++;
      CRTS_dma_wait_value(&dma_rply, D_COUNT);
    }
  }

}

void slave_zdf4_(var4 *v4){
  var4 v;
  CRTS_dma_iget(&v, v4, sizeof(var4), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  double *pta=v.loc_pta, *zwt=v.loc_zwt, *tmask=v.loc_tmask, *zws=v.loc_zws;
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1, jn = v.jn-1;
  // local
  int ji, jj, jk, off_0, off_1;
	int row_id, col_id, j_st, j_block;
  int curr, nt, bais0;
   
	int slave_tasks  = 13;
  int mod  = jpjm1 / slave_tasks;
  int res = jpjm1 % slave_tasks;

  if(res == 0){
	  j_block = mod;  
    j_st = _PEN * mod;
  }else {
	  if(_PEN < res){
	    j_block = mod +1;
      j_st = _PEN * j_block;
	  }else{
		  j_block = mod;  
	    j_st = _PEN * mod + res;
	  }	
  }

  dma_len = jpi*8;
  n_bais = jn*jpj*jpi*jpk;

  int ldm_size = j_block * jpi;
  double local_pta[2][ldm_size], local_zwt[ldm_size], local_tmask[ldm_size], local_zws[ldm_size] __attribute__ ((aligned(64)));

	if(_PEN < slave_tasks){
    dma_len = j_block*jpi*8;
    n_bais = jn*jpj*jpi*jpk;
    for(jk = jpkm1-1; jk >=0; jk--) {
      bais = jk*jpi*jpj + j_st*jpi;
      off_0 = jk & 1; off_1 = (jk+1) & 1;
      CRTS_dma_iget(&local_pta[off_0][0], pta+n_bais+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_zwt[0], zwt+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_tmask[0], tmask+bais, dma_len, &dma_rply);
			D_COUNT += 3 ;

      if(jk == jpkm1-1){
          CRTS_dma_wait_value(&dma_rply, D_COUNT);
          for(ji = 0; ji < ldm_size; ji++){
            local_pta[off_0][ji] = local_pta[off_0][ji] / local_zwt[ji] * local_tmask[ji];
					}
			}else{
        CRTS_dma_iget(&local_zws[0], zws+bais, dma_len, &dma_rply);
				D_COUNT++;
        CRTS_dma_wait_value(&dma_rply, D_COUNT);
        for(ji = 0; ji < ldm_size; ji++) {
          local_pta[off_0][ji] = (local_pta[off_0][ji] - local_zws[ji] * local_pta[off_1][ji]) / local_zwt[ji] * local_tmask[ji];
        }
			}

      CRTS_dma_iput(pta+n_bais+bais, &local_pta[off_0][0], dma_len, &dma_rply);
      D_COUNT++;
      CRTS_dma_wait_value(&dma_rply, D_COUNT);
		  
	  }
	}

}

void slave_slp1_(var5 *v5) {
  // get para
  var5 v;
  CRTS_dma_iget(&v, v5, sizeof(var5), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");
  double *zgru = v.loc_zgru, *zgrv = v.loc_zgrv, *vmask = v.loc_vmask, *umask = v.loc_umask, *prd = v.loc_prd;
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  
  // local var
  int ji, jj, k;
  int k_id, j_id, k_st, j_st, k_block, j_block, k_ed, kcnt;

  // devide
  devide_sid(0, &k_id, &j_id);
  area(6, jpk, k_id, &k_st, &k_block);
  //area(0, jpjm1, j_id, &j_st, &j_block);
  j_id = 0; j_st = 0; j_block = jpjm1;

  k_ed = k_st + k_block;
  dma_len = jpi * j_block * 8;
  
  double local_zgru[j_block][jpi], local_zgrv[j_block][jpi], local_umask[j_block][jpi], local_vmask[j_block][jpi], local_prd[j_block+1][jpi];

  for(k = k_st; k < k_ed; k++) {
    bias_c = k*jpi*jpj + j_st*jpi;

    CRTS_dma_iget(&local_umask[0][0], umask+bias_c, dma_len,                 &dma_rply);
    CRTS_dma_iget(&local_vmask[0][0], vmask+bias_c, dma_len,                 &dma_rply);
    CRTS_dma_iget(&local_prd[0][0],   prd+bias_c,   (j_block + 1) * jpi * 8, &dma_rply);
    D_COUNT+=3;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jj = 0; jj < j_block; jj++) {
      for(ji = 0; ji < fs_jpim1; ji++) {
        local_zgru[jj][ji] = local_umask[jj][ji] * ( local_prd[jj][ji+1] - local_prd[jj][ji] );
        local_zgrv[jj][ji] = local_vmask[jj][ji] * ( local_prd[jj+1][ji] - local_prd[jj][ji] );
      }
    }
    CRTS_dma_iput(zgru+bias_c, &local_zgru[0][0], dma_len, &dma_rply);
    CRTS_dma_iput(zgrv+bias_c, &local_zgrv[0][0], dma_len, &dma_rply);
    D_COUNT+=2;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

void slave_slp2_(var5 *v5) {
  var5 v;
  CRTS_dma_iget(&v, v5, sizeof(var5), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  double *zdzr=v.loc_zdzr, *pn2=v.loc_pn2, *tmask=v.loc_tmask, *prd=v.loc_prd;
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  double zm1_g=v.zm1_g;
  // local
  int ji, jj, i, jk, k_id, k_ed, k_block;
  int nt_bais, off_0, off_1, cnt=0;
	k_id = _PEN; j_len = jpj;
	area(6, jpkm1, k_id, &k_st, &k_block);
	k_ed = k_st + k_block;
  dma_len = jpi*j_len*8;
	int len = dma_len >> 3;

  double local_zdzr[len], local_pn2[2][len], local_tmask[len], local_prd[len] __attribute__ ((aligned(64)));
	memset(&local_zdzr[0], 0, dma_len);
   
  for(jk = k_st; jk < k_ed; jk++) {
    if(jk == 0) {
			CRTS_dma_put(zdzr, &local_zdzr[0], dma_len);
			continue;
		}	

		bais = jk * jpi * jpj ;
		nt_bais = (jk+1) * jpi * jpj;
    off_0 = jk & 1; off_1  = (jk+1) & 1;

    if(cnt == 0){
		  CRTS_dma_iget(&local_pn2[off_0][0], pn2+bais, dma_len, &dma_rply);
			D_COUNT++;
			cnt = 1;
		}

	  CRTS_dma_iget(&local_pn2[off_1][0], pn2+nt_bais, dma_len, &dma_rply);
    CRTS_dma_iget(&local_prd[0], prd+bais, dma_len, &dma_rply);
    CRTS_dma_iget(&local_tmask[0], tmask+nt_bais, dma_len, &dma_rply);
		D_COUNT += 3;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  
		doublev8 vzdzr, vprd, vpn2, vpn2_n, vtmask;
		doublev8 vzm1_g = zm1_g, vone  = 1.0, vhalf = 0.5;
#ifdef dsimd
		for(ji = 0; ji < len-8; ji+=8){
			simd_load(vprd, &local_prd[ji]);
			simd_load(vtmask, &local_tmask[ji]);
			simd_loadu(vpn2, &local_pn2[off_0][ji]);
			simd_loadu(vpn2_n, &local_pn2[off_1][ji]);
		  vzdzr = vzm1_g * (vprd + vone) * \
											 (vpn2 + vpn2_n) * (vone - vhalf * vtmask);
			simd_store(vzdzr, &local_zdzr[ji]);
		}
		for(; ji < len; ji++){
		  local_zdzr[ji] = zm1_g * (local_prd[ji] + 1.) * \
											 (local_pn2[off_0][ji] + local_pn2[off_1][ji]) * \
											 (1. - 0.5 * local_tmask[ji]);
		}
#else
		for(ji = 0; ji < len; ji++){
		  local_zdzr[ji] = zm1_g * (local_prd[ji] + 1.) * \
											 (local_pn2[off_0][ji] + local_pn2[off_1][ji]) * \
											 (1. - 0.5 * local_tmask[ji]);
		}
#endif    

    CRTS_dma_iput(zdzr+bais, &local_zdzr[0], dma_len, &dma_rply);
    D_COUNT+=1;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

void slave_slp3_(var5 *v5) {
  var5 v;
  CRTS_dma_iget(&v, v5, sizeof(var5), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  double *zgru=v.loc_zgru, *zgrv=v.loc_zgrv, *r1_e1u=v.loc_r1_e1u, *r1_e2v=v.loc_r1_e2v, *zdzr=v.loc_zdzr, *e3u_n=v.loc_e3u_n, *e3v_n=v.loc_e3v_n, *omlmask=v.loc_omlmask, *gdept_n=v.loc_gdept_n, *risfdep=v.loc_risfdep, *zwz=v.loc_zwz, *zslpml_hmlpu=v.loc_zslpml_hmlpu, *umask=v.loc_umask, *vmask=v.loc_vmask, *zslpml_hmlpv=v.loc_zslpml_hmlpv, *zww=v.loc_zww, *e3un=v.loc_e3un, *e3vn=v.loc_e3vn;
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  double z1_slpmax = v.z1_slpmax, zeps = v.zeps;
  // local
  int ji, jj, i;
  double zau, zav, zbu, zbv, zfi, zfj, zdepu, zdepv;
  for(i = tid; i < 3*jpkm1; i+=64) {
    j_od = i % 3;
    k_idx = i / 3; // get时k维度索引
    j_len = (j_od < jpjm1%3) ? jpjm1/3+1 : jpjm1/3; // 每一块j维度的大小
    if(j_od < jpjm1%3) j_idx = j_od * j_len;
    else j_idx = j_od * j_len + jpjm1%3;
    if(j_od!=0) {
      j_len += 1;
      j_idx -= 1;
    }
    bais = k_idx*jpi*jpj + j_idx*jpi;
    dma_len = jpi*j_len*8;

    if(k_idx==0) continue;
  
    double local_zwz[j_len][jpi], local_zww[j_len][jpi], local_zgru[j_len][jpi], local_r1_e1u[j_len][jpi], local_zgrv[j_len][jpi], local_r1_e2v[j_len][jpi], local_zdzr[j_len+1][jpi], local_e3u_n[j_len][jpi], local_e3v_n[j_len][jpi], local_omlmask[j_len+1][jpi], local_gdept_n[j_len+1][jpi], local_risfdep[j_len+1][jpi], local_zslpml_hmlpu[j_len][jpi], local_umask[j_len][jpi], local_zslpml_hmlpv[j_len][jpi], local_vmask[j_len][jpi], local_e3un[j_len][jpi], local_e3vn[j_len][jpi];

    CRTS_dma_iget(&local_zgru[0][0], zgru+bais, dma_len, &dma_rply);
    CRTS_dma_iget(&local_zgrv[0][0], zgrv+bais, dma_len, &dma_rply);
    CRTS_dma_iget(&local_r1_e1u[0][0], r1_e1u+j_idx*jpi, dma_len, &dma_rply);
    CRTS_dma_iget(&local_r1_e2v[0][0], r1_e2v+j_idx*jpi, dma_len, &dma_rply);
    CRTS_dma_iget(&local_zdzr[0][0], zdzr+bais, (j_len+1)*jpi*8, &dma_rply);
    CRTS_dma_iget(&local_e3u_n[0][0], e3u_n+bais, dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3v_n[0][0], e3v_n+bais, dma_len, &dma_rply);
    CRTS_dma_iget(&local_omlmask[0][0], omlmask+bais, (j_len+1)*jpi*8, &dma_rply);
    CRTS_dma_iget(&local_gdept_n[0][0], gdept_n+bais, (j_len+1)*jpi*8, &dma_rply);
    CRTS_dma_iget(&local_risfdep[0][0], risfdep+j_idx*jpi, (j_len+1)*jpi*8, &dma_rply);
    CRTS_dma_iget(&local_zslpml_hmlpu[0][0], zslpml_hmlpu+j_idx*jpi, dma_len, &dma_rply);
    CRTS_dma_iget(&local_zslpml_hmlpv[0][0], zslpml_hmlpv+j_idx*jpi, dma_len, &dma_rply);
    CRTS_dma_iget(&local_umask[0][0], umask+bais, dma_len, &dma_rply);
    CRTS_dma_iget(&local_vmask[0][0], vmask+bais, dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3un[0][0], e3un+j_idx*jpi, dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3vn[0][0], e3vn+j_idx*jpi, dma_len, &dma_rply);
    D_COUNT+=16;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
    for(jj = 1; jj < j_len; jj++) {
      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
        zau = local_zgru[jj][ji] * local_r1_e1u[jj][ji];
        zav = local_zgrv[jj][ji] * local_r1_e2v[jj][ji];
        zbu = (double)0.5 * (local_zdzr[jj][ji] + local_zdzr[jj][ji+1]);
        zbv = (double)0.5 * (local_zdzr[jj][ji] + local_zdzr[jj+1][ji]);
        zbu = min3(  zbu, - z1_slpmax * fabs( zau ) , (double)-7.e+3/local_e3u_n[jj][ji]* fabs( zau )  );
        zbv = min3(  zbv, - z1_slpmax * fabs( zav ) , (double)-7.e+3/local_e3v_n[jj][ji]* fabs( zav )  );
        zfi = dmax( local_omlmask[jj][ji], local_omlmask[jj][ji+1]);
        zfj = dmax( local_omlmask[jj][ji], local_omlmask[jj+1][ji]);
        zdepu = (double)0.5 * ( ( local_gdept_n[jj][ji] + local_gdept_n[jj][ji+1]) \
                        - 2 * dmax( local_risfdep[jj][ji], local_risfdep[jj][ji+1]) - local_e3un[jj][ji]);
        zdepv = (double)0.5 * ( ( local_gdept_n[jj][ji] + local_gdept_n[jj+1][ji]) \
                        - 2 * dmax( local_risfdep[jj][ji], local_risfdep[jj+1][ji]) - local_e3vn[jj][ji]);
        local_zwz[jj][ji] = ( ( (double)1.0 - zfi) * zau / ( zbu - zeps ) \
                                  + zfi  * zdepu * local_zslpml_hmlpu[jj][ji]) * local_umask[jj][ji];
        local_zww[jj][ji] = ( ( (double)1.0 - zfj) * zav / ( zbv - zeps ) \
                                  + zfj  * zdepv * local_zslpml_hmlpv[jj][ji]) * local_vmask[jj][ji];
      }
    }
    CRTS_dma_iput(zwz+k_idx*jpi*jpj+(j_idx+1)*jpi, &local_zwz[1][0], jpi*(j_len-1)*8, &dma_rply);
    CRTS_dma_iput(zww+k_idx*jpi*jpj+(j_idx+1)*jpi, &local_zww[1][0], jpi*(j_len-1)*8, &dma_rply);
    D_COUNT+=2;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

void slave_slp4_(var5 *v5) {
  var5 v;
  CRTS_dma_iget(&v, v5, sizeof(var5), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  double *zgru=v.loc_zgru, *zgrv=v.loc_zgrv, *omlmask=v.loc_omlmask, *zwz=v.loc_zwz, *umask=v.loc_umask, *vmask=v.loc_vmask, *zww=v.loc_zww, *pn2=v.loc_pn2, *prd=v.loc_prd, *e1t=v.loc_e1t, *e2t=v.loc_e2t, *wmask=v.loc_wmask, *e3w_n=v.loc_e3w_n, *gdepw_n=v.loc_gdepw_n, *gdepw_n_tmp=v.loc_gdepw_n_tmp, *hmlp=v.loc_hmlp, *wslpiml=v.loc_wslpiml, *wslpjml=v.loc_wslpjml, *uslp=v.loc_uslp, *vslp=v.loc_vslp;
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  double zm1_2g = v.zm1_2g, zeps = v.zeps, z1_16=v.z1_16;
  // local
  int ji, jj, i;
  double zbw, zci, zcj, zai, zaj, zbi, zbj, zfk, zck, uslp_tmp, vslp_tmp;
  int up_bais, nt_bais;

  for(i = tid; i < 3*jpkm1; i+=64) {
    j_od = i % 3;
    k_idx = i / 3; // get时k维度索引
    j_len = (j_od < jpjm1%3) ? jpjm1/3+1 : jpjm1/3; // 每一块j维度的大小
    if(j_od < jpjm1%3) j_idx = j_od * j_len;
    else j_idx = j_od * j_len + jpjm1%3;
    if(j_od==1 || j_od==2) {
      j_len += 1;
      j_idx -= 1;
    }
    bais = k_idx*jpi*jpj + j_idx*jpi;
    nt_bais = (k_idx+1)*jpi*jpj + j_idx*jpi;
    dma_len = jpi*j_len*8;

    if(k_idx==0) continue;
  
    double local_zwz[j_len+1][jpi], local_zww[j_len+1][jpi], local_umask[j_len+1][jpi], local_umask_nt[j_len][jpi], local_vmask[j_len][jpi], local_vmask_nt[j_len][jpi], local_uslp[j_len][jpi], local_vslp[j_len][jpi];

    CRTS_dma_iget(&local_umask[0][0],       umask+bais,            (j_len+1)*jpi*8, &dma_rply);
    CRTS_dma_iget(&local_umask_nt[0][0],    umask+nt_bais,         dma_len, &dma_rply);
    CRTS_dma_iget(&local_vmask[0][0],       vmask+bais,            dma_len, &dma_rply);
    CRTS_dma_iget(&local_vmask_nt[0][0],    vmask+nt_bais,         dma_len, &dma_rply);
    CRTS_dma_iget(&local_zwz[0][0],         zwz+bais,              (j_len+1)*jpi*8, &dma_rply);
    CRTS_dma_iget(&local_zww[0][0],         zww+bais,              (j_len+1)*jpi*8, &dma_rply);
    D_COUNT+=6;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jj = 1; jj < j_len; jj++) {
      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
        uslp_tmp = z1_16 * (        local_zwz[jj-1][ji-1] + local_zwz[jj-1][ji+1] \
                             +      local_zwz[jj+1][ji-1] + local_zwz[jj+1][ji+1] \
                             + 2.*( local_zwz[jj-1][ji]   + local_zwz[jj][ji-1]   \
                             +      local_zwz[jj][ji+1]   + local_zwz[jj+1][ji] ) \
                             + 4.*  local_zwz[jj][ji]                       );
        vslp_tmp = z1_16 * (        local_zww[jj-1][ji-1] + local_zww[jj-1][ji+1] \
                             +      local_zww[jj+1][ji-1] + local_zww[jj+1][ji+1] \
                             + 2.*( local_zww[jj-1][ji]   + local_zww[jj][ji-1]   \
                             +      local_zww[jj][ji+1]   + local_zww[jj+1][ji] ) \
                             + 4.*  local_zww[jj][ji]                       );

        local_uslp[jj][ji] = uslp_tmp * ( local_umask[jj+1][ji] + local_umask[jj-1][ji]  ) * (double)0.5 \
                                      * ( local_umask[jj][ji]   + local_umask_nt[jj][ji] ) * (double)0.5;
        local_vslp[jj][ji] = vslp_tmp * ( local_vmask[jj][ji+1] + local_vmask[jj][ji-1]  ) * (double)0.5 \
                                      * ( local_vmask[jj][ji]   + local_vmask_nt[jj][ji] ) * (double)0.5;
      }
    }
    CRTS_dma_iput(uslp+k_idx*jpi*jpj+(j_idx+1)*jpi, &local_uslp[1][0], jpi*(j_len-1)*8, &dma_rply);
    CRTS_dma_iput(vslp+k_idx*jpi*jpj+(j_idx+1)*jpi, &local_vslp[1][0], jpi*(j_len-1)*8, &dma_rply);
    D_COUNT+=2;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
  ALLSYN;
  for(i = tid; i < 3*jpkm1; i+=64) {
    j_od = i % 3;
    k_idx = i / 3; // get时k维度索引
    j_len = (j_od < jpjm1%3) ? jpjm1/3+1 : jpjm1/3; // 每一块j维度的大小
    if(j_od < jpjm1%3) j_idx = j_od * j_len;
    else j_idx = j_od * j_len + jpjm1%3;
    if(j_od==1 || j_od==2) {
      j_len += 1;
      j_idx -= 1;
    }
    bais = k_idx*jpi*jpj + j_idx*jpi;
    up_bais = (k_idx-1)*jpi*jpj + j_idx*jpi;
    nt_bais = (k_idx+1)*jpi*jpj + j_idx*jpi;
    dma_len = jpi*j_len*8;

    if(k_idx==0) continue;
  
    double local_zwz[j_len][jpi], local_zww[j_len][jpi], local_zgru[j_len][jpi], local_zgru_up[j_len][jpi], local_zgrv[j_len][jpi], local_zgrv_up[j_len][jpi], local_pn2[j_len][jpi], local_prd[j_len][jpi], local_prd_up[j_len][jpi], local_umask[j_len][jpi], local_umask_up[j_len][jpi], local_vmask[j_len][jpi], local_vmask_up[j_len][jpi], local_e1t[j_len][jpi], local_e2t[j_len][jpi], local_wmask[j_len][jpi], local_e3w_n[j_len][jpi], local_omlmask[j_len][jpi], local_omlmask_up[j_len][jpi], local_gdepw_n[j_len][jpi], local_gdepw_n_tmp[j_len][jpi], local_hmlp[j_len][jpi], local_wslpiml[j_len][jpi], local_wslpjml[j_len][jpi];

    CRTS_dma_iget(&local_zgru[0][0],        zgru+bais,             dma_len, &dma_rply);
    CRTS_dma_iget(&local_zgru_up[0][0],     zgru+up_bais,          dma_len, &dma_rply);
    CRTS_dma_iget(&local_zgrv[0][0],        zgrv+bais,             dma_len, &dma_rply);
    CRTS_dma_iget(&local_zgrv_up[0][0],     zgrv+up_bais,          dma_len, &dma_rply);
    CRTS_dma_iget(&local_pn2[0][0],         pn2+bais,              dma_len, &dma_rply);
    CRTS_dma_iget(&local_prd[0][0],         prd+bais,              dma_len, &dma_rply);
    CRTS_dma_iget(&local_prd_up[0][0],      prd+up_bais,           dma_len, &dma_rply);
    CRTS_dma_iget(&local_umask[0][0],       umask+bais,            dma_len, &dma_rply);
    CRTS_dma_iget(&local_umask_up[0][0],    umask+up_bais,         dma_len, &dma_rply);
    CRTS_dma_iget(&local_vmask[0][0],       vmask+bais,            dma_len, &dma_rply);
    CRTS_dma_iget(&local_vmask_up[0][0],    vmask+up_bais,         dma_len, &dma_rply);
    CRTS_dma_iget(&local_e1t[0][0],         e1t+j_idx*jpi,         dma_len, &dma_rply);
    CRTS_dma_iget(&local_e2t[0][0],         e2t+j_idx*jpi,         dma_len, &dma_rply);
    CRTS_dma_iget(&local_wmask[0][0],       wmask+bais,            dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3w_n[0][0],       e3w_n+bais,            dma_len, &dma_rply);
    CRTS_dma_iget(&local_omlmask[0][0],     omlmask+bais,          dma_len, &dma_rply);
    CRTS_dma_iget(&local_omlmask_up[0][0],  omlmask+up_bais,       dma_len, &dma_rply);
    CRTS_dma_iget(&local_gdepw_n[0][0],     gdepw_n+bais,          dma_len, &dma_rply);
    CRTS_dma_iget(&local_gdepw_n_tmp[0][0], gdepw_n_tmp+j_idx*jpi, dma_len, &dma_rply);
    CRTS_dma_iget(&local_hmlp[0][0],        hmlp+j_idx*jpi,        dma_len, &dma_rply);
    CRTS_dma_iget(&local_wslpiml[0][0],     wslpiml+j_idx*jpi,     dma_len, &dma_rply);
    CRTS_dma_iget(&local_wslpjml[0][0],     wslpjml+j_idx*jpi,     dma_len, &dma_rply);
    D_COUNT+=22;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jj = 1; jj < j_len; jj++) {
      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
        zbw = zm1_2g * local_pn2[jj][ji] * (local_prd[jj][ji] + local_prd_up[jj][ji] + 2.);
        zci = dmax(local_umask[jj][ji-1] + local_umask[jj][ji] + local_umask_up[jj][ji-1] + local_umask_up[jj][ji], zeps) * local_e1t[jj][ji];
        zcj = dmax(local_vmask[jj-1][ji] + local_vmask_up[jj][ji] + local_vmask_up[jj-1][ji] + local_vmask[jj][ji], zeps) * local_e2t[jj][ji];
        zai = (local_zgru[jj][ji-1] + local_zgru_up[jj][ji] + local_zgru_up[jj][ji-1] + local_zgru[jj][ji]) / zci * local_wmask[jj][ji];
        zaj = (local_zgrv[jj-1][ji] + local_zgrv_up[jj][ji] + local_zgrv_up[jj-1][ji] + local_zgrv[jj][ji]) / zcj * local_wmask[jj][ji];
        zbi = min3(zbw, (double)-100 * fabs(zai) , (double)-7.e+3 / local_e3w_n[jj][ji] * fabs(zai));
        zbj = min3(zbw, (double)-100 * fabs(zaj) , (double)-7.e+3 / local_e3w_n[jj][ji] * fabs(zaj));
        zfk = dmax(local_omlmask[jj][ji], local_omlmask_up[jj][ji]);
        zck = (local_gdepw_n[jj][ji] - local_gdepw_n_tmp[jj][ji]) / dmax(local_hmlp[jj][ji] - local_gdepw_n_tmp[jj][ji], (double)10);
        local_zwz[jj][ji] = (zai / (zbi - zeps) * ((double)1 - zfk) + zck * local_wslpiml[jj][ji] * zfk) * local_wmask[jj][ji];
        local_zww[jj][ji] = (zaj / (zbj - zeps) * ((double)1 - zfk) + zck * local_wslpjml[jj][ji] * zfk) * local_wmask[jj][ji];
      }
    }
    CRTS_dma_iput(zwz+k_idx*jpi*jpj+(j_idx+1)*jpi, &local_zwz[1][0], jpi*(j_len-1)*8, &dma_rply);
    CRTS_dma_iput(zww+k_idx*jpi*jpj+(j_idx+1)*jpi, &local_zww[1][0], jpi*(j_len-1)*8, &dma_rply);
    D_COUNT+=2;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

void slave_slp5_(var5 *v5) {
  var5 v;
  CRTS_dma_iget(&v, v5, sizeof(var5), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  double *zwz=v.loc_zwz, *umask=v.loc_umask, *vmask=v.loc_vmask, *zww=v.loc_zww, *wslpi=v.loc_wslpi, *wslpj=v.loc_wslpj, *wmask=v.loc_wmask;
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  double z1_16=v.z1_16;
  // local
  int ji, jj, i;
  double wslpi_tmp, wslpj_tmp, zck;
  int nt_bais;

  for(i = tid; i < 2*jpkm1; i+=64) {
    j_od = i % 2;
    k_idx = i / 2; // get时k维度索引
    j_len = (j_od < jpjm1%2) ? jpjm1/2+1 : jpjm1/2; // 每一块j维度的大小
    if(j_od < jpjm1%2) j_idx = j_od * j_len;
    else j_idx = j_od * j_len + jpjm1%2;
    if(j_od==1) {
      j_len += 1;
      j_idx -= 1;
    }
    bais = k_idx*jpi*jpj + j_idx*jpi;
    dma_len = jpi*j_len*8;

    if(k_idx==0) continue;
  
    double local_zwz[j_len+1][jpi], local_zww[j_len+1][jpi], local_umask[j_len][jpi], local_vmask[j_len][jpi], local_wmask[j_len][jpi], local_wslpi[j_len][jpi], local_wslpj[j_len][jpi];

    CRTS_dma_iget(&local_umask[0][0],       umask+bais,            dma_len, &dma_rply);
    CRTS_dma_iget(&local_vmask[0][0],       vmask+bais,            dma_len, &dma_rply);
    CRTS_dma_iget(&local_wmask[0][0],       wmask+bais,            dma_len, &dma_rply);
    CRTS_dma_iget(&local_zwz[0][0],         zwz+bais,              (j_len+1)*jpi*8, &dma_rply);
    CRTS_dma_iget(&local_zww[0][0],         zww+bais,              (j_len+1)*jpi*8, &dma_rply);
    D_COUNT+=5;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jj = 1; jj < j_len; jj++) {
      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
        wslpi_tmp = z1_16 * (       local_zwz[jj-1][ji-1] + local_zwz[jj-1][ji+1] \
                             +      local_zwz[jj+1][ji-1] + local_zwz[jj+1][ji+1] \
                             + 2.*( local_zwz[jj-1][ji]   + local_zwz[jj][ji-1]   \
                             +      local_zwz[jj][ji+1]   + local_zwz[jj+1][ji] ) \
                             + 4.*  local_zwz[jj][ji]                       ) * local_wmask[jj][ji];
        wslpj_tmp = z1_16 * (       local_zww[jj-1][ji-1] + local_zww[jj-1][ji+1] \
                             +      local_zww[jj+1][ji-1] + local_zww[jj+1][ji+1] \
                             + 2.*( local_zww[jj-1][ji]   + local_zww[jj][ji-1]   \
                             +      local_zww[jj][ji+1]   + local_zww[jj+1][ji] ) \
                             + 4.*  local_zww[jj][ji]                       ) * local_wmask[jj][ji];

        zck =    ( local_umask[jj][ji] + local_umask[jj][ji-1] ) \
               * ( local_vmask[jj][ji] + local_vmask[jj-1][ji] ) * 0.25;
        local_wslpi[jj][ji] = wslpi_tmp * zck;
        local_wslpj[jj][ji] = wslpj_tmp * zck;
      }
    }
    CRTS_dma_iput(wslpi+k_idx*jpi*jpj+(j_idx+1)*jpi, &local_wslpi[1][0], jpi*(j_len-1)*8, &dma_rply);
    CRTS_dma_iput(wslpj+k_idx*jpi*jpj+(j_idx+1)*jpi, &local_wslpj[1][0], jpi*(j_len-1)*8, &dma_rply);
    D_COUNT+=2;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

/* double cache version  */
// TODO
void slave_p2z1_(p2z_var *p_v) {
  p2z_var v;
  CRTS_dma_iget(&v, p_v, sizeof(p2z_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, jpkm1 = v.jpkm1, jpphy=v.jpphy - 1;
  double *trn=v.loc_trn, *zparr=v.loc_zparr, *e3t_n=v.loc_e3t_n, *zparg=v.loc_zparg;
  double zcoef=v.zcoef, xkr0=v.xkr0, xkrp=v.xkrp, xlr=v.xlr, xkg0=v.xkg0, xkgp=v.xkgp, xlg=v.xlg, tiny=v.tiny0;
  // local
  int ji, jj, jk, up_bais;
  double zpig, zkr, zkg;

  /* zparr zparg soft cache */
  double local_trn[2][jpi], local_zparr[2][jpi], local_zparg[2][jpi], local_e3t_n[2][jpi];
  int last = 0, curr = 0, next = 0, up = 0;
  dma_len = jpi*8;
  n_bais = jpphy*jpj*jpi*jpk;

  for(jj = tid; jj < jpj; jj+=64) {
    up = 1; curr = 0;

    CRTS_dma_iget(&local_zparr[up][0], zparr+jj*jpi, dma_len, &dma_rply);
    CRTS_dma_iget(&local_zparg[up][0], zparg+jj*jpi, dma_len, &dma_rply);
    D_COUNT+=2;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    get_rply[curr] = 0;
    CRTS_dma_iget(&local_trn[curr][0], trn+n_bais+jj*jpi, dma_len, &get_rply[curr]);
    CRTS_dma_iget(&local_e3t_n[curr][0], e3t_n+jj*jpi, dma_len, &get_rply[curr]);

    for(jk = 1; jk < jpk; jk++) {
      bais = jk*jpi*jpj + jj*jpi;
      curr = (jk - 1) % 2; next = 1 - curr; last = next; up = 1 - curr;
      
      if(jk < jpk - 1) {
        get_rply[next] = 0;
        CRTS_dma_iget(&local_trn[next][0],   trn+n_bais+bais, dma_len, &get_rply[next]);
        CRTS_dma_iget(&local_e3t_n[next][0], e3t_n+bais,      dma_len, &get_rply[next]);
      }
      while(get_rply[curr] !=  2); // 等待本轮数据到达
      for(ji = 0; ji < jpi; ji++) {
        zpig = log(dmax( tiny, local_trn[curr][ji] ) * zcoef  );
        zkr  = xkr0 + xkrp * exp( xlr * zpig );
        zkg  = xkg0 + xkgp * exp( xlg * zpig );
        local_zparr[curr][ji] = local_zparr[up][ji] * exp( -zkr * local_e3t_n[curr][ji] );
        local_zparg[curr][ji] = local_zparg[up][ji] * exp( -zkg * local_e3t_n[curr][ji] );
      }

      put_rply[curr] = 0;
      CRTS_dma_iput(zparr+bais, &local_zparr[curr][0], dma_len, &put_rply[curr]);
      CRTS_dma_iput(zparg+bais, &local_zparg[curr][0], dma_len, &put_rply[curr]);
      while(jk != 1 && put_rply[last] != 2);
    }
    while(put_rply[curr] != 2);
  }
}

void slave_p2z2_(p2z_var *p_v) {
  // get var
  p2z_var v;
  CRTS_dma_iget(&v, p_v, sizeof(p2z_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, jpkm1 = v.jpkm1, jpphy=v.jpphy - 1;
  double *trn=v.loc_trn, *zparr=v.loc_zparr, *e3t_n=v.loc_e3t_n, *zparg=v.loc_zparg, *etot=v.loc_etot;
  double zcoef=v.zcoef, xkr0=v.xkr0, xkrp=v.xkrp, xlr=v.xlr, xkg0=v.xkg0, xkgp=v.xkgp, xlg=v.xlg, tiny=v.tiny0;

  // local
  int ji, jj, k, n_bias_c;
  double zpig, zkr, zkg;

  // devide
  devide_sid(1, &k_id, &j_id);
  area(5, jpkm1, k_id, &k_st, &k_block);
  area(1, jpj, j_id, &j_st, &j_block);
  k_ed = k_st + k_block;
  dma_len = j_block * jpi * 8;
  
  double local_trn[j_block][jpi], local_zparr[j_block][jpi], local_zparg[j_block][jpi], local_e3t_n[j_block][jpi], local_etot[j_block][jpi] __attribute__ ((aligned(64)));
  
  doublev8 vzparr, ve3t_n, vzparg, vtrn, vzpig, vtmp, vtmp2, vzkr, vzkg, vetot;
  doublev8 vtiny = tiny, vzcoef = zcoef, vxlr = xlr, vxlg = xlg, vxkr0 = xkr0, vxkrp = xkrp, vxkg0 = xkg0, vxkgp = xkgp, vdoubleone = 1.e-15, vone = 1;
  double tmp[8], tmp_zpig[8] __attribute__ ((aligned(64)));

  for(k = k_st; k < k_ed; k++) {
    bias_c = k*jpi*jpj + j_st*jpi; n_bias_c = jpphy*jpi*jpj*jpk;

    CRTS_dma_iget(&local_trn[0][0],   trn+n_bias_c+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_zparr[0][0], zparr+bias_c,        dma_len, &dma_rply);
    CRTS_dma_iget(&local_zparg[0][0], zparg+bias_c,        dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3t_n[0][0], e3t_n+bias_c,        dma_len, &dma_rply);
    D_COUNT+=4;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

# ifdef dsimd_p2z2
    for(jj = 0; jj < j_block; jj++) {
      for(ji = 0; ji < jpi-8; ji+=8) {
        simd_loadu(vzparr, &local_zparr[jj][ji]);
        simd_loadu(ve3t_n, &local_e3t_n[jj][ji]);
        simd_loadu(vzparg, &local_zparg[jj][ji]);
        simd_loadu(vtrn, &local_trn[jj][ji]);

        // zpig = log(dmax(tiny, local_trn[jj][ji]) * zcoef);
        vtmp = simd_smaxd(vtiny, vtrn);
        vtmp = vtmp * vzcoef;
        simd_storeu(vtmp, &tmp[0]);
        tmp_zpig[0] = log(tmp[0]);
        tmp_zpig[1] = log(tmp[1]);
        tmp_zpig[2] = log(tmp[2]);
        tmp_zpig[3] = log(tmp[3]);
        tmp_zpig[4] = log(tmp[4]);
        tmp_zpig[5] = log(tmp[5]);
        tmp_zpig[6] = log(tmp[6]);
        tmp_zpig[7] = log(tmp[7]);
        simd_loadu(vzpig, &tmp_zpig[0]);

        //zkr  = xkr0 + xkrp * exp( xlr * zpig );
        vtmp = vxlr * vzpig;
        simd_storeu(vtmp, &tmp[0]);
        vexp(&tmp[0], &tmp[0], 8);
        //tmp[0] = exp(tmp[0]);
        //tmp[1] = exp(tmp[1]);
        //tmp[2] = exp(tmp[2]);
        //tmp[3] = exp(tmp[3]);
        //tmp[4] = exp(tmp[4]);
        //tmp[5] = exp(tmp[5]);
        //tmp[6] = exp(tmp[6]);
        //tmp[7] = exp(tmp[7]);
        simd_loadu(vtmp, &tmp[0]);
        vzkr = vxkr0 + vxkrp * vtmp;

        zkg  = xkg0 + xkgp * exp( xlg * zpig );
        vtmp = vxlg * vzpig;
        simd_storeu(vtmp, &tmp[0]);
        vexp(&tmp[0], &tmp[0], 8);
        //tmp[0] = exp(tmp[0]);
        //tmp[1] = exp(tmp[1]);
        //tmp[2] = exp(tmp[2]);
        //tmp[3] = exp(tmp[3]);
        //tmp[4] = exp(tmp[4]);
        //tmp[5] = exp(tmp[5]);
        //tmp[6] = exp(tmp[6]);
        //tmp[7] = exp(tmp[7]);
        simd_loadu(vtmp, &tmp[0]);
        vzkg = vxkg0 + vxkgp * vtmp;

        vtmp = -vzkr * ve3t_n;
        simd_storeu(vtmp, &tmp[0]);
        vexp(&tmp[0], &tmp[0], 8);
        //tmp[0] = exp(tmp[0]);
        //tmp[1] = exp(tmp[1]);
        //tmp[2] = exp(tmp[2]);
        //tmp[3] = exp(tmp[3]);
        //tmp[4] = exp(tmp[4]);
        //tmp[5] = exp(tmp[5]);
        //tmp[6] = exp(tmp[6]);
        //tmp[7] = exp(tmp[7]);
        simd_loadu(vtmp, &tmp[0]);

        vtmp2 = -vzkg * ve3t_n;
        simd_storeu(vtmp2, &tmp[0]);
        vexp(&tmp[0], &tmp[0], 8);
        //tmp[0] = exp(tmp[0]);
        //tmp[1] = exp(tmp[1]);
        //tmp[2] = exp(tmp[2]);
        //tmp[3] = exp(tmp[3]);
        //tmp[4] = exp(tmp[4]);
        //tmp[5] = exp(tmp[5]);
        //tmp[6] = exp(tmp[6]);
        //tmp[7] = exp(tmp[7]);
        simd_loadu(vtmp2, &tmp[0]);

        vzparr = vzparr / (vzkr * ve3t_n) * (vone - vtmp);
        vzparg = vzparg / (vzkg * ve3t_n) * (vone - vtmp2);
        vetot = simd_smaxd(vzparr + vzparg, vdoubleone);

        simd_storeu(vzparr, &local_zparr[jj][ji]);
        simd_storeu(vzparg, &local_zparg[jj][ji]);
        simd_storeu(vetot, &local_etot[jj][ji]);
      }
      for( ; ji < jpi; ji++) {
        zpig = log(dmax(tiny, local_trn[jj][ji]) * zcoef);
        zkr  = xkr0 + xkrp * exp( xlr * zpig );
        zkg  = xkg0 + xkgp * exp( xlg * zpig );
        local_zparr[jj][ji] = local_zparr[jj][ji] / ( zkr * local_e3t_n[jj][ji] ) * ( 1 - exp( -zkr * local_e3t_n[jj][ji] ));
        local_zparg[jj][ji] = local_zparg[jj][ji] / ( zkg * local_e3t_n[jj][ji] ) * ( 1 - exp( -zkg * local_e3t_n[jj][ji] ));
        local_etot[jj][ji]  = dmax( local_zparr[jj][ji] + local_zparg[jj][ji], 1.e-15 );
      }
    }
# else
    for(jj = 0; jj < j_block; jj++) {
      for(ji = 0; ji < jpi; ji++) {
        zpig = log(dmax(tiny, local_trn[jj][ji]) * zcoef);
        zkr  = xkr0 + xkrp * exp( xlr * zpig );
        zkg  = xkg0 + xkgp * exp( xlg * zpig );
        local_zparr[jj][ji] = local_zparr[jj][ji] / ( zkr * local_e3t_n[jj][ji] ) * ( 1 - exp( -zkr * local_e3t_n[jj][ji] ));
        local_zparg[jj][ji] = local_zparg[jj][ji] / ( zkg * local_e3t_n[jj][ji] ) * ( 1 - exp( -zkg * local_e3t_n[jj][ji] ));
        local_etot[jj][ji]  = dmax( local_zparr[jj][ji] + local_zparg[jj][ji], 1.e-15 );
      }
    }
# endif
# undef dsimd
    CRTS_dma_iput(etot+bias_c,  &local_etot[0][0],  dma_len, &dma_rply);
    CRTS_dma_iput(zparr+bias_c, &local_zparr[0][0], dma_len, &dma_rply);
    CRTS_dma_iput(zparg+bias_c, &local_zparg[0][0], dma_len, &dma_rply);
    D_COUNT+=3;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

// e
void slave_rab1_(rab_var *r_v) {

  rab_var v;
  CRTS_dma_iget(&v, r_v, sizeof(rab_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, jpkm1 = v.jpkm1, jp_tem = v.jp_tem - 1, jp_sal = v.jp_sal - 1;
  double r1_Z0 = v.r1_Z0, r1_T0 = v.r1_T0, rdeltaS = v.rdeltaS, r1_S0 = v.r1_S0, r1_rau0=v.r1_rau0;
  double *gdept_n = v.loc_gdept_n, *pts = v.loc_pts, *tmask = v.loc_tmask, *pab=v.loc_pab;

  // local
  int ji, jj, k, tem_bias, sal_bias;
  double zh, zt, zs, ztm, zn3, zn2, zn1, zn0, zn;

  // devide
  devide_sid(0, &k_id, &j_id);
  area(6, jpkm1, k_id, &k_st, &k_block);
  area(0, jpj, j_id, &j_st, &j_block);

  k_ed = k_st + k_block;
  dma_len = jpi * j_block * 8;
  tem_bias = jp_tem*jpi*jpj*jpk;
  sal_bias = jp_sal*jpi*jpj*jpk;
  
  double local_pab_sal[j_block][jpi], local_pab_tem[j_block][jpi], local_tmask[j_block][jpi], local_pts_tem[j_block][jpi], local_pts_sal[j_block][jpi], local_gdept_n[j_block][jpi];
  
  for(k = k_st; k < k_ed; k++) {
    bias_c = k*jpi*jpj + j_st*jpi;

    CRTS_dma_iget(&local_gdept_n[0][0], gdept_n+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_pts_tem[0][0], pts+tem_bias+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_pts_sal[0][0], pts+sal_bias+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_tmask[0][0], tmask+bias_c, dma_len, &dma_rply);
    D_COUNT+=4;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jj = 0; jj < j_block; jj++) {
      for(ji = 0; ji < jpi; ji++) {
         zh  = local_gdept_n[jj][ji] * r1_Z0;
         zt  = local_pts_tem[jj][ji] * r1_T0;
         zs  = sqrt( fabs( local_pts_sal[jj][ji] + rdeltaS ) * r1_S0 );
         ztm = local_tmask[jj][ji];
         zn3 = v.ALP003;
         zn2 = v.ALP012 * zt + v.ALP102 * zs + v.ALP002;
         zn1 = ((v.ALP031 * zt \
               + v.ALP121*zs+v.ALP021)*zt \
               + (v.ALP211*zs+v.ALP111)*zs+v.ALP011)*zt \
               + ((v.ALP301*zs+v.ALP201)*zs+v.ALP101)*zs+v.ALP001;
         zn0 = ((((v.ALP050*zt \
               + v.ALP140*zs+v.ALP040)*zt \
               + (v.ALP230*zs+v.ALP130)*zs+v.ALP030)*zt \
               + ((v.ALP320*zs+v.ALP220)*zs+v.ALP120)*zs+v.ALP020)*zt \
               + (((v.ALP410*zs+v.ALP310)*zs+v.ALP210)*zs+v.ALP110)*zs+v.ALP010)*zt \
               + ((((v.ALP500*zs+v.ALP400)*zs+v.ALP300)*zs+v.ALP200)*zs+v.ALP100)*zs+v.ALP000;
         zn  = ( ( zn3 * zh + zn2 ) * zh + zn1 ) * zh + zn0;
         local_pab_tem[jj][ji] = zn * r1_rau0 * ztm;
         
         zn3 = v.BET003;
         zn2 = v.BET012*zt + v.BET102*zs+v.BET002;
         zn2 = v.BET012*zt + v.BET102*zs+v.BET002;
         zn1 = ((v.BET031*zt \
               + v.BET121*zs+v.BET021)*zt \
               + (v.BET211*zs+v.BET111)*zs+v.BET011)*zt \
               + ((v.BET301*zs+v.BET201)*zs+v.BET101)*zs+v.BET001;
         zn0 = ((((v.BET050*zt \
               + v.BET140*zs+v.BET040)*zt \
               + (v.BET230*zs+v.BET130)*zs+v.BET030)*zt \
               + ((v.BET320*zs+v.BET220)*zs+v.BET120)*zs+v.BET020)*zt \
               + (((v.BET410*zs+v.BET310)*zs+v.BET210)*zs+v.BET110)*zs+v.BET010)*zt \
               + ((((v.BET500*zs+v.BET400)*zs+v.BET300)*zs+v.BET200)*zs+v.BET100)*zs+v.BET000;
         zn  = ( ( zn3 * zh + zn2 ) * zh + zn1 ) * zh + zn0;
         local_pab_sal[jj][ji] = zn / zs * r1_rau0 * ztm;
      }
    }
    CRTS_dma_iput(pab+tem_bias+bias_c, &local_pab_tem[0][0], dma_len, &dma_rply);
    CRTS_dma_iput(pab+sal_bias+bias_c, &local_pab_sal[0][0], dma_len, &dma_rply);
    D_COUNT+=2;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

void slave_pot1_(pot_var *pot_v) {
  pot_var v;
  CRTS_dma_iget(&v, pot_v, sizeof(pot_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, jpkm1 = v.jpkm1, jp_tem = v.jp_tem - 1, jp_sal = v.jp_sal - 1;
  double r1_Z0 = v.r1_Z0, r1_T0 = v.r1_T0, rdeltaS = v.rdeltaS, r1_S0 = v.r1_S0, r1_rau0=v.r1_rau0;
  double *pdep = v.loc_pdep, *pts = v.loc_pts, *tmask = v.loc_tmask, *prhop=v.loc_prhop, *prd=v.loc_prd;

  // local
  int ji, jj, k, tem_bias, sal_bias;
  double zh, zt, zs, ztm, zn3, zn2, zn1, zn0, zn;

  // devide
  devide_sid(0, &k_id, &j_id);
  area(6, jpkm1, k_id, &k_st, &k_block);
  area(0, jpj, j_id, &j_st, &j_block);

  k_ed = k_st + k_block;
  dma_len = jpi * j_block * 8;
  tem_bias = jp_tem*jpi*jpj*jpk;
  sal_bias = jp_sal*jpi*jpj*jpk;

  double local_prhop[j_block][jpi], local_prd[j_block][jpi], local_tmask[j_block][jpi], local_pts_tem[j_block][jpi], local_pts_sal[j_block][jpi], local_pdep[j_block][jpi];

  for(k = k_st; k < k_ed; k++) {
    bias_c = k*jpi*jpj + j_st*jpi;

    CRTS_dma_iget(&local_pdep[0][0], pdep+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_pts_tem[0][0], pts+tem_bias+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_pts_sal[0][0], pts+sal_bias+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_tmask[0][0], tmask+bias_c, dma_len, &dma_rply);
    D_COUNT+=4;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jj = 0; jj < j_block; jj++) {
      for(ji = 0; ji < jpi; ji++) {
        zh  = local_pdep[jj][ji] * r1_Z0;
        zt  = local_pts_tem[jj][ji] * r1_T0;
        zs  = sqrt( fabs( local_pts_sal[jj][ji] + rdeltaS ) * r1_S0 );
        ztm = local_tmask[jj][ji];
        zn3 = v.EOS013*zt + v.EOS103*zs+v.EOS003;
        zn2 = (v.EOS022*zt + v.EOS112*zs+v.EOS012)*zt + (v.EOS202*zs+v.EOS102)*zs+v.EOS002;
        zn1 = (((v.EOS041*zt \
              + v.EOS131*zs+v.EOS031)*zt \
              + (v.EOS221*zs+v.EOS121)*zs+v.EOS021)*zt \
              + ((v.EOS311*zs+v.EOS211)*zs+v.EOS111)*zs+v.EOS011)*zt \
              + (((v.EOS401*zs+v.EOS301)*zs+v.EOS201)*zs+v.EOS101)*zs+v.EOS001;
        zn0 = (((((v.EOS060*zt \
              + v.EOS150*zs+v.EOS050)*zt \
              + (v.EOS240*zs+v.EOS140)*zs+v.EOS040)*zt \
              + ((v.EOS330*zs+v.EOS230)*zs+v.EOS130)*zs+v.EOS030)*zt \
              + (((v.EOS420*zs+v.EOS320)*zs+v.EOS220)*zs+v.EOS120)*zs+v.EOS020)*zt \
              + ((((v.EOS510*zs+v.EOS410)*zs+v.EOS310)*zs+v.EOS210)*zs+v.EOS110)*zs+v.EOS010)*zt \
              + (((((v.EOS600*zs+v.EOS500)*zs+v.EOS400)*zs+v.EOS300)*zs+v.EOS200)*zs+v.EOS100)*zs+v.EOS000;
        zn  = ( ( zn3 * zh + zn2 ) * zh + zn1 ) * zh + zn0;
        local_prhop[jj][ji] = zn0 * ztm;
        local_prd[jj][ji] = (  zn * r1_rau0 - (double)1.  ) * ztm;
      }
    }
    CRTS_dma_iput(prhop+bias_c, &local_prhop[0][0], dma_len, &dma_rply);
    CRTS_dma_iput(prd+bias_c, &local_prd[0][0], dma_len, &dma_rply);
    D_COUNT+=2;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

//e (data reusing)
void slave_bn21_(bn2_var *bn2_v) {
  // dma var addr
  bn2_var v;
  CRTS_dma_iget(&v, bn2_v, sizeof(bn2_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, jpkm1 = v.jpkm1, jp_tem = v.jp_tem - 1, jp_sal = v.jp_sal - 1;
  double grav = v.grav;
  double *gdepw_n = v.loc_gdepw_n, *gdept_n = v.loc_gdept_n, *pab = v.loc_pab, *pts=v.loc_pts, *e3w_n=v.loc_e3w_n, *wmask=v.loc_wmask, *pn2=v.loc_pn2;

  // local var
  int ji, jj, k, tem_bias, sal_bias;
  int off_l, off_c; // n-1, n
  double zrw, zaw, zbw;

  // devide
  devide_sid(1, &k_id, &j_id);
  area(5, jpkm1, k_id, &k_st, &k_block);
  area(1, jpj, j_id, &j_st, &j_block);

  k_ed = k_st + k_block;
  dma_len = jpi * j_block * 8;
  tem_bias = jp_tem*jpi*jpj*jpk;
  sal_bias = jp_sal*jpi*jpj*jpk;

  double local_gdepw_n[j_block][jpi], local_gdept_n[2][j_block][jpi], local_pab_tem[2][j_block][jpi], local_pab_sal[2][j_block][jpi], local_pts_tem[2][j_block][jpi], local_pts_sal[2][j_block][jpi], local_wmask[j_block][jpi], local_e3w_n[j_block][jpi], local_pn2[j_block][jpi];

  // data reusing first get
  k = k_st == 0 ? 1: k_st;
	off_l = k & 1; off_c = (k+1) & 1;
  
  bias_c = k*jpi*jpj + j_st*jpi; bias_l = (k-1)*jpi*jpj + j_st*jpi;
  CRTS_dma_iget(&local_gdept_n[off_l][0][0], gdept_n+bias_l, dma_len, &dma_rply);
  CRTS_dma_iget(&local_pab_tem[off_l][0][0], pab+tem_bias+bias_l, dma_len, &dma_rply);
  CRTS_dma_iget(&local_pab_sal[off_l][0][0], pab+sal_bias+bias_l, dma_len, &dma_rply);
  CRTS_dma_iget(&local_pts_tem[off_l][0][0], pts+tem_bias+bias_l, dma_len, &dma_rply);
  CRTS_dma_iget(&local_pts_sal[off_l][0][0], pts+sal_bias+bias_l, dma_len, &dma_rply);
  D_COUNT+=5;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  for(k = k_st; k < k_ed; k++) {
    if(k == 0) continue;

	  off_l = k & 1; off_c = (k+1) & 1;
    bias_c = k*jpi*jpj + j_st*jpi; bias_l = (k-1)*jpi*jpj + j_st*jpi;

    CRTS_dma_iget(&local_gdepw_n[0][0], gdepw_n+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_gdept_n[off_c][0][0], gdept_n+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_pab_tem[off_c][0][0], pab+tem_bias+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_pab_sal[off_c][0][0], pab+sal_bias+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_pts_tem[off_c][0][0], pts+tem_bias+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_pts_sal[off_c][0][0], pts+sal_bias+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_wmask[0][0], wmask+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3w_n[0][0], e3w_n+bias_c, dma_len, &dma_rply);
    D_COUNT+=8;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jj = 0; jj < j_block; jj++) {
      for(ji = 0; ji < jpi; ji++) {
        zrw =   ( local_gdepw_n[jj][ji] - local_gdept_n[off_c][jj][ji] ) \
             / ( local_gdept_n[off_l][jj][ji] - local_gdept_n[off_c][jj][ji] ); 
        zaw = local_pab_tem[off_c][jj][ji] * (1. - zrw) + local_pab_tem[off_l][jj][ji] * zrw;
        zbw = local_pab_sal[off_c][jj][ji] * (1. - zrw) + local_pab_sal[off_l][jj][ji] * zrw;
        local_pn2[jj][ji] = grav * (  zaw * ( local_pts_tem[off_l][jj][ji] - local_pts_tem[off_c][jj][ji] ) \
                                    - zbw * ( local_pts_sal[off_l][jj][ji] - local_pts_sal[off_c][jj][ji] )  ) \
                                    / local_e3w_n[jj][ji] * local_wmask[jj][ji];
      }
    }
    CRTS_dma_iput(pn2+bias_c, &local_pn2[0][0], dma_len, &dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

void slave_dyn1_(dyn_var *d_v) {
  // get var
  dyn_var v;
  CRTS_dma_iget(&v, d_v, sizeof(dyn_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, jpkm1 = v.jpkm1;
  double r2dt = v.r2dt;
  double *ub=v.loc_ub, *ua=v.loc_ua, *umask=v.loc_umask, *vmask=v.loc_vmask, *vb=v.loc_vb, *va=v.loc_va, *ua_b=v.loc_ua_b, *va_b=v.loc_va_b;

  // local
  int ji, jj, k;

  // devide
  devide_sid(1, &k_id, &j_id);
  area(5, jpkm1, k_id, &k_st, &k_block);
  area(1, jpj, j_id, &j_st, &j_block);
  k_ed = k_st + k_block;
  dma_len = j_block * jpi * 8;

  double local_ua[j_block][jpi], local_ub[j_block][jpi], local_umask[j_block][jpi], local_va[j_block][jpi], local_vb[j_block][jpi], local_vmask[j_block][jpi], local_ua_b[j_block][jpi], local_va_b[j_block][jpi];

  CRTS_dma_iget(&local_ua_b[0][0], ua_b+j_st*jpi, dma_len, &dma_rply);
  CRTS_dma_iget(&local_va_b[0][0], va_b+j_st*jpi, dma_len, &dma_rply);
  D_COUNT+=2;

  for(k = k_st; k < k_ed; k++) {
    bias_c = k*jpi*jpj + j_st*jpi;

    CRTS_dma_iget(&local_ua[0][0], ua+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_ub[0][0], ub+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_va[0][0], va+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_vb[0][0], vb+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_umask[0][0], umask+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_vmask[0][0], vmask+bias_c, dma_len, &dma_rply);
    D_COUNT+=6;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jj = 0; jj < j_block; jj++) {
      for(ji = 0; ji < jpi; ji++) {
        local_ua[jj][ji] = ( local_ub[jj][ji] + r2dt * local_ua[jj][ji] ) * local_umask[jj][ji];
        local_va[jj][ji] = ( local_vb[jj][ji] + r2dt * local_va[jj][ji] ) * local_vmask[jj][ji];
        local_ua[jj][ji] = ( local_ua[jj][ji] - local_ua_b[jj][ji] ) * local_umask[jj][ji];
        local_va[jj][ji] = ( local_va[jj][ji] - local_va_b[jj][ji] ) * local_vmask[jj][ji];
      }
    }
    CRTS_dma_iput(ua+bias_c, &local_ua[0][0], dma_len, &dma_rply);
    CRTS_dma_iput(va+bias_c, &local_va[0][0], dma_len, &dma_rply);
    D_COUNT+=2;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

void slave_dyn2_(dyn_var *d_v) {
  // get var
  dyn_var v;
  CRTS_dma_iget(&v, d_v, sizeof(dyn_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  double zdt = v.zdt, r_vvl = v.r_vvl;
  double *e3u_n = v.loc_e3u_n, *e3u_a = v.loc_e3u_a, *avm = v.loc_avm, *e3uw_n = v.loc_e3uw_n, *wumask = v.loc_wumask, *zwi = v.loc_zwi, *zws = v.loc_zws, *zwd = v.loc_zwd;

  // local
  int ji, jj, k, off_c, off_n;
  double ze3ua, zzwi, zzws;

  // devide
  devide_sid(1, &k_id, &j_id);
  area(5, jpkm1, k_id, &k_st, &k_block);
  area(1, jpjm1, j_id, &j_st, &j_block);
  if(j_st != 0) {
    j_block += 1;
    j_st -= 1;
  }
  k_ed = k_st + k_block;
  dma_len = j_block * jpi * 8;

  double local_zwi[j_block][jpi], local_zws[j_block][jpi], local_zwd[j_block][jpi], local_e3u_n[j_block][jpi], local_e3u_a[j_block][jpi], local_avm[2][j_block][jpi], local_e3uw_n[2][j_block][jpi], local_wumask[2][j_block][jpi];
  
  // data reusing prefetch
  k = k_st;
  off_c = k & 1; off_n = (k+1) & 1;
  bias_c = k*jpi*jpj + j_st*jpi; bias_n = (k+1)*jpi*jpj + j_st*jpi;
  CRTS_dma_iget(&local_avm[off_c][0][0],       avm+bias_c,    dma_len, &dma_rply);
  CRTS_dma_iget(&local_e3uw_n[off_c][0][0],    e3uw_n+bias_c, dma_len, &dma_rply);
  CRTS_dma_iget(&local_wumask[off_c][0][0],    wumask+bias_c, dma_len, &dma_rply);
  D_COUNT+=3;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);

  for(k = k_st; k < k_ed; k++) {
    off_c = k & 1; off_n = (k+1) & 1;
    bias_c = k*jpi*jpj + j_st*jpi; bias_n = (k+1)*jpi*jpj + j_st*jpi;

    CRTS_dma_iget(&local_e3u_n[0][0],     e3u_n+bias_c,  dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3u_a[0][0],     e3u_a+bias_c,  dma_len, &dma_rply);
    CRTS_dma_iget(&local_avm[off_n][0][0],    avm+bias_n,    dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3uw_n[off_n][0][0], e3uw_n+bias_n, dma_len, &dma_rply);
    CRTS_dma_iget(&local_wumask[off_n][0][0], wumask+bias_n, dma_len, &dma_rply);
    D_COUNT+=5;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jj = 1; jj < j_block; jj++) {
      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
        ze3ua =  ((double)1. - r_vvl) * local_e3u_n[jj][ji] + r_vvl * local_e3u_a[jj][ji];
        zzwi = - zdt * (local_avm[off_c][jj][ji+1] + local_avm[off_c][jj][ji]) / (ze3ua * local_e3uw_n[off_c][jj][ji]) * local_wumask[off_c][jj][ji];
        zzws = - zdt * (local_avm[off_n][jj][ji+1] + local_avm[off_n][jj][ji]) / (ze3ua * local_e3uw_n[off_n][jj][ji]) * local_wumask[off_n][jj][ji];
        local_zwi[jj][ji] = zzwi;
        local_zws[jj][ji] = zzws;
        local_zwd[jj][ji] = (double)1. - zzwi - zzws;
      }
    }
    CRTS_dma_iput(zwi+k*jpi*jpj+(j_st+1)*jpi, &local_zwi[1][0], jpi*(j_block-1)*8, &dma_rply);
    CRTS_dma_iput(zws+k*jpi*jpj+(j_st+1)*jpi, &local_zws[1][0], jpi*(j_block-1)*8, &dma_rply);
    CRTS_dma_iput(zwd+k*jpi*jpj+(j_st+1)*jpi, &local_zwd[1][0], jpi*(j_block-1)*8, &dma_rply);
    D_COUNT+=3;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

// TODO
void slave_dyn3_(dyn_var *d_v) {
  dyn_var v;
  CRTS_dma_iget(&v, d_v, sizeof(dyn_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  double *zwd=v.loc_zwd, *zwi=v.loc_zwi, *zws=v.loc_zws;
  // local
  int ji, jj, jk;
  dma_len = jpi*8;
  int curr, up;
  double local_zwd[2][jpi], local_zwi[jpi], local_zws_up[jpi];

  for(jj = tid; jj < jpjm1; jj+=64) {
    if(jj == 0) continue;
    // 计算最顶层
    up = curr = 0;
    CRTS_dma_iget(&local_zwd[up][0], zwd+jj*jpi, dma_len, &dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
    
    for(jk = 1; jk < jpkm1; jk++) {
      bais = jk*jpi*jpj + jj*jpi;
      curr = jk % 2; up = 1 - curr;

      CRTS_dma_iget(&local_zwi[0], zwi+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_zwd[curr][0], zwd+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_zws_up[0], zws+(jk-1)*jpi*jpj+jj*jpi, dma_len, &dma_rply);
      D_COUNT+=3;
      CRTS_dma_wait_value(&dma_rply, D_COUNT);
      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
        local_zwd[curr][ji] = local_zwd[curr][ji] - local_zwi[ji] * local_zws_up[ji] / local_zwd[up][ji];
      }
      CRTS_dma_iput(zwd+bais, &local_zwd[curr][0], dma_len, &dma_rply);
      D_COUNT++;
      CRTS_dma_wait_value(&dma_rply, D_COUNT);
    }
  }
}

// TODO
void slave_dyn4_(dyn_var *d_v) {
  dyn_var v;
  CRTS_dma_iget(&v, d_v, sizeof(dyn_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  double *ua=v.loc_ua, *e3u_n=v.loc_e3u_n, *e3u_a=v.loc_e3u_a, *utau_b = v.loc_utau_b, *utau=v.loc_utau, *umask=v.loc_umask, *zwi = v.loc_zwi, *zwd = v.loc_zwd;
  double r_vvl = v.r_vvl, r2dt = v.r2dt, rau0 = v.rau0;
  // local
  int ji, jj, jk;
  double ze3ua;

  dma_len = jpi*8;
  int curr, up, bais0;
  double local_ua[2][jpi], local_e3u_n[jpi], local_e3u_a[jpi], local_utau_b[jpi], local_utau[jpi], local_umask[jpi], local_zwi[jpi], local_zwd_up[jpi];

  for(jj = tid; jj < jpjm1; jj+=64) {
    if(jj == 0) continue;
    // 计算最底层
    up = curr = 0;
    bais0 = jj*jpi;
    CRTS_dma_iget(&local_ua[curr][0], ua+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_utau_b[0], utau_b+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_utau[0], utau+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3u_n[0], e3u_n+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3u_a[0], e3u_a+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_umask[0], umask+bais0, dma_len, &dma_rply);
    D_COUNT+=6;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
    for(ji = fs_2-1; ji < fs_jpim1; ji++) {
      ze3ua =  ( (double)1. - r_vvl ) * local_e3u_n[ji] + r_vvl * local_e3u_a[ji];
      local_ua[curr][ji] = local_ua[curr][ji] + r2dt * (double)0.5 * ( local_utau_b[ji] + local_utau[ji] ) / ( ze3ua * rau0 ) * local_umask[ji];
    }

    CRTS_dma_iput(ua+bais0, &local_ua[curr][0], dma_len, &dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jk = 1; jk < jpkm1; jk++) {
      bais = jk*jpi*jpj + jj*jpi;
      curr = jk % 2; up = 1 - curr;

      CRTS_dma_iget(&local_ua[curr][0], ua+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_zwi[0], zwi+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_zwd_up[0], zwd+(jk-1)*jpi*jpj+jj*jpi, dma_len, &dma_rply);
      D_COUNT+=3;
      CRTS_dma_wait_value(&dma_rply, D_COUNT);

      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
        local_ua[curr][ji] = local_ua[curr][ji] - local_zwi[ji] / local_zwd_up[ji] * local_ua[up][ji];
      }
      CRTS_dma_iput(ua+bais, &local_ua[curr][0], dma_len, &dma_rply);
      D_COUNT++;
      CRTS_dma_wait_value(&dma_rply, D_COUNT);
    }
  }
}

// TODO
void slave_dyn5_(dyn_var *d_v) {
  dyn_var v;
  CRTS_dma_iget(&v, d_v, sizeof(dyn_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  double *ua=v.loc_ua, *zws = v.loc_zws, *zwd = v.loc_zwd;
  // local
  int ji, jj, jk;

  dma_len = jpi*8;
  int curr, nt, bais0;
  double local_ua[2][jpi], local_zwd[jpi], local_zws[jpi];

  for(jj = tid; jj < jpjm1; jj+=64) {
    if(jj == 0) continue;
    // 计算最顶层
    nt = curr = (jpkm1 - 1) % 2;
    bais0 = (jpkm1-1)*jpi*jpj + jj*jpi;
    CRTS_dma_iget(&local_ua[curr][0], ua+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_zwd[0], zwd+bais0, dma_len, &dma_rply);
    D_COUNT+=2;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
    
    for(ji = fs_2-1; ji < fs_jpim1; ji++)
      local_ua[curr][ji] = local_ua[curr][ji] / local_zwd[ji];

    CRTS_dma_iput(ua+bais0, &local_ua[curr][0], dma_len, &dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jk = jpkm1-2; jk >= 0; jk--) {
      bais = jk*jpi*jpj + jj*jpi;
      curr = jk % 2; nt = 1 - curr;

      CRTS_dma_iget(&local_ua[curr][0], ua+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_zws[0], zws+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_zwd[0], zwd+bais, dma_len, &dma_rply);
      D_COUNT+=3;
      CRTS_dma_wait_value(&dma_rply, D_COUNT);
      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
        local_ua[curr][ji] = ( local_ua[curr][ji] - local_zws[ji] * local_ua[nt][ji] ) / local_zwd[ji];
      }
      CRTS_dma_iput(ua+bais, &local_ua[curr][0], dma_len, &dma_rply);
      D_COUNT++;
      CRTS_dma_wait_value(&dma_rply, D_COUNT);
    }
  }
}

void slave_dyn6_(dyn_var *d_v) {
  // get var
  dyn_var v;
  CRTS_dma_iget(&v, d_v, sizeof(dyn_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  double zdt = v.zdt, r_vvl = v.r_vvl;
  double *e3v_n = v.loc_e3v_n, *e3v_a = v.loc_e3v_a, *avm = v.loc_avm, *e3vw_n = v.loc_e3vw_n, *wvmask = v.loc_wvmask, *zwi = v.loc_zwi, *zws = v.loc_zws, *zwd = v.loc_zwd;
  
  // local var
  int ji, jj, k, dma_len_add1, off_c, off_n;
  double ze3va, zzwi, zzws;

  // devide
  devide_sid(1, &k_id, &j_id);
  area(5, jpkm1, k_id, &k_st, &k_block);
  area(1, jpjm1, j_id, &j_st, &j_block);
  if(j_st != 0) {
    j_block += 1;
    j_st -= 1;
  }
  k_ed = k_st + k_block;
  dma_len = j_block * jpi * 8;
  dma_len_add1 = (j_block+1) * jpi * 8;

  double local_zwi[j_block][jpi], local_zws[j_block][jpi], local_zwd[j_block][jpi], local_e3v_n[j_block][jpi], local_e3v_a[j_block][jpi], local_avm[2][j_block+1][jpi], local_e3vw_n[2][j_block][jpi], local_wvmask[2][j_block][jpi];
  
  // data reusing prefetch
  k = k_st;
  off_c = k & 1; off_n = (k+1) & 1;
  bias_c = k*jpi*jpj + j_st*jpi; bias_n = (k+1)*jpi*jpj + j_st*jpi;
  CRTS_dma_iget(&local_avm[off_c][0][0], avm+bias_c, dma_len_add1, &dma_rply);
  CRTS_dma_iget(&local_e3vw_n[off_c][0][0], e3vw_n+bias_c, dma_len, &dma_rply);
  CRTS_dma_iget(&local_wvmask[off_c][0][0], wvmask+bias_c, dma_len, &dma_rply);
  D_COUNT+=3;

  for(k = k_st; k < k_ed; k++) {
    off_c = k & 1; off_n = (k+1) & 1;
    bias_c = k*jpi*jpj + j_st*jpi; bias_n = (k+1)*jpi*jpj + j_st*jpi;

    CRTS_dma_iget(&local_avm[off_n][0][0], avm+bias_n, dma_len_add1, &dma_rply);
    CRTS_dma_iget(&local_e3vw_n[off_n][0][0], e3vw_n+bias_n, dma_len, &dma_rply);
    CRTS_dma_iget(&local_wvmask[off_n][0][0], wvmask+bias_n, dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3v_n[0][0], e3v_n+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3v_a[0][0], e3v_a+bias_c, dma_len, &dma_rply);
    D_COUNT+=5;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jj = 1; jj < j_block; jj++) {
      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
        ze3va =  ( (double)1. - r_vvl ) * local_e3v_n[jj][ji] + r_vvl * local_e3v_a[jj][ji];
        zzwi = - zdt * ( local_avm[off_c][jj+1][ji] + local_avm[off_c][jj][ji]) / (ze3va * local_e3vw_n[off_c][jj][ji]) * local_wvmask[off_c][jj][ji];
        zzws = - zdt * ( local_avm[off_n][jj+1][ji] + local_avm[off_n][jj][ji]) / (ze3va * local_e3vw_n[off_n][jj][ji]) * local_wvmask[off_n][jj][ji];
        local_zwi[jj][ji] = zzwi;
        local_zws[jj][ji] = zzws;
        local_zwd[jj][ji] = (double)1. - zzwi - zzws;
      }
    }
    CRTS_dma_iput(zwi+k*jpi*jpj+(j_st+1)*jpi, &local_zwi[1][0], jpi*(j_block-1)*8, &dma_rply);
    CRTS_dma_iput(zws+k*jpi*jpj+(j_st+1)*jpi, &local_zws[1][0], jpi*(j_block-1)*8, &dma_rply);
    CRTS_dma_iput(zwd+k*jpi*jpj+(j_st+1)*jpi, &local_zwd[1][0], jpi*(j_block-1)*8, &dma_rply);
    D_COUNT+=3;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

// TODO
void slave_dyn7_(dyn_var *d_v) {
  dyn_var v;
  CRTS_dma_iget(&v, d_v, sizeof(dyn_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  double *va=v.loc_va, *e3v_n=v.loc_e3v_n, *e3v_a=v.loc_e3v_a, *vtau_b = v.loc_vtau_b, *vtau=v.loc_vtau, *vmask=v.loc_vmask, *zwi = v.loc_zwi, *zwd = v.loc_zwd;
  double r_vvl = v.r_vvl, r2dt = v.r2dt, rau0 = v.rau0;
  // local
  int ji, jj, jk;
  double ze3va;

  dma_len = jpi*8;
  int curr, up, bais0;
  double local_va[2][jpi], local_e3v_n[jpi], local_e3v_a[jpi], local_vtau_b[jpi], local_vtau[jpi], local_vmask[jpi], local_zwi[jpi], local_zwd_up[jpi];

  for(jj = tid; jj < jpjm1; jj+=64) {
    if(jj == 0) continue;
    // 计算最底层
    up = curr = 0;
    bais0 = jj*jpi;
    CRTS_dma_iget(&local_va[curr][0], va+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_vtau_b[0], vtau_b+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_vtau[0], vtau+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3v_n[0], e3v_n+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3v_a[0], e3v_a+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_vmask[0], vmask+bais0, dma_len, &dma_rply);
    D_COUNT+=6;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
    for(ji = fs_2-1; ji < fs_jpim1; ji++) {
      ze3va =  ( (double)1. - r_vvl ) * local_e3v_n[ji] + r_vvl * local_e3v_a[ji];
      local_va[curr][ji] = local_va[curr][ji] + r2dt * (double)0.5 * ( local_vtau_b[ji] + local_vtau[ji] ) / ( ze3va * rau0 ) * local_vmask[ji];
    }

    CRTS_dma_iput(va+bais0, &local_va[curr][0], dma_len, &dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jk = 1; jk < jpkm1; jk++) {
      bais = jk*jpi*jpj + jj*jpi;
      curr = jk % 2; up = 1 - curr;

      CRTS_dma_iget(&local_va[curr][0], va+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_zwi[0], zwi+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_zwd_up[0], zwd+(jk-1)*jpi*jpj+jj*jpi, dma_len, &dma_rply);
      D_COUNT+=3;
      CRTS_dma_wait_value(&dma_rply, D_COUNT);

      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
        local_va[curr][ji] = local_va[curr][ji] - local_zwi[ji] / local_zwd_up[ji] * local_va[up][ji];
      }
      CRTS_dma_iput(va+bais, &local_va[curr][0], dma_len, &dma_rply);
      D_COUNT++;
      CRTS_dma_wait_value(&dma_rply, D_COUNT);
    }
  }
}

// TODO
void slave_dyn8_(dyn_var *d_v) {
  dyn_var v;
  CRTS_dma_iget(&v, d_v, sizeof(dyn_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  double *va=v.loc_va, *zws = v.loc_zws, *zwd = v.loc_zwd;
  // local
  int ji, jj, jk;

  dma_len = jpi*8;
  int curr, nt, bais0;
  double local_va[2][jpi], local_zwd[jpi], local_zws[jpi];

  for(jj = tid; jj < jpjm1; jj+=64) {
    if(jj == 0) continue;
    // 计算最顶层
    nt = curr = (jpkm1 - 1) % 2;
    bais0 = (jpkm1-1)*jpi*jpj + jj*jpi;
    CRTS_dma_iget(&local_va[curr][0], va+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_zwd[0], zwd+bais0, dma_len, &dma_rply);
    D_COUNT+=2;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
    
    for(ji = fs_2-1; ji < fs_jpim1; ji++)
      local_va[curr][ji] = local_va[curr][ji] / local_zwd[ji];

    CRTS_dma_iput(va+bais0, &local_va[curr][0], dma_len, &dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jk = jpkm1-2; jk >= 0; jk--) {
      bais = jk*jpi*jpj + jj*jpi;
      curr = jk % 2; nt = 1 - curr;

      CRTS_dma_iget(&local_va[curr][0], va+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_zws[0], zws+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_zwd[0], zwd+bais, dma_len, &dma_rply);
      D_COUNT+=3;
      CRTS_dma_wait_value(&dma_rply, D_COUNT);
      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
        local_va[curr][ji] = ( local_va[curr][ji] - local_zws[ji] * local_va[nt][ji] ) / local_zwd[ji];
      }
      CRTS_dma_iput(va+bais, &local_va[curr][0], dma_len, &dma_rply);
      D_COUNT++;
      CRTS_dma_wait_value(&dma_rply, D_COUNT);
    }
  }
}

// e
void slave_tke1_(tke_var *t_v) {
  // get var
  tke_var v;
  CRTS_dma_iget(&v, t_v, sizeof(tke_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  double rn_bshear = v.rn_bshear, ri_cri = v.ri_cri;
  double *rn2b = v.loc_rn2b, *p_avm = v.loc_p_avm, *p_sh2 = v.loc_p_sh2, *apdlr = v.loc_apdlr;
  
  // local var
  int ji, jj, k;
  double zri;

  // devide
  devide_sid(0, &k_id, &j_id);
  area(6, jpkm1, k_id, &k_st, &k_block);
  area(0, jpjm1, j_id, &j_st, &j_block);
  if(j_st != 0) {
    j_block += 1;
    j_st -= 1;
  }

  k_ed = k_st + k_block;
  dma_len = jpi*j_block*8;

  double local_rn2b[j_block][jpi], local_p_avm[j_block][jpi], local_p_sh2[j_block][jpi], local_apdlr[j_block][jpi];

  for(k = k_st; k < k_ed; k++) {
    if(k == 0) continue;

    bias_c = k*jpi*jpj + j_st*jpi;

    CRTS_dma_iget(&local_rn2b[0][0],  rn2b+bias_c,  dma_len, &dma_rply);
    CRTS_dma_iget(&local_p_avm[0][0], p_avm+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_p_sh2[0][0], p_sh2+bias_c, dma_len, &dma_rply);
    D_COUNT+=3;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jj = 1; jj < j_block; jj++) {
      for(ji = 1; ji < fs_jpim1; ji++) {
        zri = dmax( local_rn2b[jj][ji], (double)0. ) * local_p_avm[jj][ji] / ( local_p_sh2[jj][ji] + rn_bshear );
        local_apdlr[jj][ji] = dmax(  (double)0.1,  ri_cri / dmax( ri_cri , zri )  );
      }
    }
    CRTS_dma_iput(apdlr+k*jpi*jpj+(j_st+1)*jpi, &local_apdlr[1][0], jpi*(j_block-1)*8, &dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

void slave_tke2_(tke_var *t_v) {
  // get var
  tke_var v;
  CRTS_dma_iget(&v, t_v, sizeof(tke_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  double zfact1 = v.zfact1, zfact2 = v.zfact2, zfact3 = v.zfact3, rdt = v.rdt;
  double *p_avm = v.loc_p_avm, *p_sh2 = v.loc_p_sh2, *tmask = v.loc_tmask, *p_e3t = v.loc_p_e3t, *p_e3w = v.loc_p_e3w, *zd_up = v.loc_zd_up, *zd_lw = v.loc_zd_lw, *zdiag = v.loc_zdiag, *dissl = v.loc_dissl, *wmask = v.loc_wmask, *en = v.loc_en, *p_avt = v.loc_p_avt, *rn2 = v.loc_rn2;
  
  // local
  int ji, jj, k;
  double zcof, zzd_up, zzd_lw;
  int off_l, off_c, off_n;
  int off_ll, off_cc;

  // devide
  devide_sid(1, &k_id, &j_id);
  area(5, jpkm1, k_id, &k_st, &k_block);
  area(1, jpjm1, j_id, &j_st, &j_block);
  if(j_st != 0) {
    j_block += 1;
    j_st -= 1;
  }
  k_ed = k_st + k_block;
  dma_len = j_block * jpi * 8;

  double local_tmask[j_block][jpi], local_p_avm[3][j_block][jpi], local_p_e3t[2][j_block][jpi], local_p_e3w[j_block][jpi], local_wmask[j_block][jpi], local_dissl[j_block][jpi], local_en[j_block][jpi], local_p_sh2[j_block][jpi], local_p_avt[j_block][jpi], local_rn2[j_block][jpi], local_zd_up[j_block][jpi], local_zd_lw[j_block][jpi], local_zdiag[j_block][jpi];

  // data reusing prefetch
  k = k_st == 0 ? 1:k_st;

  off_l = (k-1) % 3; off_c = k % 3; off_n = (k+1) % 3;
  off_ll = (k-1) & 1; off_cc = k & 1;
  bias_c = k*jpi*jpj + j_st*jpi; bias_l = (k-1)*jpi*jpj + j_st*jpi; bias_n = (k+1)*jpi*jpj + j_st*jpi;
  CRTS_dma_iget(&local_p_avm[off_l][0][0], p_avm+bias_l, dma_len, &dma_rply);
  CRTS_dma_iget(&local_p_avm[off_c][0][0], p_avm+bias_c, dma_len, &dma_rply);
  CRTS_dma_iget(&local_p_e3t[off_ll][0][0], p_e3t+bias_l, dma_len, &dma_rply);
  D_COUNT+=3;

  for(k = k_st; k < k_ed; k++) {
    if(k == 0) continue;
    off_l = (k-1) % 3; off_c = k % 3; off_n = (k+1) % 3;
    off_ll = (k-1) & 1; off_cc = k & 1;
    bias_c = k*jpi*jpj + j_st*jpi; bias_l = (k-1)*jpi*jpj + j_st*jpi; bias_n = (k+1)*jpi*jpj + j_st*jpi;
    
    CRTS_dma_iget(&local_tmask[0][0], tmask+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_p_e3w[0][0], p_e3w+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_wmask[0][0], wmask+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_dissl[0][0], dissl+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_p_sh2[0][0], p_sh2+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_p_avt[0][0], p_avt+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_rn2[0][0],   rn2+bias_c,   dma_len, &dma_rply);
    CRTS_dma_iget(&local_en[0][0],    en+bias_c,    dma_len, &dma_rply);
    CRTS_dma_iget(&local_p_avm[off_n][0][0], p_avm+bias_n, dma_len, &dma_rply);
    CRTS_dma_iget(&local_p_e3t[off_cc][0][0], p_e3t+bias_c, dma_len, &dma_rply);
    D_COUNT+=10;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jj = 1; jj < j_block; jj++) {
      for(ji = 1; ji < fs_jpim1; ji++) {
        zcof   = zfact1 * local_tmask[jj][ji];
        zzd_up = zcof * dmax(  local_p_avm[off_n][jj][ji] + local_p_avm[off_c][jj][ji] , (double)2.e-5  ) \
                       /    (  local_p_e3t[off_cc][jj][ji] * local_p_e3w[jj][ji]  );
        zzd_lw = zcof * dmax(  local_p_avm[off_c][jj][ji] + local_p_avm[off_l][jj][ji] , (double)2.e-5  ) \
                       /    (  local_p_e3t[off_ll][jj][ji] * local_p_e3w[jj][ji]  );
        local_zd_up[jj][ji] = zzd_up;
        local_zd_lw[jj][ji] = zzd_lw;
        local_zdiag[jj][ji] = (double)1. - zzd_lw - zzd_up + zfact2 * local_dissl[jj][ji] * local_wmask[jj][ji];
        local_en[jj][ji] = local_en[jj][ji] + rdt * (  local_p_sh2[jj][ji] - local_p_avt[jj][ji] * local_rn2[jj][ji] + zfact3 * local_dissl[jj][ji] * local_en[jj][ji]) * local_wmask[jj][ji];
      }
    }
    CRTS_dma_iput(zd_up+k*jpi*jpj+(j_st+1)*jpi, &local_zd_up[1][0], jpi*(j_block-1)*8, &dma_rply);
    CRTS_dma_iput(zd_lw+k*jpi*jpj+(j_st+1)*jpi, &local_zd_lw[1][0], jpi*(j_block-1)*8, &dma_rply);
    CRTS_dma_iput(zdiag+k*jpi*jpj+(j_st+1)*jpi, &local_zdiag[1][0], jpi*(j_block-1)*8, &dma_rply);
    CRTS_dma_iput(en+k*jpi*jpj+(j_st+1)*jpi, &local_en[1][0],       jpi*(j_block-1)*8, &dma_rply);
    D_COUNT+=4;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

// TODO
void slave_tke3_(tke_var *t_v) {
  tke_var v;
  CRTS_dma_iget(&v, t_v, sizeof(tke_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  double *zd_up = v.loc_zd_up, *zd_lw = v.loc_zd_lw, *zdiag = v.loc_zdiag, *en = v.loc_en;

  // local
  int ji, jj, jk;
  dma_len = jpi*8;
  int curr, up, bais0;

  double local_zd_lw[2][jpi], local_en[jpi], local_en_up[jpi], local_zdiag[2][jpi], local_zd_up_up[jpi];
  for(jj = tid; jj < jpjm1; jj+=64) {
    if(jj == 0) continue;
    // 计算第二层
    up = curr = 1;
    bais0 = jpj*jpi+jj*jpi;
    CRTS_dma_iget(&local_zd_lw[curr][0], zd_lw+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_en[0],          en+bais0,    dma_len, &dma_rply);
    CRTS_dma_iget(&local_en_up[0],       en+jj*jpi,   dma_len, &dma_rply);
    CRTS_dma_iget(&local_zdiag[curr][0], zdiag+bais0, dma_len, &dma_rply);
    D_COUNT+=4;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
    for(ji = fs_2-1; ji < fs_jpim1; ji++) {
      local_zd_lw[curr][ji] = local_en[ji] - local_zd_lw[curr][ji] * local_en_up[ji];
    }

    CRTS_dma_iput(zd_lw+bais0, &local_zd_lw[curr][0], dma_len, &dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jk = 2; jk < jpkm1; jk++) {
      bais = jk*jpi*jpj + jj*jpi;
      curr = jk % 2; up = 1 - curr;

      CRTS_dma_iget(&local_zdiag[curr][0], zdiag+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_zd_lw[curr][0], zd_lw+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_zd_up_up[0], zd_up+(jk-1)*jpi*jpj+jj*jpi, dma_len, &dma_rply);
      CRTS_dma_iget(&local_en[0], en+bais, dma_len, &dma_rply);
      D_COUNT+=4;
      CRTS_dma_wait_value(&dma_rply, D_COUNT);

      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
        local_zdiag[curr][ji] = local_zdiag[curr][ji] - local_zd_lw[curr][ji] * local_zd_up_up[ji] / local_zdiag[up][ji];
        local_zd_lw[curr][ji] = local_en[ji] - local_zd_lw[curr][ji] / local_zdiag[up][ji] * local_zd_lw[up][ji];
      }
      CRTS_dma_iput(zdiag+bais, &local_zdiag[curr][0], dma_len, &dma_rply);
      CRTS_dma_iput(zd_lw+bais, &local_zd_lw[curr][0], dma_len, &dma_rply);
      D_COUNT+=2;
      CRTS_dma_wait_value(&dma_rply, D_COUNT);
    }
  }
}

// TODO
void slave_tke4_(tke_var *t_v) {
  tke_var v;
  CRTS_dma_iget(&v, t_v, sizeof(tke_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  double *zd_up = v.loc_zd_up, *zd_lw = v.loc_zd_lw, *zdiag = v.loc_zdiag, *en = v.loc_en;
  // local
  int ji, jj, jk;
  dma_len = jpi*8;
  int curr, nt, bais0;

  double local_en[2][jpi], local_zd_lw[jpi], local_zdiag[jpi], local_zd_up[jpi];
  for(jj = tid; jj < jpjm1; jj+=64) {
    if(jj == 0) continue;
    // 计算最顶层
    nt = curr = (jpkm1 - 1) % 2;
    bais0 = (jpkm1-1)*jpi*jpj + jj*jpi;
    CRTS_dma_iget(&local_zd_lw[0], zd_lw+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_zdiag[0], zdiag+bais0, dma_len, &dma_rply);
    D_COUNT+=2;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
    
    for(ji = fs_2-1; ji < fs_jpim1; ji++)
      local_en[curr][ji] = local_zd_lw[ji] / local_zdiag[ji];

    CRTS_dma_iput(en+bais0, &local_en[curr][0], dma_len, &dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jk = jpkm1-2; jk >= 1; jk--) {
      bais = jk*jpi*jpj + jj*jpi;
      curr = jk % 2; nt = 1 - curr;

      CRTS_dma_iget(&local_zd_lw[0], zd_lw+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_zd_up[0], zd_up+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_zdiag[0], zdiag+bais, dma_len, &dma_rply);
      D_COUNT+=3;
      CRTS_dma_wait_value(&dma_rply, D_COUNT);
      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
        local_en[curr][ji] = ( local_zd_lw[ji] - local_zd_up[ji] * local_en[nt][ji] ) / local_zdiag[ji];
      }
      CRTS_dma_iput(en+bais, &local_en[curr][0], dma_len, &dma_rply);
      D_COUNT++;
      CRTS_dma_wait_value(&dma_rply, D_COUNT);
    }
  }
}

void slave_tke5_(tke_var *t_v) {
  // get var
  tke_var v;
  CRTS_dma_iget(&v, t_v, sizeof(tke_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  double rn_emin = v.rn_emin;
  double *en = v.loc_en, *wmask = v.loc_wmask;
  
  // local var
  int ji, jj, k;

  // devide
  devide_sid(0, &k_id, &j_id);
  area(6, jpkm1, k_id, &k_st, &k_block);
  area(0, jpjm1, j_id, &j_st, &j_block);
  if(j_st != 0) {
    j_block += 1;
    j_st -= 1;
  }
  k_ed = k_st + k_block;
  dma_len = j_block * jpi * 8;
  
  double local_en[j_block][jpi], local_wmask[j_block][jpi];

  for(k = k_st; k < k_ed; k++) {
    if(k == 0) continue;
    bias_c = k*jpi*jpj + j_st*jpi;

    CRTS_dma_iget(&local_en[0][0], en+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_wmask[0][0], wmask+bias_c, dma_len, &dma_rply);
    D_COUNT+=2;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jj = 1; jj < j_block; jj++) {
      for(ji = 1; ji < fs_jpim1; ji++) {
        local_en[jj][ji] = dmax( local_en[jj][ji], rn_emin ) * local_wmask[jj][ji];
      }
    }
    CRTS_dma_iput(en+k*jpi*jpj+(j_st+1)*jpi, &local_en[1][0], jpi*(j_block-1)*8, &dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

// TODO
void slave_tke6_(tke_var *t_v) {
  tke_var v;
  CRTS_dma_iget(&v, t_v, sizeof(tke_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  double *zpelc = v.loc_zpelc, *pdepw = v.loc_pdepw, *p_e3w = v.loc_p_e3w, *rn2b = v.loc_rn2b;

  // local
  int ji, jj, jk;
  dma_len = jpi*8;
  int curr, up, bais0;

  double local_zpelc[2][jpi], local_rn2b[jpi], local_pdepw[jpi], local_p_e3w[jpi];
  for(jj = tid; jj < jpjm1; jj+=64) {
    if(jj == 0) continue;
    // 计算第二层
    up = curr = 0;
    bais0 = jj*jpi;
    CRTS_dma_iget(&local_zpelc[curr][0], zpelc+bais0,  dma_len, &dma_rply);
    CRTS_dma_iget(&local_rn2b[0],        rn2b+bais0,   dma_len, &dma_rply);
    CRTS_dma_iget(&local_pdepw[0],       pdepw+jj*jpi, dma_len, &dma_rply);
    CRTS_dma_iget(&local_p_e3w[0],       p_e3w+bais0,  dma_len, &dma_rply);
    D_COUNT+=4;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
    for(ji = fs_2-1; ji < fs_jpim1; ji++) {
      local_zpelc[curr][ji] =  dmax( local_rn2b[ji], (double)0. ) * local_pdepw[ji] * local_p_e3w[ji];
    }

    CRTS_dma_iput(zpelc+bais0, &local_zpelc[curr][0], dma_len, &dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jk = 1; jk < jpkm1; jk++) {
      bais = jk*jpi*jpj + jj*jpi;
      curr = jk % 2; up = 1 - curr;

      CRTS_dma_iget(&local_rn2b[0],  rn2b+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_pdepw[0], pdepw+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_p_e3w[0], p_e3w+bais, dma_len, &dma_rply);
      D_COUNT+=3;
      CRTS_dma_wait_value(&dma_rply, D_COUNT);

      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
        local_zpelc[curr][ji]  = local_zpelc[up][ji] + dmax( local_rn2b[ji], (double)0. ) * local_pdepw[ji] * local_p_e3w[ji];
      }

      CRTS_dma_iput(zpelc+bais, &local_zpelc[curr][0], dma_len, &dma_rply);
      D_COUNT++;
      CRTS_dma_wait_value(&dma_rply, D_COUNT);
    }
  }
}

void slave_nxt_fix_(var_nxt_fix *var){
	var_nxt_fix v;
	CRTS_dma_get(&v, var, sizeof(var_nxt_fix));
	asm volatile("memb\n\t");

	int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1, kjpt = v.kjpt, jn = v.jn, fs_jpim1 = v.fs_jpim1, fs_2 = v.fs_2; 
	double ztn = v.ztn, atfp = v.atfp;
	double *ptn = v.ptn, *pta = v.pta, *ptb = v.ptb;
	double ztd;
	int k_id, k_st, k_ed, k_block;
	int i,j,k,l;
  
	area(6, jpkm1, _PEN, &k_st, &k_block);
	k_ed = k_st + k_block;
	dma_len = jpjm1 * jpi * 8;
	int size = dma_len >> 3;
//	double s_pta[jpjm1][jpi], s_ptb[jpjm1][jpi], s_ptn[jpjm1][jpi];
	double s_pta[size], s_ptb[size], s_ptn[size] __attribute__ ((aligned(64)));

	for(l = 0; l < kjpt; l++){
	  for(k = k_st; k < k_ed; k++){
		  bais = l*jpk*jpj*jpi + k * jpi* jpj;

		  CRTS_dma_iget(&s_pta[0], pta+bais, dma_len, &dma_rply);	
		  CRTS_dma_iget(&s_ptb[0], ptb+bais, dma_len, &dma_rply);	
		  CRTS_dma_iget(&s_ptn[0], ptn+bais, dma_len, &dma_rply);	
			D_COUNT+=3;
			CRTS_dma_wait_value(&dma_rply, D_COUNT);
	    
			for(j = 0; j < size; j++){
				ztd = s_pta[j] - 2.0 * s_ptn[j] + s_ptb[j];
				s_ptb[j] = s_ptn[j]  + atfp * ztd ;
				s_ptn[j] = s_pta[j] ;
			}

		  CRTS_dma_iput(ptb+bais, &s_ptb[0], dma_len, &dma_rply);	
		  CRTS_dma_iput(ptn+bais, &s_ptn[0], dma_len, &dma_rply);	
			D_COUNT += 2;
			CRTS_dma_wait_value(&dma_rply, D_COUNT);

		}	
	}
  	
}

void slave_div_hor_(var_div_hor *v1){
//	CRTS_ssync_array();
  var_div_hor v;
	CRTS_dma_get(&v, v1, sizeof(var_div_hor));
	asm volatile("memb\n\t");

	int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, \
	    jpkm1 = v.jpkm1, jpjm1 = v.jpjm1, fs_jpim1 = v.fs_jpim1, fs_2 = v.fs_2;
	double *hdivn = v.hdivn, *e2u = v.e2u, *e3u_n = v.e3u_n, \
	       *un = v.un, *e1v = v.e1v, *e3v_n = v.e3v_n, *vn = v.vn,
				 *r1_e1e2t = v.r1_e1e2t, *e3t_n = v.e3t_n;
	int k_id, k_st, k_ed, k_block;
	int i,j,k,l, j_st, j_ed, j_bais;
  
	area(6, jpkm1, _PEN, &k_st, &k_block);
	k_ed = k_st + k_block;
	dma_len = jpjm1 * jpi * 8;
	int hf_j = (jpjm1 >>1) +1;
//	int size = dma_len >> 3;
	double s_e2u[jpjm1][jpi], s_e3u_n[jpjm1][jpi], s_un[jpjm1][jpi], s_e1v[jpjm1][jpi],
	       s_e3v_n[jpjm1][jpi], s_vn[jpjm1][jpi], s_r1_e1e2t[jpjm1][jpi], s_e3t_n[jpjm1][jpi];
  double s_hdivn[hf_j][jpi];
  
	dma_rply = 0; D_COUNT = 0;

	CRTS_dma_iget(&s_e2u[0][0], e2u, dma_len, &dma_rply);	
	CRTS_dma_iget(&s_e1v[0][0], e1v, dma_len, &dma_rply);	
	CRTS_dma_iget(&s_r1_e1e2t[0][0], r1_e1e2t, dma_len, &dma_rply);	
	D_COUNT += 3;
	CRTS_dma_wait_value(&dma_rply, D_COUNT);

	for(k = k_st; k < k_ed; k++){
		bais = k*jpi*jpj;

	  CRTS_dma_iget(&s_e3u_n[0][0], e3u_n+bais, dma_len, &dma_rply);	
	  CRTS_dma_iget(&s_e3v_n[0][0], e3v_n+bais, dma_len, &dma_rply);	
	  CRTS_dma_iget(&s_un[0][0], un+bais, dma_len, &dma_rply);	
	  CRTS_dma_iget(&s_vn[0][0], vn+bais, dma_len, &dma_rply);	
	  CRTS_dma_iget(&s_e3t_n[0][0], e3t_n+bais, dma_len, &dma_rply);	
		D_COUNT += 5;
		CRTS_dma_wait_value(&dma_rply, D_COUNT);

		double eu_lst, eu; 	
		for(l =0; l< 2; l++){
			if(l == 0){
			  j_st = 1;
				j_ed = hf_j;
			}else{
			  j_st = hf_j;
				j_ed = jpjm1;
			}
			j = j_st;
			for(; j < j_ed; j++){
        j_idx = j-j_st;
        i = fs_2-1;

				eu_lst = s_e2u[j][i-1] * s_e3u_n[j][i-1] * s_un[j][i-1];
				for(; i < fs_jpim1; i++){
          eu = s_e2u[j][i] * s_e3u_n[j][i] * s_un[j][i];
			    s_hdivn[j_idx][i] = ( eu - eu_lst 
															+ s_e1v[j][i] * s_e3v_n[j][i] * s_vn[j][i]
															- s_e1v[j-1][i] * s_e3v_n[j-1][i] * s_vn[j-1][i] 
					                     ) * s_r1_e1e2t[j][i] / s_e3t_n[j][i];	
					eu_lst = eu;

				}
			}
      CRTS_dma_put(hdivn+ bais + j_st*jpi, &s_hdivn[0][0], (j_ed-j_st)*jpi*8);
		}

	}
}

void slave_lap_(lap_var *l_v) {
  // get var
  lap_var v;
  CRTS_dma_iget(&v, l_v, sizeof(lap_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  double zsign = v.zsign;
  double *ahmf = v.loc_ahmf, *e3f_n = v.loc_e3f_n, *r1_e1e2f = v.loc_r1_e1e2f, *e2v = v.loc_e2v, \
         *pvb = v.loc_pvb, *e1u = v.loc_e1u, *pub = v.loc_pub, *ahmt = v.loc_ahmt, *e3t_b = v.loc_e3t_b, \
         *r1_e1e2t = v.loc_r1_e1e2t, *e2u = v.loc_e2u, *e3u_b = v.loc_e3u_b, *e1v = v.loc_e1v, *e3v_b = v.loc_e3v_b, \
         *pua = v.loc_pua, *pva = v.loc_pva, *r1_e2u = v.loc_r1_e2u, *r1_e1u = v.loc_r1_e1u, *r1_e1v = v.loc_r1_e1v, \
         *r1_e2v = v.loc_r1_e2v, *e3u_n = v.loc_e3u_n, *e3v_n = v.loc_e3v_n;
  
  // local var
  int ji, jj, k;

  // devide
  devide_sid(2, &k_id, &j_id);
  area(4, jpkm1, k_id, &k_st, &k_block);
  area(2, jpj, j_id, &j_st, &j_block);
  if(j_id != 0) {
    j_block += 1;
    j_st -= 1;
  }
  k_ed = k_st + k_block;
  dma_len = j_block * jpi * 8;
  int dma_len_add1 = (j_block + 1) * jpi * 8;
  
  double local_ahmf[j_block][jpi], local_e3f_n[j_block][jpi], local_r1_e1e2f[j_block][jpi], local_e2v[j_block][jpi], \
         local_pvb[j_block+1][jpi], local_e1u[j_block+1][jpi], local_pub[j_block+1][jpi], local_ahmt[j_block+1][jpi], \
         local_e3t_b[j_block+1][jpi], local_r1_e1e2t[j_block+1][jpi], local_e2u[j_block+1][jpi], local_e3u_b[j_block+1][jpi], \
         local_e1v[j_block+1][jpi], local_e3v_b[j_block+1][jpi], local_pua[j_block][jpi], local_pva[j_block][jpi], \
         local_r1_e2u[j_block][jpi], local_r1_e1u[j_block][jpi], local_r1_e1v[j_block][jpi], local_r1_e2v[j_block][jpi], \
         local_e3u_n[j_block][jpi], local_e3v_n[j_block][jpi];
  double local_zcur[j_block+1][jpi], local_zdiv[j_block+1][jpi];

  int bias_c_2d = j_st*jpi;
  CRTS_dma_iget(&local_r1_e1e2f[0][0], r1_e1e2f+bias_c_2d, dma_len, &dma_rply);
  CRTS_dma_iget(&local_e2v[0][0], e2v+bias_c_2d, dma_len, &dma_rply);
  CRTS_dma_iget(&local_e1u[0][0], e1u+bias_c_2d, dma_len_add1, &dma_rply);
  CRTS_dma_iget(&local_r1_e1e2t[0][0], r1_e1e2t+bias_c_2d, dma_len_add1, &dma_rply);
  CRTS_dma_iget(&local_e2u[0][0], e2u+bias_c_2d, dma_len_add1, &dma_rply);
  CRTS_dma_iget(&local_e1v[0][0], e1v+bias_c_2d, dma_len_add1, &dma_rply);
  CRTS_dma_iget(&local_r1_e2u[0][0], r1_e2u+bias_c_2d, dma_len, &dma_rply);
  CRTS_dma_iget(&local_r1_e1u[0][0], r1_e1u+bias_c_2d, dma_len, &dma_rply);
  CRTS_dma_iget(&local_r1_e1v[0][0], r1_e1v+bias_c_2d, dma_len, &dma_rply);
  CRTS_dma_iget(&local_r1_e2v[0][0], r1_e2v+bias_c_2d, dma_len, &dma_rply);
  D_COUNT+=10;
  for(k = k_st; k < k_ed; k++) {
    bias_c = k*jpi*jpj + j_st*jpi;

    CRTS_dma_iget(&local_ahmf[0][0], ahmf+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3f_n[0][0], e3f_n+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3v_n[0][0], e3v_n+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_ahmt[0][0], ahmt+bias_c, dma_len_add1, &dma_rply);
    CRTS_dma_iget(&local_pvb[0][0], pvb+bias_c, dma_len_add1, &dma_rply);
    CRTS_dma_iget(&local_pub[0][0], pub+bias_c, dma_len_add1, &dma_rply);
    CRTS_dma_iget(&local_e3t_b[0][0], e3t_b+bias_c, dma_len_add1, &dma_rply);
    CRTS_dma_iget(&local_e3u_b[0][0], e3u_b+bias_c, dma_len_add1, &dma_rply);
    CRTS_dma_iget(&local_e3v_b[0][0], e3v_b+bias_c, dma_len_add1, &dma_rply);
    D_COUNT+=9;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    CRTS_dma_iget(&local_pua[0][0], pua+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3u_n[0][0], e3u_n+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_pva[0][0], pva+bias_c, dma_len, &dma_rply);
    D_COUNT+=3;

    for(jj = 1; jj < j_block+1; jj++) {
      for(ji = fs_2-1; ji < jpi; ji++) {
        local_zcur[jj-1][ji-1] = local_ahmf[jj-1][ji-1] * local_e3f_n[jj-1][ji-1] * local_r1_e1e2f[jj-1][ji-1] \
                              * (local_e2v[jj-1][ji] * local_pvb[jj-1][ji] - local_e2v[jj-1][ji-1] * local_pvb[jj-1][ji-1] \
                              -  local_e1u[jj][ji-1] * local_pub[jj][ji-1] + local_e1u[jj-1][ji-1] * local_pub[jj-1][ji-1]);
        local_zdiv[jj][ji] = local_ahmt[jj][ji] * local_r1_e1e2t[jj][ji] / local_e3t_b[jj][ji] \
                          * (local_e2u[jj][ji] * local_e3u_b[jj][ji] * local_pub[jj][ji] - local_e2u[jj][ji-1] * local_e3u_b[jj][ji-1] * local_pub[jj][ji-1] \
                           + local_e1v[jj][ji] * local_e3v_b[jj][ji] * local_pvb[jj][ji] - local_e1v[jj-1][ji] * local_e3v_b[jj-1][ji] * local_pvb[jj-1][ji]);
      }
    }

    CRTS_dma_wait_value(&dma_rply, D_COUNT);
    for(jj = 1; jj < j_block; jj++) {
      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
        local_pua[jj][ji] = local_pua[jj][ji] + zsign * ( \
                         - (local_zcur[jj][ji] - local_zcur[jj-1][ji]) * local_r1_e2u[jj][ji] / local_e3u_n[jj][ji] \
                         + (local_zdiv[jj][ji+1] - local_zdiv[jj][ji]) * local_r1_e1u[jj][ji]);
        local_pva[jj][ji] = local_pva[jj][ji] + zsign * ( \
                           (local_zcur[jj][ji] - local_zcur[jj][ji-1]) * local_r1_e1v[jj][ji] / local_e3v_n[jj][ji] \
                         + (local_zdiv[jj+1][ji] - local_zdiv[jj][ji]) * local_r1_e2v[jj][ji]);
      }
    }

    CRTS_dma_iput(pua+k*jpi*jpj+(j_st+1)*jpi, &local_pua[1][0], jpi*(j_block-1)*8, &dma_rply);
    CRTS_dma_iput(pva+k*jpi*jpj+(j_st+1)*jpi, &local_pva[1][0], jpi*(j_block-1)*8, &dma_rply);
    D_COUNT+=2;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

void slave_dynnxt_(dynnxt_var *d_v) {
  dynnxt_var v;
  CRTS_dma_iget(&v, d_v, sizeof(dynnxt_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  double *e3u_a = v.loc_e3u_a, *e3v_a = v.loc_e3v_a, *ua = v.loc_ua, *va = v.loc_va, *umask = v.loc_umask, \
         *vmask = v.loc_vmask, *r1_hu_a = v.loc_r1_hu_a, *r1_hv_a = v.loc_r1_hv_a, *ua_b = v.loc_ua_b, \
         *va_b = v.loc_va_b;
  // local
  int ji, jj, jk;

  dma_len = jpi*8;
  double local_e3u_a[jpi], local_e3v_a[jpi], local_ua[jpkm1][jpi], local_va[jpkm1][jpi], local_umask[jpkm1][jpi], local_vmask[jpkm1][jpi], local_r1_hu_a[jpi], local_r1_hv_a[jpi], local_ua_b[jpi], local_va_b[jpi];
  double local_zue[jpi], local_zve[jpi];

  for(jj = tid; jj < jpj; jj+=64) {
    int bais0 = jj*jpi;
    CRTS_dma_iget(&local_e3u_a[0], e3u_a+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3v_a[0], e3v_a+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_umask[0][0], umask+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_vmask[0][0], vmask+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_ua[0][0], ua+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_va[0][0], va+bais0, dma_len, &dma_rply);
    D_COUNT+=6;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    // compute zue zve
    for(ji = 0; ji < jpi; ji++) {
      local_zue[ji] = local_e3u_a[ji] * local_ua[0][ji] * local_umask[0][ji];
      local_zve[ji] = local_e3v_a[ji] * local_va[0][ji] * local_vmask[0][ji];
    }
    for(jk = 1; jk < jpkm1; jk++) {
      bais = jk*jpi*jpj + jj*jpi;

      CRTS_dma_iget(&local_e3u_a[0], e3u_a+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_e3v_a[0], e3v_a+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_umask[jk][0], umask+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_vmask[jk][0], vmask+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_ua[jk][0], ua+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_va[jk][0], va+bais, dma_len, &dma_rply);
      D_COUNT+=6;
      CRTS_dma_wait_value(&dma_rply, D_COUNT);

      for(ji = 0; ji < jpi; ji++) {
        local_zue[ji] += local_e3u_a[ji] * local_ua[jk][ji] * local_umask[jk][ji];
        local_zve[ji] += local_e3v_a[ji] * local_va[jk][ji] * local_vmask[jk][ji];
      }
    }

    // get r1_hu_a r1_hv_a
    CRTS_dma_iget(&local_r1_hu_a[0], r1_hu_a+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_r1_hv_a[0], r1_hv_a+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_ua_b[0],    ua_b+bais0,    dma_len, &dma_rply);
    CRTS_dma_iget(&local_va_b[0],    va_b+bais0,    dma_len, &dma_rply);
    D_COUNT+=4;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    // compute ua va
    if(v.flag == 1) {
      for(jk = 0; jk < jpkm1; jk++) {
        bais = jk*jpi*jpj + jj*jpi;
        for(ji = 0; ji < jpi; ji++) {
          local_ua[jk][ji] = (local_ua[jk][ji] - local_zue[ji] * local_r1_hu_a[ji] + local_ua_b[ji]) * local_umask[jk][ji];
          local_va[jk][ji] = (local_va[jk][ji] - local_zve[ji] * local_r1_hv_a[ji] + local_va_b[ji]) * local_vmask[jk][ji];
        }
        CRTS_dma_iput(ua+bais, &local_ua[jk][0], dma_len, &dma_rply);
        CRTS_dma_iput(va+bais, &local_va[jk][0], dma_len, &dma_rply);
        D_COUNT+=2;
        CRTS_dma_wait_value(&dma_rply, D_COUNT);
      }
    } else {
      for(jk = 0; jk < jpkm1; jk++) {
        bais = jk*jpi*jpj + jj*jpi;
        for(ji = 0; ji < jpi; ji++) {
          local_ua[jk][ji] = local_ua[jk][ji] - (local_zue[ji] * local_r1_hu_a[ji] - local_ua_b[ji]) * local_umask[jk][ji];
          local_va[jk][ji] = local_va[jk][ji] - (local_zve[ji] * local_r1_hv_a[ji] - local_va_b[ji]) * local_vmask[jk][ji];
        }
        CRTS_dma_iput(ua+bais, &local_ua[jk][0], dma_len, &dma_rply);
        CRTS_dma_iput(va+bais, &local_va[jk][0], dma_len, &dma_rply);
        D_COUNT+=2;
        CRTS_dma_wait_value(&dma_rply, D_COUNT);
      }
    }
  }
}

void slave_dynnxt2_(dynnxt_var *d_v) {
  dynnxt_var v;
  CRTS_dma_iget(&v, d_v, sizeof(dynnxt_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  double *e3u_a = v.loc_e3u_a, *e3v_a = v.loc_e3v_a, *ua = v.loc_ua, *va = v.loc_va, *umask = v.loc_umask, \
         *vmask = v.loc_vmask, *r1_hu_a = v.loc_r1_hu_a, *r1_hv_a = v.loc_r1_hv_a, *ua_b = v.loc_ua_b, *va_b = v.loc_va_b, \
         *e3u_b = v.loc_e3u_b, *e3v_b = v.loc_e3v_b, *un = v.loc_un, *vn = v.loc_vn, *ub = v.loc_ub, *vb = v.loc_vb, \
         *un_b = v.loc_un_b, *vn_b = v.loc_vn_b, *ub_b = v.loc_ub_b, *vb_b = v.loc_vb_b, *r1_hu_b = v.loc_r1_hu_b, \
         *r1_hv_b = v.loc_r1_hv_b;
  // local
  int ji, jj, jk;

  dma_len = jpi*8;
  double local_un_b[jpi], local_ub_b[jpi], local_vn_b[jpi], local_vb_b[jpi], local_umask[jpi], local_vmask[jpi], local_e3u_a[jpi], local_e3u_b[jpi], local_e3v_a[jpi], local_e3v_b[jpi], local_un[jpi], local_ub[jpi], local_vn[jpi], local_vb[jpi], local_r1_hu_a[jpi], local_r1_hv_a[jpi], local_r1_hu_b[jpi], local_r1_hv_b[jpi];

  for(jj = tid; jj < jpj; jj+=64) {
    int bais0 = jj*jpi;
    // compute un_b ub_b vn_b vb_b
    CRTS_dma_iget(&local_e3u_a[0], e3u_a+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3v_a[0], e3v_a+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3u_b[0], e3u_b+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3v_b[0], e3v_b+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_umask[0], umask+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_vmask[0], vmask+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_un[0],    un+bais0,    dma_len, &dma_rply);
    CRTS_dma_iget(&local_vn[0],    vn+bais0,    dma_len, &dma_rply);
    CRTS_dma_iget(&local_ub[0],    ub+bais0,    dma_len, &dma_rply);
    CRTS_dma_iget(&local_vb[0],    vb+bais0,    dma_len, &dma_rply);
    D_COUNT+=10;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(ji = 0; ji < jpi; ji++) {
      local_un_b[ji] = local_e3u_a[ji] * local_un[ji] * local_umask[ji];
      local_ub_b[ji] = local_e3u_b[ji] * local_ub[ji] * local_umask[ji];
      local_vn_b[ji] = local_e3v_a[ji] * local_vn[ji] * local_vmask[ji];
      local_vb_b[ji] = local_e3v_b[ji] * local_vb[ji] * local_vmask[ji];
    }

    for(jk = 1; jk < jpkm1; jk++) {
      bais = jk*jpi*jpj + jj*jpi;

      CRTS_dma_iget(&local_e3u_a[0], e3u_a+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_e3v_a[0], e3v_a+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_e3u_b[0], e3u_b+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_e3v_b[0], e3v_b+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_umask[0], umask+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_vmask[0], vmask+bais, dma_len, &dma_rply);
      CRTS_dma_iget(&local_un[0],    un+bais,    dma_len, &dma_rply);
      CRTS_dma_iget(&local_vn[0],    vn+bais,    dma_len, &dma_rply);
      CRTS_dma_iget(&local_ub[0],    ub+bais,    dma_len, &dma_rply);
      CRTS_dma_iget(&local_vb[0],    vb+bais,    dma_len, &dma_rply);
      D_COUNT+=10;
      CRTS_dma_wait_value(&dma_rply, D_COUNT);

      for(ji = 0; ji < jpi; ji++) {
        local_un_b[ji] += local_e3u_a[ji] * local_un[ji] * local_umask[ji];
        local_ub_b[ji] += local_e3u_b[ji] * local_ub[ji] * local_umask[ji];
        local_vn_b[ji] += local_e3v_a[ji] * local_vn[ji] * local_vmask[ji];
        local_vb_b[ji] += local_e3v_b[ji] * local_vb[ji] * local_vmask[ji];
      }
    }

    // compute un_b vn_b ub_b vb_b
    CRTS_dma_iget(&local_r1_hu_a[0], r1_hu_a+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_r1_hv_a[0], r1_hv_a+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_r1_hu_b[0], r1_hu_b+bais0, dma_len, &dma_rply);
    CRTS_dma_iget(&local_r1_hv_b[0], r1_hv_b+bais0, dma_len, &dma_rply);
    D_COUNT+=4;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(ji = 0; ji < jpi; ji++) {
      local_un_b[ji] = local_un_b[ji] * local_r1_hu_a[ji];
      local_vn_b[ji] = local_vn_b[ji] * local_r1_hv_a[ji];
      local_ub_b[ji] = local_ub_b[ji] * local_r1_hu_b[ji];
      local_vb_b[ji] = local_vb_b[ji] * local_r1_hv_b[ji];
    }
    CRTS_dma_iput(un_b+bais0, &local_un_b[0], dma_len, &dma_rply);
    CRTS_dma_iput(vn_b+bais0, &local_vn_b[0], dma_len, &dma_rply);
    CRTS_dma_iput(ub_b+bais0, &local_ub_b[0], dma_len, &dma_rply);
    CRTS_dma_iput(vb_b+bais0, &local_vb_b[0], dma_len, &dma_rply);
    D_COUNT+=4;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

void slave_dynnxt3_(dynnxt_var *d_v) {
  dynnxt_var v;
  CRTS_dma_iget(&v, d_v, sizeof(dynnxt_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  double *ua = v.loc_ua, *va = v.loc_va, *un = v.loc_un, *vn = v.loc_vn, *ub = v.loc_ub, *vb = v.loc_vb;
  double atfp = v.atfp;
  // local var
  int ji, jj, k;

  // devide
  devide_sid(0, &k_id, &j_id);
  area(6, jpkm1, k_id, &k_st, &k_block);
  area(0, jpj, j_id, &j_st, &j_block);
  k_ed = k_st + k_block;
  dma_len = j_block * jpi * 8;
  
  double local_un[j_block][jpi], local_vn[j_block][jpi], local_ua[j_block][jpi], local_va[j_block][jpi], local_ub[j_block][jpi], local_vb[j_block][jpi];

  for(k = k_st; k < k_ed; k++) {
    bias_c = k*jpi*jpj + j_st*jpi;

    CRTS_dma_iget(&local_un[0][0], un+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_vn[0][0], vn+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_ua[0][0], ua+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_va[0][0], va+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_ub[0][0], ub+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_vb[0][0], vb+bias_c, dma_len, &dma_rply);
    D_COUNT+=6;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jj = 0; jj < j_block; jj++) {
      for(ji = 0; ji < jpi; ji++) {
        local_ub[jj][ji] = local_un[jj][ji] + atfp * (local_ub[jj][ji] - (double)2. * local_un[jj][ji] + local_ua[jj][ji]);
        local_vb[jj][ji] = local_vn[jj][ji] + atfp * (local_vb[jj][ji] - (double)2. * local_vn[jj][ji] + local_va[jj][ji]);
        local_un[jj][ji] = local_ua[jj][ji];
        local_vn[jj][ji] = local_va[jj][ji];
      }
    }
    CRTS_dma_iput(ub+bias_c, &local_ub[0][0], dma_len, &dma_rply);
    CRTS_dma_iput(vb+bias_c, &local_vb[0][0], dma_len, &dma_rply);
    CRTS_dma_iput(un+bias_c, &local_un[0][0], dma_len, &dma_rply);
    CRTS_dma_iput(vn+bias_c, &local_vn[0][0], dma_len, &dma_rply);
    D_COUNT+=4;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

void slave_vor_(vor_var *v_v) {
  // get var
  vor_var v;
  CRTS_dma_iget(&v, v_v, sizeof(vor_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  double r1_4 = v.r1_4, zy1, zx1, zy2, zx2;
  double *ff_f = v.loc_ff_f, *e2v = v.loc_e2v, *e1u = v.loc_e1u, *r1_e1e2f = v.loc_r1_e1e2f, *e2u = v.loc_e2u, *e1v = v.loc_e1v, *r1_e1u = v.loc_r1_e1u, *r1_e2v = v.loc_r1_e2v, *pvn = v.loc_pvn, *pun = v.loc_pun, *pua = v.loc_pua, *pva = v.loc_pva;
  
  // local var
  int ji, jj, k;

  // devide
  devide_sid(2, &k_id, &j_id);
  area(4, jpkm1, k_id, &k_st, &k_block);
  area(2, jpjm1, j_id, &j_st, &j_block);
  if(j_id != 0) {
    j_block += 1;
    j_st -= 1;
  }
  k_ed = k_st + k_block;
  dma_len = j_block * jpi * 8;
  int dma_len_add1 = (j_block + 1) * jpi * 8;
  
  double local_ff_f[j_block][jpi], local_e2v[j_block][jpi], local_e1u[j_block+1][jpi], local_r1_e1e2f[j_block][jpi], local_e2u[j_block+1][jpi], local_e1v[j_block+1][jpi], local_r1_e1u[j_block][jpi], local_r1_e2v[j_block][jpi], local_pvn[j_block+1][jpi], local_pun[j_block+1][jpi], local_pua[j_block][jpi], local_pva[j_block][jpi];
  double local_zwz[j_block][jpi], local_zwy[j_block+1][jpi], local_zwx[j_block+1][jpi];

  int bias_c_2d = j_st*jpi;
  CRTS_dma_iget(&local_ff_f[0][0],     ff_f+bias_c_2d,     dma_len,      &dma_rply);
  CRTS_dma_iget(&local_e2v[0][0],      e2v+bias_c_2d,      dma_len,      &dma_rply);
  CRTS_dma_iget(&local_e1u[0][0],      e1u+bias_c_2d,      dma_len_add1, &dma_rply);
  CRTS_dma_iget(&local_r1_e1e2f[0][0], r1_e1e2f+bias_c_2d, dma_len,      &dma_rply);
  CRTS_dma_iget(&local_e2u[0][0],      e2u+bias_c_2d,      dma_len_add1, &dma_rply);
  CRTS_dma_iget(&local_e1v[0][0],      e1v+bias_c_2d,      dma_len_add1, &dma_rply);
  CRTS_dma_iget(&local_r1_e1u[0][0],   r1_e1u+bias_c_2d,   dma_len,      &dma_rply);
  CRTS_dma_iget(&local_r1_e2v[0][0],   r1_e2v+bias_c_2d,   dma_len,      &dma_rply);
  D_COUNT+=8;
  for(k = k_st; k < k_ed; k++) {
    bias_c = k*jpi*jpj + j_st*jpi;

    CRTS_dma_iget(&local_pvn[0][0], pvn+bias_c, dma_len_add1, &dma_rply);
    CRTS_dma_iget(&local_pun[0][0], pun+bias_c, dma_len_add1, &dma_rply);
    D_COUNT+=2;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    // compute zwz
    for(jj = 0; jj < j_block; jj++) {
      for(ji = 0; ji < fs_jpim1; ji++) {
        local_zwz[jj][ji] = local_ff_f[jj][ji] + (local_e2v[jj][ji+1] * local_pvn[jj][ji+1] - local_e2v[jj][ji] * local_pvn[jj][ji] \
                          - local_e1u[jj+1][ji] * local_pun[jj+1][ji] + local_e1u[jj][ji] * local_pun[jj][ji]) * local_r1_e1e2f[jj][ji];
      }
    }
    // compute zwy zwx
    for(jj = 0; jj < j_block+1; jj++) {
      for(ji = 0; ji < jpi; ji++) {
        local_zwx[jj][ji] = local_e2u[jj][ji] * local_pun[jj][ji];
        local_zwy[jj][ji] = local_e1v[jj][ji] * local_pvn[jj][ji];
      }
    }

    CRTS_dma_iget(&local_pua[0][0], pua+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_pva[0][0], pva+bias_c, dma_len, &dma_rply);
    D_COUNT+=2;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
    for(jj = 1; jj < j_block; jj++) {
      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
        zy1 = local_zwy[jj-1][ji] + local_zwy[jj-1][ji+1];
        zy2 = local_zwy[jj][ji] + local_zwy[jj][ji+1];
        zx1 = local_zwx[jj][ji-1] + local_zwx[jj+1][ji-1];
        zx2 = local_zwx[jj][ji] + local_zwx[jj+1][ji];
        local_pua[jj][ji] = local_pua[jj][ji] + r1_4 * local_r1_e1u[jj][ji] * (local_zwz[jj-1][ji] * zy1 + local_zwz[jj][ji] * zy2);
        local_pva[jj][ji] = local_pva[jj][ji] - r1_4 * local_r1_e2v[jj][ji] * (local_zwz[jj][ji-1] * zx1 + local_zwz[jj][ji] * zx2);
      }
    }

    CRTS_dma_iput(pua+k*jpi*jpj+(j_st+1)*jpi, &local_pua[1][0], jpi*(j_block-1)*8, &dma_rply);
    CRTS_dma_iput(pva+k*jpi*jpj+(j_st+1)*jpi, &local_pva[1][0], jpi*(j_block-1)*8, &dma_rply);
    D_COUNT+=2;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

void slave_wzv_(wzv_var *w_v) {
  wzv_var v;
  CRTS_dma_iget(&v, w_v, sizeof(wzv_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, jpkm1 = v.jpkm1;
  double z1_2dt = v.z1_2dt;
  double *wn = v.loc_wn, *e3t_n = v.loc_e3t_n, *hdivn = v.loc_hdivn, *e3t_a = v.loc_e3t_a, *e3t_b = v.loc_e3t_b, *tmask = v.loc_tmask;
  // local
  int ji, jj, jk;

  int bias_k;

  // how many are slave_cores ?
  int task_num = 13;
  if(tid >= task_num) return;
  int res = jpj % task_num;
  j_block = jpj / task_num;
  if(res != 0) {
    if(tid < res) {
      j_block++;
      j_st = tid * j_block;
    } else {
      j_st = tid * j_block + res;
    }
  } else {
      j_st = tid * j_block;
  }

  // get wn(k)
  dma_len = j_block*jpi*8;
  double local_wn[j_block][jpi], local_e3t_b[j_block][jpi], local_e3t_a[j_block][jpi], local_e3t_n[j_block][jpi], local_hdivn[j_block][jpi], local_tmask[j_block][jpi];
  bias_k = (jpk-1)*jpi*jpj + j_st*jpi;
  CRTS_dma_iget(&local_wn[0][0],    wn+bias_k,    dma_len, &dma_rply);
  D_COUNT++;

  for(jk = jpkm1-1; jk >= 0; jk--) {
    bias_c = jk*jpi*jpj + j_st*jpi;

    CRTS_dma_iget(&local_e3t_b[0][0], e3t_b+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3t_n[0][0], e3t_n+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3t_a[0][0], e3t_a+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_tmask[0][0], tmask+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_hdivn[0][0], hdivn+bias_c, dma_len, &dma_rply);
    D_COUNT+=5;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(jj = 0; jj < j_block; jj++) {
      for(ji = 0; ji < jpi; ji++) {
        local_wn[jj][ji] = local_wn[jj][ji] - (local_e3t_n[jj][ji] * local_hdivn[jj][ji] \
                     + z1_2dt * (local_e3t_a[jj][ji] - local_e3t_b[jj][ji])) * local_tmask[jj][ji];
      }
    }
    CRTS_dma_iput(wn+bias_c, &local_wn[0][0], dma_len, &dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

__thread_local volatile unsigned long tu_addr[16], tv_addr[16];
__thread_local volatile int l_tag = 0;
void slave_spgts1_(var_dynspg_ts *v1){
	var_dynspg_ts v;
	CRTS_dma_get(&v, v1, sizeof(var_dynspg_ts));
	asm volatile("memb\n\t");

  volatile int *p;
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, jpkm1 = v.jpkm1; 
	double *e3u_n = v.e3u_n, *ua = v.ua, *umask = v.umask, *r1_hu_n = v.r1_hu_n,\
	       *e3v_n = v.e3v_n, *va = v.va, *vmask = v.vmask, *r1_hv_n = v.r1_hv_n;
	double *zu_frc = v.zu_frc, *zv_frc=v.zv_frc;

	int ji, jj, jk, j_st, j_block;
  int slave_tasks = 16;
	int mod = jpj / slave_tasks;
	int res = jpj % slave_tasks;

  if(res == 0){
	  j_block = mod;  
    j_st = _PEN * mod;
  }else {
	  if(_PEN < res){
	    j_block = mod +1;
      j_st = _PEN * j_block;
	  }else{
		  j_block = mod;  
	    j_st = _PEN * mod + res;
	  }	
  }

	dma_len = jpi*j_block*8;

  int ldm_size = j_block * jpi;
	double s_zu_frc[ldm_size], s_e3u_n[ldm_size], s_ua[ldm_size], s_umask[ldm_size], s_r1_hu_n[ldm_size], s_zv_frc[ldm_size], s_e3v_n[ldm_size], s_va[ldm_size], s_vmask[ldm_size], s_r1_hv_n[ldm_size] __attribute__ ((aligned(64)));

	if(_PEN < slave_tasks){
#ifdef dsimd 
      doublev8 tvv, tuv, evv, euv, uav, vav, umaskv, vmaskv  __attribute__ ((aligned(64)));
#endif
	
	  double tmpu[ldm_size], tmpv[ldm_size] __attribute__ ((aligned(64)));
		int j_disp = j_st*jpi;

		CRTS_dma_iget(&s_r1_hu_n[0], r1_hu_n + j_disp, dma_len, &dma_rply);
		CRTS_dma_iget(&s_r1_hv_n[0], r1_hv_n + j_disp, dma_len, &dma_rply);
		D_COUNT += 2;

		tu_addr[_PEN] = &s_zu_frc[0];
		tv_addr[_PEN] = &s_zv_frc[0];
    memset(&tmpu[0], 0, dma_len);
    memset(&tmpv[0], 0, dma_len);

		CRTS_rma_bcast(&tu_addr[_PEN], &tu_addr[_PEN], sizeof(unsigned long), &rma_rply_r);
		CRTS_rma_bcast(&tv_addr[_PEN], &tv_addr[_PEN], sizeof(unsigned long), &rma_rply_r);
	  for(jk = 0; jk < jpk; jk++){
		  bais = jk*jpi*jpj + j_st*jpi;
      
		  CRTS_dma_iget(&s_e3u_n[0], e3u_n + bais, dma_len, &dma_rply);
		  CRTS_dma_iget(&s_e3v_n[0], e3v_n + bais, dma_len, &dma_rply);
		  CRTS_dma_iget(&s_ua[0], ua + bais, dma_len, &dma_rply);
		  CRTS_dma_iget(&s_va[0], va + bais, dma_len, &dma_rply);
		  CRTS_dma_iget(&s_umask[0], umask + bais, dma_len, &dma_rply);
		  CRTS_dma_iget(&s_vmask[0], vmask + bais, dma_len, &dma_rply);
			D_COUNT += 6;

			CRTS_dma_wait_value(&dma_rply, D_COUNT);
#ifdef dsimd
			for(ji = 0; ji < ldm_size-8; ji+=8){
				simd_load(tuv, &tmpu[ji]);
				simd_load(tvv, &tmpv[ji]);
				simd_load(euv, &s_e3u_n[ji]);
				simd_load(evv, &s_e3v_n[ji]);
				simd_load(uav, &s_ua[ji]);
				simd_load(vav, &s_va[ji]);
				simd_load(umaskv, &s_umask[ji]);
				simd_load(vmaskv, &s_vmask[ji]);
				tuv = tuv + euv * uav * umaskv;
				tvv = tvv + evv * vav * vmaskv;
				simd_store(tuv, &tmpu[ji]);
				simd_store(tvv, &tmpv[ji]);
			}

			for(; ji < ldm_size; ji++){
			  tmpu[ji] = tmpu[ji] + s_e3u_n[ji] * s_ua[ji] * s_umask[ji];
			  tmpv[ji] = tmpv[ji] + s_e3v_n[ji] * s_va[ji] * s_vmask[ji];	
			}
#else			
			for(ji = 0; ji < ldm_size; ji++){
			  tmpu[ji] = tmpu[ji] + s_e3u_n[ji] * s_ua[ji] * s_umask[ji];
			  tmpv[ji] = tmpv[ji] + s_e3v_n[ji] * s_va[ji] * s_vmask[ji];	
			}
#endif		

		}
 
#ifdef dsimd

   for(ji = 0; ji < ldm_size-8; ji+=8){
	   simd_load(tuv, &tmpu[ji]);	 
		 simd_load(tvv, &tmpv[ji]);
		 simd_load(euv, &s_r1_hu_n[ji]);
		 simd_load(evv, &s_r1_hv_n[ji]);
     uav = tuv * euv;
		 vav = tvv * evv;
		 simd_store(uav, &s_zu_frc[ji]);
		 simd_store(vav, &s_zv_frc[ji]);
	 }

	  for(; ji < ldm_size; ji++){
		  s_zu_frc[ji] = tmpu[ji]	 * s_r1_hu_n[ji];
		  s_zv_frc[ji] = tmpv[ji]	 * s_r1_hv_n[ji];
		}
#else 
	  for(ji = 0; ji < ldm_size; ji ++){
		  s_zu_frc[ji] = tmpu[ji]	 * s_r1_hu_n[ji];
		  s_zv_frc[ji] = tmpv[ji]	 * s_r1_hv_n[ji];
		}
#endif		
    
    l_tag = 1; 
		CRTS_dma_iput(zu_frc + j_disp, &s_zu_frc[0], dma_len, &dma_rply);
		CRTS_dma_iput(zv_frc + j_disp, &s_zv_frc[0], dma_len, &dma_rply);
		D_COUNT += 2;
		//CRTS_dma_wait_value(&dma_rply, D_COUNT);	

	}

  int k_st, k_block, k_ed;
  area(6, jpkm1, _PEN, &k_st, &k_block);
	k_ed = k_st + k_block;
  
	int dma_len1 = jpi*jpj*8; 
	int ldm_size1 = jpi*jpj;
	int disp;
  double s_ua_1[ldm_size1], s_va_1[ldm_size1], s_umask_1[ldm_size1], s_vmask_1[ldm_size1] __attribute__ ((aligned(64)));
	double rma_zu_frc[ldm_size1], rma_zv_frc[ldm_size1]  __attribute__ ((aligned(64)));

	for(jk = k_st; jk < k_ed; jk++){
    disp = jk * jpi * jpj;
    CRTS_dma_iget(&s_ua_1[0], ua + disp, dma_len1, &dma_rply);
    CRTS_dma_iget(&s_va_1[0], va + disp, dma_len1, &dma_rply);
    CRTS_dma_iget(&s_umask_1[0], umask + disp, dma_len1, &dma_rply);
    CRTS_dma_iget(&s_vmask_1[0], vmask + disp, dma_len1, &dma_rply);
		D_COUNT+=4;

    if(jk == k_st){
			int sumlen=0, sum_st=0;;
		  for(int id= 0; id < slave_tasks; id++){
        p = remote_ldm_addr(id, l_tag);
				while(*p != 1);
        if(res == 0){
					sumlen = mod*jpi; rma_rply_l = 0;
					CRTS_rma_iget(&rma_zu_frc[sum_st], &rma_rply_l, sumlen*8, id, tu_addr[id], &rma_rply_r);
					CRTS_rma_iget(&rma_zv_frc[sum_st], &rma_rply_l, sumlen*8, id, tv_addr[id], &rma_rply_r);
					sum_st += mod*jpi;
					while(rma_rply_l !=2);
        }else {
	        if(id < res){
					  sumlen = (mod+1)*jpi;
					  CRTS_rma_get(&rma_zu_frc[sum_st], sumlen*8, id, tu_addr[id], &rma_rply_r);
					  CRTS_rma_get(&rma_zv_frc[sum_st], sumlen*8, id, tv_addr[id], &rma_rply_r);
					  sum_st += (mod+1)*jpi;
	        }else{
						sumlen = mod*jpi;
					  CRTS_rma_get(&rma_zu_frc[sum_st], sumlen*8, id, tu_addr[id], &rma_rply_r);
					  CRTS_rma_get(&rma_zv_frc[sum_st], sumlen*8, id, tv_addr[id], &rma_rply_r);
            sum_st += mod*jpi;
	        }	
        }
			}	
		}

		CRTS_dma_wait_value(&dma_rply, D_COUNT);

#ifdef dsimd
    doublev8 uav, vav, zuv,zvv, umaskv ,vmaskv __attribute__ ((aligned(64)));
		for(ji = 0; ji < ldm_size1-8; ji+=8){
			simd_load(uav, &s_ua_1[ji]);
			simd_load(vav, &s_va_1[ji]);
			simd_load(zuv, &rma_zu_frc[ji]);
			simd_load(zvv, &rma_zv_frc[ji]);
			simd_load(umaskv, &s_umask_1[ji]);
	  	simd_load(vmaskv, &s_vmask_1[ji]);
			uav = (uav - zuv) * umaskv;
			vav = (vav - zvv) * vmaskv;
			simd_store(uav, &s_ua_1[ji]);
			simd_store(vav, &s_va_1[ji]);
//		  s_ua_1[ji] = (s_ua_1[ji] - rma_zu_frc[ji]) * s_umask_1[ji];
//		  s_va_1[ji] = (s_va_1[ji] - rma_zv_frc[ji]) * s_vmask_1[ji];
		}
		for(; ji < ldm_size1; ji++){
		  s_ua_1[ji] = (s_ua_1[ji] - rma_zu_frc[ji]) * s_umask_1[ji];
		  s_va_1[ji] = (s_va_1[ji] - rma_zv_frc[ji]) * s_vmask_1[ji];
		}
		
#else
		for(ji = 0; ji < ldm_size1; ji++){
		  s_ua_1[ji] = (s_ua_1[ji] - rma_zu_frc[ji]) * s_umask_1[ji];
		  s_va_1[ji] = (s_va_1[ji] - rma_zv_frc[ji]) * s_vmask_1[ji];
		}
#endif		

		CRTS_dma_iput(ua + disp, &s_ua_1[0], dma_len1, &dma_rply);
		CRTS_dma_iput(va + disp, &s_va_1[0], dma_len1, &dma_rply);
		D_COUNT += 2;
		CRTS_dma_wait_value(&dma_rply, D_COUNT);

	}

  CRTS_ssync_array();  
	l_tag = 0;
}

void slave_spgts2_(var_dynspg_ts *v1){
	var_dynspg_ts v;
	CRTS_dma_get(&v, v1, sizeof(var_dynspg_ts));
	asm volatile("memb\n\t");

  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, jpkm1 = v.jpkm1; 
  double *zhu = v.zhu, *zhv = v.zhv, *un_b = v.un_b, *vn_b = v.vn_b, \
	       *hu_n = v.hu_n, *hv_n = v.hv_n, *e2u = v.e2u, *e2v = v.e1v;
	
	int ji, jj, jk, j_st, j_block;
  int slave_tasks = 8;
	//int mod = jpj / slave_tasks;
	//int res = jpj % slave_tasks;
	int mod = jpj / slave_tasks;
	int res = jpj % slave_tasks;

  if(res == 0){
	  j_block = mod;  
    j_st = _PEN * mod;
  }else {
	  if(_PEN < res){
	    j_block = mod +1;
      j_st = _PEN * j_block;
	  }else{
		  j_block = mod;  
	    j_st = _PEN * mod + res;
	  }	
  }
  
	dma_len = jpi*j_block*8;

  if(_PEN < slave_tasks){
    int ldm_size = j_block * jpi;
	  double s_zhu[ldm_size], s_zhv[ldm_size], s_hu_n[ldm_size], s_hv_n[ldm_size], s_un_b[ldm_size], s_vn_b[ldm_size], s_e2u[ldm_size], s_e2v[ldm_size] __attribute__ ((aligned(64)));
    bais = j_st * jpi;
	  CRTS_dma_iget(&s_un_b[0], un_b + bais, dma_len, &dma_rply);
	  CRTS_dma_iget(&s_vn_b[0], vn_b + bais, dma_len, &dma_rply);
	  CRTS_dma_iget(&s_hu_n[0], hu_n + bais, dma_len, &dma_rply);
  	CRTS_dma_iget(&s_hv_n[0], hv_n + bais, dma_len, &dma_rply);
  	CRTS_dma_iget(&s_e2u[0], e2u + bais, dma_len, &dma_rply);
  	CRTS_dma_iget(&s_e2v[0], e2v + bais, dma_len, &dma_rply);
  	D_COUNT += 6;
 
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
#ifdef dsimd 
		doublev8 unv,vnv, huv, hvv, euv ,evv, zhuv, zhvv __attribute__ ((aligned(64)));
		for(ji = 0; ji < ldm_size - 8; ji+=8){
		  	simd_load(unv, &s_un_b[ji]);
				simd_load(vnv, &s_vn_b[ji]);
				simd_load(huv, &s_hu_n[ji]);
				simd_load(hvv, &s_hv_n[ji]);
				simd_load(euv, &s_e2u[ji]);
				simd_load(evv, &s_e2v[ji]);
        zhuv = unv * huv * euv;
				zhvv = vnv * hvv * evv;
				simd_store(zhuv, &s_zhu[ji]);
				simd_store(zhvv, &s_zhv[ji]);

		}
    for(; ji < ldm_size; ji++){
	  	s_zhu[ji] = s_un_b[ji] * s_hu_n[ji] * s_e2u[ji];
		  s_zhv[ji] = s_vn_b[ji] * s_hv_n[ji] * s_e2v[ji];
	  }
#else		
    for(ji=0; ji < ldm_size; ji++){
	  	s_zhu[ji] = s_un_b[ji] * s_hu_n[ji] * s_e2u[ji];
		  s_zhv[ji] = s_vn_b[ji] * s_hv_n[ji] * s_e2v[ji];
	  }
#endif		
  
	  CRTS_dma_iput(zhu + bais, &s_zhu[0], dma_len, &dma_rply);
	  CRTS_dma_iput(zhv + bais, &s_zhv[0], dma_len, &dma_rply);
  	D_COUNT += 2;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

void spgts3_fun1(var_dynspg_ts *v1);
void spgts3_fun2(var_dynspg_ts *v1);
void slave_spgts3_(var_dynspg_ts *v1){
  spgts3_fun1(v1);
	spgts3_fun2(v1);
}

void spgts3_fun1(var_dynspg_ts *v1){
	unsigned long t1,t2;
// CRTS_ssync_array();
// t1 = CRTS_stime_cycle();

  var_dynspg_ts v;
	CRTS_dma_get(&v, v1, sizeof(var_dynspg_ts));
	asm volatile("memb\n\t");
  
	int ji,jj,jk;
	int jpi = v.jpi, jpj = v.jpj, jpkm1 = v.jpkm1;
	double *ua_b = v.ua_b, *ub_b = v.ub_b, *va_b = v.va_b, *vb_b = v.vb_b, \
	       *un_adv = v.un_adv, *vn_adv = v.vn_adv, *ua = v.ua, *va = v.va;
	int ldm_size = jpi * jpj;			 
	dma_len = ldm_size * 8;

	double s_ua[ldm_size], s_ua_b[ldm_size], s_ub_b[ldm_size], s_va[ldm_size], s_va_b[ldm_size], s_vb_b[ldm_size] __attribute__ ((aligned(64)));

   double r1_2dt_b = v.r1_2dt_b;	

	 CRTS_dma_iget(&s_ua_b[0], ua_b , dma_len, &dma_rply); 
	 CRTS_dma_iget(&s_ub_b[0], ub_b , dma_len, &dma_rply); 
	 CRTS_dma_iget(&s_va_b[0], va_b , dma_len, &dma_rply); 
	 CRTS_dma_iget(&s_vb_b[0], vb_b , dma_len, &dma_rply); 
	 D_COUNT += 4;
	 CRTS_dma_wait_value(&dma_rply, D_COUNT);

	 for(jk=_PEN; jk < jpkm1; jk+=64){
		 bais = jk * jpi *jpj;
	   CRTS_dma_iget(&s_ua[0], ua + bais, dma_len, &dma_rply);
	   CRTS_dma_iget(&s_va[0], va + bais, dma_len, &dma_rply);
		 D_COUNT += 2;
    
		 CRTS_dma_wait_value(&dma_rply, D_COUNT);
#ifdef dsimd
		 doublev8 r1_2dt_bv = r1_2dt_b;
		 doublev8 s_uav, s_ua_bv, s_ub_bv, s_vav, s_va_bv, s_vb_bv;
     for(ji = 0; ji < ldm_size - 8; ji+=8){
			 simd_load(s_uav, &s_ua[ji]);
			 simd_load(s_ua_bv, &s_ua_b[ji]);
			 simd_load(s_ub_bv, &s_ub_b[ji]);
			 simd_load(s_vav, &s_va[ji]);
			 simd_load(s_va_bv, &s_va_b[ji]);
			 simd_load(s_vb_bv, &s_vb_b[ji]);
			 s_uav = s_uav + (s_ua_bv - s_ub_bv) * r1_2dt_bv;
			 s_vav = s_vav + (s_va_bv - s_vb_bv) * r1_2dt_bv;
			 simd_store(s_uav, &s_ua[ji]);
			 simd_store(s_vav, &s_va[ji]);
		 }
     for(; ji < ldm_size; ji++){
		   s_ua[ji] = s_ua[ji] + (s_ua_b[ji] - s_ub_b[ji]) * r1_2dt_b;
		   s_va[ji] = s_va[ji] + (s_va_b[ji] - s_vb_b[ji]) * r1_2dt_b;
		 }

#else		 
     for(ji = 0; ji < ldm_size; ji++){
		   s_ua[ji] = s_ua[ji] + (s_ua_b[ji] - s_ub_b[ji]) * r1_2dt_b;
		   s_va[ji] = s_va[ji] + (s_va_b[ji] - s_vb_b[ji]) * r1_2dt_b;
		 }
#endif		 

     CRTS_dma_iput(ua + bais, &s_ua[0], dma_len, &dma_rply);
     CRTS_dma_iput(va + bais, &s_va[0], dma_len, &dma_rply);
		 D_COUNT += 2;
		 CRTS_dma_wait_value(&dma_rply, D_COUNT);
	 }
 //  CRTS_ssync_array();
//	 t2 = CRTS_stime_cycle();

	 //if(v.rank == 1 && _PEN == 0) \
		 printf("S fun 1 DMA : %f GB/s, cyc: %ld\n", (64*jpi*jpj*4+jpi*jpj*jpkm1*2)*8*2.25/(t2-t1), t2-t1);
}

void spgts3_fun2(var_dynspg_ts *v1){
	unsigned long t1,t2;
//	CRTS_ssync_array();
//	t1 = CRTS_stime_cycle();
  var_dynspg_ts v;
	CRTS_dma_get(&v, v1, sizeof(var_dynspg_ts));
	asm volatile("memb\n\t");
  
	int ji,jj,jk;
	int jpi = v.jpi, jpj = v.jpj, jpkm1 = v.jpkm1;
	double *un = v.un, *un_adv = v.un_adv, *r1_hu_n = v.r1_hu_n, *un_b = v.un_b,  
	       *vn = v.vn, *vn_adv = v.vn_adv, *r1_hv_n = v.r1_hv_n, *vn_b = v.vn_b, 
				 *umask = v.umask, *vmask = v.vmask;
  int k_id, j_id, k_st, j_st, k_block, j_block, k_ed, kcnt;

  // devide
  devide_sid(1, &k_id, &j_id);
  area(5, jpkm1, k_id, &k_st, &k_block);
  area(1, jpj, j_id, &j_st, &j_block);
  //j_id = 0; j_st = 0; j_block = jpjm1;

  k_ed = k_st + k_block;
	int ldm_size = j_block * jpi;			 
	dma_len = ldm_size * 8;

	double s_un_adv[ldm_size], s_r1_hu_n[ldm_size], s_un_b[ldm_size], s_vn_adv[ldm_size], s_r1_hv_n[ldm_size], s_vn_b[ldm_size], s_un[ldm_size], s_vn[ldm_size], s_umask[ldm_size], s_vmask[ldm_size] __attribute__ ((aligned(64)));

   int j_bais = j_st * jpi;
	 CRTS_dma_iget(&s_un_adv[0],  un_adv + j_bais , dma_len, &dma_rply);
	 CRTS_dma_iget(&s_r1_hu_n[0], r1_hu_n + j_bais, dma_len, &dma_rply);
	 CRTS_dma_iget(&s_un_b[0],    un_b + j_bais, dma_len, &dma_rply);
	 CRTS_dma_iget(&s_vn_adv[0],  vn_adv + j_bais, dma_len, &dma_rply);
	 CRTS_dma_iget(&s_r1_hv_n[0], r1_hv_n + j_bais, dma_len, &dma_rply);
	 CRTS_dma_iget(&s_vn_b[0],    vn_b + j_bais, dma_len, &dma_rply);
	 D_COUNT += 6;
	 CRTS_dma_wait_value(&dma_rply, D_COUNT);

	 for(jk= k_st; jk < k_ed; jk++){
		 bais = jk * jpi *jpj + j_st * jpi;
	   CRTS_dma_iget(&s_un[0], un + bais, dma_len, &dma_rply);
	   CRTS_dma_iget(&s_vn[0], vn + bais, dma_len, &dma_rply);
	   CRTS_dma_iget(&s_umask[0], umask + bais, dma_len, &dma_rply);
	   CRTS_dma_iget(&s_vmask[0], vmask + bais, dma_len, &dma_rply);
		 D_COUNT += 4;
    
		 CRTS_dma_wait_value(&dma_rply, D_COUNT);

     for(ji = 0; ji < ldm_size; ji++){
		   s_un[ji] = (s_un[ji] + s_un_adv[ji] * s_r1_hu_n[ji] - s_un_b[ji] ) * s_umask[ji];
		   s_vn[ji] = (s_vn[ji] + s_vn_adv[ji] * s_r1_hv_n[ji] - s_vn_b[ji] ) * s_vmask[ji];
		 }

     CRTS_dma_iput(un + bais, &s_un[0], dma_len, &dma_rply);
     CRTS_dma_iput(vn + bais, &s_vn[0], dma_len, &dma_rply);
		 D_COUNT += 2;
		 CRTS_dma_wait_value(&dma_rply, D_COUNT);
	 }
/*	
	CRTS_ssync_array();
	t2 = CRTS_stime_cycle();

	if(v.rank == 1 && _PEN == 0) \
		printf("S fun2 DMA :%f GB/s., cyc: %ld\n", (jpi*jpj+jpj*jpi*jpkm1)*6*8*2.25/(t2-t1), t2-t1);
*/		
}

inline void spgts4_fma(double *a, double *b, double c, int size){
	int i;
#ifdef dsimd
	doublev8 va, vb;
	doublev8 vc = c;
	for(i=0; i < size-8; i+=8){
		simd_load(va, &a[i]);
		simd_load(vb, &b[i]);
		va = va + vc *vb;
		simd_store(va, &a[i]);
	}
	for(; i < size; i++){
	  a[i] = a[i] + c * b[i]	;
	}
#else
  for(i=0; i < size; i++){
	  a[i] = a[i] + c * b[i]	;
	}
#endif	
}

inline void spgts4_Memload(int op, int myid, int size, int vec, int *idx, int *block, double *M_A, double *A){
	area(op, size, myid, idx, block);
	CRTS_dma_get(A, M_A + (*idx)*vec, (*block)*vec*8);
}
__thread_local volatile unsigned long master_Sig4_count=0;
void slave_spgts4_(var_dynspg_ts *v1){
	var_dynspg_ts v;
	CRTS_dma_get(&v, v1, sizeof(var_dynspg_ts));
	asm volatile("memb\t\n");

	int jpi = v.jpi, jpj=v.jpj;
	double *ubb_e = v.ubb_e, *ub_e=v.ub_e, *un_e=v.un_e, *ua_e=v.ua_e,  
	        *vbb_e=v.vbb_e, *vb_e = v.vb_e, *vn_e=v.vn_e, *va_e=v.va_e, 
					*sshbb_e = v.sshbb_e, *sshb_e =v.sshb_e, *sshn_e=v.sshn_e, 
					*ssha_e=v.ssha_e, *ssha=v.ssha, *ua_b=v.ua_b, *va_b= v.va_b ;
	int slave_tasks_store = 4, slave_tasks_cal = 8;				
	int myid, j_st,j_block, j_ed;
	int block_1 = ((jpj+3) >> 2) + 1;
	int block_2 = ((jpj+7) >> 3) + 1;
	double A_get[block_1*jpi], B_get[block_2*jpi], B_put[block_2*jpi] __attribute__ ((aligned(64)));

  master_Sig4_count++;
	while(*(v1->spgts4_sig) != master_Sig4_count);   // master beginning's Signal 

	double za1 = v1->za1;
	if(_PEN < 4){
		myid = _PEN;
		spgts4_Memload(2, myid, jpj, jpi, &j_st, &j_block, &ub_e[0], &A_get[0]);
	}else if(_PEN < 8){
		myid = _PEN - 4;
		spgts4_Memload(2, myid, jpj, jpi, &j_st, &j_block, un_e, &A_get[0]);

	}else if(_PEN < 12){
		myid = _PEN - 8;
		spgts4_Memload(2, myid, jpj, jpi, &j_st, &j_block, ua_e, &A_get[0]);

	}else if(_PEN < 16){
		myid = _PEN - 12;
		spgts4_Memload(2, myid, jpj, jpi, &j_st, &j_block, vb_e, &A_get[0]);

	}else if(_PEN < 20){
		myid = _PEN - 16;
		spgts4_Memload(2, myid, jpj, jpi, &j_st, &j_block, vn_e, &A_get[0]);
		
	}else if(_PEN < 24){
		myid = _PEN - 20;
		spgts4_Memload(2, myid, jpj, jpi, &j_st, &j_block, va_e, &A_get[0]);
		
	}else if(_PEN < 28){
		myid = _PEN - 24;
		spgts4_Memload(2, myid, jpj, jpi, &j_st, &j_block, sshb_e, &A_get[0]);
		
	}else if(_PEN < 32){
		myid = _PEN - 28;
		spgts4_Memload(2, myid, jpj, jpi, &j_st, &j_block, sshn_e, &A_get[0]);
		
	}else if(_PEN < 36){
		myid = _PEN - 32;
		spgts4_Memload(2, myid, jpj, jpi, &j_st, &j_block, ssha_e, &A_get[0]);
		
	}else if(_PEN < 44){
	  myid = _PEN - 36;
	  area(3, jpj, myid, &j_st, &j_block);
    CRTS_dma_iget(&B_get[0], ua_e + jpi*j_st, j_block*jpi*8, &dma_rply); 
    CRTS_dma_iget(&B_put[0], ua_b + jpi*j_st, j_block*jpi*8, &dma_rply); 
		D_COUNT+=2;
		CRTS_dma_wait_value(&dma_rply, D_COUNT);
    spgts4_fma(&B_put[0], &B_get[0], za1, j_block*jpi);

	}else if(_PEN < 52){
	  myid = _PEN - 44;
	  area(3, jpj, myid, &j_st, &j_block);
    CRTS_dma_iget(&B_get[0], va_e + jpi*j_st, j_block*jpi*8, &dma_rply); 
    CRTS_dma_iget(&B_put[0], va_b + jpi*j_st, j_block*jpi*8, &dma_rply); 
		D_COUNT+=2;
		CRTS_dma_wait_value(&dma_rply, D_COUNT);
    spgts4_fma(&B_put[0], &B_get[0], za1, j_block*jpi);
		
	}else if(_PEN < 60){
	  myid = _PEN - 52;
	  area(3, jpj, myid, &j_st, &j_block);
    CRTS_dma_iget(&B_get[0], ssha_e + jpi*j_st, j_block*jpi*8, &dma_rply); 
    CRTS_dma_iget(&B_put[0], ssha + jpi*j_st, j_block*jpi*8, &dma_rply); 
		D_COUNT+=2;
		CRTS_dma_wait_value(&dma_rply, D_COUNT);
    spgts4_fma(&B_put[0], &B_get[0], za1, j_block*jpi);
		
	}

  CRTS_ssync_array();

  if(_PEN < 4){
	  CRTS_dma_put(ubb_e + j_st*jpi, &A_get[0], j_block*jpi*8);	
	}else if(_PEN < 8){
	  CRTS_dma_put(ub_e + j_st*jpi, &A_get[0], j_block*jpi*8);	
	
	}else if(_PEN < 12){
	  CRTS_dma_put(un_e + j_st*jpi, &A_get[0], j_block*jpi*8);	
		
	}else if(_PEN < 16){
	  CRTS_dma_put(vbb_e + j_st*jpi, &A_get[0], j_block*jpi*8);	
		
	}else if(_PEN < 20){
	  CRTS_dma_put(vb_e + j_st*jpi, &A_get[0], j_block*jpi*8);	
		
	}else if(_PEN < 24){
	  CRTS_dma_put(vn_e + j_st*jpi, &A_get[0], j_block*jpi*8);	
		
	}else if(_PEN < 28){
	  CRTS_dma_put(sshbb_e + j_st*jpi, &A_get[0], j_block*jpi*8);	
		
	}else if(_PEN < 32){
	  CRTS_dma_put(sshb_e + j_st*jpi, &A_get[0], j_block*jpi*8);	
		
	}else if(_PEN < 36){
	  CRTS_dma_put(sshn_e + j_st*jpi, &A_get[0], j_block*jpi*8);	
		
	}else if(_PEN < 44){
	  CRTS_dma_put(ua_b + j_st*jpi, &B_put[0], j_block*jpi*8);	
		
	}else if(_PEN < 52){
	  CRTS_dma_put(va_b + j_st*jpi, &B_put[0], j_block*jpi*8);	
		
	}else if(_PEN < 60){
	  CRTS_dma_put(ssha + j_st*jpi, &B_put[0], j_block*jpi*8);	
	}

}

__thread_local volatile unsigned long master_Sig5_count=0;
void slave_spgts5_(var_dynspg_ts *v1){
	var_dynspg_ts v;
	CRTS_dma_get(&v, v1, sizeof(var_dynspg_ts));
	asm volatile("memb\t\n");

	int jpi = v.jpi, jpj=v.jpj;
	double *ua_e = v.ua_e, *un_e = v.un_e, *ub_e = v.ub_e, *ubb_e = v.ubb_e, \
	       *va_e = v.va_e, *vn_e = v.vn_e, *vb_e = v.vb_e, *vbb_e = v.vbb_e, \
				 *zhu = v.zhu, *e2u = v.e2u, *zhup2_e = v.zhup2_e, \
				 *zhv = v.zhv, *e1v = v.e1v, *zhvp2_e = v.zhvp2_e;
	int op = 3;
	int tasks = 1 << op;
	int block = ((jpj + tasks-1) >> 3) + 1;
	int size = block * jpi;
	int myid, j_st, j_ed, j_block, i,j;
	double a_e[size], n_e[size], b_e[size], bb_e[size], zh[size], e[size], zhp2_e[size] __attribute__ ((aligned(64))) ;
	double za1, za2, za3;
	int ldm_size;

  master_Sig5_count++;
	while(*(v1->spgts5_sig) != master_Sig5_count);

  if(_PEN < tasks){
	 	myid = _PEN;
	  area(op, jpj, myid, &j_st, &j_block);
		bais = j_st * jpi; dma_len = j_block*jpi*8;
  	CRTS_dma_iget(&n_e[0], un_e + bais, dma_len, &dma_rply);
  	CRTS_dma_iget(&b_e[0], ub_e + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&bb_e[0], ubb_e + bais, dma_len, &dma_rply);
		D_COUNT += 3;
		ldm_size = dma_len >> 3;
		za1 = v1->za1; za2 = v1->za2; za3 = v1->za3;
		CRTS_dma_wait_value(&dma_rply, D_COUNT);
//if(v1->rank == 1 && myid == 0) \
	printf("SSSBEFORE: %.15e, %.15e, %.15e, %.15e, %p, %lld\n", bb_e[1*jpi+1], ubb_e[1*jpi+1],b_e[1*jpi+1], ub_e[1*jpi+1], ubb_e, ubb_e+1*jpi+1);
		CRTS_dma_iget(&e[i], e2u + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&zhp2_e[i], zhup2_e + bais, dma_len, &dma_rply);
		D_COUNT += 2;
    
		for(i=0; i < ldm_size; i++){
//if(v1->rank == 1 && i==(1-j_st)*jpi+1){ \
  printf("SSS:%.15e, %.15e, %.15e, %.15e, %.15e, %.15e, %.15e, %lld\n",za1, n_e[i], za2, b_e[i], za3, bb_e[i], ubb_e[1*jpi+1], &ubb_e[1*jpi+1]);	\
}			
		  a_e[i] = za1 * n_e[i] + za2 * b_e[i] + za3 * bb_e[i] ;
		}

    CRTS_dma_wait_value(&dma_rply, D_COUNT);
		CRTS_dma_iput(ua_e + bais, &a_e[0], dma_len, &dma_rply);
    D_COUNT++;

		for(i=0; i < ldm_size; i++){
		  zh[i] = e[i] * a_e[i] * zhp2_e[i]	;
		}

		CRTS_dma_iput(zhu + bais, &zh[0], dma_len, &dma_rply);
		D_COUNT++;
		CRTS_dma_wait_value(&dma_rply, D_COUNT);

	}else if(_PEN < 2*tasks){
	 	myid = _PEN - tasks;
	  area(op, jpj, myid, &j_st, &j_block);
		bais = j_st * jpi; dma_len = j_block*jpi*8;
//if(v.rank == 1) printf("myid: %d, j_st:%d, %d, %lld\n", myid, j_st, bais, va_e, va_e+bais+ldm_size);		
  	CRTS_dma_iget(&n_e[0], vn_e + bais, dma_len, &dma_rply);
  	CRTS_dma_iget(&b_e[0], vb_e + bais, dma_len, &dma_rply);
  	CRTS_dma_iget(&bb_e[0], vbb_e + bais, dma_len, &dma_rply);
		D_COUNT += 3;
		ldm_size = dma_len >> 3;
		za1 = v1->za1; za2 = v1->za2; za3 = v1->za3;
		CRTS_dma_wait_value(&dma_rply, D_COUNT);
 
		CRTS_dma_iget(&e[i], e1v + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&zhp2_e[i], zhvp2_e + bais, dma_len, &dma_rply);
		D_COUNT += 2;

		for(i=0; i < ldm_size; i++){
		  a_e[i] = za1 * n_e[i] + za2 * b_e[i] + za3 * bb_e[i] ;
		}

    CRTS_dma_wait_value(&dma_rply, D_COUNT);
		CRTS_dma_iput(va_e + bais, &a_e[0], dma_len, &dma_rply);
    D_COUNT++;

		for(i=0; i < ldm_size; i++){
		  zh[i] = e[i] * a_e[i] * zhp2_e[i]	;
		}

		CRTS_dma_iput(zhv + bais, &zh[0], dma_len, &dma_rply);
		D_COUNT++;
		CRTS_dma_wait_value(&dma_rply, D_COUNT);
	}else{
	  ;	
	}				 
}

void slave_zdfsh2_(zdf_sh2_t *v_sh2){
    rma_rply_l = 0;
    zdf_sh2_t ldm_v;
    CRTS_dma_iget(&ldm_v, v_sh2, sizeof(zdf_sh2_t), &dma_rply);
    D_COUNT++;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
    asm volatile("memb\n\t");

    //local
    int ji, jj, jk, i;
    int off_0, off_1;
    int k_id, j_id, k_st, k_ed, j_st, j_ed, k_block, j_block;
    int op     = 2;                            // important, decide how many cores in a row 
    int op_div = 6 - op;
    int kcnt   = 0;
    // if op = 2, _PEN = 23, we can get
    // row_sig = 4, row_id = 5, row_st = 20
    int row_sig = 1 << op;                     // how many cores in a row
    int row_id  = _PEN / row_sig;               // row id of a core
    int row_st  = row_sig * row_id;             // the first core in a row
		volatile int *remote;

    // para
    int jpi   = ldm_v.jpi,
        jpj   = ldm_v.jpj,
        jpk   = ldm_v.jpk,
        jpjm1 = ldm_v.jpjm1,
        jpkm1 = ldm_v.jpkm1,
        jpim1 = ldm_v.jpim1;
    double  *mast_pub    = ldm_v.loc_pub,
            *mast_pvb    = ldm_v.loc_pvb,
            *mast_pun    = ldm_v.loc_pun,
            *mast_pvn    = ldm_v.loc_pvn,
            *mast_e3uw_n = ldm_v.loc_e3uw_n,
            *mast_e3uw_b = ldm_v.loc_e3uw_b,
            *mast_p_avm  = ldm_v.loc_p_avm,
            *mast_e3vw_n = ldm_v.loc_e3vw_n,
            *mast_e3vw_b = ldm_v.loc_e3vw_b,
            *mast_wumask = ldm_v.loc_wumask,
            *mast_wvmask = ldm_v.loc_wvmask,
            *mast_p_sh2  = ldm_v.loc_p_sh2,
            *mast_umask  = ldm_v.loc_umask,
            *mast_vmask  = ldm_v.loc_vmask;

    devide_sid(op, &k_id, &j_id);
    area(op_div, jpkm1, k_id, &k_st, &k_block);
    area(op, jpjm1, j_id, &j_st, &j_block);
    k_ed = k_st + k_block;
    //printf("id: %d \t jpkm1: %d \t jpjm1: %d \t k_id: %d \t j_id: %d \t k_st: %d \t k_block: %d \t j_st: %d \t j_block: %d\n", _PEN, jpkm1, jpjm1, k_id, j_id, k_st, k_block, j_st, j_block);

    double zsh2u[j_block][jpi],
           zsh2v[j_block][jpi];
    double pub[2][j_block][jpi],
           pvb[2][j_block][jpi],
           pun[2][j_block][jpi],
           pvn[2][j_block][jpi],
           e3uw_n[j_block][jpi],
           e3uw_b[j_block][jpi],
           p_avm[j_block+1][jpi],
           e3vw_n[j_block][jpi],
           e3vw_b[j_block][jpi],
           wumask[j_block][jpi],
           wvmask[j_block][jpi],
           result[j_block][jpi],
           umask[j_block][jpi],
           vmask[j_block][jpi];

    if(k_st==0) k_st = 1;
    for(jk=k_st;jk<k_ed;jk++){
        //if(jk==0) continue;
        off_0 = jk & 1;
        off_1 = (jk+1) & 1;
        if(kcnt==0){
            int bais = (jk-1)*jpj*jpi + j_st*jpi;
            int dma_len = jpi*j_block*sizeof(double);
            CRTS_dma_iget(&pub[off_0][0][0], mast_pub+bais, dma_len, &dma_rply);
            CRTS_dma_iget(&pun[off_0][0][0], mast_pun+bais, dma_len, &dma_rply);
            CRTS_dma_iget(&pvb[off_0][0][0], mast_pvb+bais, dma_len, &dma_rply);
            CRTS_dma_iget(&pvn[off_0][0][0], mast_pvn+bais, dma_len, &dma_rply);
            D_COUNT += 4;
            kcnt = 1;
        }
        int bais1 = jk*jpj*jpi + j_st*jpi;
        int dma_len1 = jpi*j_block*sizeof(double);
        CRTS_dma_iget(&pub[off_1][0][0], mast_pub+bais1, dma_len1, &dma_rply);
        CRTS_dma_iget(&pun[off_1][0][0], mast_pun+bais1, dma_len1, &dma_rply);
        CRTS_dma_iget(&pvb[off_1][0][0], mast_pvb+bais1, dma_len1, &dma_rply);
        CRTS_dma_iget(&pvn[off_1][0][0], mast_pvn+bais1, dma_len1, &dma_rply);
        CRTS_dma_iget(e3uw_n, mast_e3uw_n+bais1, dma_len1, &dma_rply);
        CRTS_dma_iget(e3uw_b, mast_e3uw_b+bais1, dma_len1, &dma_rply);
        CRTS_dma_iget(e3vw_n, mast_e3vw_n+bais1, dma_len1, &dma_rply);
        CRTS_dma_iget(e3vw_b, mast_e3vw_b+bais1, dma_len1, &dma_rply);
        CRTS_dma_iget(wumask, mast_wumask+bais1, dma_len1, &dma_rply);
        CRTS_dma_iget(wvmask, mast_wvmask+bais1, dma_len1, &dma_rply);
        CRTS_dma_iget(umask, mast_umask+bais1, dma_len1, &dma_rply);
        CRTS_dma_iget(vmask, mast_vmask+bais1, dma_len1, &dma_rply);

        int dma_len2 = jpi*(j_block+1)*sizeof(double);
        CRTS_dma_iget(p_avm, mast_p_avm+bais1, dma_len2, &dma_rply);
        D_COUNT += 13;
        CRTS_dma_wait_value(&dma_rply, D_COUNT);

        for(jj=0;jj<j_block;jj++){
            for(ji=0;ji<jpim1;ji++){
                zsh2u[jj][ji] = ( p_avm[jj][ji+1] + p_avm[jj][ji] ) \
                              * (   pun[off_0][jj][ji]   -   pun[off_1][jj][ji] ) \
                              * (   pub[off_0][jj][ji]   -   pub[off_1][jj][ji] ) / ( e3uw_n[jj][ji] * e3uw_b[jj][ji] ) * wumask[jj][ji];
                zsh2v[jj][ji] = ( p_avm[jj+1][ji] + p_avm[jj][ji] ) \
                              * (   pvn[off_0][jj][ji]   -   pvn[off_1][jj][ji] ) \
                              * (   pvb[off_0][jj][ji]   -   pvb[off_1][jj][ji] ) / ( e3vw_n[jj][ji] * e3vw_b[jj][ji] ) * wvmask[jj][ji];
            }
        }

        if(j_id < row_sig-1){
            remote = remote_ldm_addr(_PEN+1, zdfsh2_rply);
            while(*remote != 0);
            //CRTS_rma_put(&zsh2v[j_block-1][0], jpi*sizeof(double), _PEN+1, stbu, &zdfsh2_rply);
            //CRTS_rma_put(&vmask[j_block-1][0], jpi*sizeof(double), _PEN+1, stbv, &zdfsh2_rply);
            CRTS_rma_iput(&zsh2v[j_block-1][0], &rma_rply_l, jpi*sizeof(double), _PEN+1, stbu, &zdfsh2_rply);
            CRTS_rma_iput(&vmask[j_block-1][0], &rma_rply_l, jpi*sizeof(double), _PEN+1, stbv, &zdfsh2_rply);
        }
        for(jj=1;jj<j_block;jj++){
            for(ji=1;ji<jpim1;ji++){
                result[jj][ji] = 0.25 * (    ( zsh2u[jj][ji-1] + zsh2u[jj][ji] ) * ( 2. - umask[jj][ji-1] * umask[jj][ji] )   \
                                           + ( zsh2v[jj-1][ji] + zsh2v[jj][ji] ) * ( 2. - vmask[jj-1][ji] * vmask[jj][ji] )   );
            }
        }
        if(j_id > 0){
						while(zdfsh2_rply != 2);
            for(ji=1;ji<jpim1;ji++){
                result[0][ji] = 0.25 * (    ( zsh2u[0][ji-1] + zsh2u[0][ji] ) * ( 2. - umask[0][ji-1] * umask[0][ji] )   \
                                           + ( stbu[ji] + zsh2v[0][ji] ) * ( 2. - stbv[ji] * vmask[0][ji] )   );
            }
            zdfsh2_rply = 0;
        }
        CRTS_dma_iput(mast_p_sh2+bais1, result, dma_len1, &dma_rply);
        D_COUNT++;
        CRTS_dma_wait_value(&dma_rply, D_COUNT);
    }
}

void slave_dynzad2_(dynzad_var *v_z) 
{
  dynzad_var v;
  CRTS_dma_iget(&v, v_z, sizeof(dynzad_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

	double *un = v.loc_un, *vn = v.loc_vn, *wn = v.loc_wn, *e1e2t = v.loc_e1e2t, *zwuw = v.loc_zwuw, *zwvw = v.loc_zwvw;
	double *r1e1e2u = v.loc_r1e1e2u, *r1e1e2v = v.loc_r1e1e2v, *e3un = v.loc_e3un, *e3vn = v.loc_e3vn, *ua = v.loc_ua, *va = v.loc_va;
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  
  int ji, jj, k;
  int k_id, j_id, k_st, j_st, k_block, j_block, k_ed, kcnt = 0;
  int off_l, off_c, off_n, off_temp;
	int off_d, off_u;
  int bias_l, bias_c, bias_n, bias;
	int dma_len_a;
  
	devide_sid(2, &k_id, &j_id);
  area(4, jpkm1-1, k_id, &k_st, &k_block);
  area(2, jpjm1, j_id, &j_st, &j_block);
	if(j_st == 0) { j_st = 1; j_block = j_block - 1; }
	if(k_st == 0) { k_st = 1; k_block = k_block - 1; }
	k_ed = k_st + k_block;
	dma_len = j_block*jpi*8;
	dma_len_a = (j_block+1)*jpi*8;
  double local_un[3][j_block][jpi], local_vn[3][j_block][jpi], local_wn[2][j_block+1][jpi], local_e1e2t[j_block+1][jpi];
	double local_r1e1e2u[j_block][jpi], local_r1e1e2v[j_block][jpi], local_e3un[2][j_block][jpi], local_e3vn[2][j_block][jpi], local_ua[2][j_block][jpi], local_va[2][j_block][jpi];
	double zwuwc, zwuwn, zwvwc, zwvwn;

  off_l = (k_st-1)%3; off_c = k_st%3; off_n = (k_st+1)%3;
	//off_d = k_st%2; off_u = (k_st+1)%2;
	off_d = k_st & 1; off_u = (k_st+1) & 1;
	bias_l = (k_st-1)*jpj*jpi; bias_c = k_st*jpj*jpi; bias_n = (k_st+1)*jpj*jpi; bias = j_st*jpi;
	CRTS_dma_iget(&local_un[off_l][0][0], un+bias_l+bias, dma_len, &dma_rply);
	CRTS_dma_iget(&local_un[off_c][0][0], un+bias_c+bias, dma_len, &dma_rply);
	CRTS_dma_iget(&local_vn[off_l][0][0], vn+bias_l+bias, dma_len, &dma_rply);
	CRTS_dma_iget(&local_vn[off_c][0][0], vn+bias_c+bias, dma_len, &dma_rply);
	CRTS_dma_iget(&local_wn[off_d][0][0], wn+bias_c+bias, dma_len_a, &dma_rply);
	CRTS_dma_iget(&local_e1e2t[0][0], e1e2t+bias, dma_len_a, &dma_rply);
	CRTS_dma_iget(&local_r1e1e2u[0][0], r1e1e2u+bias, dma_len, &dma_rply);
	CRTS_dma_iget(&local_r1e1e2v[0][0], r1e1e2v+bias, dma_len, &dma_rply);
	CRTS_dma_iget(&local_e3un[off_d][0][0], e3un+bias_c+bias, dma_len, &dma_rply);
	CRTS_dma_iget(&local_e3vn[off_d][0][0], e3vn+bias_c+bias, dma_len, &dma_rply);
	CRTS_dma_iget(&local_ua[off_d][0][0], ua+bias_c+bias, dma_len, &dma_rply);
	CRTS_dma_iget(&local_va[off_d][0][0], va+bias_c+bias, dma_len, &dma_rply);
	D_COUNT+=12;
	CRTS_dma_wait_value(&dma_rply, D_COUNT);
	for(k = k_st; k < k_ed; k++)
	{
  	off_l = (k-1)%3; off_c = k%3; off_n = (k+1)%3;
		//off_d = k%2; off_u = (k+1)%2;
		off_d = k & 1; off_u = (k+1) & 1;
		bias_c = k*jpj*jpi; bias_n = (k+1)*jpj*jpi;
		CRTS_dma_iget(&local_un[off_n][0][0], un+bias_n+bias, dma_len, &dma_rply);
		CRTS_dma_iget(&local_vn[off_n][0][0], vn+bias_n+bias, dma_len, &dma_rply);
		CRTS_dma_iget(&local_wn[off_u][0][0], wn+bias_n+bias, dma_len_a, &dma_rply);
		D_COUNT+=3;
		CRTS_dma_wait_value(&dma_rply, D_COUNT);
		if(k+1 < k_ed)
		{
			CRTS_dma_iget(&local_e3un[off_u][0][0], e3un+bias_n+bias, dma_len, &dma_rply);
			CRTS_dma_iget(&local_e3vn[off_u][0][0], e3vn+bias_n+bias, dma_len, &dma_rply);
			CRTS_dma_iget(&local_ua[off_u][0][0], ua+bias_n+bias, dma_len, &dma_rply);
			CRTS_dma_iget(&local_va[off_u][0][0], va+bias_n+bias, dma_len, &dma_rply);
			D_COUNT+=4;
		}
		for(jj = 0; jj < j_block; jj++)
		{
			for(ji = fs_2-1; ji < fs_jpim1; ji++)
			{
				zwuwc = 0.25*(local_e1e2t[jj][ji+1]*local_wn[off_d][jj][ji+1] + local_e1e2t[jj][ji]*local_wn[off_d][jj][ji])*(local_un[off_l][jj][ji] - local_un[off_c][jj][ji]);
				zwvwc = 0.25*(local_e1e2t[jj+1][ji]*local_wn[off_d][jj+1][ji] + local_e1e2t[jj][ji]*local_wn[off_d][jj][ji])*(local_vn[off_l][jj][ji] - local_vn[off_c][jj][ji]);
				zwuwn = 0.25*(local_e1e2t[jj][ji+1]*local_wn[off_u][jj][ji+1] + local_e1e2t[jj][ji]*local_wn[off_u][jj][ji])*(local_un[off_c][jj][ji] - local_un[off_n][jj][ji]);
				zwvwn = 0.25*(local_e1e2t[jj+1][ji]*local_wn[off_u][jj+1][ji] + local_e1e2t[jj][ji]*local_wn[off_u][jj][ji])*(local_vn[off_c][jj][ji] - local_vn[off_n][jj][ji]);
				local_ua[off_d][jj][ji] = local_ua[off_d][jj][ji] - (zwuwc + zwuwn)*local_r1e1e2u[jj][ji]/local_e3un[off_d][jj][ji];
				local_va[off_d][jj][ji] = local_va[off_d][jj][ji] - (zwvwc + zwvwn)*local_r1e1e2v[jj][ji]/local_e3vn[off_d][jj][ji];
			}
		}
		CRTS_dma_iput(ua+bias_c+bias, &local_ua[off_d][0][0], dma_len, &dma_rply);
		CRTS_dma_iput(va+bias_c+bias, &local_va[off_d][0][0], dma_len, &dma_rply);
		D_COUNT+=2;
		CRTS_dma_wait_value(&dma_rply, D_COUNT);
	}
 
}

void slave_insitu_(insitu_var *v_z) 
{
  insitu_var v;
 	CRTS_dma_iget(&v, v_z, sizeof(insitu_var), &dma_rply);
 	D_COUNT++;
 	CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
	int jp_tem = v.jp_tem, jp_sal = v.jp_sal;
  double r1_S0 = v.r1_S0, r1_T0 = v.r1_T0, r1_Z0 = v.r1_Z0, rdeltaS = v.rdeltaS, r1_rau0 = v.r1_rau0;
	double *pts = v.loc_pts, *pdep = v.loc_pdep, *tmask = v.loc_tmask, *prd = v.loc_prd;
 
  double EOS013 = v.EOS013, EOS103 = v.EOS103, EOS003 = v.EOS003, \
         EOS022 = v.EOS022, EOS112 = v.EOS112, EOS012 = v.EOS012, \
         EOS202 = v.EOS202, EOS102 = v.EOS102, EOS002 = v.EOS002, \
         EOS041 = v.EOS041, EOS131 = v.EOS131, EOS031 = v.EOS031, \
         EOS221 = v.EOS221, EOS121 = v.EOS121, EOS021 = v.EOS021, \
         EOS311 = v.EOS311, EOS211 = v.EOS211, EOS111 = v.EOS111, \
         EOS011 = v.EOS011, EOS401 = v.EOS401, EOS301 = v.EOS301, \
         EOS201 = v.EOS201, EOS101 = v.EOS101, EOS001 = v.EOS001, \
         EOS060 = v.EOS060, EOS150 = v.EOS150, EOS050 = v.EOS050, \
         EOS240 = v.EOS240, EOS140 = v.EOS140, EOS040 = v.EOS040, \
         EOS330 = v.EOS330, EOS230 = v.EOS230, EOS130 = v.EOS130, \
         EOS030 = v.EOS030, EOS420 = v.EOS420, EOS320 = v.EOS320, \
         EOS220 = v.EOS220, EOS120 = v.EOS120, EOS020 = v.EOS020, \
         EOS510 = v.EOS510, EOS410 = v.EOS410, EOS310 = v.EOS310, \
         EOS210 = v.EOS210, EOS110 = v.EOS110, EOS010 = v.EOS010, \
         EOS600 = v.EOS600, EOS500 = v.EOS500, EOS400 = v.EOS400, \
         EOS300 = v.EOS300, EOS200 = v.EOS200, EOS100 = v.EOS100, \
         EOS000 = v.EOS000;

# if defined dsimd 
	doublev8 vEOS013 = EOS013, vEOS103 = EOS103, vEOS003 = EOS003, \
           vEOS022 = EOS022, vEOS112 = EOS112, vEOS012 = EOS012, \
           vEOS202 = EOS202, vEOS102 = EOS102, vEOS002 = EOS002, \
           vEOS041 = EOS041, vEOS131 = EOS131, vEOS031 = EOS031, \
           vEOS221 = EOS221, vEOS121 = EOS121, vEOS021 = EOS021, \
           vEOS311 = EOS311, vEOS211 = EOS211, vEOS111 = EOS111, \
           vEOS011 = EOS011, vEOS401 = EOS401, vEOS301 = EOS301, \
           vEOS201 = EOS201, vEOS101 = EOS101, vEOS001 = EOS001, \
           vEOS060 = EOS060, vEOS150 = EOS150, vEOS050 = EOS050, \
           vEOS240 = EOS240, vEOS140 = EOS140, vEOS040 = EOS040, \
           vEOS330 = EOS330, vEOS230 = EOS230, vEOS130 = EOS130, \
           vEOS030 = EOS030, vEOS420 = EOS420, vEOS320 = EOS320, \
           vEOS220 = EOS220, vEOS120 = EOS120, vEOS020 = EOS020, \
           vEOS510 = EOS510, vEOS410 = EOS410, vEOS310 = EOS310, \
           vEOS210 = EOS210, vEOS110 = EOS110, vEOS010 = EOS010, \
           vEOS600 = EOS600, vEOS500 = EOS500, vEOS400 = EOS400, \
           vEOS300 = EOS300, vEOS200 = EOS200, vEOS100 = EOS100, \
           vEOS000 = EOS000;
  
	doublev8 vr1_S0 = r1_S0, vr1_T0 = r1_T0, vr1_Z0 = r1_Z0, vrdeltaS = rdeltaS, vr1_rau0 = r1_rau0, vcnum = 1.0;
	doublev8 vzh, vzt, vzs, vdep, vtem, vsal, vabs, vztm, vzn3, vzn2, vzn1, vzn0, vzn, vprd;
# endif

  int ji, jj, k;
  int k_id, j_id, k_st, j_st, k_block, j_block, k_ed;
	int off_c, off_n;
  int bias_c, bias_n, bias, bias_t, bias_s;
  
	devide_sid(1, &k_id, &j_id);
  area(5, jpkm1, k_id, &k_st, &k_block);
  area(1, jpj, j_id, &j_st, &j_block);
	k_ed = k_st + k_block;
	dma_len = j_block*jpi*8;

	double zh, zt, zs, ztm, zn3, zn2, zn1, zn0, zn;

	double local_pts_t[2][j_block][jpi], local_pts_s[2][j_block][jpi], local_pdep[2][j_block][jpi], local_tmask[2][j_block][jpi], local_prd[j_block][jpi];
	//off_c = k_st%2; bias_c = k_st*jpj*jpi; bias = j_st*jpi; bias_t = (jp_tem-1)*jpk*jpj*jpi; bias_s = (jp_sal-1)*jpk*jpj*jpi;
	off_c = k_st & 1; bias_c = k_st*jpj*jpi; bias = j_st*jpi; bias_t = (jp_tem-1)*jpk*jpj*jpi; bias_s = (jp_sal-1)*jpk*jpj*jpi;
	CRTS_dma_iget(&local_pts_t[off_c][0][0], pts+bias_t+bias_c+bias, dma_len, &dma_rply);
	CRTS_dma_iget(&local_pts_s[off_c][0][0], pts+bias_s+bias_c+bias, dma_len, &dma_rply);
	CRTS_dma_iget(&local_pdep[off_c][0][0],  pdep+bias_c+bias,       dma_len, &dma_rply);
	CRTS_dma_iget(&local_tmask[off_c][0][0], tmask+bias_c+bias,      dma_len, &dma_rply);
	D_COUNT+=4;
	CRTS_dma_wait_value(&dma_rply, D_COUNT);
	for(k = k_st; k < k_ed; k++)
	{
		int k_n = k+1;
		//off_c = k%2; off_n = (k+1)%2;
		off_c = k & 1; off_n = (k+1) & 1;
		bias_c = k*jpj*jpi; bias_n = (k+1)*jpj*jpi;
		if(k_n < k_ed)
		{
			CRTS_dma_iget(&local_pts_t[off_n][0][0], pts+bias_t+bias_n+bias, dma_len, &dma_rply);
			CRTS_dma_iget(&local_pts_s[off_n][0][0], pts+bias_s+bias_n+bias, dma_len, &dma_rply);
			CRTS_dma_iget(&local_pdep[off_n][0][0],  pdep+bias_n+bias,       dma_len, &dma_rply);
			CRTS_dma_iget(&local_tmask[off_n][0][0], tmask+bias_n+bias,      dma_len, &dma_rply);
			D_COUNT+=4;
		}
		for(jj = 0; jj < j_block; jj++)
		{
			ji = 0;
# if defined dsimd
			for(; ji < jpi-7; ji+=8)                       // ???
			{
				  simd_loadu(vdep, &local_pdep[off_c][jj][ji]);
				  simd_loadu(vtem, &local_pts_t[off_c][jj][ji]);
				  simd_loadu(vsal, &local_pts_s[off_c][jj][ji]);
				  simd_loadu(vztm, &local_tmask[off_c][jj][ji]);
          vzh = vdep*vr1_Z0;                                  
          vzt = vtem*vr1_T0;
					vabs = vsal+vrdeltaS;
          vzs = simd_vsqrtd(simd_vfselltd(vabs,-vabs,vabs)*vr1_S0);
          vzn3 = vEOS013*vzt \
                + vEOS103*vzs+vEOS003;
          vzn2 = (vEOS022*vzt \
                + vEOS112*vzs+vEOS012)*vzt \
                + (vEOS202*vzs+vEOS102)*vzs+vEOS002;
          vzn1 = (((vEOS041*vzt \
                + vEOS131*vzs+vEOS031)*vzt \
                + (vEOS221*vzs+vEOS121)*vzs+vEOS021)*vzt \
                + ((vEOS311*vzs+vEOS211)*vzs+vEOS111)*vzs+vEOS011)*vzt \
                + (((vEOS401*vzs+vEOS301)*vzs+vEOS201)*vzs+vEOS101)*vzs+vEOS001;
          vzn0 = (((((vEOS060*vzt \
                + vEOS150*vzs+vEOS050)*vzt \
                + (vEOS240*vzs+vEOS140)*vzs+vEOS040)*vzt \
                + ((vEOS330*vzs+vEOS230)*vzs+vEOS130)*vzs+vEOS030)*vzt \
                + (((vEOS420*vzs+vEOS320)*vzs+vEOS220)*vzs+vEOS120)*vzs+vEOS020)*vzt \
                + ((((vEOS510*vzs+vEOS410)*vzs+vEOS310)*vzs+vEOS210)*vzs+vEOS110)*vzs+vEOS010)*vzt \
                + (((((vEOS600*vzs+vEOS500)*vzs+vEOS400)*vzs+vEOS300)*vzs+vEOS200)*vzs+vEOS100)*vzs+vEOS000;
          vzn  = ((vzn3*vzh + vzn2)*vzh + vzn1)*vzh + vzn0;
					vprd = (vzn*vr1_rau0 - vcnum)*vztm;
					simd_storeu(vprd, &local_prd[jj][ji]);
			}
# endif
			for(; ji < jpi; ji++)
			{
          zh = local_pdep[off_c][jj][ji]*r1_Z0;                                  
          zt = local_pts_t[off_c][jj][ji]*r1_T0;    
          zs = sqrt(fabs(local_pts_s[off_c][jj][ji] + rdeltaS )*r1_S0);
          ztm = local_tmask[off_c][jj][ji];
          zn3 = EOS013*zt \
                + EOS103*zs+EOS003;
          zn2 = (EOS022*zt \
                + EOS112*zs+EOS012)*zt \
                + (EOS202*zs+EOS102)*zs+EOS002;
          zn1 = (((EOS041*zt \
                + EOS131*zs+EOS031)*zt \
                + (EOS221*zs+EOS121)*zs+EOS021)*zt \
                + ((EOS311*zs+EOS211)*zs+EOS111)*zs+EOS011)*zt \
                + (((EOS401*zs+EOS301)*zs+EOS201)*zs+EOS101)*zs+EOS001;
          zn0 = (((((EOS060*zt \
                + EOS150*zs+EOS050)*zt \
                + (EOS240*zs+EOS140)*zs+EOS040)*zt \
                + ((EOS330*zs+EOS230)*zs+EOS130)*zs+EOS030)*zt \
                + (((EOS420*zs+EOS320)*zs+EOS220)*zs+EOS120)*zs+EOS020)*zt \
                + ((((EOS510*zs+EOS410)*zs+EOS310)*zs+EOS210)*zs+EOS110)*zs+EOS010)*zt \
                + (((((EOS600*zs+EOS500)*zs+EOS400)*zs+EOS300)*zs+EOS200)*zs+EOS100)*zs+EOS000;
          zn  = ((zn3*zh + zn2)*zh + zn1)*zh + zn0;
          local_prd[jj][ji] = (zn*r1_rau0 - 1.0)*ztm; 
			}
		}
		CRTS_dma_iput(prd+bias_c+bias, &local_prd[0][0], dma_len, &dma_rply);
		D_COUNT+=1;
		CRTS_dma_wait_value(&dma_rply, D_COUNT);
	}

}

void slave_keg_(keg_var *k_v) {
  // get var
  keg_var v;
  CRTS_dma_iget(&v, k_v, sizeof(keg_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  double *un = v.loc_un, *vn = v.loc_vn, *ua = v.loc_ua, *va = v.loc_va, *e1u = v.loc_e1u, *e2v = v.loc_e2v;
  
  // local var
  int ji, jj, k;
  double zu, zv;

  // devide
  devide_sid(0, &k_id, &j_id);
  area(6, jpkm1, k_id, &k_st, &k_block);
  area(0, jpj, j_id, &j_st, &j_block);

  k_ed = k_st + k_block;
  dma_len = j_block * jpi * 8;
  
  double local_un[j_block][jpi], local_vn[j_block][jpi], local_ua[j_block][jpi], local_va[j_block][jpi], local_e1u[j_block][jpi], local_e2v[j_block][jpi];
  double tmp_zhke[2][jpi];

  int bias_c_2d = j_st*jpi;
  CRTS_dma_iget(&local_e1u[0][0], e1u+bias_c_2d, dma_len, &dma_rply);
  CRTS_dma_iget(&local_e2v[0][0], e2v+bias_c_2d, dma_len, &dma_rply);
  D_COUNT+=2;

  int off_n, off_c;

  for(k = k_st; k < k_ed; k++) {
    bias_c = k*jpi*jpj + j_st*jpi;

    CRTS_dma_iget(&local_un[0][0], un+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_vn[0][0], vn+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_ua[0][0], ua+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_va[0][0], va+bias_c, dma_len, &dma_rply);
    D_COUNT+=4;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

    // data prefetch
    jj = 1;
    off_c = jj & 1;
    for(ji = fs_2 - 1; ji < jpi; ji++) {
      zu = local_un[jj][ji-1] * local_un[jj][ji-1] + local_un[jj][ji] * local_un[jj][ji];
      zv = local_vn[jj-1][ji] * local_vn[jj-1][ji] + local_vn[jj][ji] * local_vn[jj][ji];
      tmp_zhke[off_c][ji] = 0.25 * (zv + zu);
    }
    // j_block = jpj
    for(jj = 1; jj < j_block-1; jj++) {
      off_c = jj & 1; off_n = (jj+1) & 1;
      for(ji = fs_2 - 1; ji < jpi; ji++) {
        zu = local_un[jj+1][ji-1] * local_un[jj+1][ji-1] + local_un[jj+1][ji] * local_un[jj+1][ji];
        zv = local_vn[jj][ji] * local_vn[jj][ji] + local_vn[jj+1][ji] * local_vn[jj+1][ji];
        tmp_zhke[off_n][ji] = 0.25 * (zv + zu);
      }

      for(ji = fs_2 - 1; ji < fs_jpim1; ji++) {
        local_ua[jj][ji] = local_ua[jj][ji] - ( tmp_zhke[off_c][ji+1] - tmp_zhke[off_c][ji]) / local_e1u[jj][ji];
        local_va[jj][ji] = local_va[jj][ji] - ( tmp_zhke[off_n][ji]   - tmp_zhke[off_c][ji]) / local_e2v[jj][ji];
      }
    }
    CRTS_dma_iput(ua+k*jpi*jpj+(j_st+1)*jpi, &local_ua[1][0], jpi*(j_block-1)*8, &dma_rply);
    CRTS_dma_iput(va+k*jpi*jpj+(j_st+1)*jpi, &local_va[1][0], jpi*(j_block-1)*8, &dma_rply);
    D_COUNT+=2;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

void slave_qsr4_(qsr_var *q_v) {
  qsr_var v;
  CRTS_dma_iget(&v, q_v, sizeof(qsr_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  // para
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1, nksr = v.nksr, jp_tem = v.jp_tem;
  double zz0 = v.zz0, xsi0r = v.xsi0r, zz1 = v.zz1, xsi1r = v.xsi1r, z1_2 = v.z1_2;
  double *gdepw_n = v.loc_gdepw_n, *qsr = v.loc_qsr, *wmask = v.loc_wmask, *qsr_hc_b = v.loc_qsr_hc_b, *e3t_n = v.loc_e3t_n, *qsr_hc = v.loc_qsr_hc, *tsa = v.loc_tsa;
  // local
  int ji, jj, jk;
  double zc0, zc1;

  // devide
  devide_sid(4, &k_id, &j_id);
  area(2, nksr, k_id, &k_st, &k_block);
  area(4, jpj, j_id, &j_st, &j_block);
  k_ed = k_st + k_block;
  dma_len = j_block * jpi * 8;

  double local_gdepw_n[2][j_block][jpi], local_qsr[j_block][jpi], \
         local_wmask[2][j_block][jpi], local_tsa[j_block][jpi], \
         local_qsr_hc_b[j_block][jpi], local_qsr_hc[j_block][jpi], \
         local_e3t_n[j_block][jpi] __attribute__ ((aligned(64)));
  int off_c, off_n;
  int bias_c_2d = j_st*jpi;
  CRTS_dma_iget(&local_qsr[0][0], qsr+bias_c_2d, dma_len, &dma_rply);
  D_COUNT++;

  // data prefetch
  jk = k_st; off_c = jk & 1; bias_c = jk*jpi*jpj + j_st*jpi;
  CRTS_dma_iget(&local_gdepw_n[off_c][0][0], gdepw_n+bias_c, dma_len, &dma_rply);
  CRTS_dma_iget(&local_wmask[off_c][0][0], wmask+bias_c, dma_len, &dma_rply);
  D_COUNT+=2;

  doublev8 vgdepw_n, vgdepw_n_n, vzc0, vzc1, vqsr_hc, vqsr, vwmask, vwmask_n, vtsa, vqsr_hc_b, ve3t_n;
  doublev8 vzz0 = zz0, vzz1 = zz1, vxsi0r = xsi0r, vxsi1r = xsi1r, vz1_2 = z1_2;
  doublev8 vtmp, vtmp2, vtmp3, vtmp4;
  double tmp[8] __attribute__ ((aligned(64)));

  for(jk = k_st; jk < k_ed; jk++) {
    off_c = jk & 1; off_n = (jk+1) & 1;
    bias_c = jk*jpi*jpj + j_st*jpi; bias_n = (jk+1)*jpi*jpj + j_st*jpi;

    CRTS_dma_iget(&local_gdepw_n[off_n][0][0], gdepw_n+bias_n, dma_len, &dma_rply);
    CRTS_dma_iget(&local_wmask[off_n][0][0], wmask+bias_n, dma_len, &dma_rply);
    CRTS_dma_iget(&local_tsa[0][0], tsa+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_qsr_hc_b[0][0], qsr_hc_b+bias_c, dma_len, &dma_rply);
    CRTS_dma_iget(&local_e3t_n[0][0], e3t_n+bias_c, dma_len, &dma_rply);
    D_COUNT+=5;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
# ifdef dsimd
    for(jj = 0; jj < j_block; jj++) {
      for(ji = fs_2-1; ji < fs_jpim1-8; ji+=8) {
        simd_loadu(vgdepw_n, &local_gdepw_n[off_c][jj][ji]);
        simd_loadu(vgdepw_n_n, &local_gdepw_n[off_n][jj][ji]);

        vtmp = -vgdepw_n * vxsi0r;
        simd_storeu(vtmp, &tmp[0]);
        vexp(&tmp[0], &tmp[0], 8);
        simd_loadu(vtmp, &tmp[0]);

        vtmp2 = -vgdepw_n*vxsi1r;
        simd_storeu(vtmp2, &tmp[0]);
        vexp(&tmp[0], &tmp[0], 8);
        simd_loadu(vtmp2, &tmp[0]);

        vtmp3 = -vgdepw_n_n*vxsi0r;
        simd_storeu(vtmp3, &tmp[0]);
        vexp(&tmp[0], &tmp[0], 8);
        simd_loadu(vtmp3, &tmp[0]);

        vtmp4 = -vgdepw_n_n*vxsi1r;
        simd_storeu(vtmp4, &tmp[0]);
        vexp(&tmp[0], &tmp[0], 8);
        simd_loadu(vtmp4, &tmp[0]);

        simd_loadu(vqsr, &local_qsr[jj][ji]);
        simd_loadu(vwmask, &local_wmask[off_c][jj][ji]);
        simd_loadu(vwmask_n, &local_wmask[off_n][jj][ji]);
        simd_loadu(vtsa, &local_tsa[jj][ji]);
        simd_loadu(vqsr_hc_b, &local_qsr_hc_b[jj][ji]);
        simd_loadu(ve3t_n, &local_e3t_n[jj][ji]);

        vzc0 = vzz0 * vtmp   + vzz1 * vtmp2;
        vzc1 = vzz0 * vtmp3 + vzz1 * vtmp4;
        vqsr_hc = vqsr * (vzc0 * vwmask - vzc1 * vwmask_n);
        vtsa = vtsa + vz1_2 * (vqsr_hc_b + vqsr_hc) / ve3t_n;

        simd_storeu(vqsr_hc, &local_qsr_hc[jj][ji]);
        simd_storeu(vtsa, &local_tsa[jj][ji]);
      }
      for( ; ji < fs_jpim1; ji++) {
        zc0 = zz0 * exp(-local_gdepw_n[off_c][jj][ji]*xsi0r) + zz1 * exp(-local_gdepw_n[off_c][jj][ji]*xsi1r);
        zc1 = zz0 * exp(-local_gdepw_n[off_n][jj][ji]*xsi0r) + zz1 * exp(-local_gdepw_n[off_n][jj][ji]*xsi1r);
        local_qsr_hc[jj][ji] = local_qsr[jj][ji] * (zc0 * local_wmask[off_c][jj][ji] - zc1 * local_wmask[off_n][jj][ji]);
        local_tsa[jj][ji] = local_tsa[jj][ji] + z1_2 * (local_qsr_hc_b[jj][ji] + local_qsr_hc[jj][ji]) / local_e3t_n[jj][ji];
      }
    }
# else
    for(jj = 0; jj < j_block; jj++) {
      for(ji = fs_2-1; ji < fs_jpim1; ji++) {
        zc0 = zz0 * exp(-local_gdepw_n[off_c][jj][ji]*xsi0r) + zz1 * exp(-local_gdepw_n[off_c][jj][ji]*xsi1r);
        zc1 = zz0 * exp(-local_gdepw_n[off_n][jj][ji]*xsi0r) + zz1 * exp(-local_gdepw_n[off_n][jj][ji]*xsi1r);
        local_qsr_hc[jj][ji] = local_qsr[jj][ji] * (zc0 * local_wmask[off_c][jj][ji] - zc1 * local_wmask[off_n][jj][ji]);
        local_tsa[jj][ji] = local_tsa[jj][ji] + z1_2 * (local_qsr_hc_b[jj][ji] + local_qsr_hc[jj][ji]) / local_e3t_n[jj][ji];
      }
    }
# endif
    CRTS_dma_iput(qsr_hc+bias_c, &local_qsr_hc[0][0], dma_len, &dma_rply);
    CRTS_dma_iput(tsa+bias_c, &local_tsa[0][0], dma_len, &dma_rply);
    D_COUNT+=2;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);
  }
}

void bio1_exp(p2zbio_var *v1, p2zbio_var *s_v);
void bio1_accu_cal(p2zbio_var *s_v);

void slave_bio1_(p2zbio_var *v1){
	p2zbio_var v;
//if(_PEN == 0) printf("s_v fun before:%ld\n", &v);
  bio1_exp(v1, &v);
//if(_PEN == 0) printf("s_v 1 fun: %ld, xksi:%p\n", &v, v.xksi);
	bio1_accu_cal(&v);
}

void bio1_exp(p2zbio_var *v1, p2zbio_var *s_v){
	p2zbio_var v;
	CRTS_dma_get(&v, v1, sizeof(p2zbio_var));
	asm volatile("memb\n\t");

  CRTS_dma_iget(s_v, v1, sizeof(p2zbio_var), &dma_rply);
	D_COUNT++;

  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, jpkbm1 = v.jpkbm1, jpjm1 = v.jpjm1, fs_jpim1 = v.fs_jpim1, \
	    fs_2 = v.fs_2, jpdet = v.jpdet - 1 , jpzoo = v.jpzoo - 1, jpphy = v.jpphy - 1, jpno3= v.jpno3 - 1, jpnh4 = v.jpnh4 - 1, jpdom = v.jpdom - 1;
	double aki = v.aki, psinut = v.psinut, tmumax= v.tmumax, rgamma= v.rgamma, fphylab=v.fphylab, rppz=v.rppz, \
	       taus=v.taus, aks=v.aks, rpnaz=v.rpnaz, rdnaz=v.rdnaz, tauzn=v.tauzn, fzoolab=v.fzoolab, tmminp=v.tmminp, tmminz=v.tmminz, \
				 fdbod=v.fdbod, taudn=v.taudn, fdetlab=v.fdetlab, redf=v.redf, reddom=v.reddom, taunn=v.taunn, akno3 = v.akno3, aknh4= v.aknh4, \
				 taudomn = v.taudomn;
  double *trn = v.trn, *etot =v.etot, *tra = v.tra, *xksi = v.xksi, *e3t_n = v.e3t_n;
// local args
  double zlt=1., zppz = rppz, zpdz = 1. - rppz; 
	double zdet, zzoo,zphy,zno3,znh4, zdom, zlno3,zlnh4,zno3phy,znh4phy, zphydom,zphynh4, zpppz, zppdz, zfood, zfilpz, zfildz, \
	       zphyzoo, zdetzoo, zzoodet, zzoonh4, zzoodom, zphydet, zzoobod, zboddet, zdetnh4, zdetdom, zdomnh4, zdomaju, znh4no3, zphya, zzooa, zno3a, znh4a, zdeta, zdoma ;			 
  int ji, jj, jk, k_id, j_id, k_st, k_ed, j_st, j_ed, k_block, j_block;
	int op = 3;
	devide_sid(6-op, &k_id, &j_id);
	area(op,  jpkbm1, k_id, &k_st, &k_block);
	area(6-op,  jpjm1, j_id, &j_st, &j_block);
	
  k_ed = k_st + k_block; 
	j_ed = j_st + j_block;
 
  int ldm_size = j_block * jpi;
	double s_trn_det[ldm_size], s_trn_zoo[ldm_size], s_trn_phy[ldm_size], s_trn_no3[ldm_size],s_trn_nh4[ldm_size], s_trn_dom[ldm_size] __attribute__ ((aligned(64))); 
	double s_tra_det[ldm_size], s_tra_zoo[ldm_size], s_tra_phy[ldm_size], s_tra_no3[ldm_size],s_tra_nh4[ldm_size], s_tra_dom[ldm_size] __attribute__ ((aligned(64))); 
	double s_etot[ldm_size], s_xksi[ldm_size], s_e3t_n[ldm_size] __attribute__ ((aligned(64)));
	double s_tmp_exp_etot[ldm_size] __attribute__ ((aligned(64)));

	volatile double yyj_tmp, zle;
	double tmp_div_refred = (1 - redf/reddom) ;
  int jp_bais = jpi*jpj*jpk; 
	int j_bais = j_st * jpi;
	dma_len = jpi * j_block * 8;
	for(jk = k_st; jk < k_ed; jk++){
    bais = jk * jpi *jpj + j_bais;		
		CRTS_dma_iget(&s_trn_det[0], trn + jpdet*jp_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_trn_zoo[0], trn + jpzoo*jp_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_trn_phy[0], trn + jpphy*jp_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_trn_no3[0], trn + jpno3*jp_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_trn_nh4[0], trn + jpnh4*jp_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_trn_dom[0], trn + jpdom*jp_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_tra_det[0], tra + jpdet*jp_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_tra_zoo[0], tra + jpzoo*jp_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_tra_phy[0], tra + jpphy*jp_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_tra_no3[0], tra + jpno3*jp_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_tra_nh4[0], tra + jpnh4*jp_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_tra_dom[0], tra + jpdom*jp_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_etot[0], etot + bais, dma_len, &dma_rply);
		D_COUNT += 13;
		CRTS_dma_wait_value(&dma_rply, D_COUNT);

#ifdef dsimd
	doublev8 vzdet, vzzoo, vzphy, vzno3, vznh4, vzdom, vzlno3, vzlnh4, vzno3phy, vznh4phy, vzphydom, vzphynh4, vzpppz, vzppdz, vzfood, vzfilpz, vzfildz, \
	         vzphyzoo, vzdetzoo, vzzoodet, vzzoonh4, vzzoodom, vzphydet, vzzoobod, vzboddet, vzdetnh4, vzdetdom, vzdomnh4, vzdomaju, vznh4no3, vzphya, vzzooa, vzno3a, vznh4a, vzdeta, vzdoma, vzle ;			 
	doublev8 vaki = aki, vpsinut = psinut, vtmumax= tmumax, vrgamma= rgamma, vfphylab=fphylab, vrppz=rppz, \
	         vtaus=taus, vaks=aks, vrpnaz=rpnaz, vrdnaz=rdnaz, vtauzn=tauzn, vfzoolab=fzoolab, vtmminp=tmminp, vtmminz=tmminz, \
				   vfdbod=fdbod, vtaudn=taudn, vfdetlab=fdetlab, vredf=redf, vreddom=reddom, vtaunn=taunn, vakno3 = akno3, vaknh4= aknh4, \
				   vtaudomn = taudomn;
	 doublev8 vstrn, vzero=0., vone=1., veis = 1.e-13 ;
   doublev8 vzlt=1., vzppz = vrppz, vzpdz;
	 doublev8 vtmp_div_refred = tmp_div_refred ;
	 double tmp_exp_ar[8] __attribute__ ((aligned(64)));
	 vzpdz = vone - vrppz; 
	 for(ji = 0; ji < ldm_size - 8; ji+=8){
		 simd_load(vstrn, &s_trn_det[ji]);
		 vzdet = simd_smaxd(vzero, vstrn);
		 simd_load(vstrn, &s_trn_zoo[ji]);
		 vzzoo = simd_smaxd(vzero, vstrn);
		 simd_load(vstrn, &s_trn_phy[ji]);
		 vzphy = simd_smaxd(vzero, vstrn);
		 simd_load(vstrn, &s_trn_no3[ji]);
		 vzno3 = simd_smaxd(vzero, vstrn);
		 simd_load(vstrn, &s_trn_nh4[ji]);
		 vznh4 = simd_smaxd(vzero, vstrn);
		 simd_load(vstrn, &s_trn_dom[ji]);
		 vzdom = simd_smaxd(vzero, vstrn);

     /*** exp( -s_etot[ji] / aki ); *****/
     simd_load(vstrn, &s_etot[ji]);
		 vstrn = -vstrn/vaki;
		 simd_store(vstrn, &tmp_exp_ar[0]);
		 vexp(&tmp_exp_ar[0], &tmp_exp_ar[0], 8);
     simd_load(vstrn, &tmp_exp_ar[0]);
     
     vzle =  vone - vstrn;
		 vstrn =  -vpsinut * vznh4;
		 simd_store(vstrn, &tmp_exp_ar[0]);
		 vexp(&tmp_exp_ar[0], &tmp_exp_ar[0], 8);
		 simd_load(vstrn, &tmp_exp_ar[0]);
		 vzlno3 = vzno3 * vstrn / (vakno3 + vzno3);
     vzlnh4 = vznh4 / (vznh4+vaknh4)  ;
		 vzno3phy = vtmumax * vzle * vzlt * vzlno3 * vzphy;
        vznh4phy = vtmumax * vzle * vzlt *  vzlnh4 * vzphy;
        //zphydom = rgamma * (1 - fphylab) * (zno3phy + znh4phy); // vzlno3, vzlnh4 err have done
        vzphydom = vrgamma *(vone - vfphylab) * (vzno3phy + vznh4phy);
        vzphynh4 = vrgamma * vfphylab * (  vzno3phy + vznh4phy);

        //zppz = rpvpz
        //zpdz = 1.v - rppz
        vzpppz = ( vzppz *   vzphy ) / ( ( vzppz * vzphy + vzpdz * vzdet ) + veis );
        vzppdz = ( vzpdz *   vzdet ) / ( ( vzppz * vzphy + vzpdz * vzdet ) + veis );
        vzfood =   vzpppz *  vzphy + vzppdz *vzdet;

        vzfilpz =  vtaus *   vzpppz / (vaks +  vzfood);
        vzfildz =  vtaus *   vzppdz / (vaks +  vzfood);

        vzphyzoo = vzfilpz * vzphy * vzzoo;
        vzdetzoo = vzfildz * vzdet * vzzoo;
             
        vzzoodet = vrpnaz *  vzphyzoo + vrdnaz *vzdetzoo;
            
        vzzoonh4 = vtauzn *  vfzoolab * vzzoo  ;
        vzzoodom = vtauzn * (vone - vfzoolab) * vzzoo;

        vzphydet = vtmminp * vzphy;

        vzzoobod = vtmminz * vzzoo * vzzoo;
        //s_xksi[jiv] = s_xksvi[ji] + (1-fdvbod) * zzoobod * s_e3t_n[ji];
        vzboddet = vfdbod *  vzzoobod;
        vzdetnh4 = vtaudn *  vfdetlab * vzdet;
        vzdetdom = vtaudn * (vone - vfdetlab) * vzdet;
        vzdomnh4 = vtaudomn *vzdom;

        vzdomaju = vtmp_div_refred * (vzphydom + vzzoodom + vzdetdom);

        vznh4no3 = vtaunn * vznh4;
        vzphya =   vzno3phy + vznh4phy - vzphynh4 - vzphydom - vzphyzoo - vzphydet;
        vzzooa =   vzphyzoo + vzdetzoo - vzzoodet - vzzoodom - vzzoonh4 - vzzoobod;
        vzno3a = - vzno3phy + vznh4no3;
        vznh4a = - vznh4phy - vznh4no3 + vzphynh4 + vzzoonh4 + vzdomnh4 + vzdetnh4 + vzdomaju;
        vzdeta =   vzphydet + vzzoodet - vzdetzoo - vzdetnh4 - vzdetdom + vzboddet;
        vzdoma =   vzphydom + vzzoodom + vzdetdom - vzdomnh4 - vzdomaju;
         
				simd_load(vstrn, &s_tra_det[ji]);
				vstrn = vstrn + vzdeta;
				simd_store(vstrn, &s_tra_det[ji]);

				simd_load(vstrn, &s_tra_zoo[ji]);
				vstrn = vstrn + vzzooa;
				simd_store(vstrn, &s_tra_zoo[ji]);

				simd_load(vstrn, &s_tra_phy[ji]);
				vstrn = vstrn + vzphya;
				simd_store(vstrn, &s_tra_phy[ji]);

				simd_load(vstrn, &s_tra_no3[ji]);
				vstrn = vstrn + vzno3a;
				simd_store(vstrn, &s_tra_no3[ji]);

				simd_load(vstrn, &s_tra_nh4[ji]);
				vstrn = vstrn + vznh4a;
				simd_store(vstrn, &s_tra_nh4[ji]);

				simd_load(vstrn, &s_tra_dom[ji]);
				vstrn = vstrn + vzdoma;
				simd_store(vstrn, &s_tra_dom[ji]);
				
	 }
    
    for(; ji < ldm_size; ji++){
        zdet = dmax( 0., s_trn_det[ji] );
        zzoo = dmax( 0., s_trn_zoo[ji] );
        zphy = dmax( 0., s_trn_phy[ji] );
        zno3 = dmax( 0., s_trn_no3[ji] );
        znh4 = dmax( 0., s_trn_nh4[ji] );
        zdom = dmax( 0., s_trn_dom[ji] );

        //zlt   = 1.
        //zle = 1.0 - exp( -s_etot[ji] / aki / zlt );
        zle = 1.0 - exp( -s_etot[ji] / aki );
               
        zlno3 = zno3 * exp( -psinut * znh4 ) / ( akno3 + zno3 );
        zlnh4 = znh4 / (znh4+aknh4)  ;

        zno3phy = tmumax * zle * zlt * zlno3 * zphy;
        znh4phy = tmumax * zle * zlt * zlnh4 * zphy;
             
        zphydom = rgamma * (1 - fphylab) * (zno3phy + znh4phy);
        zphynh4 = rgamma * fphylab * (zno3phy + znh4phy);

        //zppz = rppz
        //zpdz = 1. - rppz
        zpppz = ( zppz * zphy ) / ( ( zppz * zphy + zpdz * zdet ) + 1.e-13 );
        zppdz = ( zpdz * zdet ) / ( ( zppz * zphy + zpdz * zdet ) + 1.e-13 );
        zfood = zpppz * zphy + zppdz * zdet;

        zfilpz = taus * zpppz / (aks + zfood);
        zfildz = taus * zppdz / (aks + zfood);

        zphyzoo = zfilpz * zphy * zzoo;
        zdetzoo = zfildz * zdet * zzoo;
             
        zzoodet = rpnaz * zphyzoo + rdnaz * zdetzoo;
            
        zzoonh4 = tauzn * fzoolab * zzoo  ;
        zzoodom = tauzn * (1 - fzoolab) * zzoo;

        zphydet = tmminp * zphy;

        zzoobod = tmminz * zzoo * zzoo;
        //s_xksi[ji] = s_xksi[ji] + (1-fdbod) * zzoobod * s_e3t_n[ji];
        zboddet = fdbod * zzoobod;
        zdetnh4 = taudn * fdetlab * zdet;
        zdetdom = taudn * (1 - fdetlab) * zdet;
        zdomnh4 = taudomn * zdom;

        //zdomaju = (1 - redf/reddom) * (zphydom + zzoodom + zdetdom);
        zdomaju = tmp_div_refred * (zphydom + zzoodom + zdetdom);

        znh4no3 = taunn * znh4;
        zphya =   zno3phy + znh4phy - zphynh4 - zphydom - zphyzoo - zphydet;
        zzooa =   zphyzoo + zdetzoo - zzoodet - zzoodom - zzoonh4 - zzoobod;
        zno3a = - zno3phy + znh4no3;
        znh4a = - znh4phy - znh4no3 + zphynh4 + zzoonh4 + zdomnh4 + zdetnh4 + zdomaju;
        zdeta =   zphydet + zzoodet - zdetzoo - zdetnh4 - zdetdom + zboddet;
        zdoma =   zphydom + zzoodom + zdetdom - zdomnh4 - zdomaju;

        s_tra_det[ji] = s_tra_det[ji] + zdeta;
        s_tra_zoo[ji] = s_tra_zoo[ji] + zzooa;
        s_tra_phy[ji] = s_tra_phy[ji] + zphya;
        s_tra_no3[ji] = s_tra_no3[ji] + zno3a;
        s_tra_nh4[ji] = s_tra_nh4[ji] + znh4a;
        s_tra_dom[ji] = s_tra_dom[ji] + zdoma;
				
		}
#else
    for(ji = 0; ji < ldm_size; ji++){
        zdet = dmax( 0., s_trn_det[ji] );
        zzoo = dmax( 0., s_trn_zoo[ji] );
        zphy = dmax( 0., s_trn_phy[ji] );
        zno3 = dmax( 0., s_trn_no3[ji] );
        znh4 = dmax( 0., s_trn_nh4[ji] );
        zdom = dmax( 0., s_trn_dom[ji] );

        //zlt   = 1.
        //zle = 1.0 - exp( -s_etot[ji] / aki / zlt );
        zle = 1.0 - exp( -s_etot[ji] / aki );
               
        zlno3 = zno3 * exp( -psinut * znh4 ) / ( akno3 + zno3 );
        zlnh4 = znh4 / (znh4+aknh4)  ;

        zno3phy = tmumax * zle * zlt * zlno3 * zphy;
        znh4phy = tmumax * zle * zlt * zlnh4 * zphy;
             
        zphydom = rgamma * (1 - fphylab) * (zno3phy + znh4phy);
        zphynh4 = rgamma * fphylab * (zno3phy + znh4phy);

        //zppz = rppz
        //zpdz = 1. - rppz
        zpppz = ( zppz * zphy ) / ( ( zppz * zphy + zpdz * zdet ) + 1.e-13 );
        zppdz = ( zpdz * zdet ) / ( ( zppz * zphy + zpdz * zdet ) + 1.e-13 );
        zfood = zpppz * zphy + zppdz * zdet;

        zfilpz = taus * zpppz / (aks + zfood);
        zfildz = taus * zppdz / (aks + zfood);

        zphyzoo = zfilpz * zphy * zzoo;
        zdetzoo = zfildz * zdet * zzoo;
             
        zzoodet = rpnaz * zphyzoo + rdnaz * zdetzoo;
            
        zzoonh4 = tauzn * fzoolab * zzoo  ;
        zzoodom = tauzn * (1 - fzoolab) * zzoo;

        zphydet = tmminp * zphy;

        zzoobod = tmminz * zzoo * zzoo;
        //s_xksi[ji] = s_xksi[ji] + (1-fdbod) * zzoobod * s_e3t_n[ji];
        zboddet = fdbod * zzoobod;
        zdetnh4 = taudn * fdetlab * zdet;
        zdetdom = taudn * (1 - fdetlab) * zdet;
        zdomnh4 = taudomn * zdom;

        //zdomaju = (1 - redf/reddom) * (zphydom + zzoodom + zdetdom);
        zdomaju = tmp_div_refred * (zphydom + zzoodom + zdetdom);

        znh4no3 = taunn * znh4;
        zphya =   zno3phy + znh4phy - zphynh4 - zphydom - zphyzoo - zphydet;
        zzooa =   zphyzoo + zdetzoo - zzoodet - zzoodom - zzoonh4 - zzoobod;
        zno3a = - zno3phy + znh4no3;
        znh4a = - znh4phy - znh4no3 + zphynh4 + zzoonh4 + zdomnh4 + zdetnh4 + zdomaju;
        zdeta =   zphydet + zzoodet - zdetzoo - zdetnh4 - zdetdom + zboddet;
        zdoma =   zphydom + zzoodom + zdetdom - zdomnh4 - zdomaju;

        s_tra_det[ji] = s_tra_det[ji] + zdeta;
        s_tra_zoo[ji] = s_tra_zoo[ji] + zzooa;
        s_tra_phy[ji] = s_tra_phy[ji] + zphya;
        s_tra_no3[ji] = s_tra_no3[ji] + zno3a;
        s_tra_nh4[ji] = s_tra_nh4[ji] + znh4a;
        s_tra_dom[ji] = s_tra_dom[ji] + zdoma;
				
		}
#endif		

		CRTS_dma_iput(tra + jpdet*jp_bais + bais, &s_tra_det[0], dma_len, &dma_rply);
		CRTS_dma_iput(tra + jpzoo*jp_bais + bais, &s_tra_zoo[0], dma_len, &dma_rply);
		CRTS_dma_iput(tra + jpphy*jp_bais + bais, &s_tra_phy[0], dma_len, &dma_rply);
		CRTS_dma_iput(tra + jpno3*jp_bais + bais, &s_tra_no3[0], dma_len, &dma_rply);
		CRTS_dma_iput(tra + jpnh4*jp_bais + bais, &s_tra_nh4[0], dma_len, &dma_rply);
		CRTS_dma_iput(tra + jpdom*jp_bais + bais, &s_tra_dom[0], dma_len, &dma_rply);
    D_COUNT += 6;
		CRTS_dma_wait_value(&dma_rply, D_COUNT);
	}
  	
}

void bio1_accu_cal(p2zbio_var *s_v){
  int jpi = s_v->jpi, jpj = s_v->jpj, jpk = s_v->jpk, jpkbm1 = s_v->jpkbm1, jpjm1 = s_v->jpjm1, fs_jpim1 = s_v->fs_jpim1, \ 
	    jpzoo = s_v->jpzoo - 1;
	double *trn = s_v->trn, *xksi = s_v->xksi, *e3t_n=s_v->e3t_n;
	double tmminz = s_v->tmminz, fdbod = s_v->fdbod;
/********* local var ***************/
  int ji, jj, jk, j_st, j_ed, j_block, k_st, k_ed, k_block;
	int jn_bais, ldm_size, j_bais;
  double zzoo, zzoobod;

  int slave_tasks = 13;
	if(_PEN >= slave_tasks ) return ;

  int mod  = jpjm1 / slave_tasks;
  int res = jpjm1 % slave_tasks;
  if(res == 0){
	  j_block = mod;  
    j_st = _PEN * mod;
  }else {
	  if(_PEN < res){
	    j_block = mod +1;
      j_st = _PEN * j_block;
	  }else{
		  j_block = mod;  
	    j_st = _PEN * mod + res;
	  }	
  }
  ldm_size = j_block * jpi;
	dma_len = ldm_size << 3;

	double s_trn[ldm_size], s_e3t_n[ldm_size], s_xksi[ldm_size] __attribute__ ((aligned(64)));
  jn_bais = jpzoo *jpi *jpj * jpk;
  j_bais = j_st *jpi;

  CRTS_dma_iget(&s_xksi[0], xksi + j_bais, dma_len, &dma_rply);
	D_COUNT++;

  for(jk = 0; jk < jpkbm1; jk++){
	  bais = jk * jpi *jpj + j_st *jpi;	
		CRTS_dma_iget(&s_trn[0], trn + jn_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_e3t_n[0], e3t_n + bais, dma_len, &dma_rply);
		D_COUNT += 2; 
		CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(ji = 0; ji < ldm_size; ji++){
			zzoo = dmax( 0.e0, s_trn[ji] );
			zzoobod = tmminz * zzoo * zzoo;
			s_xksi[ji] = s_xksi[ji] + (1-fdbod) * zzoobod * s_e3t_n[ji];
		}
	}
  CRTS_dma_put(xksi + j_bais, &s_xksi[0], dma_len);

}

void slave_bio2_(p2zbio_var *v1){
	p2zbio_var v;
	CRTS_dma_get(&v, v1, sizeof(p2zbio_var));
	asm volatile("memb\n\t");

	int jpi = v.jpi, jpj=v.jpj, jpk=v.jpk, jpkb =v.jpkb, jpjm1=v.jpjm1, jpkm1=v.jpkm1 , \
	    jpdet = v.jpdet - 1, jpzoo = v.jpzoo-1, jpphy=v.jpphy -1, jpno3=v.jpno3 -1, \
			jpnh4=v.jpnh4 -1, jpdom=v.jpdom -1;
	double aki = v.aki, psinut = v.psinut, tmumax= v.tmumax, rgamma= v.rgamma, fphylab=v.fphylab, rppz=v.rppz, \
	       taus=v.taus, aks=v.aks, rpnaz=v.rpnaz, rdnaz=v.rdnaz, tauzn=v.tauzn, fzoolab=v.fzoolab, tmminp=v.tmminp, tmminz=v.tmminz, \
				 fdbod=v.fdbod, taudn=v.taudn, fdetlab=v.fdetlab, redf=v.redf, reddom=v.reddom, taunn=v.taunn, akno3 = v.akno3, aknh4= v.aknh4, \
				 taudomn = v.taudomn;
  double *trn=v.trn, *tra=v.tra;
// local args
  double zlt=0. , zle ; 
	double zdet, zzoo,zphy,zno3,znh4, zdom, zlno3,zlnh4,zno3phy,znh4phy, zphydom,zphynh4, zpppz, zppdz, zfood, zfilpz, zfildz, \
	       zphyzoo, zdetzoo, zzoodet, zzoonh4, zzoodom, zphydet, zzoobod, zboddet, zdetnh4, zdetdom, zdomnh4, zdomaju, znh4no3, zphya, zzooa, zno3a, znh4a, zdeta, zdoma ;			 
  int ji, jj, jk, k_id, j_id, k_st, k_ed, j_st, j_ed, k_block, j_block;
	int op = 1;

	devide_sid(op, &k_id, &j_id);
	int k_dim = jpkm1 -jpkb + 1;
	area(6-op,  k_dim, k_id, &k_st, &k_block);
	area(op,  jpjm1, j_id, &j_st, &j_block);
	
	k_st = k_st + jpkb - 1;
  k_ed = k_st + k_block; 
	j_ed = j_st + j_block;
//if(_PEN == 0 && v.rank == 1) printf("jpkb: %d, jpkm1: %d, k_st:%d, k_block:%d, k_ed:%d\n", jpkb,jpkm1,k_st,k_block,k_ed); 
  int ldm_size = j_block * jpi;
	double s_trn_det[ldm_size], s_trn_zoo[ldm_size], s_trn_phy[ldm_size], s_trn_no3[ldm_size],s_trn_nh4[ldm_size], s_trn_dom[ldm_size] __attribute__ ((aligned(64))); 
	double s_tra_det[ldm_size], s_tra_zoo[ldm_size], s_tra_phy[ldm_size], s_tra_no3[ldm_size],s_tra_nh4[ldm_size], s_tra_dom[ldm_size] __attribute__ ((aligned(64))); 
	double tmp_div_refred = (1 - redf/reddom) ;  //(1 - redf/reddom)
  int jp_bais = jpi*jpj*jpk; 
	int j_bais = j_st * jpi;
	dma_len = jpi * j_block * 8;
  zlt   = 0.0  ;   
  zle   = 0.0  ;    
  zlno3 = 0.0  ;  
  zlnh4 = 0.0  ;

  zno3phy = 0.0;
  znh4phy = 0.0;
  zphydom = 0.0;
  zphynh4 = 0.0;

  zphyzoo = 0.0;      
  zdetzoo = 0.e0;

  zzoodet = 0.0;      
  zzoobod = 0.0;              
  zboddet = 0.0;             

	for(jk = k_st; jk < k_ed; jk++){
    bais = jk * jpi *jpj + j_bais;		
		CRTS_dma_iget(&s_trn_det[0], trn + jpdet*jp_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_trn_zoo[0], trn + jpzoo*jp_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_trn_phy[0], trn + jpphy*jp_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_trn_no3[0], trn + jpno3*jp_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_trn_nh4[0], trn + jpnh4*jp_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_trn_dom[0], trn + jpdom*jp_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_tra_det[0], tra + jpdet*jp_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_tra_zoo[0], tra + jpzoo*jp_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_tra_phy[0], tra + jpphy*jp_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_tra_no3[0], tra + jpno3*jp_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_tra_nh4[0], tra + jpnh4*jp_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_tra_dom[0], tra + jpdom*jp_bais + bais, dma_len, &dma_rply);
		D_COUNT += 12;
		CRTS_dma_wait_value(&dma_rply, D_COUNT);

    for(ji = 0; ji < ldm_size; ji++){
        zdet = dmax( 0., s_trn_det[ji] );
        zzoo = dmax( 0., s_trn_zoo[ji] );
        zphy = dmax( 0., s_trn_phy[ji] );
        zno3 = dmax( 0., s_trn_no3[ji] );
        znh4 = dmax( 0., s_trn_nh4[ji] );
        zdom = dmax( 0., s_trn_dom[ji] );


        zzoonh4 = tauzn * fzoolab * zzoo ; 
        zzoodom = tauzn * (1 - fzoolab) * zzoo;

        zphydet = tmminp * zphy ;                

        zdetnh4 = taudn * fdetlab * zdet;
        zdetdom = taudn * (1 - fdetlab) * zdet;

        zdomnh4 = taudomn * zdom;
        zdomaju = (1 - redf/reddom) * (zphydom + zzoodom + zdetdom);

        znh4no3 = taunn * znh4;

        zphya =   zno3phy + znh4phy - zphynh4 - zphydom - zphyzoo - zphydet;
        zzooa =   zphyzoo + zdetzoo - zzoodet - zzoodom - zzoonh4 - zzoobod;
        zno3a = - zno3phy + znh4no3 ;
        znh4a = - znh4phy - znh4no3 + zphynh4 + zzoonh4 + zdomnh4 + zdetnh4 + zdomaju;
        zdeta = zphydet + zzoodet  - zdetzoo - zdetnh4 - zdetdom + zboddet;
        zdoma = zphydom + zzoodom + zdetdom - zdomnh4 - zdomaju;

        s_tra_det[ji] = s_tra_det[ji] + zdeta;
        s_tra_zoo[ji] = s_tra_zoo[ji] + zzooa;
        s_tra_phy[ji] = s_tra_phy[ji] + zphya;
        s_tra_no3[ji] = s_tra_no3[ji] + zno3a;
        s_tra_nh4[ji] = s_tra_nh4[ji] + znh4a;
        s_tra_dom[ji] = s_tra_dom[ji] + zdoma;
				
		}

		CRTS_dma_iput(tra + jpdet*jp_bais + bais, &s_tra_det[0], dma_len, &dma_rply);
		CRTS_dma_iput(tra + jpzoo*jp_bais + bais, &s_tra_zoo[0], dma_len, &dma_rply);
		CRTS_dma_iput(tra + jpphy*jp_bais + bais, &s_tra_phy[0], dma_len, &dma_rply);
		CRTS_dma_iput(tra + jpno3*jp_bais + bais, &s_tra_no3[0], dma_len, &dma_rply);
		CRTS_dma_iput(tra + jpnh4*jp_bais + bais, &s_tra_nh4[0], dma_len, &dma_rply);
		CRTS_dma_iput(tra + jpdom*jp_bais + bais, &s_tra_dom[0], dma_len, &dma_rply);
    D_COUNT += 6;
		CRTS_dma_wait_value(&dma_rply, D_COUNT);
	}	
}

void slave_p2zsed_(p2zsed_var *v1){
	p2zsed_var v;
	CRTS_dma_get(&v, v1, sizeof(p2zsed_var));
	asm volatile("memb\n\t");;

	int jpi= v.jpi, jpj = v.jpj, jpk =v.jpk, jpkm1 =v.jpkm1, \
	    jpdet = v.jpdet - 1;
	double vsed = v.vsed;
  double *trn = v.trn, *tra =v.tra, *ztra=v.ztra, *e3t_n = v.e3t_n;
  
// local -->
  int ji,jj,jk, k_st, k_block, k_ed, ldm_size, jp_bais, n_bais;
	int offset_cur, offset_lst;

	area(6, jpkm1, _PEN, &k_st, &k_block);
	k_ed = k_st + k_block;

  ldm_size = jpi*jpj;
	dma_len = ldm_size << 3;
  jp_bais = jpdet *jpi * jpj *jpk;
	int pref_bais;
  double s_trn[2][ldm_size], s_tra[ldm_size], s_e3t_n[ldm_size] __attribute__ ((aligned(64)));
  double s_ztra;

#ifdef dsimd			
			doublev8 vtrn, ven, vztra, vtra;
			doublev8 vvsed=vsed;
#endif			

	for(jk = k_st; jk < k_ed; jk++){
		offset_cur = jk & 1; offset_lst = (jk + 1) & 1;
		if(jk == k_st){
			if(jk == 0){
				;
			}else{
			  pref_bais = (jk - 1) *jpi *jpj;
				CRTS_dma_iget(&s_trn[offset_lst][0], trn + jp_bais + pref_bais, dma_len, &dma_rply);
				D_COUNT++;
				CRTS_dma_wait_value(&dma_rply, D_COUNT);
			}
		}

		bais = jk *jpi *jpj;
		CRTS_dma_iget(&s_trn[offset_cur][0], trn + jp_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_tra[0], tra + jp_bais + bais, dma_len, &dma_rply);
		CRTS_dma_iget(&s_e3t_n[0], e3t_n + bais, dma_len, &dma_rply);
		D_COUNT +=3;
    CRTS_dma_wait_value(&dma_rply, D_COUNT);

		if(jk == 0){
#ifdef dsimd			
			doublev8 vtrn, ven, vztra, vtra;
			doublev8 vvsed=vsed;
			for(ji=0; ji < ldm_size - 8; ji+=8){
				simd_loadu(vtrn, &s_trn[offset_cur][ji]);
				simd_load(ven, &s_e3t_n[ji]);
				vztra = -vvsed * vtrn / ven;
				vtra = vtra + vztra;
				simd_store(vtra, &s_tra[ji]);
			}
		  for(; ji < ldm_size; ji++){
			  s_ztra = -vsed * s_trn[offset_cur][ji] / s_e3t_n[ji];
				s_tra[ji] = s_tra[ji] + s_ztra ;
			}	
#else			
		  for(ji = 0; ji < ldm_size; ji++){
			  s_ztra = -vsed * s_trn[offset_cur][ji] / s_e3t_n[ji];
				s_tra[ji] = s_tra[ji] + s_ztra ;
			}	
#endif			
		}else if(jk == jpkm1-1){
#ifdef dsimd
      for(ji=0; ji < ldm_size-8; ji+=8){
			  simd_loadu(vtrn, &s_trn[offset_lst][ji]);	
				simd_load(ven, &s_e3t_n[ji]);
        vztra = vvsed * vtrn / ven;
				vtra = vtra + vztra;
				simd_store(vtra, &s_tra[ji]);
			}
			for(; ji < ldm_size; ji++){
			  s_ztra = vsed * s_trn[offset_lst][ji] / s_e3t_n[ji];
				s_tra[ji] = s_tra[ji] + s_ztra ;
			}
#else			
			for(ji=0; ji < ldm_size; ji++){
			  s_ztra = vsed * s_trn[offset_lst][ji] / s_e3t_n[ji];
				s_tra[ji] = s_tra[ji] + s_ztra ;
			}
#endif			
    }else{
#ifdef dsimd
			doublev8 vtrnl;
      for(ji=0; ji < ldm_size-8; ji+=8){
			  simd_loadu(vtrn, &s_trn[offset_cur][ji]);	
			  simd_loadu(vtrnl, &s_trn[offset_lst][ji]);	
				simd_load(ven, &s_e3t_n[ji]);
        vztra = -vvsed * (vtrn - vtrnl) / ven;
				vtra = vtra + vztra;
				simd_store(vtra, &s_tra[ji]);
			}
		  for(; ji < ldm_size; ji++){
			  s_ztra = -vsed * (s_trn[offset_cur][ji] - s_trn[offset_lst][ji]) / s_e3t_n[ji] ;
				s_tra[ji] = s_tra[ji] + s_ztra  ;
			}	
#else			
		  for(ji = 0; ji < ldm_size; ji++){
			  s_ztra = -vsed * (s_trn[offset_cur][ji] - s_trn[offset_lst][ji]) / s_e3t_n[ji] ;
				s_tra[ji] = s_tra[ji] + s_ztra  ;
			}	
#endif			
		}
	  bais =  jk *jpi *jpj;
		CRTS_dma_put(tra + jp_bais + bais, &s_tra[0], dma_len );
	}
}

void slave_p2zexp1_(p2zexp_var *v1){
	p2zexp_var v;
	CRTS_dma_get(&v, v1, sizeof(p2zexp_var));
	asm volatile("memb\n\t");

	int jpi=v.jpi, jpj=v.jpj, jpk=v.jpk, fs_2=v.fs_2, fs_jpim1 = v.fs_jpim1, jpjm1 =v.jpjm1, jpkm1 =v.jpkm1;
	int jpno3 = v.jpno3-1;
  double *e3t_n=v.e3t_n, *tra=v.tra, *dmin3=v.dmin3, *xksi=v.xksi;
/********** local var **************/
  double ze3t;
	int ji,jj,jk,k_st, k_ed, k_block, d4_bais;
	area(6, jpkm1, _PEN, &k_st, &k_block);
  k_ed = k_st + k_block;
	int ldm_size = jpjm1 * jpi;
  dma_len = ldm_size << 3;
	double s_e3t_n[ldm_size], s_tra[ldm_size], s_dmin3[ldm_size], s_xksi[ldm_size] __attribute__ ((aligned(64)));
	d4_bais = jpno3* jpi *jpj *jpk;
	while(1){
	  if(*(v1->p2zexp1_sig) == 1)
			break;
	}
	CRTS_dma_iget(&s_xksi[0], xksi, dma_len, &dma_rply);
	D_COUNT++;
	for(jk=k_st; jk < k_ed; jk++){
		  bais = jk *jpi *jpj;
	  	CRTS_dma_iget(&s_e3t_n[0], e3t_n + bais, dma_len, &dma_rply);
	  	CRTS_dma_iget(&s_tra[0], tra + d4_bais + bais, dma_len, &dma_rply);
	  	CRTS_dma_iget(&s_dmin3[0], dmin3 + bais, dma_len, &dma_rply);
			D_COUNT += 3;
			CRTS_dma_wait_value(&dma_rply, D_COUNT);
#ifdef dsimd
			doublev8 vet3, vdmin3, vsi, vtra;
			doublev8 vone=1., vzet;  
			for(ji=0; ji < ldm_size-8; ji+=8){
			  	simd_load(vet3, &s_e3t_n[ji]);
					simd_load(vtra, &s_tra[ji]);
					simd_load(vdmin3, &s_dmin3[ji]);
					simd_load(vsi, &s_xksi[ji]);
					vzet = simd_vdivd(vone, vet3);
					vtra = vtra + vzet * vdmin3 * vsi;
					simd_store(vtra, &s_tra[ji]);
			}
			for(; ji < ldm_size; ji++){
			   ze3t = 1. / s_e3t_n[ji];
				 s_tra[ji] = s_tra[ji] + ze3t * s_dmin3[ji] * s_xksi[ji];
			}
#else			
			for(ji=0; ji < ldm_size; ji++){
			   ze3t = 1. / s_e3t_n[ji];
				 s_tra[ji] = s_tra[ji] + ze3t * s_dmin3[ji] * s_xksi[ji];
			}
#endif			

			CRTS_dma_put(tra + d4_bais + bais, &s_tra[0], dma_len);
	}
	
}

void slave_p2zexp2_(p2zexp_var *v1){
	p2zexp_var sv;
	CRTS_dma_get(&sv, v1, sizeof(p2zexp_var));
	asm volatile("memb\n\t");
	double zgeolpoc;

	int jpi = sv.jpi, jpj=sv.jpj, jpk=sv.jpk, fs_2=sv.fs_2, fs_jpim1 = sv.fs_jpim1, jpjm1 =sv.jpjm1, jpkm1 =sv.jpkm1;
  int jpno3 = sv.jpno3-1, jpdet = sv.jpdet-1, ikt = sv.ikt-1;
	double sedlam =sv.sedlam, vsed = sv.vsed, rdt=sv.rdt, sedlostpoc=sv.sedlostpoc, areacot = sv.areacot;
	double *tra=sv.tra, *sedpocn=sv.sedpocn, *e3t_n=sv.e3t_n, *trn=sv.trn, *zsedpoca=sv.zsedpoca, *dminl=sv.dminl,
	       *xksi=sv.xksi, *e1e2t=sv.e1e2t, *cmask=sv.cmask;
// local var --				 
  int ji, jj, jk, j_st,j_block, j_ed, k_st, k_block, k_ed;
	double zwork;
  int slave_tasks = 13;
	int mod = jpj / slave_tasks;
	int res = jpj % slave_tasks;

  if(res == 0){
	  j_block = mod;  
    j_st = _PEN * mod;
  }else {
	  if(_PEN < res){
	    j_block = mod +1;
      j_st = _PEN * j_block;
	  }else{
		  j_block = mod;  
	    j_st = _PEN * mod + res;
	  }	
  }
	
	if(_PEN >= slave_tasks) return;

	int ldm_size = j_block*jpi;
	dma_len = ldm_size *8 ;
	double s_tra[ldm_size], s_sedpocn[ldm_size], s_e3t_n[ldm_size], s_trn[ldm_size], s_zsedpoca[ldm_size], s_dminl[ldm_size], \
	       s_xksi[ldm_size], s_e1e2t[ldm_size]  __attribute__ ((aligned(64)));
	double s_tra_i1[ldm_size], s_cmask[ldm_size], s_e3t_n_i1[ldm_size] __attribute__ ((aligned(64))) ;			 
  int j_bais = j_st *jpi; 
	int k_bais = ikt *jpi *jpj;
	int d4_bais = jpno3 * jpi *jpj *jpk;
   
	CRTS_dma_iget(&s_tra[0], tra + d4_bais + k_bais + j_bais, dma_len, &dma_rply);
	CRTS_dma_iget(&s_trn[0], trn + d4_bais + k_bais + j_bais, dma_len, &dma_rply);
	CRTS_dma_iget(&s_sedpocn[0], sedpocn + j_bais, dma_len, &dma_rply);
	CRTS_dma_iget(&s_e3t_n[0], e3t_n + k_bais + j_bais, dma_len, &dma_rply);
	CRTS_dma_iget(&s_dminl[0], dminl + j_bais, dma_len, &dma_rply);
	CRTS_dma_iget(&s_xksi[0],  xksi+ j_bais, dma_len, &dma_rply);
	CRTS_dma_iget(&s_e1e2t[0], e1e2t + j_bais, dma_len, &dma_rply);
	D_COUNT += 7;
  zgeolpoc = 0.;
	CRTS_dma_wait_value(&dma_rply, D_COUNT);

	CRTS_dma_iget(&s_cmask[0], cmask + j_bais, dma_len, &dma_rply);
	D_COUNT++;
	if(ikt != 0) {
		CRTS_dma_iget(&s_tra_i1[0], tra + d4_bais + j_bais, dma_len, &dma_rply);		
	  CRTS_dma_iget(&s_e3t_n_i1[0], e3t_n + j_bais, dma_len, &dma_rply);
		D_COUNT+=2;
	}else{
	  memcpy(&s_e3t_n_i1[0], &s_e3t_n[0], dma_len);	
	}	
#ifdef dsimd1
	doublev8 vtra, vsedpocn, ve3t_n, vtrn, vdminl, vzsedpoca, vxksi, ve1e2t;
	doublev8 vsedlam=sedlam, vvsed=vsed, vrdt=rdt, vzgeolpoc=zgeolpoc,vsedlostpoc=sedlostpoc;
	doublev8 vzwork;
	for(ji=0; ji < ldm_size-8; ji+=8){
	  simd_load(vtra, &s_tra[ji]);	
		simd_load(vsedpocn, &s_sedpocn[ji]);
		simd_load(ve3t_n, &s_e3t_n[ji]);
		simd_load(vtrn, &s_trn[ji]);
		simd_load(vxksi, &s_xksi[ji]);
		simd_load(ve1e2t, &s_e1e2t[ji]);
		vtra = vtra + vsedlam * vsedpocn / ve3t_n;
		vzwork = vvsed * vtrn;
    vzsedpoca = (vzwork + vdminl * vxksi - vsedlam * vsedpocn - vsedlostpoc * vsedpocn ) *vrdt;
		vzgeolpoc = vzgeolpoc + vsedlostpoc * vsedpocn * ve1e2t;
		simd_store(vtra, &s_tra[ji]);
		simd_store(vzsedpoca, &s_zsedpoca[ji]);
	}
  zgeolpoc = simd_reduc_plusd(vzgeolpoc);

	for(; ji < ldm_size-8; ji++){
		 s_tra[ji] = s_tra[ji] + sedlam * s_sedpocn[ji] / s_e3t_n[ji];
     zwork = vsed * s_trn[ji];
		 s_zsedpoca[ji] = ( zwork + s_dminl[ji] * s_xksi[ji] \
		                  - sedlam * s_sedpocn[ji] - sedlostpoc * s_sedpocn[ji] ) * rdt;
		 zgeolpoc = zgeolpoc + sedlostpoc * s_sedpocn[ji] * s_e1e2t[ji];
	}
#else
	for(ji = 0; ji < ldm_size; ji++){
		 s_tra[ji] = s_tra[ji] + sedlam * s_sedpocn[ji] / s_e3t_n[ji];
     zwork = vsed * s_trn[ji];
		 s_zsedpoca[ji] = ( zwork + s_dminl[ji] * s_xksi[ji] \
		                  - sedlam * s_sedpocn[ji] - sedlostpoc * s_sedpocn[ji] ) * rdt;
		 zgeolpoc = zgeolpoc + sedlostpoc * s_sedpocn[ji] * s_e1e2t[ji];
	}
#endif	
    
	CRTS_dma_wait_value(&dma_rply, D_COUNT);
	CRTS_dma_iput(zsedpoca + j_bais, &s_zsedpoca[0], dma_len, &dma_rply);
	D_COUNT++;
	CRTS_dma_wait_value(&dma_rply, D_COUNT);

	if(ikt !=0 ){
	  CRTS_dma_iput(tra + d4_bais + k_bais + j_bais, &s_tra[0], dma_len, &dma_rply);
	  D_COUNT++;
	}else{
	  memcpy(&s_tra_i1[0], &s_tra[0], dma_len);	
	}

#ifdef dsimd1
	vzgeolpoc=zgeolpoc;
  for(ji = 0; ji < ldm_size - 8; ji+=8){
	  	simd_load(vtra_i1, &s_tra_i1[ji]);
			simd_load(vcmask, &s_cmask[ji]);
			simd_load(ve3t_n_i1, &s_e3t_n_i1[ji]);
			vtra_i1 = vtra_i1 + vzgeolpoc * vcmask /vareacot/ve3t_n_i1;
      simd_store(vtra_i1, &s_tra_i1[ji]);
	}
	for(; ji < ldm_size; ji++){		
	  s_tra_i1[ji] = s_tra_i1[ji] + zgeolpoc * s_cmask[ji] / areacot/ s_e3t_n_i1[ji];
	}
#else
	for(ji = 0; ji < ldm_size; ji++){		
	  s_tra_i1[ji] = s_tra_i1[ji] + zgeolpoc * s_cmask[ji] / areacot/ s_e3t_n_i1[ji];
	}
#endif	

	CRTS_dma_iput(tra + d4_bais + j_bais, &s_tra_i1[0], dma_len, &dma_rply);
	D_COUNT++;
	CRTS_dma_wait_value(&dma_rply, D_COUNT);

}

void zdfmxl_cache(zdfmxl_var *v_z) 
{
	//set_cache_size(0);
  zdfmxl_var v;
 	CRTS_dma_iget(&v, v_z, sizeof(zdfmxl_var), &dma_rply);
 	D_COUNT++;
 	CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");
	
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, jpim1 = v.jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1, nlb10 = v.nlb10;
  double zN2_c = v.zN2_c, avt_c = v.avt_c;
	int *mbkt = v.loc_mbkt, *nmln = v.loc_nmln;
	double *rn2b = v.loc_rn2b, *e3wn = v.loc_e3wn, *avt = v.loc_avt, *wmask = v.loc_wmask, *gdepwi = v.loc_gdepwi, *gdepwn = v.loc_gdepwn, *gdeptn = v.loc_gdeptn, *ssmask = v.loc_ssmask, *hmld = v.loc_hmld, *hmlp = v.loc_hmlp, *hmlpt = v.loc_hmlpt;
 
  int ji, jj, k, kn = jpkm1-1;
  int j_id, j_st, j_block;
	int off_c_p, off_n_p, off_c_n, off_n_n;
  int bias_c_p, bias_n_p, bias_c_n, bias_n_n, bias;
  int dma_len_i, dma_len_d;
	int iikn, iiki;

  area(6, jpjm1, _PEN, &j_st, &j_block);
	dma_len_d = j_block*jpi*8;
  dma_len_i = j_block*jpi*4;

  int local_mbkt[j_block][jpi], local_nmln[j_block][jpi];
	double local_rn2b[2][j_block][jpi], local_e3wn[2][j_block][jpi], local_avt[2][j_block][jpi], local_wmask[2][j_block][jpi], local_gdepwi[j_block][jpi], local_gdepwn[j_block][jpi], local_gdeptn[j_block][jpi], local_ssmask[j_block][jpi], local_hmld[j_block][jpi], local_hmlp[j_block][jpi], local_hmlpt[j_block][jpi];
  
	off_c_p = (nlb10-1) & 1; bias_c_p = (nlb10-1)*jpj*jpi; bias = j_st*jpi;
	off_c_n = (jpkm1-1) & 1; bias_c_n = (jpkm1-1)*jpj*jpi; 
	CRTS_dma_iget(&local_mbkt[0][0], mbkt+bias, dma_len_i, &dma_rply);
	CRTS_dma_iget(&local_ssmask[0][0], ssmask+bias, dma_len_d, &dma_rply);
	CRTS_dma_iget(&local_rn2b[off_c_p][0][0], rn2b+bias_c_p+bias, dma_len_d, &dma_rply);
	CRTS_dma_iget(&local_e3wn[off_c_p][0][0], e3wn+bias_c_p+bias, dma_len_d, &dma_rply);
	CRTS_dma_iget(&local_avt[off_c_n][0][0], avt+bias_c_n+bias, dma_len_d, &dma_rply);
	CRTS_dma_iget(&local_wmask[off_c_n][0][0], wmask+bias_c_n+bias, dma_len_d, &dma_rply);
	CRTS_dma_iget(&local_gdepwn[0][0], gdepwn+(nlb10-1)*jpj*jpi+bias, dma_len_d, &dma_rply);
	CRTS_dma_iget(&local_gdeptn[0][0], gdeptn+(nlb10-2)*jpj*jpi+bias, dma_len_d, &dma_rply);
	D_COUNT+=8;
	for(jj = 0; jj < j_block; jj++){
		for(ji = 0; ji < jpi; ji++){
			local_nmln[jj][ji] = nlb10;
			local_hmlp[jj][ji] = 0.0;
		}
	}
	CRTS_dma_wait_value(&dma_rply, D_COUNT);
	
	for(k = nlb10-1; k < jpkm1; k++){
		off_c_p = k & 1; off_n_p = (k+1) & 1; bias_n_p = (k+1)*jpj*jpi;
		if(k+1 < jpkm1){
			CRTS_dma_iget(&local_rn2b[off_n_p][0][0], rn2b+bias_n_p+bias, dma_len_d, &dma_rply);
			CRTS_dma_iget(&local_e3wn[off_n_p][0][0], e3wn+bias_n_p+bias, dma_len_d, &dma_rply);
			D_COUNT+=2;
		}
	  for(jj = 0; jj < j_block; jj++){
			for(ji = 0; ji < jpi; ji++){
				local_hmlp[jj][ji] = local_hmlp[jj][ji] + dmax(local_rn2b[off_c_p][jj][ji], 0.0)*local_e3wn[off_c_p][jj][ji];
				if(local_hmlp[jj][ji] < zN2_c){
					iikn = dmin(k, local_mbkt[jj][ji]-1) + 1;
					local_nmln[jj][ji] = iikn+1;
					local_gdepwn[jj][ji] = gdepwn[iikn*jpj*jpi+jj*jpi+ji]; // request Master Memory
					local_gdeptn[jj][ji] = gdeptn[(iikn-1)*jpj*jpi+jj*jpi+ji]; // request Master Memory
				}
			}
		}
		CRTS_dma_wait_value(&dma_rply, D_COUNT);
	}

	while(v_z->signal != 1){
  	flush_slave_cache();
	}
	CRTS_dma_get(&local_gdepwi[0][0], gdepwi+bias, dma_len_d);

	for(kn = jpkm1-1; kn >= nlb10-1; kn--){
		off_c_n = kn & 1; off_n_n = (kn-1) & 1; bias_n_n = (kn-1)*jpj*jpi;
		if(kn-1 >= nlb10-1){
			CRTS_dma_iget(&local_avt[off_n_n][0][0],  avt+bias_n_n+bias, dma_len_d, &dma_rply);
			CRTS_dma_iget(&local_wmask[off_n_n][0][0],  wmask+bias_n_n+bias, dma_len_d, &dma_rply);
			D_COUNT+=2;
		}
	  for(jj = 0; jj < j_block; jj++){
			for(ji = 0; ji < jpi; ji++){
				if(local_avt[off_c_n][jj][ji] < avt_c*local_wmask[off_c_n][jj][ji]){
					local_gdepwi[jj][ji] = gdepwn[kn*jpj*jpi+jj*jpi+ji]; // request Master Memory
				}
			}
		}
		CRTS_dma_wait_value(&dma_rply, D_COUNT);
	}

	for(jj = 0; jj < j_block; jj++){
		for(ji = 0; ji < jpi; ji++){
			local_hmld[jj][ji] = local_gdepwi[jj][ji]*local_ssmask[jj][ji];
			local_hmlp[jj][ji] = local_gdepwn[jj][ji]*local_ssmask[jj][ji];
			local_hmlpt[jj][ji] = local_gdeptn[jj][ji]*local_ssmask[jj][ji];
		}
	}

	CRTS_dma_iput(nmln+bias, &local_nmln[0][0], dma_len_i, &dma_rply);
	CRTS_dma_iput(hmld+bias, &local_hmld[0][0], dma_len_d, &dma_rply);
	CRTS_dma_iput(hmlp+bias, &local_hmlp[0][0], dma_len_d, &dma_rply);
	CRTS_dma_iput(hmlpt+bias, &local_hmlpt[0][0], dma_len_d, &dma_rply);
	D_COUNT+=4;
	CRTS_dma_wait_value(&dma_rply, D_COUNT);
	//set_cache_size(3);
}

void slave_zdfmxl_(zdfmxl_var *v_z){
  set_cache_size(0);
	zdfmxl_cache(v_z);
  set_cache_size(3);
}

void slave_zdfevd_(zdfevd_var *v_z) 
{
  zdfevd_var v;
 	CRTS_dma_iget(&v, v_z, sizeof(zdfevd_var), &dma_rply);
 	D_COUNT++;
 	CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, jpim1 = v.jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  double rn_evd = v.rn_evd;
	double *rn2 = v.loc_rn2, *rn2b = v.loc_rn2b, *wmask = v.loc_wmask, *pavm = v.loc_pavm, *pavt = v.loc_pavt, *zavm = v.loc_zavm, *zavt = v.loc_zavt;
 
  int ji, jj, k;
  int k_id, j_id, k_st, j_st, k_block, j_block, k_ed;
	int off_c, off_n;
  int bias_c, bias_n, bias;
	int dma_len_a;
	int jf = 0,jd;
 
  // must be 151 mode, other error
	devide_sid(1, &k_id, &j_id);
  area(5, jpkm1, k_id, &k_st, &k_block);
  area(1, jpjm1, j_id, &j_st, &j_block);
	k_ed = k_st + k_block;
	if(j_st == 0) { j_st = 1; j_block = j_block - 1; jf = 1; }
	dma_len = j_block*jpi*8;
	dma_len_a = (j_block+1)*jpi*8;

	double local_rn2[2][j_block][jpi], local_rn2b[2][j_block][jpi], local_wmask[2][j_block][jpi], local_pavm[2][j_block][jpi], local_pavt[2][j_block][jpi];
	double local_zavm[j_block+1][jpi], local_zavt[j_block+1][jpi];
	off_c = k_st & 1; bias_c = k_st*jpj*jpi; bias = j_st*jpi; 
	CRTS_dma_iget(&local_rn2[off_c][0][0], rn2+bias_c+bias, dma_len, &dma_rply);
	CRTS_dma_iget(&local_rn2b[off_c][0][0], rn2b+bias_c+bias, dma_len, &dma_rply);
	CRTS_dma_iget(&local_wmask[off_c][0][0],  wmask+bias_c+bias, dma_len, &dma_rply);
	CRTS_dma_iget(&local_pavm[off_c][0][0], pavm+bias_c+bias, dma_len, &dma_rply);
	CRTS_dma_iget(&local_pavt[off_c][0][0], pavt+bias_c+bias, dma_len, &dma_rply);
	D_COUNT+=5;
	CRTS_dma_wait_value(&dma_rply, D_COUNT);
	double tmp;
	for(k = k_st; k < k_ed; k++)
	{
		int k_n = k+1;
		off_c = k & 1; off_n = (k+1) & 1;
		bias_c = k*jpj*jpi; bias_n = (k+1)*jpj*jpi;
		if(k_n < k_ed)
		{
			CRTS_dma_iget(&local_rn2[off_n][0][0], rn2+bias_n+bias, dma_len, &dma_rply);
			CRTS_dma_iget(&local_rn2b[off_n][0][0], rn2b+bias_n+bias, dma_len, &dma_rply);
			CRTS_dma_iget(&local_wmask[off_n][0][0],  wmask+bias_n+bias, dma_len, &dma_rply);
			CRTS_dma_iget(&local_pavm[off_n][0][0], pavm+bias_n+bias, dma_len, &dma_rply);
			CRTS_dma_iget(&local_pavt[off_n][0][0], pavt+bias_n+bias, dma_len, &dma_rply);
			D_COUNT+=5;
		}
		for(jj = 0; jj < j_block; jj++)
		{
			jd = jj+jf;
			local_zavm[jd][0] = 0.0;
			local_zavm[jd][jpim1] = 0.0;
			local_zavt[jd][0] = 0.0;
			local_zavt[jd][jpim1] = 0.0;
			for(ji = 1; ji < jpim1; ji++)
			{
				if(dmin(local_rn2[off_c][jj][ji], local_rn2b[off_c][jj][ji]) <= -1.e-12)
				{
					tmp = rn_evd * local_wmask[off_c][jj][ji];
					local_zavm[jd][ji] = tmp - local_pavm[off_c][jj][ji]; 
					local_zavt[jd][ji] = tmp - local_pavt[off_c][jj][ji];
					local_pavm[off_c][jj][ji] = tmp;
					local_pavt[off_c][jj][ji] = tmp;
				}
				else
				{
					local_zavm[jj][ji] = 0.0; 
					local_zavt[jj][ji] = 0.0;
				}
			}
		}
		jd = (1-jf)*j_block;
		for(ji = 0; ji < jpi; ji++)
		{
			local_zavm[jd][ji] = 0.0;
			local_zavt[jd][ji] = 0.0;
		}
		CRTS_dma_iput(zavm+bias_c+(j_st-jf)*jpi, &local_zavm[0][0], dma_len_a, &dma_rply);
		CRTS_dma_iput(zavt+bias_c+(j_st-jf)*jpi, &local_zavt[0][0], dma_len_a, &dma_rply);
		CRTS_dma_iput(pavm+bias_c+bias, &local_pavm[off_c][0][0], dma_len, &dma_rply);
		CRTS_dma_iput(pavt+bias_c+bias, &local_pavt[off_c][0][0], dma_len, &dma_rply);
		D_COUNT+=4;
		CRTS_dma_wait_value(&dma_rply, D_COUNT);
	}

}

// NEW
void slave_dynzad_(dynzad_var *v_z) 
{
  dynzad_var v;
  CRTS_dma_iget(&v, v_z, sizeof(dynzad_var), &dma_rply);
  D_COUNT++;
  CRTS_dma_wait_value(&dma_rply, D_COUNT);
  asm volatile("memb\n\t");

	double *un = v.loc_un, *vn = v.loc_vn, *wn = v.loc_wn, *e1e2t = v.loc_e1e2t, *zwuw = v.loc_zwuw, *zwvw = v.loc_zwvw;
	double *r1e1e2u = v.loc_r1e1e2u, *r1e1e2v = v.loc_r1e1e2v, *e3un = v.loc_e3un, *e3vn = v.loc_e3vn, *ua = v.loc_ua, *va = v.loc_va;
  int jpi = v.jpi, jpj = v.jpj, jpk = v.jpk, fs_2 = v.fs_two, fs_jpim1 = v.fs_jpim1, jpjm1 = v.jpjm1, jpkm1 = v.jpkm1;
  
  int ji, jj, k, kf = 0;
  int k_id, j_id, k_st, j_st, k_block, j_block, k_ed, k_edd, kcnt = 0;
  int off_l, off_c, off_n, off_temp;
	int off_d, off_u;
  int bias_l, bias_c, bias_n, bias;
	int dma_len_a;
  
	devide_sid(3, &k_id, &j_id);
  area(3, jpkm1-1, k_id, &k_st, &k_block);
  area(3, jpjm1, j_id, &j_st, &j_block);
	if(j_st == 0) { j_st = 1; j_block = j_block - 1; }
	if(k_st == 0) { k_st = 1; k_block = k_block - 1; }
	k_ed = k_st + k_block;
	if(k_ed == jpkm1-1) kf = 1; 
	k_edd = k_ed + kf;
	dma_len = j_block*jpi*8;
	dma_len_a = (j_block+1)*jpi*8;
  double local_un[2][j_block][jpi], local_vn[2][j_block][jpi], local_wn[2][j_block+1][jpi], local_e1e2t[j_block+1][jpi], local_zwuw[j_block][jpi], local_zwvw[j_block][jpi];
	double local_r1e1e2u[j_block][jpi], local_r1e1e2v[j_block][jpi], local_e3un[2][j_block][jpi], local_e3vn[2][j_block][jpi], local_ua[2][j_block][jpi], local_va[2][j_block][jpi];
	double zwuwc, zwvwc;

  off_c = (k_st-1) & 1; off_n = k_st & 1;
	off_d = k_st & 1; off_u = (k_st+1) & 1;
	bias_c = (k_st-1)*jpj*jpi; bias_n = k_st*jpj*jpi; bias = j_st*jpi;
	CRTS_dma_iget(&local_un[off_c][0][0], un+bias_c+bias, dma_len, &dma_rply);
	CRTS_dma_iget(&local_un[off_n][0][0], un+bias_n+bias, dma_len, &dma_rply);
	CRTS_dma_iget(&local_vn[off_c][0][0], vn+bias_c+bias, dma_len, &dma_rply);
	CRTS_dma_iget(&local_vn[off_n][0][0], vn+bias_n+bias, dma_len, &dma_rply);
	CRTS_dma_iget(&local_wn[off_d][0][0], wn+bias_n+bias, dma_len_a, &dma_rply);
	CRTS_dma_iget(&local_e1e2t[0][0], e1e2t+bias, dma_len_a, &dma_rply);
	D_COUNT+=6;
	CRTS_dma_wait_value(&dma_rply, D_COUNT);
	CRTS_dma_iget(&local_r1e1e2u[0][0], r1e1e2u+bias, dma_len, &dma_rply);
	CRTS_dma_iget(&local_r1e1e2v[0][0], r1e1e2v+bias, dma_len, &dma_rply);
	CRTS_dma_iget(&local_e3un[off_d][0][0], e3un+bias_n+bias, dma_len, &dma_rply);
	CRTS_dma_iget(&local_e3vn[off_d][0][0], e3vn+bias_n+bias, dma_len, &dma_rply);
	CRTS_dma_iget(&local_ua[off_d][0][0], ua+bias_n+bias, dma_len, &dma_rply);
	CRTS_dma_iget(&local_va[off_d][0][0], va+bias_n+bias, dma_len, &dma_rply);
	D_COUNT+=6;
	for(jj = 0; jj < j_block; jj++)
	{
		for(ji = fs_2-1; ji < fs_jpim1; ji++)
		{
			local_zwuw[jj][ji] = 0.25*(local_e1e2t[jj][ji+1]*local_wn[off_d][jj][ji+1] + local_e1e2t[jj][ji]*local_wn[off_d][jj][ji])*(local_un[off_c][jj][ji] - local_un[off_n][jj][ji]);
			local_zwvw[jj][ji] = 0.25*(local_e1e2t[jj+1][ji]*local_wn[off_d][jj+1][ji] + local_e1e2t[jj][ji]*local_wn[off_d][jj][ji])*(local_vn[off_c][jj][ji] - local_vn[off_n][jj][ji]);
		}
	}
	CRTS_dma_wait_value(&dma_rply, D_COUNT);
	for(k = k_st; k < k_ed; k++)
	{
  	off_c = k & 1; off_n = (k+1) & 1;
		off_d = off_c; off_u = off_n;
		bias_c = k*jpj*jpi; bias_n = (k+1)*jpj*jpi;
		CRTS_dma_iget(&local_un[off_n][0][0], un+bias_n+bias, dma_len, &dma_rply);
		CRTS_dma_iget(&local_vn[off_n][0][0], vn+bias_n+bias, dma_len, &dma_rply);
		CRTS_dma_iget(&local_wn[off_u][0][0], wn+bias_n+bias, dma_len_a, &dma_rply);
		D_COUNT+=3;
		CRTS_dma_wait_value(&dma_rply, D_COUNT);
		if(k+1 < k_edd)
		{
			CRTS_dma_iget(&local_e3un[off_u][0][0], e3un+bias_n+bias, dma_len, &dma_rply);
			CRTS_dma_iget(&local_e3vn[off_u][0][0], e3vn+bias_n+bias, dma_len, &dma_rply);
			CRTS_dma_iget(&local_ua[off_u][0][0], ua+bias_n+bias, dma_len, &dma_rply);
			CRTS_dma_iget(&local_va[off_u][0][0], va+bias_n+bias, dma_len, &dma_rply);
			D_COUNT+=4;
		}
		for(jj = 0; jj < j_block; jj++)
		{
			for(ji = fs_2-1; ji < fs_jpim1; ji++)
			{
				zwuwc = local_zwuw[jj][ji]; 
				zwvwc = local_zwvw[jj][ji];
				local_zwuw[jj][ji] = 0.25*(local_e1e2t[jj][ji+1]*local_wn[off_u][jj][ji+1] + local_e1e2t[jj][ji]*local_wn[off_u][jj][ji])*(local_un[off_c][jj][ji] - local_un[off_n][jj][ji]);
				local_zwvw[jj][ji] = 0.25*(local_e1e2t[jj+1][ji]*local_wn[off_u][jj+1][ji] + local_e1e2t[jj][ji]*local_wn[off_u][jj][ji])*(local_vn[off_c][jj][ji] - local_vn[off_n][jj][ji]);
				local_ua[off_d][jj][ji] = local_ua[off_d][jj][ji] - (zwuwc + local_zwuw[jj][ji])*local_r1e1e2u[jj][ji]/local_e3un[off_d][jj][ji];
				local_va[off_d][jj][ji] = local_va[off_d][jj][ji] - (zwvwc + local_zwvw[jj][ji])*local_r1e1e2v[jj][ji]/local_e3vn[off_d][jj][ji];
			}
		}
		CRTS_dma_iput(ua+bias_c+bias, &local_ua[off_d][0][0], dma_len, &dma_rply);
		CRTS_dma_iput(va+bias_c+bias, &local_va[off_d][0][0], dma_len, &dma_rply);
		D_COUNT+=2;
		CRTS_dma_wait_value(&dma_rply, D_COUNT);
	}
	if(kf == 1)
	{
		for(jj = 0; jj < j_block; jj++)
		{
			for(ji = fs_2-1; ji < fs_jpim1; ji++)
			{
				zwuwc = local_zwuw[jj][ji]; 
				zwvwc = local_zwvw[jj][ji];
				local_ua[off_u][jj][ji] = local_ua[off_u][jj][ji] - zwuwc*local_r1e1e2u[jj][ji]/local_e3un[off_u][jj][ji];
				local_va[off_u][jj][ji] = local_va[off_u][jj][ji] - zwvwc*local_r1e1e2v[jj][ji]/local_e3vn[off_u][jj][ji];
			}
		}
		CRTS_dma_iput(ua+bias_n+bias, &local_ua[off_u][0][0], dma_len, &dma_rply);
		CRTS_dma_iput(va+bias_n+bias, &local_va[off_u][0][0], dma_len, &dma_rply);
		D_COUNT+=2;
		CRTS_dma_wait_value(&dma_rply, D_COUNT);
	}
 
}
