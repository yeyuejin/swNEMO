/*************************************************************************
	> File Name: src/MY/struct.h
	> Author: zhousc
	> Created Time: 2020年09月22日 星期二 20时27分58秒
 ************************************************************************/

typedef struct {
  int jpi, jpj, jpk, fs_two, fs_jpim1, jpjm1, jpkm1, jn, rank;
  double p2dt;
  double *loc_pun, *loc_pvn, *loc_ptb, *loc_zwx, *loc_zwy, *loc_zwz, \
         *loc_pwn, *loc_wmask, *loc_e3t_n, *loc_e3t_a, *loc_e3t_b, \
         *loc_r1_e1e2t, *loc_tmask, *loc_pta, *loc_zwi, *loc_ptn;
} var1;

typedef struct {
  int jpi, jpj, jpk, fs_two, fs_jpim1, jpjm1, jpkm1, jn, rank;
  double p2dt, zrtrn, zbig;
  double *loc_paa, *loc_pbb, *loc_pcc, *loc_paft, *loc_zbup, *loc_zbdo, *loc_zbetup, *loc_zbetdo, *loc_e1e2t, *loc_e3t_n, *loc_pbef, *loc_tmask;
} var2;

typedef struct {
  int jpi, jpj, jpk, fs_two, fs_jpim1, jpjm1, jpkm1, jn, rank, kt;
  double zsign;
  double *loc_wmask, *loc_umask, *loc_vmask, *loc_pahu, *loc_pahv, *loc_wslpi, *loc_wslpj, *loc_ah_wslp2, *loc_zdit, *loc_zdjt, *loc_ptb, *loc_e2_e1u, *loc_e3u_n, *loc_e1_e2v, *loc_e3v_n, *loc_e2u, *loc_uslp, *loc_e1v, *loc_vslp, *loc_pta, *loc_r1_e1e2t, *loc_e3t_n, *loc_e2t, *loc_e1t, *loc_e1e2t, *loc_e3w_n, *loc_akz, *loc_ztfw, *loc_zzz;
;
} var3;

typedef struct {
  int jpi, jpj, jpk, fs_two, fs_jpim1, jpjm1, jpkm1, jn, rank, kt;
  double p2dt;
  double *loc_pta, *loc_e3t_b, *loc_ptb, *loc_e3t_n, *loc_zwi, *loc_zwt, *loc_tmask, *loc_zws, *loc_zwd, *loc_e3w_n, *loc_e3t_a;
;
} var4;

typedef struct {
  int jpi, jpj, jpk, fs_two, fs_jpim1, jpjm1, jpkm1, rank, kt;
  double z1_slpmax, zeps, zm1_2g, z1_16, zm1_g;
  double *loc_zgru, *loc_zgrv, *loc_r1_e1u, *loc_r1_e2v, *loc_zdzr, *loc_e3u_n, *loc_e3v_n, *loc_omlmask, *loc_gdept_n, *loc_risfdep, *loc_zwz, *loc_zslpml_hmlpu, *loc_umask, *loc_vmask, *loc_zslpml_hmlpv, *loc_zww, *loc_e3un, *loc_e3vn, *loc_tmask;
  double *loc_pn2, *loc_prd, *loc_e1t, *loc_e2t, *loc_wmask, *loc_e3w_n, *loc_gdepw_n, *loc_gdepw_n_tmp, *loc_hmlp, *loc_wslpiml, *loc_wslpjml, *loc_uslp, *loc_vslp, *loc_wslpi, *loc_wslpj;
} var5;

typedef struct {
  int jpi, jpj, jpk, jpkm1, rank, kt, jpphy;
  double zcoef, xkr0, xkrp, xlr, xkg0, xkgp, xlg, tiny0;
  double *loc_trn, *loc_zparr, *loc_e3t_n, *loc_zparg, *loc_etot;
} p2z_var;

typedef struct {
  int jpi, jpj, jpk, jpkm1, narea, jp_tem, jp_sal;
  double r1_Z0, r1_T0, rdeltaS, r1_S0, r1_rau0;
  double *loc_gdept_n, *loc_pts, *loc_tmask, *loc_pab;
  double ALP000, ALP001, ALP002, ALP003, ALP010, ALP011, ALP012, ALP020, ALP021, ALP030, ALP031, ALP040, ALP050, ALP100, ALP101, ALP102, ALP110, ALP111, ALP120, ALP121, ALP130, ALP140, ALP200, ALP201, ALP210, ALP211, ALP220, ALP230, ALP301, ALP310, ALP320, ALP400, ALP410, ALP500, ALP300;
  double BET000, BET001, BET002, BET003, BET010, BET011, BET012, BET020, BET021, BET030, BET031, BET040, BET050, BET100, BET101, BET102, BET110, BET111, BET120, BET121, BET130, BET140, BET200, BET201, BET210, BET211, BET220, BET230, BET301, BET310, BET320, BET400, BET410, BET500, BET300;
} rab_var;

typedef struct {
  int jpi, jpj, jpk, jpkm1, narea, jp_tem, jp_sal;
  double r1_Z0, r1_T0, rdeltaS, r1_S0, r1_rau0;
  double *loc_pdep, *loc_pts, *loc_tmask, *loc_prhop, *loc_prd;
  double EOS000, EOS001, EOS002, EOS003, EOS010, EOS011, EOS012, EOS013, EOS020, EOS021, EOS022, EOS030, EOS031, EOS040, EOS041, EOS050, EOS060, EOS100, EOS101, EOS102, EOS103, EOS110, EOS111, EOS112, EOS120, EOS121, EOS130, EOS131, EOS140, EOS150, EOS200, EOS201, EOS202, EOS210, EOS211, EOS220, EOS221, EOS230, EOS240, EOS300, EOS301, EOS310, EOS311, EOS320, EOS330, EOS400, EOS401, EOS410, EOS420, EOS500, EOS510, EOS600;
} pot_var;

typedef struct {
  int jpi, jpj, jpk, jpkm1, narea, jp_tem, jp_sal;
  double grav;
  double *loc_gdepw_n, *loc_gdept_n, *loc_pab, *loc_pts, *loc_e3w_n, *loc_wmask, *loc_pn2;
} bn2_var;

typedef struct {
  int jpi, jpj, jpk, jpjm1, jpkm1, fs_two, fs_jpim1, narea;
  double r2dt, r_vvl, zdt, rau0;
  double *loc_ub, *loc_ua, *loc_umask, *loc_vmask, *loc_vb, *loc_va, *loc_ua_b, *loc_va_b, *loc_e3u_n, *loc_e3u_a, *loc_avm, *loc_e3uw_n, *loc_wumask, *loc_zwi, *loc_zws, *loc_zwd, *loc_utau_b, *loc_utau, *loc_e3v_n, *loc_e3v_a, *loc_e3vw_n, *loc_wvmask, *loc_vtau, *loc_vtau_b;
} dyn_var;

typedef struct {
  int jpi, jpj, jpk, fs_two, fs_jpim1, jpjm1, jpkm1, narea;
  double rn_bshear, ri_cri, zfact1, zfact2, zfact3, rdt, rn_emin;
  double *loc_rn2b, *loc_p_avm, *loc_p_sh2, *loc_apdlr, *loc_tmask, *loc_p_e3t, *loc_p_e3w, *loc_zd_up, *loc_zd_lw, *loc_zdiag, *loc_dissl, *loc_wmask, *loc_en, *loc_p_avt, *loc_rn2, *loc_zpelc, *loc_pdepw;
} tke_var;

typedef struct{
  int jpi,jpj,jpk,kjpt;
	int jn, jpkm1, jpjm1, fs_2,fs_jpim1,rank;
	double ztn, atfp;
	double *ptn, *pta, *ptb;
} var_nxt_fix;

typedef struct{
  int jpi,jpj,jpk;
	int jn, jpkm1, jpjm1, fs_2, fs_jpim1, rank;
  double *hdivn, *e2u, *e3u_n, *un, *e1v, *e3v_n, *vn,
	       *r1_e1e2t, *e3t_n; 
} var_div_hor;

typedef struct {
  int jpi, jpj, jpk, fs_two, fs_jpim1, jpjm1, jpkm1, narea;
  double zsign;
  double *loc_ahmf, *loc_e3f_n, *loc_r1_e1e2f, *loc_e2v, *loc_pvb, *loc_e1u, *loc_pub, *loc_ahmt, *loc_e3t_b, *loc_r1_e1e2t, *loc_e2u, *loc_e3u_b, *loc_e1v, *loc_e3v_b, *loc_pua, *loc_pva, *loc_r1_e2u, *loc_r1_e1u, *loc_r1_e1v, *loc_r1_e2v, *loc_e3u_n, *loc_e3v_n;
} lap_var;

typedef struct {
  int jpi, jpj, jpk, jpjm1, jpkm1, narea;
  int flag;
  double atfp;
  double *loc_e3u_a, *loc_e3v_a, *loc_ua, *loc_va, *loc_umask, *loc_vmask, *loc_r1_hu_a, *loc_r1_hv_a, *loc_ua_b, \
         *loc_va_b, *loc_e3u_b, *loc_e3v_b, *loc_un, *loc_vn, *loc_ub, *loc_vb, *loc_un_b, *loc_vn_b, *loc_ub_b, \
         *loc_vb_b, *loc_r1_hu_b, *loc_r1_hv_b;
} dynnxt_var;

typedef struct {
  int jpi, jpj, jpk, fs_two, fs_jpim1, jpjm1, jpkm1, narea;
  double r1_4;
  double *loc_ff_f, *loc_e2v, *loc_e1u, *loc_r1_e1e2f, *loc_e2u, *loc_e1v, *loc_r1_e1u, *loc_r1_e2v, *loc_pvn, *loc_pun, *loc_pua, *loc_pva;
} vor_var;

typedef struct {
  int jpi, jpj, jpk, jpkm1, narea;
  double z1_2dt;
  double *loc_wn, *loc_e3t_n, *loc_hdivn, *loc_e3t_a, *loc_e3t_b, *loc_tmask;
} wzv_var;

typedef struct{
  int jpi, jpj, jpk, fs_2, fs_jpim1, jpjm1, jpkm1, jn,kt, rank;
  volatile unsigned long *spgts4_sig, *spgts5_sig;
  double grav, r1_2dt_b, za1, za2, za3;
  double *zu_frc, *zv_frc, *e3u_n, *e3v_n, \
         *ua, *va, *umask, *vmask, \
         *r1_hu_n, *r1_hv_n, *zhu, *zhv, *un_b, \
         *vn_b, *hu_n, *hv_n, *e2u, *e1v,       \
         *zu_trd, *ssumask, *zv_trd, *ssvmask,  \
         *ua_b, *ub_b, *va_b, *vb_b,            \
         *un_adv, *vn_adv, *un, *vn, *ubb_e,    \
         *ub_e, *un_e, *ua_e, *vbb_e, *vb_e,    \
         *vn_e, *va_e, *sshbb_e, *sshb_e, *sshn_e, \
         *ssha_e, *ssha, *zhup2_e, *zhvp2_e ;
} var_dynspg_ts;

typedef struct {
	int jpi, jpj, jpk, jpjm1, jpkm1, jpim1;
	double *loc_pub, *loc_pvb, *loc_pun, *loc_pvn, *loc_e3uw_n, *loc_e3uw_b, *loc_p_avm, *loc_e3vw_n, *loc_e3vw_b, *loc_wumask, *loc_wvmask, *loc_p_sh2, *loc_umask, *loc_vmask;
} zdf_sh2_t;

typedef struct {
	int jpi, jpj, jpk, fs_two, fs_jpim1, jpjm1, jpkm1, narea;
	double *loc_un, *loc_vn, *loc_wn, *loc_e1e2t, *loc_zwuw, *loc_zwvw, *loc_r1e1e2u, *loc_r1e1e2v, *loc_e3un, *loc_e3vn, *loc_ua, *loc_va;
} dynzad_var;

typedef struct {
	int jpi, jpj, jpk, fs_two, fs_jpim1, jpjm1, jpkm1, narea;
	int jp_tem, jp_sal;
	double r1_S0, r1_T0, r1_Z0, rdeltaS, r1_rau0;
	double EOS013, EOS103, EOS003, EOS022, EOS112, EOS012, EOS202, EOS102, EOS002, EOS041, EOS131, EOS031, EOS221, EOS121, EOS021, EOS311, EOS211, EOS111, EOS011, EOS401, EOS301, EOS201, EOS101, EOS001, EOS060, EOS150, EOS050, EOS240, EOS140, EOS040, EOS330, EOS230, EOS130, EOS030, EOS420, EOS320, EOS220, EOS120, EOS020, EOS510, EOS410, EOS310, EOS210, EOS110, EOS010, EOS600, EOS500, EOS400, EOS300, EOS200, EOS100, EOS000;
	double *loc_pts, *loc_pdep, *loc_tmask, *loc_prd;
} insitu_var;

typedef struct {
  int jpi, jpj, jpk, fs_two, fs_jpim1, jpjm1, jpkm1, narea;
  double *loc_un, *loc_vn, *loc_ua, *loc_va, *loc_e1u, *loc_e2v;
} keg_var;

typedef struct {
  int jpi, jpj, jpk, fs_two, fs_jpim1, jpjm1, jpkm1, narea, nksr, jp_tem;
  double zz0, xsi0r, zz1, xsi1r, z1_2;
  double *loc_gdepw_n, *loc_qsr, *loc_wmask, *loc_qsr_hc_b, *loc_e3t_n, *loc_qsr_hc, *loc_tsa;
} qsr_var;

typedef struct{
  int jpi, jpj,jpk, jpkb, jpkbm1, jpkm1, jpjm1, fs_jpim1, fs_2, jpdet, \
      jpzoo, jpphy, jpno3, jpnh4, jpdom, rank;
  double aki, psinut, tmumax, rgamma, fphylab, rppz, taus, \
         aks, rpnaz, rdnaz,tauzn , fzoolab, tmminp, tmminz,     \
         fdbod, taudn, fdetlab, redf, reddom, taunn, akno3, \
				 aknh4, taudomn ;
  double *trn, *etot, *tra, *xksi, *e3t_n;
} p2zbio_var;

typedef struct{
  int jpi,jpj,jpk,jpkm1, jpdet, rank ;
  double vsed;
  double *trn, *tra, *ztra, *e3t_n;
} p2zsed_var ;

typedef struct{
  int jpi,jpj,jpk,fs_2,fs_jpim1,jpjm1,jpkm1, rank;
  int jpno3, jpdet, ikt;
  double sedlam, vsed, rdt, sedlostpoc, zgeolpoc, areacot;
  double *e3t_n, *tra, *dmin3, *xksi, *mbkt, *sedpocn, *trn, \
         *zsedpoca, *dminl, *e1e2t, *cmask;
	volatile int *p2zexp1_sig;			 
} p2zexp_var;										 

typedef struct {
	int jpi, jpj, jpk, jpim1, jpjm1, jpkm1, nlb10, narea, signal;
	double zN2_c, avt_c;
  int *loc_mbkt, *loc_nmln;
	double *loc_rn2b, *loc_e3wn, *loc_avt, *loc_wmask, *loc_gdepwi, *loc_gdepwn, *loc_gdeptn, *loc_ssmask, *loc_hmld, *loc_hmlp, *loc_hmlpt;
} zdfmxl_var;

typedef struct {
	int jpi, jpj, jpk, jpim1, jpjm1, jpkm1, narea;
	double rn_evd;
	double *loc_rn2, *loc_rn2b, *loc_wmask, *loc_pavm, *loc_pavt, *loc_zavm, *loc_zavt;
} zdfevd_var;
