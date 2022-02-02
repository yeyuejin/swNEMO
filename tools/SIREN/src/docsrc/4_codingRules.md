# Coding Rules

The conventions used in SIREN coding are based on the NEMO coding rules
(see [NEMO coding
conventions](http://www.nemo-ocean.eu/content/download/15483/73221/file/NEMO_coding.conv_v3.pdf)).<br/>
However some modifications were added to improve readibility of the code.<br/>
Some of the NEMO coding rules are reminded here, and extensions are described.

@tableofcontents

# Fortran Standard {#std}

SIREN software adhere to strict __FORTRAN 95__ standard.<br/>
There is only one exception. The use of functions _COMMAND_ARGUMENT_COUNT_ and
_GET_COMMAND_ARGUMENT_.<br/>
There exist no equivalent for those Fortran 03 intrinsec functions in Fortran
95.<br/> At least none convenient for compilers tested (see @ref md_src_docsrc_1_install). 

#  Free Form Source {#free}
Free Form Source will be used, however a self imposed limit of 80 should
enhance readibility.

#  Indentation {#indent}
Code as well as comments lines will be indented 3 characters for readibility.<br/>
__Indentation should be write without hard tabs__.

Example for vi  :
~~~~~~~~~~~~~~~~~~~~~
:set expandtab tabstop=3 shiftwidth=3
~~~~~~~~~~~~~~~~~~~~~

#  Naming conventions : variable {#namvar}
All variables should be named as explicitly as possible.<br/>
The naming conventions concerns prefix letters of these name, in order to
identify the variable type and status.<br/> It must be composed of two
letters defining type and status follow by an underscore.<br/>
table below list the starting letters to be used for variable naming,
depending on their type and status.

 | Type / Status     | byte (integer(1)) <br/> __b__ | short (integer(2)) <br/> __s__ | integer(4) <br/> __i__  | integer(8) <br/> __k__ | real(4) <br/> __r__ | real(8) <br/> __d__ | logical <br/> __l__ | character <br/> __c__ | complex <br/> __y__ | structure <br/> __t__  |
 | :----:            | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: | :----: |
 |global <br/> __g__           | bg_  | sg_  | ig_  | kg_  | rg_  | dg_  | lg_  | cg_  | yg_  | tg_  |
 |global parameter <br/> __p__ | bp_  | sp_  | ip_  | kp_  | rp_  | dp_  | lp_  | cp_  | yp_  | tp_  |
 |module <br/> __m__           | bm_  | sm_  | im_  | km_  | rm_  | dm_  | lm_  | cm_  | ym_  | tm_  |
 |namelist <br/> __n__         | bn_  | sn_  | in_  | kn_  | rn_  | dn_  | ln_  | cn_  | yn_  | tn_  |
 |dummy argument <br/> __d__   | bd_  | sd_  | id_  | kd_  | rd_  | dd_  | ld_  | cd_  | yd_  | td_  |
 |local <br/> __l__            | bl_  | sl_  | il_  | kl_  | rl_  | dl_  | ll_  | cl_  | yl_  | tl_  |
 |function result <br/> __f__  | bf_  | sf_  | if_  | kf_  | rf_  | df_  | lf_  | cf_  | yf_  | tf_  |
 |loop control                 |      |      | j?   |      |      |      |      |      |      |      |

# Naming conventions : structure {#namstr}
The structure name should be written in capital letter, and start with
__T__<br/> Example: TTRACER <br/>
Variables inside the structure should be named as explicitly as possible.<br/>
For those variables, the prefix naming conventions only concern the type of variable.<br/>
It must be composed of one letter defining type follows by an
underscore.<br/>
see table of variable conventions.<br/>

Example: __tl\_type\%i\_year__<br/> _year_ is an integer(4) variable in a local strucure
named _type_. <br/>

# Naming conventions : function-subroutine {#namsub}
Functions or Subroutines are defined in a module.<br/>
Their name should start with the module name then with their "functional" name. So it will be
easy to find it.<br/>
Example:<br/> a function to realise addition written in a module
__math__ should be called __math\_add__.<br/>

__PUBLIC__  function or subroutine should used one undescrore: _math_add_<br/>
__PRIVATE__ function or subroutine should used two undescrores: _math__add_<br/>

# Precision {#precision}
__All variables should make use of kinds__.<br/>
Numerical constant need to have a suffix of __kindvalue__

# Declaration for global variable and constants {#global}
All global data must be accompanied with a comment field on the same
line.<br/>
_Note that using doxygen (see [header](#header)), we will use symbol !< instead of !: as separator_

# Implicit none {#implicit}
All subroutines and functions will include an IMPLICIT NONE statement.

# Header {#header}

SIREN use __doxigen auto documentation__ tool.<br/>
Information could be find on
[doxygen](http://www.stack.nl/~dimitri/doxygen/index.html) web page.<br/>
Some basic tag are described
[here](http://www.msg.chem.iastate.edu/gamess/DoxygenRules.oct10.pdf).

 <HR>
   <b>
   - @ref index
   - @ref md_src_docsrc_1_install
   - @ref md_src_docsrc_2_quickstart
   - @ref md_src_docsrc_3_support_bug
   - @ref md_src_docsrc_5_changeLog
   - @ref todo
   </b>
