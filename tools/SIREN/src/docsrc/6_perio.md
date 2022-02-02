# NEMO periodicity

NEMO periodicity is defined as follow :<br/> 
<dl>
<dt>closed boundary (in_perio=0)</dt>
<dd>ghost cells (solid walls) are imposed at all model boundaries.</dd> 
@image html perio0_20.png 
<center>@image latex perio0_20.png 
</center>
<dt>cyclic east-west boundary (in_perio=1)</dt>
<dd>first and last rows are closed, whilst the first column is set to the value of the last-but-one column and the last column to the value of the second one.</dd>  
@image html perio1_20.png 
<center>@image latex perio1_20.png 
</center>
<dt>symmetric boundary condition across the equator. (in_perio=2)</dt>
<dd>last row, and first and last columns are closed. </dd> 
@image html perio2_20.png 
<center>@image latex perio2_20.png 
</center>
<dt>North fold boundary with a T -point pivot (in_perio=3)</dt>
<dd>first row, and first and last columns are closed. </dd> 
@image html perio3_20.png 
<center>@image latex perio3_20.png 
</center>
<dt>North fold boundary with a T -point pivot and cyclic east-west boundary (in_perio=4)</dt>
<dd>first row is closed. The first column is set to the value of the last-but-one column and the last column to the value of the second one. </dd> 
@image html perio4_20.png 
<center>@image latex perio4_20.png 
</center>
<dt>North fold boundary with a F -point pivot (in_perio=5)</dt>
<dd>first row, and first and last columns are closed. </dd> 
@image html perio5_20.png 
<center>@image latex perio5_20.png 
</center>
<dt>North fold boundary with a F -point pivot and cyclic east-west boundary (in_perio=6)</dt>
<dd>first row is closed. The first column is set to the value of the last-but-one column and the last column to the value of the second one.</dd> 
@image html perio6_20.png 
<center>@image latex perio6_20.png 
</center>
</dl>
@sa For more information about NEMO periodicity, see _Model Boundary Condition_ chapter in [NEMO documentation](http://www.nemo-ocean.eu/About-NEMO/Reference-manuals))
