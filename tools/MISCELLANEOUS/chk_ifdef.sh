#!/bin/bash
#
# check the propper syntax of C preprocessor directives.
# for example:
#if defined key_traldf_c3d && key_traldf_smag
# is not good and should be
#if defined key_traldf_c3d && defined key_traldf_smag
#
# use: go to TOOLS/MISCELLANEOUS/ and simply execute:
#  ./chk_ifdef.sh 
#
set -u
#
grep -r "^ *#if" ../../NEMO | grep -v "~:" > tmp$$      # get each lines of the code starting with #if
grep -r "^ *#elif" ../../NEMO | grep -v "~:" >> tmp$$   # get each lines of the code starting with #elif
#
for ll in $( seq 1 $( cat tmp$$ | wc -l  ) )            # for each of these lines
do
    lll=$( sed -n -e "${ll}p" tmp$$ )
    nbdef=$( echo $lll | grep -o defined  | wc -l )         # number of occurences of "defined"
    nband=$( echo $lll | grep -o   "&&"   | wc -l )         # number of occurences of "&&"
    nbor=$(  echo $lll | grep -o   "||"   | wc -l )         # number of occurences of "||"
    [ $nbdef -ne $(( $nband + $nbor + 1 )) ] && echo $lll   # print bad line
done
rm -f tmp$$ 

#
# add other basic tests
#
grep -ir  ":,:.*ji,jj" * | grep -v "~:"
grep -ir  "ji,jj.*:,:" *  | grep -v "~:"
