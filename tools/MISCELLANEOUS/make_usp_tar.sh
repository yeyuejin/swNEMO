#!/bin/bash
#set -x
set -o posix
#set -u
#set -e
#+
# ===============
# make_usp_tar.sh
# ===============
# ---------------
# Fetch and tar a unsupported configuration setup.
# This script is only needed when target systems do
# not have wget access to the internet. To configure
# unsupported conigurations on these systems it will
# first be necessary to run this script on a system that
# does have access. Then copy access the resulting tar file,
# unpack and run the enclosed set_local_uspcfg script in a 
# bash shell to complete the process. That script redefines/
# defines wget as a bash function to perform local copies 
# from this unpacked tarball. If you wish to create a new 
# configuration based on this local copy of an unsupported 
# configuration in future sessions then you may need to 
# redefine wget again before running makenemo with the 
# appropriate -u setting. A simple script: def_wget is included
# for such situations.
# ---------------
# SYNOPSIS
# ========
#  $ make_usp_tar.sh uspcfg.txt target_conf target_dir
# Note target_dir.tar will be created
#
# DESCRIPTION
# ===========
# - Extract target configuration details from uspcfg.txt
# - Create target directory
# - Recursively use wget to retrieve remote configuration files
#   into target directory
# - Copy uspcfg.txt into target directory and alter remote paths
#   to local (relative) versions
# - tar target_directory and remove originals
#----------------------------------------------------------------
#
#----------------------------------------------------------------
# Check the correct number of arguments have been provided
#----------------------------------------------------------------
#
   if [ "$#" != "3" ]; then
    echo "Expected usage: make_usp_tar.sh uspcfg.txt target_conf target_dir"
    exit
   fi
#
#----------------------------------------------------------------
# Check the named uspcfg.txt file exists
#----------------------------------------------------------------
#
   if [ ! -f $1 ]; then
    echo "named uspcfg.txt file does not exist ($1); attempt abandoned"
    exit
   fi
#
#----------------------------------------------------------------
# Check the requested configuration is listed in the named uspcfg.txt file
#----------------------------------------------------------------
#
   inthere=$( grep -c "$2" $1 )
   if [ "$inthere" -lt 1 ]; then
    echo "requested configuration is not in named uspcfg.txt file ($2); attempt abandoned"
    exit
   fi
#
#----------------------------------------------------------------
# Create the target directory if it does not already exist
# and cd into it
#----------------------------------------------------------------
#
   if [ ! -d $3 ]; then
    mkdir $3
   else
    echo "target directory already exists; attempt abandoned"
    exit
   fi
   basedir=$(pwd)
#
   cd ${3}
#
#----------------------------------------------------------------
# Copy named uspcfg.txt file into target directory
#----------------------------------------------------------------
#
   cp $basedir/$1 .
#
#----------------------------------------------------------------
# Extract information on target configuration and
# retrieve full file list from remote server
#----------------------------------------------------------------
#
   grep "$2 " $1 > ./cfg.tmp
#
   LOCAL_REF=$(cat cfg.tmp | awk 'BEGIN {FS = "#" }{print $2}')
   TAB=$(cat cfg.tmp | awk 'BEGIN {FS = "#" }{print $3}')
   REMOTE_CTL=$(cat cfg.tmp | awk 'BEGIN {FS = "#" }{print $4}')
   wget ${REMOTE_CTL} -O remote_file.list
#
#----------------------------------------------------------------
# Retrieve each remote file and create local directory structure
# At the same time prepare a local version of the control file
# by replacing http links with a string that will be replaced later
# with a local directory path
#----------------------------------------------------------------
#
   if [ -f remote_file.list ] ; then
    cat remote_file.list | grep -v '^#' |
     while
      read remfile locfile
      do
       if [ $remfile == 'create_directory' ] ;then
         mkdir $locfile
         echo $remfile "                 " $locfile >> local_template.ctl
       else
         wget $remfile -O $locfile
         remfile=$(echo $remfile | sed -e "s;.*$locfile;SET_LOCAL_DIRECTORY_HERE/$locfile;")
         echo $remfile "  " $locfile >> local_template.ctl
       fi
      done
   else
    echo "Unable to find remote_file.list. Attempt abandoned"
    echo "Files may be left in "$(pwd)
    exit
   fi
#
#----------------------------------------------------------------
# Construct a modified version of the named uspcfg.txt file
# First copy across all the other untouched configuration listed
#----------------------------------------------------------------
#
   grep -v "$2 " $1 >  uspcfg_local_template.txt
#
#----------------------------------------------------------------
# Now append the modified entry, replacing http links as before
#----------------------------------------------------------------
#
   cat cfg.tmp | sed -e "s;http.*$;SET_LOCAL_DIRECTORY_HERE/local.ctl;" >> uspcfg_local_template.txt
#
#
#----------------------------------------------------------------
# Construct a script that can be used later to complete a local installation
#----------------------------------------------------------------
#
cat > set_local_uspcfg << EOF
#!/bin/bash
   if [ "\$( echo $SHELL | grep -c bash )" -lt 1 ]; then
    echo "WARNING: This is only going to be effective in a bash shell"
    echo "since it redefines the wget command as a bash function."
    echo "(ignore this comment if you are in a bash shell)"
   fi
   basedir=\$(pwd)
#
   echo "Enter full path to the CONFIG directory on your target system: "
   read confdir
#
# Edit the local.ctl file to set the local directory path
#
   sed -e "s;SET_LOCAL_DIRECTORY_HERE;\$basedir;" local_template.ctl > local.ctl
#
# Edit the uspcfg_local.txt file to set the local directory path
#
   sed -e "s;SET_LOCAL_DIRECTORY_HERE;\$basedir;" uspcfg_local_template.txt > uspcfg_local.txt
#
# Install local versions in the named CONFIG directory
#
   if [ -f \$confdir/uspcfg.txt ] && [ ! -L \$confdir/uspcfg.txt ]; then
    mv \$confdir/uspcfg.txt \$confdir/uspcfg_remote.txt
    echo "\$confdir/uspcfg.txt moved to \$confdir/uspcfg_remote.txt"
   fi
   if [ -f \$confdir/uspcfg_local.txt ]; then
    echo "Existing uspcfg_local.txt file found in \$confdir"
    echo "This has been moved to: "\$confdir/uspcfg_local.txt\$\$
    mv \$confdir/uspcfg_local.txt \$confdir/uspcfg_local.txt\$\$
   fi
   mv uspcfg_local.txt \$confdir/uspcfg_local.txt
   ln -s \$confdir/uspcfg_local.txt \$confdir/uspcfg.txt
#
# define/redefine the wget command
#
function wget {
   if [ "\$2" != "-O" ]; then
    echo "Expected wget usage: wget src -O dest"
    echo "-O not found. No action taken"
   else
    cp \$1 \$3
   fi
}
export -f wget
EOF
#----------------------------------------------------------------
# Construct a script that can be used later to just redefine wget 
# in bash shell sessions
#----------------------------------------------------------------
#
cat > def_wget << EOFC
#!/bin/bash
   if [ "\$( echo $SHELL | grep -c bash )" -lt 1 ]; then
    echo "WARNING: This is only going to be effective in a bash shell"
    echo "since it redefines the wget command as a bash function."
    echo "(ignore this comment if you are in a bash shell)"
   fi
function wget {
   if [ "\$2" != "-O" ]; then
    echo "Expected wget usage: wget src -O dest"
    echo "-O not found. No action taken"
   else
    cp \$1 \$3
   fi
}
export -f wget
EOFC
#----------------------------------------------------------------
# Make sure these scripts have execute permission
#----------------------------------------------------------------
   chmod 755 set_local_uspcfg
   chmod 755 def_wget
#
#----------------------------------------------------------------
# Tidy up and tar the contents of the downloaded configuration
#----------------------------------------------------------------
   rm cfg.tmp
   cd $basedir
   tar cvf ${3}.tar $3
   echo ${3}.tar " file successfully created and prepared for local references. Move this"
   echo "tar file to your target system, unpack and run the set_local_uspcfg script in a "
   echo "bash shell to complete the process. This script redefines/defines wget as a bash"
   echo "function to perform local copies from this unpacked tarball. If you wish to create"
   echo "a new configuration based on this local copy of an unsupported configuration in "
   echo "future sessions then you may need to rerun the def_wget script"
exit
