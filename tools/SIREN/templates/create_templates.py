#!/usr/bin/python
# -*- coding:Utf-8 -*-
"""
This script create template of namelist for Siren.

see create_templates.py --help 
"""
import os
import argparse
import re

def get_default(i,var):
    """
    Parameters
    ----------
    i: str
       input filename
    var: str
       variable name
    """
    with open(i) as f:
        for line in f:
            line=line.strip()
            if re.match('.*::.*'+var+'.*', line):
                return line.split("::")[1].split("=")[1]


def get_nam(i,o,d):
    """
    Copy input file in output file, line by line.
    
    If line contains filterlist parameter and src:
        - copy the line without the src option
        - copy the file to be include line by line

    Parameters
    ----------
    i: str
       input filename
    o: str
       output filename
    d: bool
       added default value
    """
    with open(i) as f:
        inRecordingMode = False
        for line in f:
            line=line.strip()
            if not inRecordingMode:
                if line.startswith('NAMELIST'):
                    inRecordingMode = True
                    nam='&'+line.split('/')[1]
                    cmt=''
                    if re.match('^.*!<.*$', line):
                        cmt='!< '+line.split('!<')[1]
                    print("{0:10} {1}".format(nam,cmt),file=o)
            else:
                if line.startswith('NAMELIST') or not line:
                    inRecordingMode = False
                    print("/",file=o)
                else:
                    if not line.startswith('!'):
                        var=line.split('&')[1].split(',')[0].split()[0]
                        if re.match('^.*!<.*$', line):
                            cmt='!< '+line.split('!<')[1]
                        if d:
                            val=get_default(i,var)
                            print("\t {0:12} = {1:30} {2}".format(var,val,cmt),file=o)
                        else:
                            print("\t {0:12} = {1:30} {2}".format(var,"",cmt),file=o)



def main():
    """
    Read Fortran src file and create template of namelist from it.
    """
    # define parser
    parser = argparse.ArgumentParser(
        prog="create_templates.py", description="Create template of namelist from Fortran src."
    )
    # positional arguments
    parser.add_argument("input" , type=str, help="Fortran filename, to be read in Siren/src directory")
    parser.add_argument("output", type=str, help="Output  filename, to be written in Siren/templates directory")
    # optional arguments
    parser.add_argument("-v", "--verbose", action="store_true", help="show more things")
    parser.add_argument("-d", "--default", action="store_true", help="add default value for some variables")

    # parse arguments
    args = parser.parse_args()

    if re.match('.*/.*', args.output):
        raise NameError(
                "Output filename must not be a path (should not contains /)."
                )
        print(args.output.split("/")[1])
    fin=os.path.abspath(os.path.join("../src",args.input))
    fout=os.path.abspath(os.path.join("../templates",args.output))
    if args.verbose:
        print("Input  file : {0}".format(fin))
        print("Output file : {0}".format(fout))
        if args.default:
            print("\nAdd default value to some variables")

    if not os.path.isfile(fin):
        raise NameError(
                "Can not find {0} in src directory".format(os.path.basename(fin))
        )

    with open(args.output,'w') as fout:
        get_nam(fin,fout,args.default)

if __name__ == "__main__":
        main()
