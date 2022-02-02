"""
 script to sort and select processor layout from MPP_PREP output
 wrote by P. Mathiot 10/2018
"""

# import module
from __future__ import print_function
import argparse
import sys

# define class layout
class layout(object):
    """
    Class containing all the information about a specific model layout
   
    Output:
    self.jpnij = (i domain along i dimension, j domain along j dimension, total number of domain (land+ocean))
    self.nproc = (total number of domain (land+ocean), n ocean domain, nland domain)
    self.jpij  = (dimension along i, dimension along j) of a domain
    self.ratio = (total computed point, overhead, ratio computed point / global)
    """
    def __init__(self, txtlst):
        """
        Initialisation of a layout class object:

        Input: list of string containing 1 layout description 
           extracted from the processor layout file
        """
        self.jpnij = extract_jpnij(txtlst)  # (jpni, jpnj, jpnij)
        self.nproc = extract_nproc(txtlst)  # (ntot, noce, nland)
        self.jpij = extract_jpij(txtlst)    # (jpi , jpj)
        self.ratio = extract_point(txtlst)  # (total computed point, overhead, ratio computed point / global)

    def test_criterion(self, rmax, nocemin, nocemax):
        """
        function to decide if yes or no a specific layout has to be selected
        Input: 
        float rmax: maximal ratio (computed/global) accepted
        int   nocemin and nocemax: range of number of ocean processor accepted

        output: logical
        """
        if ( nocemin <= self.nproc[1] <= nocemax ) and ( self.ratio[2] <= rmax ):
            return True
        else:
            return False

    def print_layout(self, cidx):
        """
        function to print specific information about a specific layout
        """
        print( 'Domain decomposition {}'.format(cidx) )
        print( 'domain decomposition (jpni, jpnj) = {}'.format((self.jpnij[0], self.jpnij[1])) )
        print( 'number of ocean domain            = {}'.format(self.nproc[1]) )
        print( 'ratio computed/global             = {}'.format(self.ratio[2]) )
        print( 'domain size (jpi, jpj)            = {}'.format(self.jpij) )
        print('')

# define sorting function
def noce_proc(elem):
    """
    function used as key to sort list of layout 
    by number of ocean domain in the list sorting algorithm
    """
    return elem.nproc[1]

def ratio_proc(elem):
    """
    function used as key to sort list of layout 
    by ratio (computed/global) in the list sorting algorithm
    """
    return elem.ratio[2]

def jpij_proc(elem):
    """
    function used as key to sort list of layout 
    by domain size in the list sorting algorithm
    """
    return elem.jpij[0]*elem.jpij[1]

# txt extraction function to feed class layout
def extract_jpnij(txtlst):
    """
    function to extract total number of domain
    for a specific domain layout txt output

    Input: list of string containing 1 layout description 
           extracted from the processor layout file
    
    Output: tuple (jpni, jpnj, jpni*jpnj)
    """
    jpnij = int(txtlst[1].split()[-1])
    ctmp = txtlst[3].split()
    jpni = int(ctmp[1])
    jpnj = int(ctmp[3])
    return (jpni, jpnj, jpnij)

def extract_nproc(txtlst):
    """
    function to extract total number of ocean domain
    for a specific domain layout txt output
    
    Input: list of string containing 1 layout description 
           extracted from the processor layout file
    
    Output: tuple (jpni*jpnj, n oce domain, n land domain)
    """
    ntot = int(txtlst[1].split()[-1])
    noce = int(txtlst[5].split()[-1])
    nland = int(txtlst[6].split()[-1])
    return (ntot, noce, nland)

def extract_jpij(txtlst):
    """
    function to extract domain dimension
    for a specific domain layout txt output

    Input: list of string containing 1 layout description 
           extracted from the processor layout file

    Output: tuple (jpi, jpj)
    """
    ctmp = txtlst[4].split()
    jpi = int(ctmp[1])
    jpj = int(ctmp[3])
    return (jpi, jpj)

def extract_point(txtlst):
    """
    function to extract ration (computed/global)
    for a specific domain layout txt output

    Input: list of string containing 1 layout description 
           extracted from the processor layout file

    Output: tuple (total number of point, overhead, ratio (computed/global))
    """
    npoint = int(txtlst[13].split()[-1])
    noverh = int(txtlst[14].split()[-1])
    ratio = float(txtlst[15].split()[-1])
    return (npoint, noverh, ratio)

# main
def main():
    """
    script to sort and select processor layout from MPP_PREP output based on user constrains
    """
   
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", metavar='layout_file' , help="names of domain layout file to sort", type=str, nargs=1, required=True)
    parser.add_argument("--rmax", metavar='max_ratio' , help="max ratio allowed (computed/global)", type=float, nargs=1, required=True)
    parser.add_argument("--noce", metavar='min/max_noce' , help="min and max number of ocean domain allowed", type=int, nargs=2, required=True)
    
    args = parser.parse_args()
    
    # read file
    filein = args.f[0]
    fid = open(filein,"r") #opens file with name of "test.txt"
    txtdata = []
    for cline in fid:
        txtdata.extend([cline])

    # skip header and tail of file
    txtdata = txtdata[4:-20]
    
    # loop on different domain decomposition
    ilinepl = 17 # number of line of a specific decomposition
    ilayout = 0
    lst_layout = []
    ratio_min = 9999.0
    noce_min = 9999999
    noce_max = 0
    for iline in range(0, len(txtdata)):
        if iline % ilinepl == 0:
            # initialise layout
            conf_layout = layout( txtdata[iline:iline+ilinepl] )
            ratio_min = min( conf_layout.ratio[2], ratio_min )           
            noce_min = min( conf_layout.nproc[1], noce_min )           
            noce_max = max( conf_layout.nproc[1], noce_max )           
 
            # select layout based on condition
            if conf_layout.test_criterion(args.rmax[0], args.noce[0], args.noce[1]): 
                ilayout = ilayout + 1
                lst_layout.extend([conf_layout])

    if lst_layout == []:
        print('')
        print( 'E R R O R: constrains are too strong, no domain are found' )
        print('')
        if ratio_min > args.rmax[0] :
            print( 'min ratio is      {} and you ask for a ratio smaller than {}'.format(ratio_min, args.rmax[0]) )
        if noce_min > args.noce[1] :
            print( 'min ocean proc is {} and you ask for a number of ocean proc lower than {}'.format(noce_min, args.noce[1]) )
        if noce_max < args.noce[0] :
            print( 'max ocean proc is {} and you ask for a number of ocean proc larger than {}'.format(noce_max, args.noce[0]) )
        print('')
        sys.exit()
    
    lst_layout.sort(key=noce_proc)
    for idx, ilayout in enumerate(lst_layout):
        ilayout.print_layout(str(idx))
    
    print( '=====================================================================' )
    print('')
    print( 'Among the layouts fitting the constraint on : ratio (computed/global) < {} and {} <= number of ocean domain <= {}' \
        .format(args.rmax[0], args.noce[0], args.noce[1]) )
    print('')
    print( ' 3 layouts are highlighted : ' )
    print('')
    lst_layout.sort(key=ratio_proc)
    lst_layout[0].print_layout('SMALLEST RATIO')
#
    lst_layout.sort(key=noce_proc)
    lst_layout[-1].print_layout('LARGEST NUMBER OF OCEAN DOMAINS')
#
    lst_layout.sort(key=jpij_proc)
    lst_layout[0].print_layout('SMALLEST COMPUTED DOMAIN')
    
if __name__ == "__main__":
    main()
