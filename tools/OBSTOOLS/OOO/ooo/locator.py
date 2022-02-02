"""Locates input files and calculates namelist attributes

    Typically a user will have files distributed throughout the hard drive
    and it makes little or no sense to try to capture all use cases.

    >>> date = "20130101"
    >>> namobs = observations(date, types=["profbfiles"])

    >>> namooo, namcl4 = forecasts(date, types=["forecast", "persistence"], lead_times=[12, 36, 60])
"""

def observations(date, types=None):
    """Responsible for locating observation files

    Valid **namobs** observation types.

    * profbfiles
    * sstfbfiles
    * slafbfiles

    :param date: The verification date in string format ``'%Y%m%d'``
    :param types: A list of namobs observation types

    :returns: namobs namelist dictionary
    """
    namobs = {"ln_t3d": False,
              "ln_s3d": False,
              "ln_ena": False,
              "ln_profb": False,
              "ln_sst": False,
              "ln_sstfb": False,
              "ln_reysst": False,
              "ln_ghrsst": False,
              "ln_sla": False,
              "ln_slafb": False,
              "ln_sladt": False}
    if types is None: types = []
    for obtype in types: 
        if obtype == "profbfiles":
            namobs[obtype] = profbfiles(date)
            namobs["ln_t3d"] = True
            namobs["ln_s3d"] = True
            namobs["ln_profb"] = True
        elif obtype == "sstfbfiles":
            namobs[obtype] = sstfbfiles(date)
            namobs["ln_sst"] = True
            namobs["ln_sstfb"] = True
        elif obtype == "slafbfiles":
            namobs[obtype] = slafbfiles(date)
            namobs["ln_sla"] = True
            namobs["ln_slafb"] = True
    return namobs

def profbfiles(date):
    """Observation file locator stub

    .. note:: User-specified stub

    :param date: The verification date in string format ``'%Y%m%d'``
    :returns: List of files
    """
    files = ['profb.nc']
    return files

def sstfbfiles(date):
    """Observation file locator stub

    .. note:: User-specified stub

    :param date: The verification date in string format ``'%Y%m%d'``
    :returns: List of files
    """
    files = ['sstfb.nc']
    return files

def slafbfiles(date):
    """Observation file locator stub

    .. note:: User-specified stub

    :param date: The verification date in string format ``'%Y%m%d'``
    :returns: List of files
    """
    files = ['slafb.nc']
    return files

def forecasts(date, types=None, lead_times=None):
    """Responsible for locating forecast fields

    :param date: The verification date in string format ``'%Y%m%d'``
    :param types: A list of forecast system types
    :param lead_times: A list of lead_times to search for

    :returns: tuple of namelist data, (namooo, namcl4)
    """
    namooo = {}
    namcl4 = {}
    if types is None: types = []
    if lead_times is None: lead_times = []

    # Initialise data
    ooo_files = []
    nn_ooo_idx = []
    cl4_vars = []
    cl4_fcst_idx = []

    # Search for files
    for type in types:
        files = []
        in_indices = []
        out_indices = []
        for ilt, lead_time in enumerate(lead_times):
            file, index = field(date, type=type, lead_time=lead_time)
            files.append(file)
            in_indices.append(index)
            out_indices.append(ilt + 1)
        # Expand return lists
        ooo_files += files
        nn_ooo_idx += in_indices
        cl4_fcst_idx += out_indices
        cl4_vars += len(files) * [type]

    # Namoo
    namooo["ooo_files"] = ooo_files
    namooo["nn_ooo_idx"] = nn_ooo_idx

    # Namcl4
    namcl4["cl4_vars"] = cl4_vars
    namcl4["cl4_fcst_idx"] = cl4_fcst_idx

    return namooo, namcl4

def field(date, type=None, lead_time=None):
    """Forecast field locator

    Maps verification date and lead time off set to file name and
    index along file *time_counter*

    .. note:: User-specified stub

    :param date: The verification date in string format ``'%Y%m%d'``
    :param type: Forecast type
    :param lead_time: Forecast off set

    :returns: (**path**, **index**)
    """
    # Worker function
    file = 'nofile'
    index = -1
    return file, index

