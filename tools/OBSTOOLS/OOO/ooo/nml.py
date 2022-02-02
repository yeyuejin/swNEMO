"""Collection of Fortran 90 namelist helper functions.

    A common way to interface with a Fortran executable is via
    an input file called a namelist. This module defines
    functions which simplify the process of updating and
    extending namelist data.

    .. note:: This module is especially lightweight and follows the
              batteries included philosophy. As such, only standard
              library modules are required to use this code.

    Walkthrough
    ===========

    New namelist
    ------------

    A typical usage is to create and update a Fortran namelist on the fly.

    >>> import nml
    >>> namid = "namfoo"
    >>> text = nml.new(namid)
    >>> data = {"x": nml.tostring([1, 2, 3])}
    >>> text = nml.update(namid, text, data)
    >>> print text
    &namfoo
       x = 1 2 3
    /
    <BLANKLINE>

    In the above snippet :func:`tostring` has been used to sanitize the input
    Python list. This function cleverly maps string data and numeric data to
    the correct Fortran syntax.

    However, the :func:`new` function takes care of many of the above steps automatically.
    Where appropriate :func:`sanitize` has been embedded to reduce the need
    to worry about data format problems. Take for example,

    >>> print nml.new("namfoo", data={"x": range(3)})
    &namfoo
       x = 0 1 2
    /
    <BLANKLINE>

    Parse existing namelist
    -----------------------

    In order to update a namelist it is necessary to convert the namelist text into
    a dictionary of *key, value* pairs which can be manipulated in the usual Pythonic
    fashion before being piped back out to disk.

    In everyday usage text will be read from files, here however for illustration
    purposes I have hand written a namelist.

    >>> text = '''
    ... &namfoo
    ...     x = y  ! A description of the variables
    ... /
    ... &nambar
    ...     ln_on = .TRUE. ! A description of the variables
    ... /
    ... '''

    This can be parsed by invoking the :func:`variables` command.

    >>> nml.variables(text)
    {'x': 'y', 'ln_on': '.TRUE.'}

    Or by using the :func:`namelists` function to split the file into sub-lists.

    >>> nml.namelists(text)
    {'namfoo': '&namfoo\\n    x = y  ! A description of the variables\\n/', 'nambar': '&nambar\\n    ln_on = .TRUE. ! A description of the variables\\n/'}
    >>> sublists = nml.namelists(text)
    >>> print sublists["nambar"]
    &nambar
        ln_on = .TRUE. ! A description of the variables
    /

    Which can be parsed into a dictionary as before.

    >>> print nml.variables(sublists["nambar"])
    {'ln_on': '.TRUE.'}


    Update/replace data
    -------------------

    There are two ways of modifying values inside a Fortran namelist. 

    Replace
        The first is to simply replace a set of variables with new values. This behaviour is accomplished
        via the :func:`replace` function. This approach simply overwrites existing variables. No knowledge
        of sub-namelist structure is required to modify a string of text. 

    .. note:: Additional variables will not be added to a namelist via this approach

    Update
        The second is to extend the set of variables contained within a namelist. This functionality is
        controlled by the :func:`update` function. Here, variables which are not already specified are
        added using a templated namelist line.

    .. note:: It is essential to specify which sub-namelist is to be updated before modification takes place

    Pipe to/from file
    -----------------

    As typical NEMO namelists are no larger than a few tens of kilobytes
    it makes sense to process namelists as single strings instead of
    line by line.

    >>> path = "foo.nml"
    >>> text = nml.new("namfoo")

    To write to a file simply invoke the writer.

    >>> # Write to file
    >>> nml.writer(path, text)

    To read from a file specify the path to be read.

    >>> # Read from file
    >>> text = nml.reader(path)

    Join multiple namelists
    -----------------------

    Since the namelists are regular Python strings there is no need for a 
    specific *join* function. Namelists can be combined in whatever manner
    is most pleasing to the eye.

    >>> namoff = nml.new("namoff")
    >>> namcl4 = nml.new("namcl4")
    >>> # new line join
    >>> print "\\n".join([namoff, namcl4])
    &namoff
    /
    <BLANKLINE>
    &namcl4
    /
    <BLANKLINE>

    >>> # Or addition
    >>> print namoff + namcl4
    &namoff
    /
    &namcl4
    /
    <BLANKLINE>

    Module functions
    ================

"""
__version__ = "0.1.0"
import re
from numbers import Number

def reader(path):
    """Reads a file into a string

    Reads whole file into single string. Typically, 
    namelists are small enough to be stored in memory
    while updates and edits are being performed.

    :param path: Path to input file
    :returns: entire file as a single string

    """
    with open(path, "r") as handle:
        text = handle.read()
    return text

def writer(path, text):
    """Writes to a file from a string

    Handy way of piping a processed namelist into
    a file.

    :param path: Path to output file
    :param text: Input text to process

    """
    with open(path, "w") as handle:
        handle.write(text)

def update(namid, text, data, convert=True):
    """Extends namelist definition.

    Similar to replace this function alters the values
    of variables defined within a namelist. In addition to
    replacing values it also creates definitions if the
    variable is not found in the namelist. As such, the
    namelist id must be specified.

    :param namid: Namelist id
    :param text: Input text to process
    :param data: Dictionary of variables
    :keyword convert: Sanitizes input data before replacement takes place

    :returns: Text

    .. seealso:: :func:`replace` :func:`sanitize`
    """
    sublists = namelists(text)
    assert namid in sublists, "Warning: invalid namid specified!"

    # Sanitize inputs
    if convert:
        data = sanitize(data)

    # Parse subsection
    namtext = sublists[namid]
    subdata = variables(namtext)
    subvars = subdata.keys()

    # Replace existing variables in namtext
    tmptext = replace(namtext, data)
    text = text.replace(namtext, tmptext)
    namtext = tmptext

    # Identify new variables
    vars = data.keys()
    newvars = list(set(vars) - set(subvars))
    newvars.sort()

    # Append new vars to namid
    lines = namtext.split("\n")
    for v in newvars:
        newline = "   %s = %s" % (v, data[v])
        lines.insert(-1, newline)
    newtext = "\n".join(lines)

    # Replace old namtext with new namtext
    text = text.replace(namtext, newtext)
    return text

def replace(text, data, convert=True):
    """Edits existing variables.

    Pattern matches and substitutes variables inside
    a string of text. This is independent of namid and
    as such is useful for modifying existing variables.
    To append new variables the :func:`update` function 
    is required.

    >>> text = '''
    ... &namobs
    ...    ln_sst = .TRUE. ! Logical switch for SST observations
    ... /
    ... '''
    >>> data = {"ln_sst": ".FALSE."}
    >>> print replace(text, data)
    <BLANKLINE>
    &namobs
       ln_sst = .FALSE. ! Logical switch for SST observations
    /
    <BLANKLINE>

    .. note :: This does not append new variables to a namelist

    :param text: string to process
    :param data: dictionary with which to modify **text**
    :keyword convert: Sanitizes input data before replacement takes place

    :returns: string with new data values

    .. seealso:: :func:`update`, :func:`sanitize`
    """
    if convert:
        data = sanitize(data)
    for k, v in data.iteritems():
        pat = r"(%s\s*=\s*).+?(\s*[!\n])" % (k,)
        repl = r"\g<1>%s\g<2>" % (v,)
        text = re.sub(pat, repl, text)
    return text

def variables(text):
    """Retrieves dictionary of variables in text.

    >>> text = '''
    ... &namobs
    ...    ln_sst = .TRUE. ! Logical switch for SST observations
    ... /
    ... '''
    >>> variables(text)
    {'ln_sst': '.TRUE.'}

    :param text: Input text to process

    :returns: A dictionary of variable, value pairs.

    """
    data = {}
    pairs = re.findall(r"\n\s*(\w+)\s*=\s*(.+?)\s*(?=[!\n])", text)
    for key, value in pairs:
        data[key] = value
    return data

def namelists(text):
    """Retrieves dictionary of namelists in text.

    Useful for isolating sub-namelists.

    >>> text = '''
    ... &namobs
    ...    ln_sst = .TRUE. ! Logical switch for SST observations
    ... /
    ... '''
    >>> namelists(text)
    {'namobs': '&namobs\\n   ln_sst = .TRUE. ! Logical switch for SST observations\\n/'}

    :param text: Input text to process

    :returns: A dictionary of id, text block key, value pairs

    """
    # Boundary case
    if text.startswith("&"):
        text = "\n" + text
    # Regular expression
    results = re.findall(r"\n(&(\w+).*?\n/)", text, re.DOTALL)
    data = {}
    for content, namid in results:
        data[namid] = content
    return data

def tostring(data):
    """Maps standard Python data to Fortran namelist format.

    >>> tostring([1, 2, 3])
    '1 2 3'
    >>> tostring(["foo.nc", "bar.nc"])
    "'foo.nc', 'bar.nc'"
    >>> tostring(True)
    '.TRUE.'

    :param data: Input Python data

    :returns: Namelist formatted string

    .. seealso:: :func:`sanitize`
    """
    if isinstance(data, list):
        if all_numeric(data):
            delim = " "
        else:
            delim = ", "
        text = delim.join([convert(item) for item in data])
    else:
        text = convert(data)
    return text

def all_numeric(inputs):
    # Checks all list entries are numbers
    flag = True
    for input in inputs:
        if not isinstance(input, Number):
            flag = False
            break
    return flag

def numeric(word):
    # Tests input string is numeric data
    parts = word.split(" ")
    try:
        map(float, parts)
        flag = True
    except ValueError:
        flag = False
    return flag

def logical(word):
    # Tests input string is numeric data
    if word.upper() in [".FALSE.", ".TRUE."]:
        flag = True
    else:
        flag = False
    return flag

def listed(word):
    # Tests input string is not a list
    if ("," in word) or (" " in word):
        flag = True
    else:
        flag = False
    return flag

def quote(word):
    word = str(word)
    if not quoted(word):
        word = "'%s'" % (word,)
    return word

def convert(word):
    # Conversion function
    if isinstance(word, str):
        if (quoted(word) or numeric(word) 
            or logical(word) or listed(word)):
            result = "%s" % (word,)
        else:
            result = quote(word)
    elif isinstance(word, bool):
        if word:
            result = ".TRUE."
        else:
            result = ".FALSE."
    else:
        result = str(word)
    return result

def quoted(word):
    # Checks if string begins/ends with quotation marks
    if (word.startswith("'") and word.endswith("'")):
        flag = True
    elif (word.startswith('"') and word.endswith('"')):
        flag = True
    else: 
        flag = False
    return flag

def same_type(data):
    # True if all entries are the same type
    types = map(type, data)
    if len(set(types)) == 1:
        flag = True
    else:
        flag = False
    return flag

def sanitize(data):
    """Converts dictionary values into Fortran namelist format.

    This is a more typical way to prepare data for inclusion in
    a Fortran namelist. Instead of manually applying :func:`tostring`
    to every element of the input data, **sanitize** fixes the entire
    data set.

    >>> sanitize({"x": True})
    {'x': '.TRUE.'}
    >>> 

    :param data: Dictionary to convert

    :returns: Dictionary whose values are in Fortran namelist format

    .. seealso:: :func:`tostring`
    """
    replacements = [(k, tostring(v)) for k, v in data.items()]
    data.update(replacements)
    return data

def new(namid, data=None, convert=True):
    """Creates a new Fortran namelist

    >>> new("namobs")
    '&namobs\\n/\\n'
    >>> print new("namobs")
    &namobs
    /
    <BLANKLINE>

    :param namid: Name for the new namelist

    :keyword data: Specifies an initial dictionary with which to
                   populate the namelist
    :type data: dict
    :keyword convert: Sanitizes input data before replacement takes place

    :returns: string representation of a Fortran namelist
    """
    text = "&{namid}\n/\n".format(namid=namid)
    if data is not None:
        text = update(namid, text, data, convert=convert)
    return text

if __name__ == '__main__':
    import doctest
    doctest.testmod()

