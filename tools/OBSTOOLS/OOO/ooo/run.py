"""

The NEMO offline observation operator is built and runs similarly to the
online NEMO model.
 
"""
import subprocess
import shlex
import os

class SubmitError(Exception):
    pass

def submit(command="./opa"):
    """Simple function that runs the code.

    This can be customised based on the particular environment
    used to submit MPI or serial tasks. For simplicity, this
    function calls opa.

    By default, this program runs ``./opa``.

    :returns: retcode
    """
    retcode = subprocess.call(shlex.split(command))
    return retcode

