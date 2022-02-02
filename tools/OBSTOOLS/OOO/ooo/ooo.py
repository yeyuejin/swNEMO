#!/usr/bin/env python2.7

import os
import shutil

# Local imports
import locator
import nml
import run

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("date", metavar="DATE", 
                        help="Run date.")
    parser.add_argument("-w", "--work-dir", default=os.getcwd())
    parser.add_argument("-f", "--forecast-types", default="forecast",
                        help="Choice of forecast,persistence,climatology")
    parser.add_argument("-l", "--lead-times", default="12",
                        help="Forecast lead times")
    parser.add_argument("-o", "--obs-types", default="profbfiles",
                        help="Choice of namobs types.")
    parser.add_argument("--class4", dest="ln_cl4", action="store_true",
                        help="Flag to choose class 4 file outputs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Flag to test namelist building without submitting.")
    parser.add_argument("--cmd", dest="command", default="./opa",
                        help="Submit task to run.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Prints difference between before and after namelists.")
    parser.add_argument("namelist", metavar="NAMELIST", 
                        help="NEMO namelist to edit.")
    args = parser.parse_args()
    args.forecast_types = args.forecast_types.split(',')
    args.obs_types = args.obs_types.split(',')
    args.lead_times = map(int, args.lead_times.split(','))
    return args

def printdiff(text1, text2):
    # Provides nice text difference summary of namelists
    import difflib
    lines1 = text1.splitlines()
    lines2 = text2.splitlines()
    d = difflib.Differ()
    result = list(d.compare(lines1, lines2))
    text = '\n'.join(result)
    print text

def main():
    args = parse_args()
    date = args.date
    print "Processing", args.namelist, " for", args.date
    
    # Move to working directory
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    os.chdir(args.work_dir)

    # Collect forecast files
    types = args.forecast_types
    lead_times = args.lead_times
    namooo, namcl4 = locator.forecasts(date=date,
                                       types=types,
                                       lead_times=lead_times)

    # Process NEMO namelist
    text = nml.reader(args.namelist)
    sublists = nml.namelists(text)

    # Verbose save original text
    if args.verbose:
        original_text = text

    # namooo
    if "namooo" not in sublists:
        # Attach boilerplate
        text += nml.new("namooo")
    text = nml.update("namooo", text, data=namooo)

    # namcl4
    if "namcl4" not in sublists:
        # Attach boilerplate
        text += nml.new("namcl4")
    namcl4["cl4_leadtime"] = lead_times
    namcl4["cl4_date"] = nml.quote(date)
    namcl4["cl4_match_len"] = len(namcl4["cl4_vars"])
    namcl4["cl4_fcst_len"] = len(namcl4["cl4_leadtime"])
    # Add naming convention
    namcl4["cl4_sys"] = "FOAM"
    namcl4["cl4_cfg"] = "orca025"
    namcl4["cl4_vn"] = "'1.0'"
    namcl4["cl4_prefix"] = "class4"
    namcl4["cl4_contact"] = "example@example.com"
    namcl4["cl4_inst"] = "institute"
    text = nml.update("namcl4", text, data=namcl4)

    # namrun
    namrun = {"nn_date0": nml.quote(date)}
    text = nml.update("namrun", text, data=namrun)

    # namobs
    namobs = locator.observations(date=date,
                                  types=args.obs_types)
    namobs["ln_cl4"] = args.ln_cl4
    text = nml.update("namobs", text, data=namobs)

    # Verbose print namelist differences
    if args.verbose:
        printdiff(original_text, text)

    # pipe text to file
    tmp = args.namelist+".tmp"
    nml.writer(tmp, text)
    shutil.move(tmp, args.namelist)

    # Run job
    if not args.dry_run:
        run.submit(command=args.command)
    


if __name__ == '__main__':
    main()

