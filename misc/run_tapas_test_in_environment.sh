#!/bin/bash
# 
# RUN_TAPAS_TEST_IN_ENVIRONMENT
#
# Clone/download tapas,spm and tapas-examples to temporal folder and 
# run testing pipeline in 'isolated environment' (clean matlab).
#
# Authors: Matthias Müller-Schrader & Lars Kasper
# Created: 2023-05-08
# Copyright (C) 2023 TNU, Institute for Biomedical Engineering,
#                    University of Zurich and ETH Zurich.
#
# This file is part of the TAPAS PhysIO Toolbox, which is released under
# the terms of the GNU General Public License (GPL), version 3. You can
# redistribute it and/or modify it under the terms of the GPL (either
# version 3 or, at your option, any later version). For further details,
# see the file COPYING or <http://www.gnu.org/licenses/>.
#

## Parameters
TAPASREPO="git@tnurepository.ethz.ch:TNU/tapas.git"
BRANCH="tapas-v-6-1-0"
BRANCH=$(git rev-parse --abbrev-ref HEAD) # Uses current branch as default
TOOLBOX="physio"
MATLABOPT='-nodesktop'
MATLAB_USE_USERWORK="0"
export MATLAB_USE_USERWORK=0
#LOGFILE="${TEMPDIR}/tapas-test.log"
LOGFILE="$HOME/tapas-test.log"
DORUN=true
DOPRINTENV=false
optstring="hdpt:r:b:f:t:o:l:"
while getopts ${optstring} arg; do
    case "${arg}" in
    h)
        echo "run_tapas_test_in_environment: Run tapas-tests in isolated environment"
        echo "Options:"
        echo "  -h Show this help text."
        echo "  -d Perform dry run (not starting matlab)."
        echo "  -p Print environment-variables."
        echo "  -t <toolbox>    Test toolbox <toolbox> (default: physio)."
        echo "  -f <tempdir>    Use <tempdir> as directory (default: new tmpdir will be created)."
        echo "  -r <tapasRepo>  Use <tapasRepo> as repository (default: $TAPASREPO). "
        echo "  -b <branch>     Checkout <branch> (default: $TOOLBOX)." 
        echo "  -o <matlabOpt> Pass <matlabOpt> to matlab (default: $MATLABOPT)."
        echo "  -l <logFile>    Use <logFile> as logfile (default: $LOGFILE)"
        echo ""
        echo "By default, matlab starts in a non-graphical environment. For a "
        echo "  graphical environment, use pass the following arguments: -o -desktop "
        exit 0;;
    d)
        DORUN=false;;
    p)
        DOPRINTENV=true;;
    t)  
        TOOLBOX="$OPTARG";;  
    f)  
        TEMPDIR="$OPTARG";;  
    r)
        TAPASREPO="$OPTARG";;
    b)
        BRANCH="$OPTARG";;
    o)
        MATLABOPT="$OPTARG";;
    l)
        LOGFILE="$OPTARG";;
    esac
done 
shift $((OPTIND -1))
if [ -z "$TEMPDIR" ]; then
    TEMPDIR=$(mktemp -d -t tapas_test.XXX) # TODO: Check if that will work under mac 
    # Under macOS, this could be a different dir.
fi

## Create Logfile
echo "Using temporal directory $TEMPDIR"
touch "$LOGFILE"
echo "$START LOGGING AT $(date)" >> "$LOGFILE"
echo "MATLAB-log will be written to $LOGFILE"
## Copy script in there
cp tapas_test_in_environment_template.m "${TEMPDIR}/tapas_test_in_environment.m"
cd "$TEMPDIR"
#touch startup.m # Overwrite different statup-file
echo "% This is an auto-generated empty startup-file to prevent the use of other startup-files." > $TEMPDIR/startup.m
## Clone tapas and spm
git clone --depth=1 --branch "$BRANCH" --single-branch "$TAPASREPO" |& tee -a "$LOGFILE"
git clone --depth=1 https://github.com/spm/spm.git |& tee -a "$LOGFILE" # Development version
## Document hashes:
cd tapas
echo "TAPAS REPO $TAPASREPO BRANCH $BRANCH hash:" |& tee -a "$LOGFILE"
git rev-parse HEAD |& tee -a "$LOGFILE"
cd ../spm
echo "SPM git hash:" |& tee -a "$LOGFILE"
git rev-parse HEAD |& tee -a "$LOGFILE"
cd ..
echo "========== COPY OF TEST SCRIPT ===========" >> "$LOGFILE"
cat tapas_test_in_environment.m >> "$LOGFILE"
echo "========== END COPY OF TEST SCRIPT =======" >> "$LOGFILE"
## Start matlab:
if $DOPRINTENV; then
    echo "========== MATLAB ENVIRONMENT ============" |& tee -a "$LOGFILE"
    matlab -e |& tee -a "$LOGFILE"
    echo "========== END OF MATLAB ENVIRONMENT ============" |& tee -a "$LOGFILE"
fi
echo "Everything is ready - starting matlab"
echo matlab -sd "$TEMPDIR" $MATLABOPT -r "tapas_test_in_environment $LOGFILE $TOOLBOX " |& tee -a "$LOGFILE"
if $DORUN; then
    matlab -sd "$TEMPDIR" $MATLABOPT -r "tapas_test_in_environment $LOGFILE $TOOLBOX "
else
    echo "Skipping execution of matlab in dry run (option -d)." |& tee -a "$LOGFILE"
fi

# We don't delete temporal directory, since a) that can be often done by UNIX 
# and b) allows us to inspect the directory for debugging.
exit
