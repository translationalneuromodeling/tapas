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
# TODO: Use getopts.
TMPDIR=`mktemp -d -t tapas_test.XXX` # TODO: Check if that will work under mac. 
TAPASREPO="git@tnurepository.ethz.ch:TNU/tapas.git"
BRANCH="tapas-v-6-1-0"
TOOLBOX="physio"
DESKTOPOPT='-nodesktop'
#LOGFILE="${TMPDIR}/tapas-test.log"
LOGFILE="$HOME/tapas-test.log"

## Create directory
echo "Using temporal directory $TMPDIR"
touch "$LOGFILE"
echo "$START LOGGING AT $(date)" >> "$LOGFILE"
echo "MATLAB-log will be written to $LOGFILE"
## Copy script in there
cp tapas_test_in_environment_template.m "${TMPDIR}/tapas_test_in_environment.m"
cd "$TMPDIR"

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
echo "Everything is ready - starting matlab"
echo matlab -sd "$TMPDIR" $DESKTOPOPT -r "tapas_test_in_environment $LOGFILE $TOOLBOX " >> "$LOGFILE"
matlab -sd "$TMPDIR" $DESKTOPOPT -r "tapas_test_in_environment $LOGFILE $TOOLBOX "

# We don't delete temporal directory, since a) that can be often done by UNIX 
# and b) allows us to inspect the directory for debugging.
exit
