#! /bin/bash

# aponteeduardo@gmail.com
# Copyright (C) 2017

set -e

DEBUGM=0

for i in "$@"
do
    case $i in
        -d|--debug)
        DEBUGM=1
        shift # past argument=value
        ;;
        #-s=*|--searchpath=*)
        #SEARCHPATH="${i#*=}"
        #shift # past argument=value
        #;;
    esac
done

#cp ../../h2gf/tapas_h2gf_data.m ./
#cp ../../h2gf/tapas_h2gf_estimate.m ./
#cp ../../h2gf/tapas_h2gf_inference.m ./
#cp ../../h2gf/tapas_h2gf_model.m ./
#cp ../../h2gf/tapas_h2gf_estimate_interface.m ./
#cp ../../h2gf/tapas_h2gf_prepare_data.m ./
#cp ../../h2gf/tapas_h2gf_prepare_model.m ./
#cp ../../h2gf/tapas_h2gf_prepare_inference.m ./
#cp ../../h2gf/tapas_h2gf_gen_state.m ./
#cp ../../h2gf/tapas_h2gf_get_stored_state.m ./
#cp ../../h2gf/tapas_h2gf_init_state.m ./
#cp ../../h2gf/tapas_h2gf_init_states.m ./
#cp ../../h2gf/tapas_h2gf_prepare_posterior.m ./
#cp ../../h2gf/tapas_h2gf_prepare_ptheta.m ./
#cp ../../h2gf/tapas_h2gf_llh.m ./

ls tapas_h2gf* | while read line
do
    fname=${line/h2gf/sem_hier}
    cat $line | sed \
        -e 's/h2gf/sem_hier/g' > $fname
    rm $line
done


exit 0

