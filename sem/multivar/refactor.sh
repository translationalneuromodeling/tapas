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

ls tapas_sem_hier* | while read line
do
    fname=${line/sem_hier/sem_multiv}
    cat $line | sed \
        -e 's/sem_hier/sem_multiv/g' > $fname
    rm $line
done


exit 0

