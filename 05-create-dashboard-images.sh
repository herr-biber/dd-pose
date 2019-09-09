#!/bin/bash

# make sure 00-activate.sh is run beforehand
if [ -z ${DD_POSE_SOURCED+x} ]; then echo "DD_POSE_SOURCED environment variable is not set. Have you sourced 00-activate.sh?"; exit 1; fi

echo "Creating dashboard images"

function create_dashboard_images()
{
    local SUBJECT=$1
    local SCENARIO=$2
    local HUMANHASH=$3

    local BASENAME=subject-${SUBJECT}-scenario-${SCENARIO}-${HUMANHASH}
    local OUTDIR=$DD_POSE_DATA_ROOT_DIR/02-dashboard-images
    local LOGFILE=${OUTDIR}/${BASENAME}.log
    local FINISHFILE=${OUTDIR}/${BASENAME}.finished

    mkdir -p $OUTDIR

    echo "About to create dashboards for $SUBJECT $SCENARIO $HUMANHASH" | tee $LOGFILE
    echo -n "start: " >> $LOGFILE
    date >> $LOGFILE

    if [ -f $FINISHFILE ]; then
        echo "Finishfile exists. Skipping. $FINISHFILE" | tee -a $LOGFILE
        exit 1
    fi

    local COMMAND="python $DD_POSE_DIR/05-create-dashboard-images.py $SUBJECT $SCENARIO $HUMANHASH"

    echo "command: $COMMAND" | tee -a $LOGFILE
    $COMMAND &>> $LOGFILE
    echo $? >> $LOGFILE

    echo -n "end: " >> $LOGFILE
    date >> $LOGFILE
    touch $FINISHFILE

    echo "ok"
    return 0
}

# make function visible to gnu parallel
export -f create_dashboard_images

cat $DD_POSE_DIR/resources/dataset-items-trainval.txt | parallel -j1 -C ' ' create_dashboard_images {1} {2} {3}
