#!/bin/bash

# make sure 00-activate.sh is run beforehand
if [ -z ${DD_POSE_SOURCED+x} ]; then echo "DD_POSE_SOURCED environment variable is not set. Have you sourced 00-activate.sh?"; exit 1; fi

echo "Encoding dashboard images to compressed video files for $OUTPUT_DIR"

function encode_dashboard_images()
{
    local SUBJECT=$1
    local SCENARIO=$2
    local HUMANHASH=$3

    local BASENAME=subject-${SUBJECT}-scenario-${SCENARIO}-${HUMANHASH}
    local IMAGE_DIR=${DD_POSE_DATA_ROOT_DIR}/02-dashboard-images/subject-$SUBJECT/scenario-$SCENARIO/$HUMANHASH
    local OUTPUT_DIR=$DD_POSE_DATA_ROOT_DIR/03-dashboard-videos
    local LOGFILE=${OUTPUT_DIR}/${BASENAME}.log
    local INPUT_FINISHED_FILE=${DD_POSE_DATA_ROOT_DIR}/02-dashboard-images/${BASENAME}.finished
    local FINISHFILE=${OUTPUT_DIR}/${BASENAME}.finished

    echo "About to encode dashboard images to videos for $SUBJECT $SCENARIO $HUMANHASH" | tee $LOGFILE
    
    if [ ! -f $INPUT_FINISHED_FILE ]; then
        echo "Create dashboard finish file does not exist. Skipping. Have you run 05-create-dashboard-images.sh?"
        return 1
    fi

    mkdir -p $OUTPUT_DIR

    echo -n "start: " >> $LOGFILE
    date >> $LOGFILE

    if [ -f $FINISHFILE ]; then
        echo "Finishfile exists. Skipping. $FINISHFILE" | tee -a $LOGFILE
        return 1
    fi

    local COMMAND="mencoder mf://$IMAGE_DIR/*.png -mf fps=15:type=png -ovc lavc -lavcopts vcodec=mpeg4:vbitrate=2000:mbd=2:trell -oac copy -o $OUTPUT_DIR/$BASENAME.avi"

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
export -f encode_dashboard_images

cat $DD_POSE_DIR/resources/dataset-items-trainval.txt | parallel -j12 -C ' ' encode_dashboard_images {1} {2} {3}
