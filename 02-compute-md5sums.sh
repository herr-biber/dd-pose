#!/bin/bash

# make sure 00-activate.sh is run beforehand
if [ -z ${DD_POSE_SOURCED+x} ]; then echo "DD_POSE_SOURCED environment variable is not set. Have you sourced 00-activate.sh?"; exit 1; fi

export DOWNLOAD_DIR=$DD_POSE_DATA_ROOT_DIR/00-download
echo "Computing md5sums of downloaded data in $DOWNLOAD_DIR"

function compute_md5sum()
{
    local FILE=$1 # basename
    local FILE_ABS=$DOWNLOAD_DIR/$FILE
    echo "About to compute md5sum of $FILE"
    if [ ! -f $FILE_ABS.downloaded ]; then
        echo "Download finish file from download script does not exist. Skipping md5 computation. $FILE_ABS.finished"
        return 1
    fi

    # make sure we're in DOWNLOAD_DIR to have basenames in md5sum files
    pushd $DOWNLOAD_DIR >/dev/null
    if [ ! -f $FILE_ABS.md5sum ]; then
        md5sum $FILE | tee $FILE_ABS.md5sum
    elif [ $FILE_ABS -nt $FILE_ABS.md5sum ]; then
        echo "md5sum outdated. recomputing"
        md5sum $FILE | tee $FILE_ABS.md5sum
    else
        echo "md5sum already computed"
    fi
    popd >/dev/null
}

# make function visible to gnu parallel
export -f compute_md5sum

cat $DD_POSE_DIR/resources/download-file-names.txt | parallel compute_md5sum {}
