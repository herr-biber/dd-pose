#!/bin/bash

# make sure 00-activate.sh is run beforehand
if [ -z ${DD_POSE_SOURCED+x} ]; then echo "DD_POSE_SOURCED environment variable is not set. Have you sourced 00-activate.sh?"; exit 1; fi

export DOWNLOAD_DIR=$DD_POSE_DATA_ROOT_DIR/00-download

echo "Extracting correctly downloaded data from $DOWNLOAD_DIR to $DD_POSE_DATA_DIR"

mkdir -p $DD_POSE_DATA_DIR

function untar()
{
    local FILE=$1 # basename without .md5sum extension
    local FILE_ABS=$DOWNLOAD_DIR/$FILE

    echo -n "About to extract $FILE: "
    if [ ! -f $FILE_ABS.md5sum-correct ]; then
        echo "md5sum check was not successful. Skipping extraction. Have you run 03-compare-md5sums.sh?"
        return 1
    fi

    if tar xzf $FILE_ABS -C $DD_POSE_DATA_DIR; then
        echo "ok"
    fi
}

# make function visible to gnu parallel
export -f untar

cat $DD_POSE_DIR/resources/download-file-names.txt | parallel untar {}
