#!/bin/bash

# make sure 00-activate.sh is run beforehand
if [ -z ${DD_POSE_SOURCED+x} ]; then echo "DD_POSE_SOURCED environment variable is not set. Have you sourced 00-activate.sh?"; exit 1; fi

export DOWNLOAD_DIR=$DD_POSE_DATA_ROOT_DIR/00-download
echo "Downloading data to $DOWNLOAD_DIR"
mkdir -p $DOWNLOAD_DIR

function download_file()
{
    local FILE=$1 # basename
    local DEST=$DOWNLOAD_DIR/$FILE
    echo "About to download file $FILE"
    if [ -f $DEST.downloaded ]; then
        echo "Finish file exists. Skipping download. $DEST.downloaded"
        return 1
    fi

    # check, whether local file and remote file have same file size
    if [ -f $DEST ]; then
        local FILE_SIZE=$(du -b $DEST | cut -f1)
    else
        local FILE_SIZE="-1"
    fi

    STATUS_CODE=$(curl --fail --basic --user "$DD_POSE_USER:$DD_POSE_PASSWORD" --head --silent --write-out "%{http_code}" $DD_POSE_DOWNLOAD_URI/$FILE  --output /dev/null)
    # echo "Status code: $STATUS_CODE"
    if [ $STATUS_CODE -eq 303 ]; then
        echo "User not approved, yet. Please trigger approval at DD-Pose team"
        return 2
    fi

    if [ -f $DEST ] && curl --fail --silent --basic --user "$DD_POSE_USER:$DD_POSE_PASSWORD" -i --head $DD_POSE_DOWNLOAD_URI/$FILE | grep "Content-Length: $FILE_SIZE"; then
        echo "File already downloaded and same size"
        touch $DEST.downloaded
        return 3
    fi

    # download with autoresume
#    if curl --fail --basic --user "$DD_POSE_USER:$DD_POSE_PASSWORD" $DD_POSE_DOWNLOAD_URI/$FILE --output $DEST --continue-at -; then
    # download without autoresume
    if curl --fail --basic --user "$DD_POSE_USER:$DD_POSE_PASSWORD" $DD_POSE_DOWNLOAD_URI/$FILE --output $DEST ; then
        # create finish file to make sure script has not aborted
        touch $DEST.downloaded
    fi
}

# make function visible to gnu parallel
export -f download_file

# download metadata first
cat $DD_POSE_DIR/resources/download-file-names.txt | grep "metadata\|head-pose" | parallel -j 4 download_file {}
cat $DD_POSE_DIR/resources/download-file-names.txt | parallel -j 4 download_file {}
