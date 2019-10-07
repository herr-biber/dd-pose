#!/bin/bash

# make sure 00-activate.sh is run beforehand
if [ -z ${DD_POSE_SOURCED+x} ]; then echo "DD_POSE_SOURCED environment variable is not set. Have you sourced 00-activate.sh?"; exit 1; fi

export DOWNLOAD_DIR=$DD_POSE_DATA_ROOT_DIR/00-download
echo "Comparing md5sums of downloaded data in $DOWNLOAD_DIR with md5sums from server"
# logfile to store file names with non-matching md5sums
export BAD_MD5SUM_FILE=$DOWNLOAD_DIR/bad-md5sums.txt
# clean file
rm -f $BAD_MD5SUM_FILE

function compare_md5sum()
{
    local FILE=$1 # basename without .md5sum extension
    local FILE_ABS=$DOWNLOAD_DIR/$FILE

    echo -n "checking $FILE: "
    if [ ! -f $FILE_ABS.md5sum ]; then
        echo "md5sum file does not exist. Skipping comparison. Have you run 02-compute-md5sums.sh?"
        return 1
    fi
    if diff <(curl --basic --user "$DD_POSE_USER:$DD_POSE_PASSWORD" $DD_POSE_DOWNLOAD_URI/$FILE.md5sum 2>/dev/null) "$FILE_ABS.md5sum"; then
        echo "ok"
        touch $FILE_ABS.md5sum-correct
    else
        echo "md5sums differ"
        # remove previous correct-file in case md5sum changed (e.g. on server)
        rm -f $FILE_ABS.md5sum-correct
        echo $FILE >> $BAD_MD5SUM_FILE
    fi
}

# make function visible to gnu parallel
export -f compare_md5sum

cat $DD_POSE_DIR/resources/download-file-names.txt | parallel compare_md5sum {}

if [ -f $BAD_MD5SUM_FILE ]; then
    echo "Found mismatching md5sums. Please check file $BAD_MD5SUM_FILE"
else
    echo "All checked md5sums okay."
fi
