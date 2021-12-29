# this file is meant to be sourced by bash to set environment variables needed for dd-pose
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "This file is meant to be sourced. Please run"
    echo "source $0"
    exit 1
fi

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)"

# variables to configure
DD_POSE_DIR=$THIS_DIR
# this path should have at least 1.8TB of free space (890GB for downloaded files and 900GB for extraction)
# see also https://github.com/herr-biber/dd-pose/issues/4
DD_POSE_DATA_ROOT_DIR=$THIS_DIR/data
DD_POSE_DATA_DIR=$DD_POSE_DATA_ROOT_DIR/01-extracted
DD_POSE_DOWNLOAD_URI=https://dd-pose-dataset.tudelft.nl/eval/downloads/download_file
# user and password you obtained by registering at https://dd-pose-dataset.tudelft.nl
# fill in an uncomment lines
#DD_POSE_USER=
#DD_POSE_PASSWORD=

if [ -z "$DD_POSE_USER" ]; then
    echo "DD_POSE_USER is not set. Please configure activate.sh according to given credentials."
elif [ -z "$DD_POSE_PASSWORD" ]; then
    echo "DD_POSE_PASSWORD is not set. Please configure activate.sh according to given credentials."
else
    # make variables known to subshells
    echo "Setting the following environment variables:"
    echo "DD_POSE_DIR=$DD_POSE_DIR"
    echo "DD_POSE_DATA_ROOT_DIR=$DD_POSE_DATA_ROOT_DIR"
    echo "DD_POSE_DATA_DIR=$DD_POSE_DATA_DIR"
    echo "DD_POSE_DOWNLOAD_URI=$DD_POSE_DOWNLOAD_URI"
    echo "DD_POSE_USER"
    echo "DD_POSE_PASSWORD"
    echo "DD_POSE_SOURCED"
    export DD_POSE_DIR
    export DD_POSE_DATA_ROOT_DIR
    export DD_POSE_DATA_DIR
    export DD_POSE_DOWNLOAD_URI
    export DD_POSE_USER
    export DD_POSE_PASSWORD
    export DD_POSE_SOURCED=

    # add python module to pythonpath
    echo
    echo "Adding \$DD_POSE_DIR module to PYTHONPATH"
    export PYTHONPATH="$DD_POSE_DIR:$PYTHONPATH"

    # activate virtualenv if existing and not within docker environment (which does not use virtualenv)
    if ! [ -z "$IS_DOCKER" ]; then
        echo "Within docker environment. Skipping virtualenv."
    elif [ -f $DD_POSE_DIR/venv/bin/activate ]; then
        echo "Activating virtualenv from $DD_POSE_DIR/venv"
        source $DD_POSE_DIR/venv/bin/activate
    else
        echo "Virtualenv not setup, yet. Please run 00-setup-environment.sh"
    fi

    # echo "Adding $DD_POSE_DIR/bin to \$PATH"
    # export PATH="$DD_POSE_DIR/bin:$PATH"
fi
