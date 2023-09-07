#!/bin/bash

# make sure 00-activate.sh is run beforehand
if [ -z ${DD_POSE_SOURCED+x} ]; then echo "DD_POSE_SOURCED environment variable is not set. Have you sourced 00-activate.sh?"; exit 1; fi

# ask for password
sudo -k
sudo apt-get update
sudo apt-get install -y python python-dev python-pip parallel curl coreutils imagemagick mencoder python-virtualenv

if [ ! -f $DD_POSE_DIR/venv/bin/activate ]; then
    echo "Setting up new virtualenv"
    virtualenv $DD_POSE_DIR/venv
fi
source $DD_POSE_DIR/venv/bin/activate

pip install pip==20.0.2 --upgrade
pip install numpy==1.16.6 --upgrade  # make sure numpy is installed before transformations.py to have proper ABI
pip install -r $DD_POSE_DIR/requirements-py27.txt --upgrade

echo "Done!"
echo "Make sure you re-source 00-activate.sh to use the virtualenv"
