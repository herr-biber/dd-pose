#!/bin/bash

# make sure 00-activate.sh is run beforehand
if [ -z ${DD_POSE_SOURCED+x} ]; then echo "DD_POSE_SOURCED environment variable is not set. Have you sourced 00-activate.sh?"; exit 1; fi

# ask for password
sudo -k
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-pip parallel curl coreutils imagemagick mencoder python3-virtualenv

if [ ! -f $DD_POSE_DIR/venv/bin/activate ]; then
    echo "Setting up new virtualenv"
    virtualenv $DD_POSE_DIR/venv
fi
source $DD_POSE_DIR/venv/bin/activate

pip3 install pip>=19.3 --upgrade
pip install setuptools --upgrade # to fix pip install behind proxy
pip install numpy --upgrade # to fix transformations install
pip install -r $DD_POSE_DIR/requirements.txt --upgrade

echo "Done!"
echo "Make sure you re-source 00-activate.sh to use the virtualenv"
