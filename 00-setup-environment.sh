#!/bin/bash

# make sure 00-activate.sh is run beforehand
if [ -z ${DD_POSE_SOURCED+x} ]; then echo "DD_POSE_SOURCED environment variable is not set. Have you sourced 00-activate.sh?"; exit 1; fi

# ask for password
sudo -k
sudo apt-get update
sudo apt-get install -y python2.7 python-pip parallel curl coreutils imagemagick mencoder python-opencv python-virtualenv
# plotly orca binary dependencies
sudo apt-get install -y libgtk2.0-0 libgconf-2-4

if [ ! -f $DD_POSE_DIR/venv/bin/activate ]; then
    echo "Setting up new virtualenv"
    virtualenv $DD_POSE_DIR/venv
fi
source $DD_POSE_DIR/venv/bin/activate
pip install pip --upgrade
pip install setuptools --upgrade # to fix pip install behind proxy
pip install numpy --upgrade # to fix transformations install
pip install -r $DD_POSE_DIR/requirements.txt --upgrade


# orca for plotly export
# https://github.com/plotly/orca/releases
# https://github.com/plotly/orca#linux
echo "Downloading orca binary..."
mkdir -p $DD_POSE_DIR/bin
curl -L https://github.com/plotly/orca/releases/download/v1.2.1/orca-1.2.1-x86_64.AppImage -o $DD_POSE_DIR/bin/orca-1.2.1-x86_64.AppImage
ln -f -s orca-1.2.1-x86_64.AppImage $DD_POSE_DIR/bin/orca
chmod +x $DD_POSE_DIR/bin/orca
