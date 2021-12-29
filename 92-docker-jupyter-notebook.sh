docker run -it --rm -v $PWD:/dd-pose -p 8888:8888/tcp --name dd_pose_jupyter dd-pose:latest \
    bash -c 'source 00-activate.sh; jupyter notebook --ip=0.0.0.0 --no-browser --allow-root'
