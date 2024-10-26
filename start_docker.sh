#!/bin/bash  
docker run -it --gpus all --rm --hostname=local-env --network=host \
    -v `pwd`:/opt/ml/env -w /opt/ml/env \
    DOCKER_PATH bash
