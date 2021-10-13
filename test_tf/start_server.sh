#/usr/bin/bash
docker run -t --runtime=nvidia --rm -p 8500:8500 -p 8501:8501 -v "/home/zhangh/Documents/mypython/tests/test_tf:/models/" \
       tensorflow/serving:latest-gpu --model_config_file=/models/model.config \
       --model_config_file_poll_wait_seconds=60
