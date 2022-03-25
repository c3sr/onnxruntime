IMAGE_NAME=onnxruntime-dev
CONTAINER_NAME=onnxruntime-agent-dev
TODAY=`date -u +"%Y.%m.%d-%H.%M"`

# If docker complains about disk space
#docker builder prune --force

docker build -t ${IMAGE_NAME}:$TODAY -t ${IMAGE_NAME}:latest .
if [ $? -eq 0 ]; then
# if you want to do container rotation, uncomment this and remove the --rm docker run
#  docker rm ${CONTAINER_NAME}-old > /dev/null
#  docker stop $CONTAINER_NAME
#  docker rename $CONTAINER_NAME ${CONTAINER_NAME}-old

docker run -it --rm --name ${CONTAINER_NAME} \
--hostname ${CONTAINER_NAME} \
-e MQ_USER=mlmodelscope -e MQ_PASSWORD=mlmodelscope -e MQ_HOST=172.17.0.5 -e  MQ_PORT=5672 -e MQ_TIMEOUT_MS=5000 \
-v ${PWD}/data:/root/data \
-v ${PWD}/Docker-bashrc:/root/.bashrc \
-v ${PWD}/.carml_config:/root/.carml_config.yaml \
${IMAGE_NAME}:latest /bin/bash
fi
