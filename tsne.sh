#!/bin/bash
#SBATCH -J diffae_tsne_asan_processing_ver3_size512_foldering_new
#SBATCH -t 7-00:00:00
#SBATCH -o logs_eda/%x_%A_%N.out
#SBATCH --mail-type END,TIME_LIMIT_90,REQUEUE,INVALID_DEPEND,BEGIN
#SBATCH --mail-user chobyeongcheon00@gmail.com
#SBATCH -p A6000
#SBATCH -w gpu120
#SBATCH --gres=gpu:1

export HTTP_PROXY="http://192.168.45.108:3128"
export HTTPS_PROXY="http://192.168.45.108:3128"

# Define vars
JOB_NAME="diffae_tsne_asan_processing_ver3_size512_foldering_new"
DOCKER_IMAGE_NAME="bc_cho/${JOB_NAME}"
DOCKER_CONTAINER_NAME="bc_cho${JOB_NAME}"
PORT_NUM=4835

# Paths inside the container
CODE_DIR="/workspace/bc_cho/1_Model/diffae-custom"


# Run containers
docker build -t ${DOCKER_IMAGE_NAME} -f Dockerfile .

# Stop running container
if docker ps -q --filter "name=${DOCKER_CONTAINER_NAME}" | grep -q .; then
    echo "Stopping running container: ${DOCKER_CONTAINER_NAME}"
    docker stop ${DOCKER_CONTAINER_NAME}
fi

# Remove existing container
if docker ps -a -q --filter "name=${DOCKER_CONTAINER_NAME}" | grep -q .; then
    echo "Removing stopped container: ${DOCKER_CONTAINER_NAME}"
    docker rm ${DOCKER_CONTAINER_NAME}
fi

# Run containers
docker run --rm \
        --name ${DOCKER_CONTAINER_NAME} \
        --shm-size 1TB \
        --device nvidia.com/gpu=all \
        -v /mnt/nas100/forGPU/bc_cho:/workspace/bc_cho \
        -v /mnt/nas203/dental/IDs:/workspace/bc_cho/0_Project/1_class3_ceph/Mydataset \
        -v /mnt/nas203/dental/new_total_research:/workspace/bc_cho/0_Project/1_class3_ceph/dataset \
        ${DOCKER_IMAGE_NAME} \
        bash -c "
            pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y && \
            rm -rf /usr/local/lib/python3.8/dist-packages/cv2 && \
            pip install opencv-python-headless==4.8.0.76 && \
            pip install seaborn tensorboardX && \
            cd ${CODE_DIR} && \
            python t-sne.py \
                --dataset-dir '/workspace/bc_cho/0_Project/1_class3_ceph/Mydataset/bccho/asan_processing_ver3_size512_foldering_new' \
                --ckpt 'diffae/my_experiment_ver3' \
                --load-itr 0100000 \
                --image-size 512 \
                --batch-size 10 
        "