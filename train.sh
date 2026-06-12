#!/bin/bash
#SBATCH -J diffae_train_kooaldam_processing_ver2_normalized_bitstored_minmax_foldering_png_ruleron_claheoff
#SBATCH -t 7-00:00:00
#SBATCH -o logs_eda/%x_%A_%N.out
#SBATCH --mail-type END,TIME_LIMIT_90,REQUEUE,INVALID_DEPEND,BEGIN
#SBATCH --mail-user chobyeongcheon00@gmail.com
#SBATCH -p A6000
#SBATCH --gres=gpu:3

export HTTP_PROXY="http://192.168.45.108:3128"
export HTTPS_PROXY="http://192.168.45.108:3128"

# Define vars
JOB_NAME="diffae_train_kooaldam_processing_ver2_normalized_bitstored_minmax_foldering_png_ruleron_claheoff"
DOCKER_IMAGE_NAME="bc_cho/${JOB_NAME}"
DOCKER_CONTAINER_NAME="bbc_cho${JOB_NAME}"
PORT_NUM=4945

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
            cd ${CODE_DIR} && \
            python train.py \
                --name my_experiment_ver2_Kooaldam_Ruler_ON_CLAHE_OFF \
                --dataset-dir \
                    "/workspace/bc_cho/0_Project/1_class3_ceph/Mydataset/bccho/kooaldam_processing_ver2_normalized_bitstored_minmax_foldering_new" \
                --image-subdir png_RulerON_CLAHE_OFF \
                --image-size 512 \
                --in-channels 1 \
                --batch-size 64 \
                --microbatch 1 \
                --num-itr 50000 \
                --n-gpu-per-node 2 \
                --log-writer tensorboard
        "





: << 'END_COMMENT'
        bash -c "
            cd ${CODE_DIR} && \
            python train.py \
                --name my_experiment_ver4_Ruler_ON_CLAHE_OFF \
                --dataset-dir \
                    "/workspace/bc_cho/0_Project/1_class3_ceph/Mydataset/bccho/kooaldam_processing_ver2_normalized_bitstored_minmax_foldering" \
                --image-subdir png_RulerON_CLAHE_OFF \
                --image-size 512 \
                --in-channels 1 \
                --batch-size 64 \
                --microbatch 1 \
                --num-itr 100000 \
                --n-gpu-per-node 1 \
                --log-writer tensorboard \
                --ckpt my_experiment_ver4_Ruler_ON_CLAHE_OFF \
                --start-itr 50000 \
        "
        bash -c "
            cd ${CODE_DIR} && \
            python train.py \
                --name my_experiment_ver1_total_data_Ruler_ON_CLAHE_OFF \
                --dataset-dir \
                    "/workspace/bc_cho/0_Project/1_class3_ceph/Mydataset/bccho/kooaldam_processing_ver2_normalized_bitstored_minmax_foldering" \
                    "/workspace/bc_cho/0_Project/1_class3_ceph/Mydataset/bccho/kooaldam_processing_ver2_normalized_bitstored_minmax_foldering" \
                --image-subdir png_RulerON_CLAHE_OFF \
                --image-size 512 \
                --in-channels 1 \
                --batch-size 64 \
                --microbatch 1 \
                --num-itr 100000 \
                --n-gpu-per-node 1 \
                --log-writer tensorboard \
                --ckpt my_experiment_ver1_total_data_Ruler_ON_CLAHE_OFF \
                --start-itr 40000 \
        "
END_COMMENT
