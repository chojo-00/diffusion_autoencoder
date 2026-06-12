#!/bin/bash
#SBATCH -J diffae_integrated_tsne_domain_analysis
#SBATCH -t 7-00:00:00
#SBATCH -o logs_eda/%x_%A_%N.out
#SBATCH --mail-type END,TIME_LIMIT_90,REQUEUE,INVALID_DEPEND,BEGIN
#SBATCH --mail-user chobyeongcheon00@gmail.com
#SBATCH -p RTX3090
#SBATCH --gres=gpu:1

export HTTP_PROXY="http://192.168.45.108:3128"
export HTTPS_PROXY="http://192.168.45.108:3128"

JOB_NAME="diffae_integrated_tsne_domain"
DOCKER_IMAGE_NAME="bc_cho/${JOB_NAME}"
DOCKER_CONTAINER_NAME="bc_cho_${JOB_NAME}"
CODE_DIR="/workspace/bc_cho/1_Model/diffae-custom"

docker build -t ${DOCKER_IMAGE_NAME} -f Dockerfile .

if docker ps -q --filter "name=${DOCKER_CONTAINER_NAME}" | grep -q .; then
    docker stop ${DOCKER_CONTAINER_NAME}
fi
if docker ps -a -q --filter "name=${DOCKER_CONTAINER_NAME}" | grep -q .; then
    docker rm ${DOCKER_CONTAINER_NAME}
fi

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
            pip install opencv-python-headless==4.8.0.76 pydicom pandas scikit-learn seaborn tensorboardX && \
            cd ${CODE_DIR} && \
            python integrated_tsne.py \
                --dataset-dir /workspace/bc_cho/0_Project/1_class3_ceph/Mydataset/bccho/asan_processing_ver2_normalized_bitstored_minmax_foldering \
                --ckpt diffae/my_experiment_ver4_Ruler_ON_CLAHE_ON \
                --image-subdir png_RulerON_CLAHE_ON \
                --load-itr 40000 \
                --image-size 512 \
                --batch-size 10 \
                --anb-csv /workspace/bc_cho/0_Project/1_class3_ceph/Mydataset/bccho/asan_processing_ver2_normalized_bitstored_minmax/integrated_anb_results_raw.csv
        "