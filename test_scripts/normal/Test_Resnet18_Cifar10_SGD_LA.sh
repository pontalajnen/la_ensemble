#!/bin/bash
cd "/Users/kathideckenbach/Documents/Machine Learning Master/Year 2/Master Thesis/laplace_approx_SAM"
export PYTHONPATH=$PWD
echo "!!Start evaluation!!"
/Users/kathideckenbach/anaconda3/envs/thesis/bin/python evaluate.py \
    --save_file_name Test_Resnet18_Cifar10_SGD_LA_full.txt \
    --model_path_file evaluate_test_resnet18_cifar10_sgd.txt \
    --model_type ResNet18 \
    --dataset CIFAR10 \
    --batch_size 128 \
    --laplace \
    --hessian_approx full \
    --use_cpu \

echo "!!Evaluation done!!"