python3 evaluate.py \
    --save_file_name resnet20_cifar10_sgd_la.json \
    --model_path_file resnet20_cifar10_sgd_batch_norm.txt \
    --model_type resnet20 \
    --dataset cifar10 \
    --batch_size 32 \
    --laplace \
    --hessian_approx kron \
    --subset_of_weights all \
    "$@"
