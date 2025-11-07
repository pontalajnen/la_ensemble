python3 evaluate.py \
    --save_file_name resnet18_cifar10_sgd.json \
    --model_path_file resnet18_cifar10_sgd.txt \
    --model_type resnet18 \
    --dataset cifar10 \
    --batch_size 8 \
    --no-eval_train \
    # --hessian_approx kron \
    # --laplace \
    "$@"
