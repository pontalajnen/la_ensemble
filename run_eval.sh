python3 evaluate.py \
    --save_file_name rn20_c10_sgd_packed_la.json \
    --model_path_file resnet20_cifar10_sgd_packed.txt \
    --model_type resnet20 \
    --dataset cifar10 \
    --batch_size 32 \
    --laplace \
    --hessian_approx kron \
    --subset_of_weights last_layer \
    --optimize_prior_precision marglik \
    "$@"

    # --subset_of_weights all \