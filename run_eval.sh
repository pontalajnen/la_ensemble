python3 evaluate.py \
    --save_file_name test.json \
    --model_path_file test.txt \
    --model_type resnet20_packed \
    --dataset cifar10 \
    --batch_size 16 \
    --laplace \
    --hessian_approx diag \
    --subset_of_weights subnetwork \
    --optimize_prior_precision marglik \
    "$@"

    # --subset_of_weights all \