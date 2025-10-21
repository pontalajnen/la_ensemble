python3 evaluate.py \
    --save_file_name test_resnet18_savefile.json \
    --model_path_file first_eval.txt \
    --model_type ResNet18 \
    --dataset CIFAR10 \
    --batch_size 8 \
    --no-eval_train \
    "$@"
