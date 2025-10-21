from models.ensemble_model import EnsembleModel
import torch.optim as optim
from models.resnet_packed import ResNet18_packed
import torch
from models.resnet import torch_resnet18, ResNet18
import timm
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import ViTImageProcessor  # , ViTForImageClassification
from torchvision.transforms import v2
# from sam import SAM
from utils.sam import SAM


def init_model(args, device, num_classes):
    if args.model == "ResNet18":
        if args.packed:
            print("[model] Using ResNet18 packed")
            models = ResNet18_packed(num_classes)
        elif args.ensemble:
            models = EnsembleModel(
                num_models=args.num_ensemble_models,
                num_classes=num_classes,
                model=torch_resnet18
            )
        else:
            models = ResNet18(num_classes)

        print("[model] loaded")
        models = models.to(device)

    elif args.model == "ViT":
        models = timm.create_model('vit_base_patch16_224.orig_in21k', pretrained=True, num_classes=num_classes)
        models = models.to(device)
        print("----- Model ready and on device -----")
    else:
        raise Exception("Requested model does not exist! Has to be one of 'ResNet18', 'ViT'")
    if args.distributed:
        models = DDP(models, device_ids=[args.local_rank])

    return models


def init_transformer(args, data_module):
    # Resize the images so that it is compatible with a ViT pretrained on ImageNet-21k
    model_name = 'google/vit-base-patch16-224-in21k'
    processor = ViTImageProcessor.from_pretrained(model_name)
    image_mean, image_std = processor.image_mean, processor.image_std
    # size = processor.size["height"]
    if args.normalize_pretrained_dataset:
        normalize = v2.Normalize(mean=image_mean, std=image_std)
    else:
        if args.dataset == "CIFAR10":
            normalize = v2.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.201])
        elif args.dataset == "CIFAR100":
            normalize = v2.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])

    # resize_transform = v2.Resize((224, 224))

    data_module.train_transform = v2.Compose([
        v2.Resize(256),
        v2.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        v2.RandomHorizontalFlip(),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ])

    data_module.test_transform = v2.Compose([
        v2.Resize(256),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ])

    return data_module


def init_optimizer(args, model):
    if args.base_optimizer == "SGD":
        base_optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
                                   weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.base_optimizer == "AdamW":
        base_optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise Exception("Requested optimizer does not exist! Optimizer has to be one of 'SGD', 'AdamW'")

    # Determine the base optimizer and SAM optimizer setup
    if args.SAM:
        print("----- Using SAM optimizer -----")
        if args.adaptive:
            print("----- Using Adaptive SAM optimizer -----")
        # Set up arguments for both SAM and the base optimizer
        optimizer_args = {
            'params': model.parameters(),
            'base_optimizer': type(base_optimizer),
            'rho': args.rho,
            'adaptive': args.adaptive,
            'lr': args.learning_rate,
            'weight_decay': args.weight_decay,
        }
        if isinstance(base_optimizer, optim.SGD):
            optimizer_args['momentum'] = args.momentum

        # Create the SAM optimizer
        opt = SAM(**optimizer_args)

        # Create the learning rate scheduler for SAM
        if args.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt.base_optimizer, args.epochs)
    else:
        # Use the base optimizer without SAM
        opt = base_optimizer
        if args.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    return opt, scheduler

