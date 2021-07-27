import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--attention_size", type=int, default=14, help="vgg16: 14; inceptionV3: 14")
    parser.add_argument("--danet", type=int, default=0, choices=[0, 1],
                        help="0: not use danet; 1: danet.")
    parser.add_argument("--update_rate", type=float, default=0.01,
                        help="update attention rate")
    parser.add_argument("--backbone_rate", default=0.1, type=float)
    parser.add_argument("--function", default='quadratic', type=str,
                        help='select CCAM function: linear, quadratic，mean')
    parser.add_argument("--mean_num", type=int, default=20,
                        help="mean num")
    parser.add_argument("--decay_epoch", type=int, default=2,
                        help="second decay epoch in imagenet")
    parser.add_argument("--decay_rate", type=float, default=0.5,
                        help="second decay rate in imagenet")
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="learning rate")
    parser.add_argument("--seg_thr", type=float, default=0.12,
                        help="segmentation threshold")

    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--time", type=str, default=None,
                        help="procedure run time")
    parser.add_argument("--dataset", type=str, default="cub", choices=["cub", "imagenet"],
                        help="cub: CUB-200-2011; imagenet: ILSVRC 2016")

    parser.add_argument("--backbone", type=str, default="vgg16", choices=["vgg16", "inceptionV3"])

    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--non_local", type=int, default=1, choices=[0, 1],
                        help="0: vgg16; 1: the vgg16 that adds non-local attention")
    parser.add_argument("--shared_classifier", type=int, default=1, choices=[0, 1],
                        help="0: not shared; 1: shared classifier")
    parser.add_argument("--combination", type=int, default=1, choices=[0, 1],
                        help="0: common cam; 1: use linear function to combination cam")
    parser.add_argument("--input_size", type=int, default=256,
                        help="image resize input size")

    parser.add_argument("--decay", type=int, default=1, choices=[0, 1],
                        help="0: learning rate is fixed; 1: learning rate is decay")
    parser.add_argument("--attention", type=int, default=1, choices=[0, 1],
                        help="0: not update attention; 1: update attention")
    parser.add_argument("--pretrain", type=int, default=1, choices=[0, 1],
                        help="0: not use pre-train vgg16 conv weight; 1: use pre-train vgg16 conv weight")
    parser.add_argument("--model_path", type=str, default=None,
                        help="None: random init. Else, load model path weight")

    args = parser.parse_args()
    if args.dataset == "cub":
        args.num_classes = 200
    else:
        args.num_classes = 1000
    args.dir = "dataset_"+str(args.dataset) \
               + "_inpSize_" + str(args.input_size) \
               + "_cropSize_" + str(args.crop_size) \
               + "_attenSize_" + str(args.attention_size) \
               + "_backbone_" + str(args.backbone) \
               + "_backbone_rate_" + str(args.backbone_rate) \
               + "_func_" + str(args.function) \
               + "_meanNum_" + str(args.mean_num) \
               + "_dEpo_" + str(args.decay_epoch) \
               + "_dRate_" + str(args.decay_rate) \
               + "_lr_" + str(args.lr) \
               + "_segThr_" + str(args.seg_thr) \
               +"_epoch_" + str(args.epoch) \
                +"_danet_" + str(args.danet)

    return args

if __name__ == '__main__':
    args_parser()

