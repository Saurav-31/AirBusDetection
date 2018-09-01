import argparse


def parse_cmd_args():
    conf = argparse.ArgumentParser()

    conf.add_argument("-data", required=True, help="Root directory for Input data")
    conf.add_argument("-model", required=False, help="Model For pre-training")
    conf.add_argument("-max_epochs", default=5, help="Number of epochs for training")
    conf.add_argument("-batch_size", default=32, help="Batch Size for training")
    conf.add_argument("-gpu", required=True, help="Gpu to be used")
    conf.add_argument("-save_path", required=True, help="Path where to save the model")
    conf.add_argument("-lr", default=0.001, required=False, help="Learning Rate")
    conf.add_argument("-scale", default=False, required=True, help="Scale Image or Not (Boolean)")
    args = vars(conf.parse_args())

    if args.model == "resnet-50":
        imsize = 224
    elif args.model == 'resnet-150':
        imsize = 299
    elif args.model == 'vgg-16':
        imsize = 256
    else:
        print("Wrong Model Chosen")
        exit()
    args['mconf'] = {'imsize': 224}

    return args


args = parse_cmd_args()
print(args)

