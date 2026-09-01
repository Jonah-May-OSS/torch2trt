import tensorrt as trt
import torch
from datasets.cifar10 import Cifar10Loaders
from models.resnet import resnet18, resnet34
from parser import parse_args
from utils.utilities import calculate_accuracy, mapping_names

from torch2trt import torch2trt

torch.set_printoptions(precision=5)


def main():
    args = parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(78543)

    if args.cuda:
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)

    loaders = Cifar10Loaders()
    loaders.train_loader()
    test_loader = loaders.test_loader()

    if args.m == "resnet18":
        model = resnet18(qat_mode=True, infer=True) if args.netqat else resnet18()
    elif args.m == "resnet34":
        model = resnet34(qat_mode=True, infer=True) if args.netqat else resnet34()
    else:
        raise NotImplementedError(f"{args.m} model not found")

    model = model.cuda().eval()

    if args.load_ckpt:
        checkpoint = torch.load(args.load_ckpt)
        if not args.netqat:
            checkpoint = mapping_names(checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(f"===>>> Checkpoint loaded successfully from {args.load_ckpt} ")

    test_accuracy = calculate_accuracy(model, test_loader)
    print(f" Test accuracy for Pytorch model: {test_accuracy} ")
    rand_in = torch.randn([128, 3, 32, 32], dtype=torch.float32).cuda()

    # Converting the model to TRT
    if args.FP16:
        trt_model_fp16 = torch2trt(
            model,
            [rand_in],
            log_level=trt.Logger.INFO,
            fp16_mode=True,
            max_batch_size=128,
        )
        test_accuracy = calculate_accuracy(trt_model_fp16, test_loader)
        print(f" TRT test accuracy at FP16: {test_accuracy}")

    if args.INT8QAT:
        trt_model_int8 = torch2trt(
            model,
            [rand_in],
            log_level=trt.Logger.INFO,
            fp16_mode=True,
            int8_mode=True,
            max_batch_size=128,
            qat_mode=True,
        )
        test_accuracy = calculate_accuracy(trt_model_int8, test_loader)
        print(f" TRT test accuracy at INT8 QAT: {test_accuracy}")

    if args.INT8PTC:
        ##preparing calib dataset
        calib_dataset = []
        for i, sam in enumerate(test_loader):
            calib_dataset.extend(sam[0])
            if i == 5:
                break

        trt_model_calib_int8 = torch2trt(
            model,
            [rand_in],
            log_level=trt.Logger.INFO,
            fp16_mode=True,
            int8_calib_dataset=calib_dataset,
            int8_mode=True,
            max_batch_size=128,
        )
        test_accuracy = calculate_accuracy(trt_model_calib_int8, test_loader)
        print(f" TRT test accuracy at INT8 PTC: {test_accuracy}")


if __name__ == "__main__":
    main()
