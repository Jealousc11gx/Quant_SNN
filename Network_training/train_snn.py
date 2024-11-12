import dill
from quant_net import *
from training_utils import *
import tracemalloc
import math
import gc
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns


def main():
    torch.manual_seed(23)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True
    cudnn.deterministic = True

    args = args_config.get_args()
    print("********** SNN simulation parameters **********")
    print(args)

    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        train_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            train=True,
            transform=transform_train,
            download=True)

        test_dataset = torchvision.datasets.CIFAR10(
            root=args.dataset_dir,
            train=False,
            transform=transform_test,
            download=True)

        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True)

        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=4,
            pin_memory=True)

        num_classes = 10

    elif args.dataset == 'svhn':

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_dataset = torchvision.datasets.SVHN(
            root=args.dataset_dir,
            split='train',
            transform=transform_train,
            download=True)
        test_dataset = torchvision.datasets.SVHN(
            root=args.dataset_dir,
            split='test',
            transform=transform_test,
            download=True)
        train_data_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=True)
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=True)

        num_classes = 10

    elif args.dataset == 'tiny':
        traindir = os.path.join('/gpfs/gibbs/project/panda/shared/tiny-imagenet-200/train')
        valdir = os.path.join('/gpfs/gibbs/project/panda/shared/tiny-imagenet-200/val')
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        train_transforms = transforms.Compose([
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])
        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])

        train_dataset = torchvision.datasets.ImageFolder(traindir, train_transforms)
        test_dataset = torchvision.datasets.ImageFolder(valdir, test_transforms)

        train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                        num_workers=4, pin_memory=True)
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                       num_workers=4, pin_memory=True)

        num_classes = 200
    elif args.dataset == 'dvs':
        train_dataset_dvs = torch.load("./train_dataset_dvs_8.pt", pickle_module=dill)
        test_dataset_dvs = torch.load("./test_dataset_dvs_8.pt", pickle_module=dill)

        train_data_loader = torch.utils.data.DataLoader(train_dataset_dvs,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        num_workers=4,
                                                        pin_memory=True)
        test_data_loader = torch.utils.data.DataLoader(test_dataset_dvs,
                                                       batch_size=args.batch_size,
                                                       shuffle=False,
                                                       num_workers=4,
                                                       pin_memory=True)
        num_classes = 10


    def visualize_weight_distribution(model, epoch):
        os.makedirs('weight_distributions', exist_ok=True)
        plt.figure(figsize=(20, 15))
        plot_index = 1
    
        for name, module in model.named_modules():
            if isinstance(module, QConv2dLIF) and name in ['ConvLif2', 'ConvLif3', 'ConvLif4', 'ConvLif5', 'ConvLif6']:
                # Assuming w_q_inference is a function that returns quantized weights
                quantized_weights, _ = w_q_inference(module.conv_module.weight, module.num_bits_w, module.beta[0])
                weights = quantized_weights.detach().cpu().numpy().flatten()
    
                plt.subplot(5, 2, plot_index)
                sns.histplot(weights, kde=True, bins=50)
                plt.title(f'{name} - Epoch {epoch}')
                plt.xlabel('Weight Value')
                plt.ylabel('Frequency')
    
                plot_index += 1
    
        plt.tight_layout()
        plt.savefig(f'weight_distributions/epoch_{epoch}.png')
        plt.close()

    
    criterion = nn.CrossEntropyLoss()
    if args.arch == 'vgg16':
        model = Q_ShareScale_VGG16(args.T, args.dataset).cuda()  # def __init__(self, time_step, dataset):
    elif args.arch == 'vgg9':
        model = Q_ShareScale_VGG9(args.T, args.dataset).cuda()
    elif args.arch == 'res19':
        model = ResNet19(num_classes, args.T).cuda()
    elif args.arch == 'vgg8':
        model = Q_ShareScale_VGG8(args.T, args.dataset).cuda()

    # print(model)

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr, 0.9, weight_decay=5e-4)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=1e-4)
    else:
        print("Current does not support other optimizers other than sgd or adam.")
        exit()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=0)


    best_accuracy = 0
    dir = f'./logs/{args.arch}/{args.dataset}/T4/4w4b_hard_IF_125epoch'
    writer = SummaryWriter(log_dir= dir)
    print(f'this test\'s result is in{dir}')

    for epoch_ in range(args.epoch):
        loss = 0
        accuracy = 0

        loss = train(args, train_data_loader, model, criterion, optimizer, epoch_)

        accuracy = test(model, test_data_loader, criterion)

        writer.add_scalar('Loss/train', loss, epoch_)
        writer.add_scalar('Accuracy/test', accuracy, epoch_)

        scheduler.step()
        if accuracy > best_accuracy:
            best_accuracy = accuracy

            checkdir(f"{os.getcwd()}/model_dumps/T4/4w4b_hard_IF_125epoch")
            torch.save(model,
                       f"{os.getcwd()}/model_dumps/T4/4w4b_hard_IF_125epoch/final_dict_4w4b_hard_IF.pth")

        if (epoch_ + 1) % args.test_display_freq == 0:
            print(
                f'Train Epoch: {epoch_}/{args.epoch} Loss: {loss:.6f} Accuracy: {accuracy:.3f}% Best Accuracy: {best_accuracy:.3f}%')
        if (epoch_ == 24  or epoch_ == 49 or epoch_ == 74 or epoch_ == 99):
            torch.save(model, f"{os.getcwd()}/model_dumps/T4/4w4b_hard_IF_125epoch/{epoch_}_dict_4w4b_hard_IF.pth")
            print(f"model saved in epoch {epoch_}")

    writer.close()


def train(args, train_data, model, criterion, optimizer, epoch):
    model.train()

    for batch_idx, (imgs, targets) in enumerate(train_data):
        train_loss = 0.0
        optimizer.zero_grad()
        imgs, targets = imgs.cuda(), targets.cuda()

        output = model(imgs)

        train_loss = sum([criterion(s, targets) for s in output]) / args.T


        train_loss.backward()
        if args.share:
            for m in model.modules():
                if isinstance(m, QConvBN2dLIF):
                    # print(m.scaling.grad)
                    m.beta[0].grad.data = m.beta[0].grad / math.sqrt(
                        torch.numel(m.conv_module.weight) * (2 ** (m.num_bits_w - 1) - 1))
                elif isinstance(m, QConvBN2d):
                    m.beta[0].grad.data = m.beta[0].grad / math.sqrt(
                        torch.numel(m.conv_module.weight) * (2 ** (m.num_bits_w - 1) - 1))
        optimizer.step()

    return train_loss.item()


if __name__ == '__main__':
    main()
