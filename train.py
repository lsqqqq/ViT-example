import os
import time
import math
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from timm import create_model
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from models import VisionTransformer
from tools import transfer_model, ImageDataSet, analyse_dataset, analyse_training_result


def train(net, train_iter, val_iter, criterion, optimizer, num_epochs, scheduler=None, result_dir='result', device='cpu', 
    visualize=True, exp_name='default'):
    """Train the model, vaildate for each epoch.
    Also generate visualization result, can be check by:
        >>> !tensorboard --logdir=result/tensorboard --port=7777

    Args:
        net (nn.Module): The network model.
        train_iter (DataLoader): Train dataloader.
        val_iter (DataLoader): Validate dataloader.
        criterion : Loss function.
        optimizer : Optimizer.
        num_epochs : Number of epochs for training.
        result_dir : Where to save training result. Defaults to 'result'.
        device (optional): The selected device. Defaults to 'cpu'.
        visualize (bool, optional): Whether to generate figures. Defaults to True.
        exp_name (str, optional): Where to store the latest result.
    """
    net = net.to(device)
    start_time = time.time()

    writer = SummaryWriter(log_dir=os.path.join(result_dir, exp_name))
    curve_data = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'lr': []}
    print(f"training on {device}")

    for epoch in range(num_epochs):
        
        net.train() # Train mode
        train_loss_sum, train_acc_sum, train_count, batch_count, lr_current = 0.0, 0.0, 0, 0, 0
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)   # Move into device
            optimizer.zero_grad()               # Clear optimizer grads, prevent gradient accumulation
            y_pred = net(X)                     # Predict
            loss = criterion(y_pred, y)         # Calculate the loss
            loss.backward()                     # Backward propagate, calculate the gradients of each parameter
            optimizer.step()                    # Update network parameters using the optimizer

            train_loss_sum += loss.cpu().item()
            train_acc_sum += (y_pred.argmax(dim=1) == y).sum().cpu().item()
            train_count += y.shape[0]
            batch_count += 1

        with torch.no_grad():
            net.eval()  # Validation mode
            val_acc_sum, val_count = 0.0, 0
            for X, y in val_iter:
                val_acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                val_count += y.shape[0]
        
        # Change learning rate by scheduler if selected
        if scheduler:
            scheduler.step()                # Change the learning rate by time
            lr_current = scheduler.get_last_lr()[0]
        else:
            lr_current = optimizer.state_dict()['param_groups'][0]['lr']

        train_loss = float("%.4f" % (train_loss_sum / batch_count))
        train_acc = float("%.3f" % (train_acc_sum / train_count))
        val_acc = float("%.3f" % (val_acc_sum / val_count))
        lr_current = float(lr_current)
        
        # Write to tensorboard
        writer.add_scalar('train_loss', train_loss, global_step=epoch)
        writer.add_scalar('train_acc', train_acc, global_step=epoch)
        writer.add_scalar('val_acc', val_acc, global_step=epoch)
        writer.add_scalar('lr_current', lr_current, global_step=epoch)

        curve_data['train_loss'].append(train_loss)
        curve_data['train_acc'].append(train_acc)
        curve_data['val_acc'].append(val_acc)
        curve_data['lr'].append(lr_current)

        print('epoch %d, loss %.4f, train acc %.3f, val acc %.3f, lr %f, time %.1f sec'
              % (epoch + 1, train_loss, train_acc, val_acc, lr_current, time.time() - start_time))
    writer.flush()
    writer.close()

    # Draw figures
    if visualize:
        analyse_training_result(curve_data, save_path=os.path.join(result_dir, exp_name))


def test(net, test_iter, device='cpu'):
    """Get the accuracy in test dataset

    Args:
        net (nn.Module): The network model.
        test_iter (DataLoader): The test dataloader.
        device (str, optional): The selected device. Defaults to 'cpu'.
    """
    net = net.to(device)
    start_time = time.time()
    test_acc_sum, test_count = 0.0, 0
    for X, y in test_iter:
        test_acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
        test_count += y.shape[0]
    
    print('test acc %.3f, time %.1f sec' % (test_acc_sum / test_count, time.time() - start_time))


def main(args):
    """Main training function.

    Args:
        batch_size (int): The batch size.
        custom_transform (str): Name of the customized transform model.
        augment (int): Dataset augmentation times (by rotating the input image).
        lr (float): Learning rate.
        num_epochs (int): Number of epochs for training.
        min_r (float): Minimum learning rate ratio.
        dataset_dir (str): Dataset directory.
    """
    # Fetch hyper-parameters
    batch_size = args.batch_size
    custom_transform = args.custom_transform
    augment = args.augment
    lr = args.lr
    num_epochs = args.num_epochs
    min_r = args.min_r
    dataset_dir = args.dataset_dir

    # Define the experiment name (to save result)
    exp_name = f'vit_{augment}_{batch_size}_{num_epochs}_{time.strftime("%Y-%m-%d_%H-%M-%S",time.localtime())}'

    # Directories
    pretrained_model_save_dir = 'base_model_params'
    trained_model_save_dir = 'trained_model_params'
    result_dir = 'result'

    # Create directories
    for dir in [pretrained_model_save_dir, trained_model_save_dir, result_dir]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    # Choose training device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate datasets
    if custom_transform == 'rotation':
        transform = transforms.Compose([
            transforms.RandomRotation(45),          # Rotate for data augmentation
            transforms.Resize(int(224 * 1.42)),     # Down sampling
            transforms.CenterCrop(224),             # Cut the center of the image
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    elif custom_transform == 'center':
        transform = transforms.Compose([
            transforms.Resize(224),     # Down sampling, fit smaller dimension to img_size
            transforms.CenterCrop(224),             # Cut the center of the image to fit the img_size
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        augment = 1     # Without rotating
    else:
        transform = None

    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')
    test_dir = os.path.join(dataset_dir, 'test')

    train_set = ImageDataSet(train_dir, transform=transform, augment=augment)
    val_set = ImageDataSet(val_dir, transform=transform, augment=augment)
    test_set = ImageDataSet(test_dir)

    # visualize dataset
    analyse_dataset(train_set, val_set, test_set, save_path=os.path.join(result_dir, exp_name))

    # Prepare the data loaders
    train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_iter = DataLoader(val_set, batch_size=batch_size)
    test_iter = DataLoader(test_set, batch_size=batch_size)

    # Load pre-trained model
    pretrained_model = create_model('vit_base_patch16_224', pretrained=True, in_chans=3)
    torch.save(
        pretrained_model.state_dict(), 
        os.path.join(pretrained_model_save_dir, 'vit_base_patch16_224.pt'))

    # Define the model, initialize parameters using transfer learning, 
    # but ignore the last layer (because we only have 2 classes)
    model = VisionTransformer(num_classes=2)
    model = transfer_model(
        model, 
        pretrained_file=os.path.join(pretrained_model_save_dir, 'vit_base_patch16_224.pt'), 
        except_list=['head.weight', 'head.bias'])

    # Use cross entrophy as the loss function
    criterion = nn.CrossEntropyLoss()

    # Use Adam as the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    
    # Cosine Scheduler by https://arxiv.org/pdf/1812.01187.pdf
    s_func = lambda epoch: 0.5 * (1+math.cos(epoch*math.pi/num_epochs)) * (1-min_r) + min_r
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=s_func)

    # Train the model
    train(model, train_iter, val_iter, criterion, optimizer, num_epochs=num_epochs, 
            scheduler=scheduler, result_dir=result_dir, device=device, exp_name=exp_name)

    # Test the model
    test(model, test_iter, device=device)

    # Save the model
    torch.save(model.state_dict(), os.path.join(trained_model_save_dir, 'vit_trained.pt'))


def get_args_parser():
    """Input parameters
    """
    parser = argparse.ArgumentParser(description='Vision transformer training example')

    parser.add_argument('--batch_size', default=8, type=int, help='Batch size.')
    parser.add_argument('--custom_transform', default='rotation', type=str, help='Where you stored the dataset.')
    parser.add_argument('--augment', default=3, type=int, help='Dataset augmentation times (by rotating the input image)')
    parser.add_argument('--lr', default=0.00001, type=float, help='Learning rate.')
    parser.add_argument('--num_epochs', default=10, type=int, help='Number of epochs for training.')
    parser.add_argument('--min_r', default=0.5, type=float, help='Minimum learning rate ratio. Used by the scheduler.')
    parser.add_argument('--dataset_dir', default='warwick_CLS', type=str, help='Where you stored the dataset.')

    return parser


def setup_seed(seed):
    """Fix up the random seed

    Args:
        seed (int): Seed to be applied
    """
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    setup_seed(1)
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)









