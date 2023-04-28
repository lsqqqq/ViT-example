"""
Here defines the transfer learning, dataset management and drawing functions
"""

import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pathlib
from matplotlib import pyplot as plt
from collections import Counter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def transfer_model(model, pretrained_model=None, pretrained_file=None, except_list=[]):
    """Set the weights of the target model based on the pre-trained model.
    Can self define a list of layers that will be ignored.

    Ref: https://blog.csdn.net/qq_42178122/article/details/117636996

    Args:
        model (nn.Module): Target model
        pretrained_model (nn.Module or bool, optional): Pre-trained model itself. Defaults to None.
        pretrained_file (str or bool, optional): Path to the state dict file of the pre-trained model. Defaults to None.
        except_list (list, optional): List of layers that will be ignored. eg: ['head.weight', 'head.bias'].

    Returns:
        model: Target model with weights.

    Example:
        >>> model = transfer_model(
        >>>     model, 
        >>>     pretrained_file='./base_model_params/vit_base_patch16_224.pt', 
        >>>     except_list=['head.weight', 'head.bias'])
    """
    assert pretrained_model or pretrained_file, 'need to select either pretrained model or pretrained file'

    # Get pretrained dict
    if pretrained_model:
        pretrained_dict = pretrained_model.load_state_dict()
    else:
        pretrained_dict = torch.load(pretrained_file)

    model_dict = model.state_dict()  # get model dict

    # Remove parameters that not exist in the traget model
    # Will also ignore layers in the except_list
    pretrained_dict = _transfer_state_dict(pretrained_dict, model_dict, except_list)

    # Update target model with parameters from the pretrained model
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    # model.load_state_dict(model_dict, False)   # better
    return model


def _transfer_state_dict(pretrained_dict, model_dict, except_list):
    """Refered by transfer_model, will not been called directly.
    Remove parameters that not exist in the traget model

    Args:
        pretrained_dict : Pre-trained model state dict.
        model_dict : Target model state dict.
        except_list : List of layers that will be ignored. eg: ['head.weight', 'head.bias'].

    Returns:
        state_dict : Filtered pre-trained model state dict
    """
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            if k not in except_list:
                state_dict[k] = v
            else:
                print(f"Ignored weights : {k}")
        else:
            print(f"Missing key(s) in state_dict : {k}")
    return state_dict


class ImageDataSet(Dataset):
    def __init__(self, root_path, transform=None, augment=1, img_size=224):
        """Generate image dataset

        Ref: https://blog.csdn.net/weixin_43917574/article/details/114625616

        The file structure of the image-folder dataset:
            root_path
            ├── class1
            │   └── data1.jpg
            └── class2
                └── data2.jpg

        Args:
            root_path (str): Where we stored the image dataset.
            transform (transforms.Compose, optional): Customized transform function. Defaults to None.
            augment (int, optional): Times of input images in the generated dataset. Defaults to 1.
            img_size (int, optional): Width and height of the image. Defaults to 224.
        
        Example:
            >>> train_set = ImageDataSet(train_path)
            >>> train_iter = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        """
        super(ImageDataSet, self).__init__()
        assert type(augment) == int and type(img_size) == int
        self.augment = augment

        # Fetch all image path
        data_root = pathlib.Path(root_path)
        all_image_paths = list(data_root.glob('*/*'))   # find all sub-directories
        self.all_image_paths = [str(path) for path in all_image_paths]

        # Generate label-to-index dict
        label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
        self.label_to_index = dict((label, index) for index, label in enumerate(label_names))

        # Fetch image index labels
        self.all_image_labels = [self.label_to_index[path.parent.name] for path in all_image_paths]

        # transform
        if not transform:
            self.transform = transforms.Compose([
                transforms.RandomRotation(90),          # Rotate for data augmentation
                transforms.Resize(int(img_size * 1.42)),     # Down sampling, fit smaller dimension to img_size
                transforms.CenterCrop(img_size),             # Cut the center of the image to fit the img_size
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        else:
            self.transform = transform

    def __getitem__(self, index):
        """Called when index is used

        Args:
            index : Selected index

        Returns:
            img : Image in shape (img_size, img_size)
            label : Image index label
        """
        img = Image.open(self.all_image_paths[index // self.augment])
        img = self.transform(img)
        label = torch.tensor(self.all_image_labels[index // self.augment])
        return img, label
    
    def __len__(self):
        """Called when fetching length of the ImageDataSet object

        Returns:
            path_len : Number of image paths
        """
        path_len = self.augment * len(self.all_image_paths)
        return path_len


def analyse_dataset(train_set, val_set, test_set, save_path='result'):
    """Display the distribution of the dataset
    The figure will be saved under save_path/data_distribution.jpg

    Args:
        train_set : Training set.
        val_set : Validate set.
        test_set : Test set.
        save_path : Save directory.
    """

    label_to_index = train_set.label_to_index
    index_to_label = dict(zip(label_to_index.values(), label_to_index.keys()))

    train_info_dict = dict(Counter(train_set.all_image_labels))
    val_info_dict = dict(Counter(val_set.all_image_labels))
    test_info_dict = dict(Counter(test_set.all_image_labels))

    for info in [train_info_dict, val_info_dict, test_info_dict]:
        for idx in index_to_label:
            label = index_to_label[idx]
            info.update({label: info.pop(idx)})

    df = pd.DataFrame(
        [train_info_dict, val_info_dict, test_info_dict], 
        index=['Train', 'Validate', 'Test'])

    # Draw bar chart
    _draw_df_figs(df, save_name='dataset_analysis.jpg', kind='bar', 
        title='Categorical distribution of the dataset', ylabel='Number of cases', save_path=save_path)


def analyse_training_result(curve_data, save_path='result'):
    """Draw curves for training result

    Args:
        curve_data (dict): Training data, eg: {'train_loss': []}.
        save_path (str, optional): Save directory. Defaults to 'result'.
    """
    df = pd.DataFrame(curve_data)

    # Draw line chart
    _draw_df_figs(df, save_name='training_result.jpg', subplots=True, kind='line', 
        xlabel='Epochs', title='Training result', save_path=save_path)


def _draw_df_figs(df, save_name, kind='bar', style='seaborn-colorblind', subplots=False, 
    xlabel=None, ylabel=None, title=None, save_path='result'):
    """Draw figures of the input dataframe.

    Args:
        df (pd.DataFrame): Dataframe dataset.
        save_name (str): Saved file name, suffix included.
        kind (str, optional): Type of the figure. Defaults to 'bar'.
        style (str, optional): Plot style. Defaults to 'seaborn-colorblind'.
        subplots (bool, optional): Whether to draw subplots. Defaults to False.
        xlabel (str, optional): X label. Defaults to None.
        ylabel (str, optional): Y label. Defaults to None.
        title (str, optional): Figure title. Defaults to None.
        save_path (str, optional): Save directory. Defaults to 'result'.
    """
    # Set style
    plt.style.use(style)
    plt.rc('font',family='Times New Roman')
    
    # Plot chart
    df.plot(kind=kind, subplots=subplots, xlabel=xlabel, ylabel=ylabel, title=title)

    # Draw numbers at the top of the bars (auto flexible for dataframe)
    if kind == 'bar':   
        df_len = len(df.columns.to_list())
        step = 1 / (df_len*2)       # The width of a single bar

        for idx in range(len(df.index.values)):
            row = df.index.values[idx]

            width = -step * (df_len-1) / 2
            for col in df.columns.to_list():
                plt.text(idx+width, df.loc[row, col], df.loc[row, col], ha = 'center')
                width += step

    plt.xticks(rotation="horizontal")

    # Save result
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = os.path.join(save_path, save_name)
    plt.savefig(save_path, bbox_inches='tight', dpi=400)