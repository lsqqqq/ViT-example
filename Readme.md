# Vision Transformer Implementation

## Background

This is a simple implementation of ViT using Timm library.

The program includes three parts:

* tools.py: defined the transfer learning, dataset management, and drawing functions
* models.py: defined the ViT model.
* train.py: defined the training function

These three files include detailed comments illustrating the Vision Transformer's data flow and its training process.

The test environment includes:
* One Nvidia A6000

To initialize a new environment, we can create a new python environment using Conda and install the required packages by `pip install -r requirements.txt`.


## Train and Visualization

To simply test the performance of ViT, we can run `train.py`:

```python
!python train.py
```

The program also provides visualization results.

The **data distribution** and the **training result** can be found under the folder `result`.

Besides, all the training results are stored on **tensorboard**. The tensorboard can be visited on `http://server_ip:7777/` after running the following command.

```python
!tensorboard --logdir=result --host=0.0.0.0 --port=7777
```

## Exploring various hyper-parameters

By default, the program will use **data augmentation** method and **scheduler** during the training process. The default settings include the following:

* batch_size    : 8
* custom_transform  : rotation
* augment       : 3
* lr            : 0.00001
* num_epochs    : 10
* min_r         : 0.5
* dataset_dir   : warwick_CLS

By running `python train.py --help`, we can acquire the meanings and the default settings of those parameters inside my code.

```bash
usage: train.py [-h] [--batch_size BATCH_SIZE]
                [--custom_transform CUSTOM_TRANSFORM] [--augment AUGMENT]
                [--lr LR] [--num_epochs NUM_EPOCHS] [--min_r MIN_R]
                [--dataset_dir DATASET_DIR]

Vision transformer training example

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Batch size.
  --custom_transform CUSTOM_TRANSFORM
                        Where you stored the dataset.
  --augment AUGMENT     Dataset augmentation times (by rotating the input
                        image)
  --lr LR               Learning rate.
  --num_epochs NUM_EPOCHS
                        Number of epochs for training.
  --min_r MIN_R         Minimum learning rate ratio. Used by the scheduler.
  --dataset_dir DATASET_DIR
                        Where you stored the dataset.
```