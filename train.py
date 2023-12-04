import torch 
from torch import nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
import torchmetrics
from torchvision import transforms, datasets
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl 
from pytorch_lightning.loggers import CSVLogger
import pandas as pd  
import os, sys
from pathlib import Path
import matplotlib.pyplot as plt
import rasterio as rio
import math

from flops_counter import get_model_complexity_info
from visualisation_utils import plot_csv_values, save_metrics_to_csv, plot_all_data

import argparse

MAX_EPOCHS = 30

# Create the parser
parser = argparse.ArgumentParser(description='Parser for training with BatchSize and LearningRate')

# Add arguments
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training (default: 32)')
parser.add_argument('--learning-rate', type=float, default=0.01,
                    help='learning rate for the optimizer (default: 0.01)')
parser.add_argument('--precision', type=int, default=32,
                    help='Precision for training the model (default: 16 floating point numbers)')

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using {torch.cuda.get_device_name()} for training.")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU for training.")

torch.set_float32_matmul_precision('medium')  # For performance
seed = pl.seed_everything(42, workers=True) # For reproducibility

# Initialize a CSVLogger
home = '/home/roberto/PythonProjects/S2RAWVessel/mmdetection/data/Venus/classification'
csv_logger = CSVLogger('.', name='lightning_logs')


# 2. Model definition
efficientnet_lite_params = {
    # width_coefficient, depth_coefficient, image_size, dropout_rate
    'efficientnet_lite0_venus': [1.0, 1.0, 128, 0.3],
    'efficientnet_lite0': [1.0, 1.0, 224, 0.2],
    'efficientnet_lite1': [1.0, 1.1, 240, 0.2],
    'efficientnet_lite2': [1.1, 1.2, 260, 0.3],
    'efficientnet_lite3': [1.2, 1.4, 280, 0.3],
    'efficientnet_lite4': [1.4, 1.8, 300, 0.3],
}

def round_filters(filters, multiplier, divisor=8, min_width=None):
    """Calculate and round number of filters based on width multiplier."""
    if not multiplier:
        return filters
    filters *= multiplier
    min_width = min_width or divisor
    new_filters = max(min_width, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, multiplier):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))

def drop_connect(x, drop_connect_rate, training):
    if not training:
        return x
    keep_prob = 1.0 - drop_connect_rate
    batch_size = x.shape[0]
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=x.dtype, device=x.device)
    binary_mask = torch.floor(random_tensor)
    x = (x / keep_prob) * binary_mask
    return x

class MBConvBlock(nn.Module):
    def __init__(self, inp, final_oup, k, s, expand_ratio, se_ratio, has_se=False):
        super(MBConvBlock, self).__init__()

        self._momentum = 0.01
        self._epsilon = 1e-3
        self.input_filters = inp
        self.output_filters = final_oup
        self.stride = s
        self.expand_ratio = expand_ratio
        self.has_se = has_se
        self.id_skip = True  # skip connection and drop connect

        # Expansion phase
        oup = inp * expand_ratio  # number of output channels
        if expand_ratio != 1:
            self._expand_conv = nn.Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._momentum, eps=self._epsilon)

        # Depthwise convolution phase
        self._depthwise_conv = nn.Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, padding=(k - 1) // 2, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._momentum, eps=self._epsilon)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(inp * se_ratio))
            self._se_reduce = nn.Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = nn.Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        self._project_conv = nn.Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._momentum, eps=self._epsilon)
        self._relu = nn.ReLU6(inplace=True)

    def forward(self, x, drop_connect_rate=None):
        """
        :param x: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        identity = x
        if self.expand_ratio != 1:
            x = self._relu(self._bn0(self._expand_conv(x)))
        x = self._relu(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._relu(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        if self.id_skip and self.stride == 1  and self.input_filters == self.output_filters:
            if drop_connect_rate:
                x = drop_connect(x, drop_connect_rate, training=self.training)
            x += identity  # skip connection
        return x

class EfficientNetLite(nn.Module):
    def __init__(self, widthi_multiplier, depth_multiplier, num_classes, drop_connect_rate, dropout_rate):
        super(EfficientNetLite, self).__init__()

        # Batch norm parameters
        momentum = 0.01
        epsilon = 1e-3
        self.drop_connect_rate = drop_connect_rate
        input_channels = 3  # RGB
        input_channels = 12 # Spectral bands of Venus
        
        
        mb_block_settings = [
            #repeat|kernal_size|stride|expand|input|output|se_ratio
                [1, 3, 1, 1, 32,  16,  0.25],
                [2, 3, 2, 6, 16,  24,  0.25],
                [2, 5, 2, 6, 24,  40,  0.25],
                [3, 3, 2, 6, 40,  80,  0.25],
                [3, 5, 1, 6, 80,  112, 0.25],
                [4, 5, 2, 6, 112, 192, 0.25],
                [1, 3, 1, 6, 192, 320, 0.25]
            ]

        # Stem
        out_channels = 32
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels, momentum=momentum, eps=epsilon),
            nn.ReLU6(inplace=True),
        )

        # Build blocks
        self.blocks = nn.ModuleList([])
        for i, stage_setting in enumerate(mb_block_settings):
            stage = nn.ModuleList([])
            num_repeat, kernal_size, stride, expand_ratio, input_filters, output_filters, se_ratio = stage_setting
            # Update block input and output filters based on width multiplier.
            input_filters = input_filters if i == 0 else round_filters(input_filters, widthi_multiplier)
            output_filters = round_filters(output_filters, widthi_multiplier)
            num_repeat= num_repeat if i == 0 or i == len(mb_block_settings) - 1  else round_repeats(num_repeat, depth_multiplier)
            

            # The first block needs to take care of stride and filter size increase.
            stage.append(MBConvBlock(input_filters, output_filters, kernal_size, stride, expand_ratio, se_ratio, has_se=False))
            if num_repeat > 1:
                input_filters = output_filters
                stride = 1
            for _ in range(num_repeat - 1):
                stage.append(MBConvBlock(input_filters, output_filters, kernal_size, stride, expand_ratio, se_ratio, has_se=False))
            
            self.blocks.append(stage)

        # Head
        in_channels = round_filters(mb_block_settings[-1][5], widthi_multiplier)
        out_channels = 1280
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=out_channels, momentum=momentum, eps=epsilon),
            nn.ReLU6(inplace=True),
        )

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        if dropout_rate > 0:
            self.dropout = nn.Dropout(dropout_rate)
        else:
            self.dropout = None
        self.fc = torch.nn.Linear(out_channels, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.stem(x)
        idx = 0
        for stage in self.blocks:
            for block in stage:
                drop_connect_rate = self.drop_connect_rate
                if drop_connect_rate:
                    drop_connect_rate *= float(idx) / len(self.blocks)
                x = block(x, drop_connect_rate)
                idx +=1
        x = self.head(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)

        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1.0/float(n))
                m.bias.data.zero_()
    
    def load_pretrain(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict, strict=True)      

def build_efficientnet_lite(name, num_classes):
    width_coefficient, depth_coefficient, _, dropout_rate = efficientnet_lite_params[name]
    model = EfficientNetLite(width_coefficient, depth_coefficient, num_classes, 0.2, dropout_rate)
    return model

# 3. Use PyTorch Lightning for Training
class StratifiedImageDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        # Ensure the custom loader is passed to the super class
        super().__init__(root, loader=self.loader, transform=transform)
        self.filepaths = self.samples

    @staticmethod
    def loader(x):
        try:
            # Open the image file using rasterio
            with rio.open(x) as src:
                # Read the data
                image_data = src.read()  # shape (channels, height, width)
            
            # Convert the NumPy array to a PyTorch tensor
            tensor = torch.from_numpy(image_data).float()
            # Normalize the tensor to 0-1 range if it's a 16-bit image
            if tensor.max() > 1.0:
                tensor /= 65535

            return tensor
        except rio.RasterioIOError as e:
            # Handle exceptions raised by rasterio
            print(f"RasterioIOError: Could not open {x}: {e}")
            raise e
        except Exception as e:
            # Handle any other exceptions
            print(f"Unexpected error occurred: {e}")
            raise e

    def __getitem__(self, index):
        path, target = self.filepaths[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

class SVenusClassifier(pl.LightningModule):
    def __init__(self, image_dir='.', batch_size=32, lr=1e-3,image_size=(128, 128), train_split=0.5, val_split=0.3):
        super().__init__()
        self.num_workers = 7

        model_name = 'efficientnet_lite0_venus'
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.image_size = image_size
        # self.model = timm.create_model(model_name, pretrained=True)
        categories = [x.name for x in Path(image_dir).glob("*") if (x.is_dir()) and ('pycache' not in x.name)]
        self.num_classes = len(categories)
        self.model = build_efficientnet_lite(model_name, self.num_classes)
        # dataset: 
        self.transform = transforms.Compose([
            transforms.Resize(image_size, antialias=True),
            # Include any other transformations and normalization here
        ])
        
        self.train_split = train_split
        self.val_split = val_split
        self.lr = lr # learning rate
        self.loss_fn = torch.nn.CrossEntropyLoss()
        # metrics:
        self.precision = torchmetrics.Precision(num_classes=self.num_classes, average='macro')
        self.recall = torchmetrics.Recall(num_classes=self.num_classes, average='macro')
        
        # self.f1_score = torchmetrics.F1(num_classes=self.num_classes, average='macro')
        self.f1_score = torchmetrics.classification.f_beta.F1Score(num_classes=self.num_classes, average='macro')
        # self.confmat = torchmetrics.ConfusionMatrix(num_classes=self.num_classes)
        self.cohen_kappa = torchmetrics.CohenKappa(num_classes=self.num_classes)
        self.balanced_accuracy = torchmetrics.Accuracy(average='macro', num_classes=self.num_classes)
        # self.prompt_mcc = MatthewsCorrCoef(num_classes=self.num_classes)
        # Instantiate the split
        self.setup()
        
    def setup(self, stage=None):
        # Load the entire dataset
        full_dataset = StratifiedImageDataset(root=self.image_dir, transform=self.transform)

        # Create a list to hold the indices for stratified split
        train_indices, val_indices, test_indices = [], [], []

        # Get the list of classes and the number of classes
        classes = full_dataset.classes
        class_to_idx = full_dataset.class_to_idx

        # Perform stratified split
        for class_name in classes:
            # Get all file indices for the current class
            class_indices = [i for i, (_, class_id) in enumerate(full_dataset.samples) if class_id == class_to_idx[class_name]]

            # Calculate split sizes for the current class
            train_size = int(self.train_split * len(class_indices))
            val_size = int(self.val_split * len(class_indices))
            test_size = len(class_indices) - train_size - val_size

            # Perform the split
            class_train_indices, temp_indices = train_test_split(class_indices, train_size=train_size, stratify=None)
            class_val_indices, class_test_indices = train_test_split(temp_indices, test_size=test_size, stratify=None)

            # Add the class split indices to the respective dataset lists
            train_indices.extend(class_train_indices)
            val_indices.extend(class_val_indices)
            test_indices.extend(class_test_indices)

        # Create subsets for each dataset
        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)
        self.test_dataset = Subset(full_dataset, test_indices)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=self.num_workers)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
        return {
                    'optimizer': optimizer,
                    'lr_scheduler': scheduler,
                    'monitor': 'cohen_kappa',  # optional key used for early stopping
                }
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)

        # Update metrics
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        metrics = {'train_loss': loss,'train_acc':acc, 'lr': self.lr}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        # Update metrics
        preds = torch.argmax(y_hat, dim=1)
        self.precision.update(preds, y)
        self.recall.update(preds, y)
        self.f1_score.update(preds, y)
        self.cohen_kappa.update(preds, y)
        self.balanced_accuracy.update(preds, y)

        metrics = {'val_loss': loss, 'val_acc': acc,'lr': self.lr, 'precision': self.precision, 'recall': self.recall, 'f1_score': self.f1_score, 'cohen_kappa': self.cohen_kappa, 'balanced_accuracy': self.balanced_accuracy}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return metrics

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.precision.update(preds, y)
        self.recall.update(preds, y)
        self.f1_score.update(preds, y)
        self.cohen_kappa.update(preds, y)
        self.balanced_accuracy.update(preds, y)
        
        metrics = {'test_loss': loss, 'test_acc': acc, 'test_precision': self.precision, 'test_recall': self.recall, 'test_f1_score': self.f1_score, 'test_cohen_kappa': self.cohen_kappa, 'test_balanced_accuracy': self.balanced_accuracy}
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.test_results = metrics
        return metrics
    
    # def on_validation_epoch_end(self):

    # def on_test_epoch_end(self):

        
    def forward(self, x):
        return self.model(x)
    


if __name__ == '__main__':
    # Parse arguments
    args = parser.parse_args()

    # Access the arguments
    BS = args.batch_size
    LR = args.learning_rate
    Pr = args.precision
    # Start training
    venus_classifier = SVenusClassifier(image_dir=f'{home}/V2C', batch_size=BS, image_size=(128, 128), train_split=0.5, val_split=0.3, lr=LR)
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, log_every_n_steps=10, logger=csv_logger, precision=Pr)
    trainer.fit(venus_classifier)
    test_results = trainer.test(venus_classifier)
    save_metrics_to_csv(test_results, f'{home}/test_results/venus_classifier_{BS}_{LR}.csv')

