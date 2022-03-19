import os.path as osp
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from modules import FeatureDataModule, MlpClassifier

import seaborn
import matplotlib.pyplot as plt


def parse_args(argv=None):
    parser = ArgumentParser(__file__, add_help=False)
    parser.add_argument('name')
    parser = FeatureDataModule.add_argparse_args(parser)
    parser = MlpClassifier.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--earlystop_patience', type=int, default=15)
    parser = ArgumentParser(parents=[parser])
    parser.set_defaults(gpus=1, default_root_dir=osp.abspath(
        osp.join(osp.dirname(__file__), '../data/mlp')))
    args = parser.parse_args(argv)
    return args
 
 
def plot_confusion_matrix(data, labels, output_filename):
    """Plot confusion matrix using heatmap.
 
    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.
 
    """
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))
 
    plt.title("Confusion Matrix")
 
    seaborn.set(font_scale=1.4)
    ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})
 
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
 
    ax.set(ylabel="True Label", xlabel="Predicted Label")
 
    plt.savefig(output_filename, bbox_inches='tight', dpi=300)
    plt.close()


def main(args):
    data_module = FeatureDataModule(args)
    model = MlpClassifier(args)
    logger = TensorBoardLogger(args.default_root_dir, args.name)
    checkpoint_callback = ModelCheckpoint(
        filename='{epoch}-{step}-{val_acc:.4f}', monitor='val_acc',
        mode='max', save_top_k=-1)
    early_stop_callback = EarlyStopping(
        'val_acc', patience=args.earlystop_patience, mode='max', verbose=True)
    trainer = pl.Trainer.from_argparse_args(
        args, logger=logger, 
        callbacks=[checkpoint_callback, early_stop_callback])
    trainer.fit(model, data_module)
    predictions = trainer.predict(datamodule=data_module, ckpt_path='best')
    df = data_module.test_df.copy()
    df['Category'] = torch.concat(predictions).numpy()
    prediction_path = osp.join(logger.log_dir, 'test_prediction.csv')
    df.to_csv(prediction_path, index=False)
    print('Output file:', prediction_path)
    

    nb_classes = 15
    val_loader = data_module.val_dataloader()
    print(val_loader)
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(val_loader):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(classes.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

    print(confusion_matrix)

    confusion_path = osp.join(logger.log_dir, 'confmat.png')
    # define data
    data = confusion_matrix.tolist()
    # define labels
    labels = [i for i in range(0,15)]  
    # create confusion matrix
    plot_confusion_matrix(data, labels, confusion_path)


if __name__ == '__main__':
    main(parse_args())
