## MoNuSeg Medical Image Segmentation 

This repository contains code for training and validating a U-Net model on the MoNuSeg dataset for medical image segmentation tasks. The dataset includes input images and corresponding masks, with the option to use XML annotations for mask creation. The U-Net model is trained using a set of experimental parameters, and the validation phase evaluates the model's performance based on various metrics.

## Getting started
Dataset Setup:

Ensure that the MoNuSeg dataset is in the same path as the code files.
Run dataset.py to initialize the MoNuSegDataset class with essential components.
Training:

Run train.py to train the U-Net model on the dataset. The training includes data loading, model initialization, loss calculation, optimization, and monitoring of training metrics over multiple epochs.
Validation:

Run validation.py to evaluate the pre-trained U-Net model on a validation dataset. The script calculates metrics such as IoU, Precision, Recall, and F1 Score.


## Experimental Parameters
Model Architecture: U-Net architecture with parameters such as input channels, output classes, number of filters, and use of batch normalization.


Data Split: Dataset split into 80% training, 20% validation and test sets.


Loss Function and Optimizer: Binary Cross-Entropy with Logits loss function, 
Adam optimizer with a learning rate of 0.001.


Dataloader and Batch Size: Data loaders for training, validation, and test sets created using DataLoader from torch.utils.data. Batch size set to 4.


Training Epochs: Model trained over 25 epochs.


Random Seed: Random state set to 42 for reproducibility.


Data Augmentation: Horizontal and vertical flip used to diversify the training dataset.

## Training Phase Results
The U-Net model was used to train the dataset. The training phase, as implemented in the
train.py script, includes data loading, model initialization, loss calculation, optimization, and
monitoring of training and validation metrics over multiple epochs.

![image](https://github.com/samuelgjy/resale-price-final/assets/110824653/9da83682-ffcf-457b-bf1f-2ab93deedd62)
A decreasing trend in Loss over Epoch graph suggests that the U-Net model is learning and
improving its performance and is generalizing well to new and unseen data.
An increasing trend in IoU epoch suggests that the model is getting better at accurately
predicting the segmentation masks on the training data and its performance

## Validation Phase
The validation phase, as implemented in the validation.py script, serves the purpose of
evaluating the performance of a pre-trained U-Net model on a validation dataset
![image](https://github.com/samuelgjy/resale-price-final/assets/110824653/536eabb1-b888-4651-925f-43d8aacab132)
Average IoU on Validation Set: 0.6200; 62% of the predicted segmentation overlaps with the actual tissue regions in the validation set.


Average Precision: 0.7684; approximately 76.84% of the model's positive predictions are correct on average.


Average Recall: 0.7621; on average, the model successfully identifies approximately 76.21% of all actual tissue regions in the validation set.


Average F1 Score : 0.7650; suggests a balanced trade-off, indicating a reasonably good overall performance in semantic segmentation tasks.
