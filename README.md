Dear Profressor Liu,

Welcome to my Hub. You can review my source code of the programming task by downloading CIFAR-10_training_script.py.

The following content is the details of my training process and outcome shown in the console:

All libraries imported successfully.

Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).

Files will be saved to: /content/drive/My Drive/CDS521_Dissertation_v10/

Running on device: cuda:0

100%|██████████| 170M/170M [00:13<00:00, 12.6MB/s]
CIFAR-10 dataset loaded successfully with data augmentation.

Model created with light dropout (p=0.2).

Using Adam optimizer with weight decay.

Starting Training...

[Epoch 1, Batch 1000] loss: 1.819

[Epoch 1] Average loss: 1.753

[Epoch 2, Batch 1000] loss: 1.542

[Epoch 2] Average loss: 1.523

[Epoch 3, Batch 1000] loss: 1.430

[Epoch 3] Average loss: 1.423

[Epoch 4, Batch 1000] loss: 1.366

[Epoch 4] Average loss: 1.362

[Epoch 5, Batch 1000] loss: 1.322

[Epoch 5] Average loss: 1.319

[Epoch 6, Batch 1000] loss: 1.297

[Epoch 6] Average loss: 1.293

[Epoch 7, Batch 1000] loss: 1.270

[Epoch 7] Average loss: 1.269

[Epoch 8, Batch 1000] loss: 1.248

[Epoch 8] Average loss: 1.252

[Epoch 9, Batch 1000] loss: 1.231

[Epoch 9] Average loss: 1.228

[Epoch 10, Batch 1000] loss: 1.230

[Epoch 10] Average loss: 1.220

[Epoch 11, Batch 1000] loss: 1.214

[Epoch 11] Average loss: 1.212

[Epoch 12, Batch 1000] loss: 1.189

[Epoch 12] Average loss: 1.197

[Epoch 13, Batch 1000] loss: 1.186

[Epoch 13] Average loss: 1.187

[Epoch 14, Batch 1000] loss: 1.178

[Epoch 14] Average loss: 1.182

[Epoch 15, Batch 1000] loss: 1.176

[Epoch 15] Average loss: 1.175

[Epoch 16, Batch 1000] loss: 1.167

[Epoch 16] Average loss: 1.166

[Epoch 17, Batch 1000] loss: 1.149

[Epoch 17] Average loss: 1.150

[Epoch 18, Batch 1000] loss: 1.156

[Epoch 18] Average loss: 1.152

[Epoch 19, Batch 1000] loss: 1.147

[Epoch 19] Average loss: 1.148

[Epoch 20, Batch 1000] loss: 1.146

[Epoch 20] Average loss: 1.146

Finished Training

Model saved to: /content/drive/My Drive/CDS521_Dissertation_v10/cifar10_model_v2.pth

Training loss plot saved to: /content/drive/My Drive/CDS521_Dissertation_v10/training_loss_v2.png

![Alt text of the image](https://github.com/zhoubojian-stevenchow/CDS521_Dissertation_CV_CIFAR10_Model/blob/main/training_loss_curve.png)

Calculating accuracy on the 10000 test images...

Accuracy of the network on the 10000 test images: 65.83 %

Per-class accuracy:

Accuracy of  plane: 70.90%

Accuracy of    car: 83.80%

Accuracy of   bird: 57.40%

Accuracy of    cat: 40.30%

Accuracy of   deer: 54.50%

Accuracy of    dog: 54.60%

Accuracy of   frog: 74.00%

Accuracy of  horse: 72.60%

Accuracy of   ship: 76.00%

Accuracy of  truck: 74.20%

Generating confusion matrix...

Confusion matrix plot saved to: /content/drive/My Drive/CDS521_Dissertation_v10/confusion_matrix_v2.png
![Alt text of the image](https://github.com/zhoubojian-stevenchow/CDS521_Dissertation_CV_CIFAR10_Model/blob/main/confusion_matrix.png)

============================================================

TRAINING SUMMARY

============================================================

Architecture: 2 Conv layers (6, 16 filters) + 3 FC layers

Regularization: Dropout (p=0.2) + Weight Decay (1e-4)

Data Augmentation: RandomHorizontalFlip + RandomCrop

Optimizer: Adam (lr=0.001)

Loss Function: CrossEntropyLoss

Batch Size: 32

Epochs: 20

Final Test Accuracy: 65.83%

============================================================
