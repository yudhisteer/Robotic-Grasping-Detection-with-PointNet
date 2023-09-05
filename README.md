# Deep Point Clouds: 3D Perception

## Problem Statement

## Dataset

## Abstract

## Plan of Action
1. Understanding PointNet
2. Coding PointNet
3. Training with ShapeNet Dataset
4. Training with custom dataset

-----------------

## 2. Coding PointNet

```python
id1[Input Point Cloud] --> id2[Conv1D 64 + BatchNorm + ReLU]
id2 --> id3[Conv1D 128 + BatchNorm + ReLU]
id3 --> id4[Conv1D 1024 + BatchNorm + ReLU] 
id4 --> id5[MaxPool]
id5 --> id6[Flatten 1024]
id6 --> id7[FC 512 + BatchNorm + ReLU]
id7 --> id8[FC 256 + BatchNorm + ReLU]
id8 --> id9[FC 9]
id9 --> id10[Add Identity 9]
id10 --> id11[Reshape 3x3]
id11 --> id12[Output Transform Matrix]
```

```python
# T-Net Architecture
class STN3d(nn.Module):

    def __init__(self):
        super(STN3d, self).__init__()  # Initializes PyTorch module

        # Conv layers
        self.conv1 = torch.nn.Conv1d(3, 64, 1)  # 1D conv, in_channels=3, out_channels=64, kernel_size=1
        self.conv2 = torch.nn.Conv1d(64, 128, 1)  # 1D conv, in_ch=64, out_ch=128, kernel=1
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)  # 1D conv, in_ch=128, out_ch=1024, kernel=1

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 512)  # FC layer, in_features=1024, out_features=512
        self.fc2 = nn.Linear(512, 256)  # FC layer, in_f=512, out_f=256
        self.fc3 = nn.Linear(256, 9)  # Output transform matrix, in_f=256, out_f=9

        # Nonlinearities
        self.relu = nn.ReLU()

        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        # Input shape
        batchsize = x.size()[0]
        print("Shape initial:", x.shape)

        # Conv layers
        x = F.relu(self.bn1(self.conv1(x)))  # Conv1 - BatchNorm - ReLU
        print("Shape after Conv1:", x.shape)

        x = F.relu(self.bn2(self.conv2(x)))  # Conv2 - BatchNorm - ReLU
        print("Shape after Conv2:", x.shape)

        x = F.relu(self.bn3(self.conv3(x)))  # Conv3 - BatchNorm - ReLU
        print("Shape after Conv3:", x.shape)

        # Pooling layer
        x = torch.max(x, 2, keepdim=True)[0]
        print("Shape after Max Pooling:", x.shape)

        # Reshape for FC layers
        x = x.view(-1, 1024)
        print("Shape after Reshape:", x.shape)

        # FC layers
        x = F.relu(self.bn4(self.fc1(x)))  # FC - BatchNorm - ReLU
        print("Shape after FC1:", x.shape)

        x = F.relu(self.bn5(self.fc2(x)))  # FC - BatchNorm - ReLU
        print("Shape after FC2:", x.shape)

        x = self.fc3(x)  # Final FC layer
        print("Shape after FC3:", x.shape)

        # Initialize identity transformation matrix
        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)

        # Move to GPU if needed
        if x.is_cuda:
            iden = iden.cuda()

        # Add identity matrix to transform pred
        x = x + iden

        # Reshape to 3x3
        x = x.view(-1, 3, 3)
        print("Shape after Reshape to 3x3:", x.shape)

        return x
```




```python
Shape initial: torch.Size([32, 3, 2500])
Shape after Conv1: torch.Size([32, 64, 2500])
Shape after Conv2: torch.Size([32, 128, 2500])
Shape after Conv3: torch.Size([32, 1024, 2500])
Shape after Max Pooling: torch.Size([32, 1024, 1])
Shape after Reshape: torch.Size([32, 1024])
Shape after FC1: torch.Size([32, 512])
Shape after FC2: torch.Size([32, 256])
Shape after FC3: torch.Size([32, 9])
Shape after Reshape to 3x3: torch.Size([32, 3, 3])
```





## References
1. https://medium.com/@luis_gonzales/an-in-depth-look-at-pointnet-111d7efdaa1a
2. https://github.com/luis-gonzales/pointnet_own
3. https://keras.io/examples/vision/pointnet/
4. https://medium.com/p/90398f880c9f
5. https://towardsdatascience.com/3d-deep-learning-python-tutorial-pointnet-data-preparation-90398f880c9f
6. https://medium.com/deem-blogs/speaking-code-pointnet-65b0a8ddb63f
7. https://arxiv.org/pdf/1506.02025.pdf
8. https://medium.com/geekculture/understanding-3d-deep-learning-with-pointnet-fe5e95db4d2d
9. https://towardsdatascience.com/deep-learning-on-point-clouds-implementing-pointnet-in-google-colab-1fd65cd3a263
10. https://github.com/nikitakaraevv/pointnet/blob/master/nbs/PointNetClass.ipynb
11. https://medium.com/@itberrios6/introduction-to-point-net-d23f43aa87d2
12. https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py
13. https://medium.com/@itberrios6/point-net-from-scratch-78935690e496
14. https://medium.com/@luis_gonzales/an-in-depth-look-at-pointnet-111d7efdaa1a
15. https://www.youtube.com/watch?v=HIRj5pH2t-Y&t=427s&ab_channel=Lights%2CCamera%2CVision%21
