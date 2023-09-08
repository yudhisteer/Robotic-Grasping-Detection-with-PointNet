# Deep Point Clouds: PointNet

## Problem Statement

## Dataset

## Abstract

## Plan of Action
1. [Understanding PointNet](#up)
2. [Coding PointNet](#cp)
3. [Training with ShapeNet Dataset](#tsd)
4. [Training with Custom Dataset](#tcd)

-----------------
<a name="up"></a>
## 1. Understanding PointNet

Before PointNet, researchers would convert point cloud data into **3D Voxel Grids**. A large portion of these voxel grids can be empty and this makes the data voluminous. 

PointNet is a neural network architecture designed for processing and understanding point cloud data. The idea behind PointNet is to take in directly the point cloud data such that it respects the ```permutation invariance``` of points in the point cloud data.

<p align="center">
  <img src="https://github.com/yudhisteer/Deep-Point-Clouds-3D-Perception/assets/59663734/f54fd8a0-901c-4743-ac91-a47303888b70" width="70%" />
</p>

The architecture is designed to directly process **unordered** point cloud data, which makes it a useful tool for various ```3D tasks``` such as ```object classification```, ```object segmentation```, and ```shape recognition```. 

<p align="center">
  <img src="https://github.com/yudhisteer/Deep-Point-Clouds-3D-Perception/assets/59663734/3d3487cf-65c5-4cc7-b744-5813d5b6e27d" width="40%" />
</p>


### 1.1 Permutation Invariance

Point clouds represent a set of points in 3D space. Each point typically has attributes such as position ```(x, y, z)``` and possibly additional features (e.g., ```color, intensity```). These point clouds are **unordered** sets of points, meaning that the order of the points in the set doesn't convey any meaningful information. For example, if we have a point cloud representing the 3D coordinates of objects in a scene, we could shuffle the order of the points without changing the scene itself. Here's an example below:

**Original Point Cloud:**
 
```python
Point 1: (0, 0, 0)
Point 2: (1, 0, 0)
Point 3: (0, 1, 0)
Point 4: (1, 1, 0)
Point 5: (0, 0, 1)
Point 6: (1, 0, 1)
Point 7: (0, 1, 1)
Point 8: (1, 1, 1)
```

**Shuffled Point Cloud:**

```python
Point 1: (0, 1, 0) # Previously point 3
Point 2: (1, 0, 0) # Previously point 2
Point 3: (1, 1, 0) # Previously point 4
Point 4: (0, 0, 0) # Previously point 1
Point 5: (0, 1, 1) # Previously point 1
Point 6: (1, 1, 1) # Previously point 8
Point 7: (1, 0, 1) # Previously point 6
Point 8: (0, 0, 1) # Previously point 5
```

Although the order of the points has changed, the ```spatial relationships``` between the points and the overall structure of the cube **remain the same**. It's about recognizing that the order of points in a point cloud **doesn't change** the ```underlying geometry``` or content being represented.


#### How does PointNet process point cloud data in a permutation-invariant manner?

- **Symmetric Functions:** ```Max-pooling``` to aggregate information from all points in the input set. The operation treats all points equally, regardless of their order.

- **Shared Weights:** The same set of weights using ```Multi-Layer Perceptron (MLP)``` is used for all points in the input. This means that the processing applied to one point is the same as that applied to any other point. This shared weight scheme ensures that the network doesn't favor any particular order of points.

### 1.2 Point Cloud Properties
Let's describe the three main properties of a point cloud:

- **Unordered:** A point cloud is a collection of 3D points, and unlike images or grids, there's ```no specific order``` to these points. This means that the order in which we feed the points to a network shouldn't matter. It should be able to handle any order we provide.

- **Interaction Among Points:** These points aren't isolated; they have a ```distance metric```. Nearby points often form meaningful structures. So, a good model should be able to capture these local patterns and understand how points interact with each other.

- **Invariance Under Transformations:** A good representation of a point cloud should stay the same even if we ```rotate``` or ```translate``` the entire point cloud. In other words, changing the viewpoint or position of the points as a whole shouldn't change the global point cloud category or segmentation of the points.

### 1.2 Input Transform

### 1.3 Feature Transform



### 1.4 Shared Multi-Layer Perceptron (MLP)






------------------------
<a name="cp"></a>
## 2. Coding PointNet

### 2.1 Input T-Net

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

<p align="center">
  <img src="https://github.com/yudhisteer/Deep-Point-Clouds-3D-Perception/assets/59663734/79a2c885-1f9a-4216-a40c-688e05bf8cf1" width="100%" />
</p>









```python
# Conv layers
conv1 = nn.Conv1d(3, 64, 1)
conv2 = nn.Conv1d(64, 128, 1)
conv3 = nn.Conv1d(128, 1024, 1)
```

```python
# Fully connected layers
fc1 = nn.Linear(1024, 512)
fc2 = nn.Linear(512, 256)
fc3 = nn.Linear(256, 9)
```

```python
# Nonlinearities
relu = nn.ReLU()
```

```python
# Batch normalization layers
bn1 = nn.BatchNorm1d(64)
bn2 = nn.BatchNorm1d(128)
bn3 = nn.BatchNorm1d(1024)
bn4 = nn.BatchNorm1d(512)
bn5 = nn.BatchNorm1d(256)
```

```python
def stn3d(x):
    # Input shape
    batchsize = x.size()[0]
    print("Shape initial:", x.shape)

    # Conv layers
    x = F.relu(bn1(conv1(x)))
    print("Shape after Conv1:", x.shape)

    x = F.relu(bn2(conv2(x)))
    print("Shape after Conv2:", x.shape)

    x = F.relu(bn3(conv3(x)))
    print("Shape after Conv3:", x.shape)

    # Pooling layer
    x = torch.max(x, 2, keepdim=True)[0]
    print("Shape after Max Pooling:", x.shape)

    # Reshape for FC layers
    x = x.view(-1, 1024)
    print("Shape after Reshape:", x.shape)

    # FC layers
    x = F.relu(bn4(fc1(x)))
    print("Shape after FC1:", x.shape)

    x = F.relu(bn5(fc2(x)))
    print("Shape after FC2:", x.shape)

    x = fc3(x)
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

### 2.2 Feature T-Net


```python
id1[Input Point Cloud] --> id2[Conv1D 64 + BatchNorm + ReLU]
id3 --> id4[Conv1D 128 + BatchNorm + ReLU] 
id5 --> id6[Conv1D 1024 + BatchNorm + ReLU]
id7 --> id8[MaxPool]
id9 --> id10[Flatten] 
id11 --> id12[FC 512 + BatchNorm + ReLU]
id13 --> id14[FC 256 + BatchNorm + ReLU]  
id15 --> id16[FC k*k]
id17 --> id18[Add Identity]
id18 --> id19[Reshape k x k]
id19 --> id20[Output Transform]
```

<p align="center">
  <img src="https://github.com/yudhisteer/Deep-Point-Clouds-3D-Perception/assets/59663734/0f3ebf0d-3eb2-4a1c-a68a-dfcce2b3c31a" width="100%" />
</p>



### 2.3 PointNet Feat





### 2.4 Classification Head


### 2.5 Segmentation Head








------------------------
<a name="tsd"></a>
## 3. Training with ShapeNet Dataset

------------------------
<a name="tcd"></a>
## 4. Training with Custom Dataset

-------------------

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
16. https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215
