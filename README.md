# Robotic Grasping Detection with PointNet

## Problem Statement
I have always been amazed by how quickly we can analyze what is the correct way to **hold an object**. As it is simple prowess, we often take this skill for granted. We are conditioned to grab our toothbrush by the handle and not the bristle, we know not to grab a knife by its sharp edge, and we also know there is no single universally correct way to grab a ball. A child typically starts developing the ability to grasp objects in the first few months of life and by 1 year old, the child is able to master the ```"pincer grasp"``` which is the use of the thumb and forefinger to pick up small objects. 

In the field of robotics, extensive research has been dedicated to teaching robots how to grasp objects, employing methods like **supervised**, **semi-supervised**, and **reinforcement learning**. Grasping objects poses a **complex** and **computationally intensive** challenge, especially in **real-time operations**. This project's focus is not on programming a robotic hand to physically grasp objects, but rather on determining **where to grasp them**. Similar to how humans learn to identify handles on objects, our goal is to train robots to recognize **optimal grasping points** on a specific set of objects, allowing them to apply this knowledge to **new objects**.

<p align="center">
  <img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/a42db359-e6d8-4d5a-ab09-d7b0d0b3933e" width="49%" />
  <img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/daeca668-1135-4c86-b83f-57e5ee29b6a6" width="49%" />
</p>
<div align="center">
    <p>Video source: <a href="https://www.youtube.com/watch?v=sDFAWnrCqKc&ab_channel=NVIDIADeveloper">Eureka! Extreme Robot Dexterity with LLMs</a></p>
    <p>Video source: <a href="https://www.youtube.com/watch?v=GyEHRXA_aA4&t=25s&ab_channel=TheTelegraph">'Kitchen robot' that will cook meals from scratch unveiled</a></p>
</div>

## Dataset

1. [ShapeNet](https://shapenet.org/)

    - **ShapeNetCore** is a subset of ShapeNet, featuring ```51,300``` meticulously verified 3D models spanning ```55``` common object categories, including the ```12``` from PASCAL 3D+.

   - **ShapeNetSem** is a more compact subset, containing ```12,000``` models across ```270``` categories, each extensively annotated with real-world dimensions, material composition estimates, and volume and weight approximations at the category level.

2. [UtensilsNet](https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/tree/main/Data)

    - Created my own custom dataset which consists of ```400``` point clouds of three object classes: **Knife**, **Pan**, and **Cup**. Data augmentation was used to increase size of dataset. More on data augmentation techniques in section 4. Part of the dataset has been uploaded to this GitHub repo.


## Abstract
This projects involves designing the PointNet model from scratch. We first collected point cloud data using an iPhone's LiDAR though the **Polycam** app. We chose three objects for our dataset: ```Cup```, ```Knife``` and ```Pan```. We cleaned our point cloud using the RANSAC algorithm to filter out any outliers. Because we have a limited dataset, we use data augmentation which is different from those we use on images. We explain why the data augmentation such as scaling or reflection do not work on point cloud data. We then labeled our point cloud using **Segments.ai**. We downloaded and processed the segmentation label. Finally, we trained our model for both **classification** and **part-segmentation**. Our goal is to be able to segment the handle on the object. We evaluated our model on an out-of-sample dataset which did not perform as well as expected but nervertheless, had good results on our test set with accuracy greater than ```98%```. 

<p align="center">
  <img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/2eccede5-8b00-4b91-99bd-fe5998b237bf" width="70%" />
</p>
<div align="center">
    <p>Video source: <a href="https://www.youtube.com/watch?v=ly32I-WLxQY&t=1s&ab_channel=NAVERLABS">AMBIDEX Grasping Demo - Perception & Force Control</a></p>
</div>


## Plan of Action
1. [Understanding PointNet](#up)
2. [Coding PointNet](#cp)
3. [Data Collection with Polycam](#dc)
4. [Data Pre-processing with Open3D](#dpo)
5. [Data Labeling with Segments.ai](#dls)
6. [Training: Classification and Part-Segmentation](#t)
7. [Evaluation](#e)

-----------------
<a name="up"></a>
## 1. Understanding PointNet

Before PointNet, researchers would convert point cloud data into **3D Voxel Grids**. The disadvantages of Voxel Grids are that they lead to **data sparsity** because there are large regions with no points (empty voxels), **increase storage and computation**, **loss of fine-grained details** in the point cloud, and have proven to be **inefficient for irregular data** whereby point clouds are inherently irregular data structures.

Introducing, ```PointNet``` which is a neural network architecture designed for processing and understanding point cloud data. The idea behind PointNet is to take in **directly** the point cloud data such that it respects the ```permutation invariance``` of points in the point cloud data and thus no longer needs to transform the data into 3D Voxel Grids.

<p align="center">
  <img src="https://github.com/yudhisteer/Deep-Point-Clouds-3D-Perception/assets/59663734/f54fd8a0-901c-4743-ac91-a47303888b70" width="70%" />
</p>
<div align="center">
    <p>Image source: <a href="https://www.youtube.com/watch?v=ly32I-WLxQY&t=1s&ab_channel=NAVERLABS">PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</a></p>
</div>

The architecture is designed to directly process **unordered** point cloud data, which makes it a useful tool for various ```3D tasks``` such as ```object classification```, ```semantic segmentation```, and ```part segmentation```. 


<p align="center">
  <img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/682fa684-af02-4f82-8231-c00fa7c5a4bc" width="40%" />
</p>
<div align="center">
    <p>Image source: <a href="https://arxiv.org/abs/1612.00593">PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation</a></p>
</div>

Our project will focus on **classification** and **part-segmentation**.


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

Let's try to visualize it. Below is an example of an image (left) whose pixel grids are shuffled randomly with others. We observe that the original shape in the image is no longer retained. On the right, we have the point cloud data of a cube and when the points are shuffled randomly, we can still observe the original shape- a cube.

 

<p align="center">
  <img src="https://github.com/yudhisteer/Classifying-ASL-with-PointNet/assets/59663734/96b1b214-86ba-473e-9afb-b420a31fbfbf" width="30%" />
  <img src="https://github.com/yudhisteer/Classifying-ASL-with-PointNet/assets/59663734/07cb1632-4e7b-469b-b509-e6e05f75411e" width="30%" />
</p>


Although the order of the points has changed, the ```spatial relationships``` between the points and the overall structure of the cube **remain the same**. It's about recognizing that the order of points in a point cloud **doesn't change** the ```underlying geometry``` or content being represented.


#### How does PointNet process point cloud data in a permutation-invariant manner?

- **Symmetric Functions:** ```Max-pooling``` to aggregate information from all points in the input set. The operation treats all points equally, regardless of their order.

- **Shared Weights:** The same set of weights using ```Multi-Layer Perceptron (MLP)``` is used for all points in the input. This means that the processing applied to one point is the same as that applied to any other point. This shared weight scheme ensures that the network doesn't favor any particular order of points.

### 1.2 Point Cloud Properties
Let's describe the three main properties of a point cloud:

- **Unordered:** A point cloud is a collection of 3D points, and unlike images or grids, there's ```no specific order``` to these points. This means that the order in which we feed the points to a network shouldn't matter. It should be able to handle any order we provide.

- **Interaction Among Points:** These points aren't isolated; they have a ```distance metric```. Nearby points often form meaningful structures. So, a good model should be able to capture these local patterns and understand how points interact with each other.

- **Invariance Under Transformations:** A good representation of a point cloud should stay the same even if we ```rotate``` or ```translate``` the entire point cloud. In other words, changing the viewpoint or position of the points as a whole shouldn't change the global point cloud category or segmentation of the points.

### 1.3 Input Transform
From the PointNet architecture, we observe that the Input Transform encompasses a ```T-Net```. So what is a T-Net? The T-Net is a type of **Spatial transformer Network (STN)** that can be seen as a ```mini-PointNet```. The first T-Net takes in ```raw point cloud data``` and outputs a ```3 x 3``` matrix.

<p align="center">
  <img src="https://github.com/yudhisteer/Classifying-Lego-with-PointNet/assets/59663734/bd6df400-744f-468e-ad68-5071db6675f6" width="100%" />
</p>

The T-Net is responsible for predicting an ```affine transformation matrix``` that aligns the input point cloud to a ```canonical space```. This alignment ensures that the network is **invariant** to certain geometric transformations, such as ```rigid transformations```.

Let's take a step back and define some of these complex technical terms:

- **Affine transformation matrix:** This matrix defines how the input points should be ```transformed``` to align with the reference space (Canonical space).


- **Canonical space:** It refers to a ```standardized``` and ```consistent reference point``` for processing 3D point clouds. It's like choosing a common starting point or orientation for all input point clouds. This standardization ensures that no matter how a point cloud is initially positioned or rotated, it gets transformed into this ```common reference frame```. This simplifies the learning process of the neural network. 


- **Rigid transformations:**  Transformation that **preserves** the ```shape``` and ```size``` of an object. It includes operations like ```translation``` (moving an object without changing its shape), ```rotation``` (turning an object around a point), and ```reflection``` (flipping an object). Essentially, rigid transformations **don't distort** or **deform** the object; they only change its ```position``` and ```orientation``` in space.

In summary, this means that no matter how the original point cloud was ```oriented``` or ```positioned```, the T-Net will transform it in such a way that it becomes ```standardized```, making it **easier** for subsequent layers of the network to process the data effectively.

### 1.4 Feature Transform
The PointNet has a second transformer network called ```Feature T-Net```.  The role of this second T-Net is to predict a ```feature transformation matrix``` to align features from different input point clouds. It captures ```fine-grained``` information about ```point-specific transformations``` that are important for capturing ```local details``` and ```patterns```.

The second transformer network has the same architecture as the first except that this one's output is a ```64 x 64``` matrix. In the diagram below we used ```k``` where k will be equal to ```64```.

<p align="center">
  <img src="https://github.com/yudhisteer/Deep-Point-Clouds-3D-Perception/assets/59663734/0f3ebf0d-3eb2-4a1c-a68a-dfcce2b3c31a" width="100%" />
</p>


In the paper, the author argues that the feature space has a ```higher dimension``` which greatly increases the complexity of the optimization process. Hence, they add a ```regularization``` term to their ```softmax``` training loss in order to constrain the feature transformation matrix to be close to ```orthogonal```. By being orthogonal. it helps ensure that the transformation applied to features during alignment doesn't introduce ```unnecessary distortion``` which could result in ```loss of information```. Below is the regularization term where ```A``` is the feature transformation matrix and ```I``` is the identity matrix: 

<p align="center">
  <img src="https://github.com/yudhisteer/Classifying-Lego-with-PointNet/assets/59663734/af430dae-0c60-4929-8b4a-b2e727dae109" />
</p>


In summary, this is the difference between the Input T-Net and the Feature T-Net:

- **Input T-Net:** By aligning the entire input point cloud to a **canonical space**, the Input T-Net captures ```global features``` that are **invariant** to transformations like **rotation** and **translation**.

- **Feature T-Net:** Operates on the **feature vectors** extracted from the point cloud **after** the initial global alignment by the **Input T-Net**. It aims to capture ```local features``` and ```fine-grained patterns``` within the aligned point cloud.

### 1.5 Shared Multi-Layer Perceptron (MLP)
Since the Input T-Net and Feature T-Net are themselves mini-PointNet, they both contain shared MLPs in their architecture. While MLP is a type of neural network architecture that consists of multiple layers of artificial neurons (perceptrons), we will use ```1D convolutional layers (Conv1D)``` for the shared MLP followed by ```Fully-Connected (FC)``` layers in both the Input and Feature Transformation networks. However, we will use only Fully Connected layers for the PointNet after the Feature Transformation network.

Note that the shared MLPs are the ```core feature extraction``` component after the initial alignment and transformation by the Input T-Net and Feature T-Net.

<p align="center">
  <img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/98c4bf22-9078-4613-a247-8838bfd86cf7" width="100%" />
</p>

In the full architecture of the PointNet, we can clearly see how the Input Transformation Network and the Feature Transformation Network are ```mini-PointNets``` themselves.


------------------------
<a name="cp"></a>
## 2. Coding PointNet
Now that we know how PointNet works, we need to code it. Since it is a big neural network architecture, we will divide it into four segments:

**1. Input T-Net**

**2. Feature T-Net**

**3. PointNet Feat**

**4. Classification Head**

**5. Segmentation Head**

For this project, we will focus on both **classification** and **part-segmentation**.


### 2.1 Input T-Net
As explained above, the Input Transform section of PointNet contains a T-Net. Below is the schema for the architecture of the T-Net consisting of 1D Convolutional Layers and Fully Connected layers as shared MLPs. 

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


We first define the **convolutional layers**, **fully connected layers**, **activation function**, and **batch normalization layers** needed:

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

The **function** for T-Net:

```python
def input_tnet(x):
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

We will create simulated data for the point cloud with the following parameters:

- batch size = ```32```
- number of points = ```2500```
- number of channels (x,y,z) = ```3```

```python
    # Generate a sample input tensor (N, 3, 2500)
    sample_input = torch.randn(32, 3, 2500)
```


Below is the size of the layers after each process. 


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

The shape of the layer after T-Net and before matrix multiplication is a ```3 x 3``` matrix.

### 2.2 Feature T-Net
As explained above, the Feature T-Net is similar to the Input T-Net except that the output is now a ```64 x 64``` matrix.  Below is the schema of its architecture:


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
Similarly, we define the **convolutional layers**, **fully connected layers**, **activation function**, and **batch normalization layers** needed. I defined a parameter k to be equal to the input size of the layer for the T-Net which is equal to ```64```. 

```python
# Hyperparameter
k = 64
```

```python
# Conv layers
conv1 = nn.Conv1d(k, 64, 1) #input, #output, # size of the convolutional kernel
conv2 = nn.Conv1d(64, 128, 1)
conv3 = nn.Conv1d(128, 1024, 1)
```

```python
# Fully connected layers
fc1 = nn.Linear(1024, 512) #input, #output
fc2 = nn.Linear(512, 256)
fc3 = nn.Linear(256, k * k)
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

Note that the output of the input T-Net is ```3 x 3``` (before matrix multiplication) but the input of the feature T-Net is ```n x 64```. We have a shared MLP in order to convert the layer to the correct size before the feature transformation. Below is the function of the Feature T-Net:

```python
def feature_tnet(x):
    print("Feature T-Net...")

    batchsize = x.size()[0]
    print("Shape initial:", x.shape)

    # Conv layers
    x = F.relu(bn1(conv1(x)))
    print("Shape after Conv1:", x.shape)

    x = F.relu(bn2(conv2(x)))
    print("Shape after Conv2:", x.shape)

    x = F.relu(bn3(conv3(x)))
    print("Shape after Conv3:", x.shape)

    # Pooling
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

    # Initialize identity
    iden = Variable(torch.from_numpy(np.eye(k).flatten().astype(np.float32))).view(1, k * k).repeat(batchsize, 1)
    if x.is_cuda:
        iden = iden.cuda()

    # Add identity
    x = x + iden
    x = x.view(-1, k, k)

    return x
```

Similarly, we created a simulated data but this time with ```64``` channels and print the results:

```python
Shape initial: torch.Size([32, 64, 2500])
Shape after Conv1: torch.Size([32, 64, 2500])
Shape after Conv2: torch.Size([32, 128, 2500])
Shape after Conv3: torch.Size([32, 1024, 2500])
Shape after Max Pooling: torch.Size([32, 1024, 1])
Shape after Reshape: torch.Size([32, 1024])
Shape after FC1: torch.Size([32, 512])
Shape after FC2: torch.Size([32, 256])
Shape after FC3: torch.Size([32, 4096])
torch.Size([32, 64, 64])
```

The output for the feature T-Net is a ```64 x 64``` matrix.

### 2.3 PointNet Feat
Next, we will code a function that will encompass the Input Transform, Feature Transform, and output the **Global Features (1024)**. We will define **wrapper functions** for the input T-Net and feature T-Net. 

```python
# Input T-Net
def stn(x): #wrapper function
    return input_tnet(x)

# Feature T-Net
def fstn(x):
    return feature_tnet(x)
```

Once again, our shared MLPs will consist of **1D Convolutional layers**.

```python
# Convolution layers
conv1 = nn.Conv1d(3, 64, 1) #input, #output, # size of the convolutional kernel
conv2 = nn.Conv1d(64, 128, 1)
conv3 = nn.Conv1d(128, 1024, 1)

# Batchnorm layers
bn1 = nn.BatchNorm1d(64)
bn2 = nn.BatchNorm1d(128)
bn3 = nn.BatchNorm1d(1024)
```

The PointNet Feat function:

```python
def pointnetfeat(x, global_feat=True, feature_transform=True):
    # Number of points
    n_pts = x.size()[2]
    print("Shape initial:", x.shape, '\n')

    # Apply Input T-Net
    trans = stn(x) #torch.Size([32, 3, 3])
    x = x.transpose(2, 1) #torch.Size([32, 2500, 3])
    x = torch.bmm(x, trans)
    print("Shape after Matrix Multiply:", x.shape)
    x = x.transpose(2, 1)
    print("Shape after Input T-Net:", x.shape) #torch.Size([32, 3, 2500])

    # conv layers
    x = F.relu(bn1(conv1(x)))
    print("Shape after Conv1:", x.shape)

    # Apply Feature T-Net
    if feature_transform:
        print("\nApply Feature T-Net...", '\n')
        trans_feat = fstn(x) #torch.Size([32, 64, 64])
        x = x.transpose(2, 1) #torch.Size([32, 2500, 64])
        x = torch.bmm(x, trans_feat)
        print("Shape after Matrix Multiply:", x.shape)
        x = x.transpose(2, 1)
        print("Shape after Feature T-Net:", x.shape, '\n') #torch.Size([32, 64, 2500])
    else:
        trans_feat = None

    # Save point features
    pointfeat = x

    # conv layers
    x = F.relu(bn2(conv2(x)))
    print("Shape after Conv2:", x.shape)

    x = bn3(conv3(x))
    print("Shape after Conv3:", x.shape)

    # Pooling
    x = torch.max(x, 2, keepdim=True)[0]
    print("Shape after Pooling:", x.shape)

    # Global Feature
    x = x.view(-1, 1024)
    print("Shape of global feature: ", x.shape)

    # classification:
    if global_feat:
        return x, trans, trans_feat
    # segmentation
    else:
        x = x.view(-1, 1024, 1).repeat(1, 1, n_pts) 
        return torch.cat([x, pointfeat], 1), trans, trans_feat #1088
```

Again with simulated data, we print our results:

```python
# Input Transform...
Shape initial: torch.Size([32, 3, 2500])
Shape after Conv1: torch.Size([32, 64, 2500])
Shape after Conv2: torch.Size([32, 128, 2500])
Shape after Conv3: torch.Size([32, 1024, 2500])
Shape after Max Pooling: torch.Size([32, 1024, 1])
Shape after Reshape: torch.Size([32, 1024])
Shape after FC1: torch.Size([32, 512])
Shape after FC2: torch.Size([32, 256])
Shape after FC3: torch.Size([32, 9])
Shape of Iden: torch.Size([32, 9])
Shape after Reshape to 3x3: torch.Size([32, 3, 3])
Shape after Matrix Multiply: torch.Size([32, 2500, 3])
Shape after Input T-Net: torch.Size([32, 3, 2500])

# MLP...
Shape after Conv1: torch.Size([32, 64, 2500])

# Feature Transform...
Shape initial: torch.Size([32, 64, 2500])
Shape after Conv1: torch.Size([32, 64, 2500])
Shape after Conv2: torch.Size([32, 128, 2500])
Shape after Conv3: torch.Size([32, 1024, 2500])
Shape after Max Pooling: torch.Size([32, 1024, 1])
Shape after Reshape: torch.Size([32, 1024])
Shape after FC1: torch.Size([32, 512])
Shape after FC2: torch.Size([32, 256])
Shape after FC3: torch.Size([32, 4096])
Shape after Matrix Multiply: torch.Size([32, 2500, 64])
Shape after Feature T-Net: torch.Size([32, 64, 2500]) 

# MLP...
Shape after Conv2: torch.Size([32, 128, 2500])
Shape after Conv3: torch.Size([32, 1024, 2500])
Shape after Pooling: torch.Size([32, 1024, 1])
Shape of global feature:  torch.Size([32, 1024])
```

The size of our global features is of size ```1024```.


### 2.4 Classification Head
Lastly, we have the classification head function which will complete the whole classification network architecture of PointNet. First, we need to define our output parameter ```k``` which is the **number of classes** our neural network will predict:

```python
k = 5 #number of classes to predict
```

Similarly, we then defined the last shared MLP which consists of only Fully Connected layers :

```python
# Fully Connected Layers
fc1 = nn.Linear(1024, 512) #input, #output
fc2 = nn.Linear(512, 256)
fc3 = nn.Linear(256, k)
```

```python
# Batchnorm layers and dropout
dropout = nn.Dropout(p=0.3)
bn1 = nn.BatchNorm1d(512)
bn2 = nn.BatchNorm1d(256)
```

```python
# Nonlinearities
relu = nn.ReLU()
```

The classification head function:

```python
def pointnetcls(x, feature_transform=False):
    print("Classification head...")
    # PointNetFeat
    x, trans, trans_feat = pointnetfeat(x, global_feat=True, feature_transform=feature_transform)
    print("\nShape after PointNetFeat: ", x.shape)

    # FC layers
    x = F.relu(bn1(fc1(x)))
    print("Shape after FC1:", x.shape)
    x = F.relu(bn2(dropout(fc2(x))))
    print("Shape after FC2:", x.shape)
    x = fc3(x)
    print("Shape after FC3:", x.shape)

    return F.log_softmax(x, dim=1), trans, trans_feat
```

With simulated data, we print the output from the beginning:

```python
# Input Transform...
Shape initial: torch.Size([32, 3, 2500])
Shape after Conv1: torch.Size([32, 64, 2500])
Shape after Conv2: torch.Size([32, 128, 2500])
Shape after Conv3: torch.Size([32, 1024, 2500])
Shape after Max Pooling: torch.Size([32, 1024, 1])
Shape after Reshape: torch.Size([32, 1024])
Shape after FC1: torch.Size([32, 512])
Shape after FC2: torch.Size([32, 256])
Shape after FC3: torch.Size([32, 9])
Shape of Iden: torch.Size([32, 9])
Shape after Reshape to 3x3: torch.Size([32, 3, 3])
Shape after Matrix Multiply: torch.Size([32, 2500, 3])
Shape after Input T-Net: torch.Size([32, 3, 2500])

# MLP...
Shape after Conv1: torch.Size([32, 64, 2500])

# Feature Transform...
Shape initial: torch.Size([32, 64, 2500])
Shape after Conv1: torch.Size([32, 64, 2500])
Shape after Conv2: torch.Size([32, 128, 2500])
Shape after Conv3: torch.Size([32, 1024, 2500])
Shape after Max Pooling: torch.Size([32, 1024, 1])
Shape after Reshape: torch.Size([32, 1024])
Shape after FC1: torch.Size([32, 512])
Shape after FC2: torch.Size([32, 256])
Shape after FC3: torch.Size([32, 4096])
Shape after Matrix Multiply: torch.Size([32, 2500, 64])
Shape after Feature T-Net: torch.Size([32, 64, 2500]) 

# MLP...
Shape after Conv2: torch.Size([32, 128, 2500])
Shape after Conv3: torch.Size([32, 1024, 2500])
Shape after Pooling: torch.Size([32, 1024, 1])
Shape of global feature:  torch.Size([32, 1024])

# Classification Head...
Shape after PointNetFeat:  torch.Size([32, 1024])
Shape after FC1: torch.Size([32, 512])
Shape after FC2: torch.Size([32, 256])
Shape after FC3: torch.Size([32, 5])
```

Our output will be **log probabilities**. We will need to **exponentiate** to get probabilities for each point cloud in our batch. We will then select the class with the **highest probability** to classify the point cloud.

### 2.5 Segmentation Head
For the segmentation head, our shared MLPs will also consist of **1D Convolutional layers**.

```python
m = 3 # number of segmentation classes
```

```python
# Convolutional Layers
conv1 = nn.Conv1d(1088, 512, 1)
conv2 = nn.Conv1d(512, 256, 1)
conv3 = nn.Conv1d(256, 128, 1)
conv4 = nn.Conv1d(128, m, 1)
```

```python
# Batch Normalization Layers
bn1 = nn.BatchNorm1d(512)
bn2 = nn.BatchNorm1d(256)
bn3 = nn.BatchNorm1d(128)
```

Note that since we need to concatenate our output ```(64)``` from the **feature transform** with the **global feature** ```(1024)```, we have an if function in our ```pointnetfeat``` function which do so and return a feature of size ```[batch size, 1088, n]```. We then pass the latter through the shared MLPs. 

```python
    if global_feat:
        return x, trans, trans_feat
    else:
        x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
        return torch.cat([x, pointfeat], 1), trans, trans_feat
```

The output from the segmentation head is of size: ```[batch size, n, m]``` where ```m``` is the number of segmentation classes. 

```python
def PointNetDenseCls(x, m=2, feature_transform=False):
    print("\nSegmentation head...")
    batchsize = x.size()[0]
    n_pts = x.size()[2]
    print("Shape initial:", x.shape, '\n')

    # PointNetFeat
    x, trans, trans_feat = pointnetfeat(x, global_feat=False, feature_transform=feature_transform)
    print("\nShape after PointNetFeat: ", x.shape)


    # Convolutional Layers
    x = F.relu(bn1(conv1(x)))
    print("Shape after Conv1:", x.shape)

    x = F.relu(bn2(conv2(x)))
    print("Shape after Conv2:", x.shape)

    x = F.relu(bn3(conv3(x)))
    print("Shape after Conv3:", x.shape)

    x = conv4(x)

    # Post-processing
    x = x.transpose(2, 1).contiguous()
    x = F.log_softmax(x.view(-1, m), dim=-1)
    x = x.view(batchsize, n_pts, m)

    return x, trans, trans_feat
```
With simulated data, we print the output from the beginning:

```python
# Input Transform...
Shape initial: torch.Size([32, 3, 2500])
Shape after Conv1: torch.Size([32, 64, 2500])
Shape after Conv2: torch.Size([32, 128, 2500])
Shape after Conv3: torch.Size([32, 1024, 2500])
Shape after Max Pooling: torch.Size([32, 1024, 1])
Shape after Reshape: torch.Size([32, 1024])
Shape after FC1: torch.Size([32, 512])
Shape after FC2: torch.Size([32, 256])
Shape after FC3: torch.Size([32, 9])
Shape of Iden: torch.Size([32, 9])
Shape after Reshape to 3x3: torch.Size([32, 3, 3])
Shape after Matrix Multiply: torch.Size([32, 2500, 3])
Shape after Input T-Net: torch.Size([32, 3, 2500])

MLP...
Shape after Conv1: torch.Size([32, 64, 2500])

# Feature Transform...
Shape initial: torch.Size([32, 64, 2500])
Shape after Conv1: torch.Size([32, 64, 2500])
Shape after Conv2: torch.Size([32, 128, 2500])
Shape after Conv3: torch.Size([32, 1024, 2500])
Shape after Max Pooling: torch.Size([32, 1024, 1])
Shape after Reshape: torch.Size([32, 1024])
Shape after FC1: torch.Size([32, 512])
Shape after FC2: torch.Size([32, 256])
Shape after FC3: torch.Size([32, 4096])
Shape after Matrix Multiply: torch.Size([32, 2500, 64])
Shape after Feature T-Net: torch.Size([32, 64, 2500]) 

MLP...
Shape after Conv2: torch.Size([32, 128, 2500])
Shape after Conv3: torch.Size([32, 1024, 2500])
Shape after Pooling: torch.Size([32, 1024, 1])
Shape of global feature:  torch.Size([32, 1024])

# Segmentation Head...
Shape after PointNetFeat:  torch.Size([32, 1088, 2500])
Shape after Conv1: torch.Size([32, 512, 2500])
Shape after Conv2: torch.Size([32, 256, 2500])
Shape after Conv3: torch.Size([32, 128, 2500])
Output torch.Size([32, 2500, 3])
```

------------------------
<a name="dc"></a>
## 3. Data Collection with Polycam
Thanks to [Polycam](https://poly.cam/), it is very easy to collect data with your iPhone's LiDAR. The app has a free trial however, we can only export the data in ```.glb``` format. Make sure that you move slowly around your object to avoid **drift** and avoid capturing the facet of your object more than once. Unfortunately, you will not be able to capture transparent objects or shiny objects. For better results, I observed that it is better to choose objects with textures and details. I placed those objects on the ground to scan as when putting them on a table, I would accidentally scan other parts of the room and this would lead to extraneous details. 


<p align="center">
  <img src=https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/df78eafc-d44d-4569-9d98-bb48a55b6061" width="49%" height="371px" />
  <img src=https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/f51b3371-2570-4d12-857d-a11c1604420e" width="49%" />
</p>


You will need to convert your files from ```.glb``` to either ```.pcd``` or ```.ply``` for further processing. I used this website: [link](https://products.aspose.app/3d/conversion/glb-to-ply). You can visualize your result with [Meshlab](https://www.meshlab.net/).


------------------------
<a name="dpo"></a>
## 4. Data Pre-processing with Open3D
After data collection, the next crucial step is data pre-processing before we move on to the labeling phase. Our current dataset consists of raw data that includes outliers and extraneous information, and our priority is to clean the data before labeling it.

### 4.1 Segmentation with RANSAC
We are interested only in the cup or the object of interest hence, we do not need the ground plane in our point cloud. How can we remove this? In my previous project [Point Clouds: 3D Perception
](https://github.com/yudhisteer/Point-Clouds-3D-Perception), I talk about how to segment the road and the vehicles using **RANSAC**. The latter algorithm is widely used as a way to **remove outliers** from our data. As such, this is exactly what we need. The ground plane is the outlier here. 

In our scenario, we choose ransac_n equal to ```3```, which is the number of points to randomly sample for each iteration of RANSAC, and num_iterations equal to ```100``` which is the number of RANSAC iterations to perform. We also fine-tuned our distance_threshold to ```0.005```.

```python
    ### ------- SEGMENTATION
    outlier_cloud, inlier_cloud, = ransac(point_cloud, distance_threshold=0.005, ransac_n=3, num_iterations=100)
```

The variable **outlier_cloud** now represents the ground plane as shown in blue below while the variable **inlier_cloud** represents the cup in red. We save inlier_cloud (cup) as a PLY file and perform the same operation for all data.


<div style="text-align: center;">
  <video src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/8426310b-8ad8-4e15-b306-b459fccd5f58" controls="controls" style="max-width: 730px;">
  </video>
</div>





### 4.2 Data Augmentation
Note that we have three object classes and we have around ```25``` point clouds for each class. In summary, we have a very limited dataset. Of course, we can still collect more data however, this will take a long time and will require more objects to scan. I have a finite number of cups in my house!

Similar to how we do data augmentation on our images - ```rotation```, ```blurring```, ```brightness adjusting```, and so on - we can do the same on the point cloud. However, there is a catch. We cannot do a ```rotation```, ```translation```, or ```reflection``` operation on our point cloud, as the PointNet is **invariant** to **rigid transformation** as explained above. And we cannot do blurring or brightness changing as we are only concerned with the spatial location of the points and not the intensity values (Also, I don't think we can do blurring). Recall, that the **Input T-Net** will align  the input point cloud to a ```canonical space``` so there is no point in doing the rigid transformations anyway.  Below we will describe three examples of data augmentation that we can do.

#### 4.2.1 Random Noise
Given our point cloud, we will add a little **jitter** on each point. That is, we will randomly sample **noise** from a **Normal Distribution** and add it to the original point cloud. This will make each point shift a little bit from its original position. We perform the operation for how many 'new' samples we want to create. 


```python
    ### ----- Data Augmentation: Noise
    augmentation_noise(outlier_cloud, num_augmented_samples=5, noise_level=0.0025, save=None)
```

<table>
  <tr>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/fdbce21e-b572-4226-a601-e3cb04cd1931" alt="Image 1" width="547"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/0f7bd8af-20a8-4008-8b0a-44e43a3caf66" alt="Image 2" width="543"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/6ff6468b-0dd7-479a-87e6-27ebfdeab586" alt="Image 3" width="548"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/01a6a721-173d-48b1-9729-41278c7346b3" alt="Image 4" width="503"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/3bdddf74-2562-417f-9207-c626ec9bc93b" alt="Image 5" width="502"></td>
  </tr>
</table>




#### 4.2.2 Random Sampling
The point cloud for this cup has around ```10,000``` points. We will first generate a random number between ```1000``` and ```5000``` which will represent the number of points we want to sample from our original point cloud. We will then sample indices with range: ```[min=num_points_to_sample, max=len(point_cloud.points)]``` using ```np.random.choice()```. With these indices, we will filter out the points from the original point cloud and create a new point cloud - ```sampled_point_cloud```. Similarly, we run the code in a loop depending on how many 'new' samples we want to create. 





```python
    ### ----- Data Augmentation: Random Sampling
    augmentation_sampling(outlier_cloud, num_augmented_samples=5, min_points_to_sample=1000, max_points_to_sample=5000, save=None)
```




<table>
  <tr>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/f754a495-9090-4764-9ebd-21db414703e2" alt="Image 1" width="550"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/fc78235a-8def-4cfb-b96b-c041d8334508" alt="Image 2" width="484"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/d4ba5c75-72df-4aba-b2a3-1d40e67b4437" alt="Image 3" width="526"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/55d5041a-46f4-4887-b6e5-cacbd62df288" alt="Image 4" width="518"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/72b5196a-683d-48d1-aed2-c1bcca525706" alt="Image 5" width="447"></td>
  </tr>
</table>

#### 4.2.2 Random Deformation
The last data augmentation we will do is to deform our point cloud about an axis. We randomly select a **bending angle** between ```10``` and ```15``` degrees. We then randomly select a point in the point cloud as the **bending axis**. We generate a **rotation matrix** using that bending axis and the bending angle. We then apply that rotation matrix to the point cloud. The result can be a bit **noisy** as shown below.

```python
    ### ----- Data Augmentation: Random Deformation
    augmentation_deformation(outlier_cloud, num_augmented_samples=5, min_bending_angle=10.0, max_bending_angle=15.0)
```

<table>
  <tr>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/7428cde4-255a-42c2-8890-0c2bc512a812" alt="Image 1" width="439"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/2eb1242b-c301-4a69-b7c5-66bc2a265a55" alt="Image 2" width="532"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/f8c4e0bc-d80b-46b3-ac81-86762031c6c5" alt="Image 3" width="472"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/69eed6d8-0143-483d-90a5-dc371014e6ec" alt="Image 4" width="541"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/2d0609c2-8f40-4e0c-b362-cbfd358165e4" alt "Image 5" width="530"></td>
  </tr>
</table>






------------------------
<a name="dls"></a>
## 5. Data Labeling with Segments.ai
For labeling, I chose the platform [Segments.ai](https://segments.ai/) which is a multi-sensor labeling platform for robotics and autonomous vehicles. I find it to be pretty user-friendly and you get ```14``` days trial. 

1. We start by creating an **account** and then a **new dataset**.
2. We will be asked to **name** the dataset.
3. We chose the data type to be **Point Cloud** from a list of Text, Image, and Point Cloud.
4. We select the task to be **Segmentation** among Keypoints, Bounding Box, Vector Labels, and Segmentation.
5. We will then need to upload our files in the format of .pcd, .bin or .ply.
   
<table>
  <tr>
    <td>
      <img width="479" alt="image" src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/69f25183-2758-4e88-8c67-0d37ad0a6047">
    </td>
    <td>
      <img width="411" alt="image" src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/bf48bcba-9755-40d1-93ec-147cdeec6162">
    </td>
  </tr>
</table>

6. In the settings tab we can add the labeling **categories**. For each category, there is an **id**, a **name** and a **color** associated with it.
7. We click on "**Start Labeling**".
8. A new window will open whereby we have the **side**, **top**, **back**, and **3D** view of our object.
9. We choose the "**Brush**" tool on the left to start selecting the points we want to label as "**Handle**".
10. We press ```spacebar``` when we are done.
11. We do the same process for the "**body**" of the object.

<table>
  <tr>
    <td>
      <img width="629" alt="image" src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/226ade25-41b0-48bb-bec0-5138242cf9d5">
    </td>
    <td>
      <img width="847" alt="image" src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/90c4f164-a442-4adb-bbf1-98bbf70c16fb">
    </td>
  </tr>
</table>

Below is a video of how we label points with the brush tool:

<div style="text-align: center;">
  <video src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/83fbcfab-c315-4f53-9e53-7430ac9e0a1e" controls="controls" style="max-width: 730px;">
  </video>
</div>

After we label all our objects, we click on the "export" button and a JSON file will be downloaded which consists of all our objects id, points coordinates, reflectivity value, and segmentation label.




------------------------
<a name="t"></a>
## 6. Training: Classification and Part-Segmentation
Now that we have coded our model architecture, collected our data, cleaned the data, augmented our dataset, and labeled it, it is now time for training!

### 6.1 Training for Classification
We can train our model for either classification or segmentation at a time. We cannot do both at the same time. Hence, we will start with the easiest one: ```Classification```. Note that for both the classification and segmentation tasks, we will create our own **custom Dataset class** that will store the **samples** and their corresponding **labels**. We then create our train and test **DataLoaders** as follows:

```python
train_dataloader_custom = DataLoader(dataset=train_data_custom,
                                     batch_size=BATCH_SIZE,
                                     num_workers=NUM_WORKERS,
                                     shuffle=True)

test_dataloader_custom = DataLoader(dataset=test_data_custom,
                                    batch_size=BATCH_SIZE,
                                    num_workers=NUM_WORKERS,
                                    shuffle=False)
```

We then create an **instance** of our **model**:

```python
# Instantiate model
model_cls = PointNetCls(len(train_data_custom.classes))
```

We use the ```negative log-likelihood``` as our **loss function** and set the ```Stochastic Gradient Descent``` as our **optimizer**. 

```
# Setup loss function and optimizer
loss_fn = nn.NLLLoss()
optimizer = torch.optim.SGD(params=model_cls.parameters(), lr=0.001, momentum=0.95)
```
We then train our model with a batch size of ```32``` and ```25``` epochs. Below are the results from our training:

```python
train_loss: 0.0258 | train_acc: 0.9955 | test_loss: 0.0286 | test_acc: 1.0000
```

<p align="center">
  <img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/f15d9aa1-f677-4d96-8f3e-31057f22d746" width="70%" />
</p>


### 6.1 Training for Part-Segmentation
Now let's train our model for part-segmentation. Note that we will use the same loss function and optimizer as we used for the classifier. With experimentation, I observed a **learning rate** of ```0.001``` and a **momentum** value of ```0.95``` works best. However, this time we will train our model for ```250``` epochs. Below are the results of our training:

```python
# Instantiate model
model_seg = PointNetDenseCls(train_data_custom.num_seg_classes)
```
Below is our segmentation results:

```python
train_loss: 0.0353 | train_acc: 0.9890 | test_loss: 0.0406 | test_acc: 0.9870
```

<p align="center">
  <img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/326f3026-4b7c-4915-90e0-613afb7d6e37" width="70%" />
</p>





<a name="e"></a>
## 7. Evaluation
Now, let's see how our model performs on an out-of-sample dataset.

### 7.1 Evaluation: Classification
Below are some results we performed on our test dataset. We observe that the model correctly predicts the class of all three objects with high probability.


<table>
  <tr>
    <td><img width="514" alt="image" src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/3f0ec298-1eac-43a1-9952-72fdbe4bdaba"></td>
    <td><img width="489" alt="image" src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/a7c1519f-e762-4d22-8295-9e8e05ddcce1"></td>
    <td><img width="564" alt="image" src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/f00871ab-2628-40a2-9c97-6e85fbdffb48"></td>
  </tr>
  <tr>
    <td><img width="445" alt="image" src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/fc1a8a66-6ca7-478e-b5ce-6c222133ecd7"></td>
    <td><img width="453" alt="image" src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/ccdf3521-f485-46df-b435-459ee967f092"></td>
    <td><img width="453" alt="image" src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/15b9ef20-5526-4b17-9071-b7222eec4aa5"></td>
  </tr>
</table>


Secondly, we scanned three more objects that were not part of our test dataset and performed inference. We can see that though these objects are not specifically cups, they do have a physical structure similar to a cup. The model wrongly classifies the first object but correctly classifies the last two objects. Note that 

<table>
  <tr>
    <td><img width="410" alt="image" src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/3f5eee7b-7d4c-4240-8b10-ff71a9be6b23"></td>
    <td><img width="427" alt="image" src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/8cc3e608-5e51-4772-b8d1-aa645f00cc69"></td>
    <td><img width="447" alt="image" src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/d4dc8ca1-223b-459b-8d98-44892903983b"></td>
  </tr>
  <tr>
    <td><img width="445" alt="image" src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/12e6bf5a-a962-4cf7-8635-ba569179c992"></td>
    <td><img width="453" alt="image" src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/b8abdc63-4062-4f3f-891a-f0e8c73c62e6"></td>
    <td><img width="446" alt="image" src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/c8f8cbaa-3943-46bd-90e7-9ceada2cc5f6"></td>
  </tr>
</table>



### 7.1 Evaluation: Part-Segmentation
Similarly, we test our model on our test dataset and we observe that at epoch 100 we start to segment part of the handle of the cup. However, from epoch ```100``` to epoch ```200```, we no longer have proper results. At epoch ```250``` we have a clear segmentation of the handle.



<table>
  <tr>
    <th>Epoch 50</th>
    <th>Epoch 100</th>
    <th>Epoch 150</th>
    <th>Epoch 200</th>
    <th>Epoch 250</th>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/47aa781d-a8be-41a6-96fd-6d0d3957f406" alt="Image 1" width="602"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/4ccf8496-8fb2-4d31-911a-d12ecaaaa88c" alt="Image 2" width="499"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/70928637-c6de-4aa7-a06d-3d637a9096e0" alt="Image 3" width="562"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/dfb707fe-8543-4324-b00b-02cf751ccfb4" alt="Image 4" width="550"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/2565cf99-c57c-4b5f-b07c-2fbf6a21e181" alt="Image 5" width="500"></td>
  </tr>
</table>


With the pan dataset, we have our best results  at epoch ```150```. After that, our results started to deteriorate. This may be due to a lack of data. 


<table>
  <tr>
    <th>Epoch 50</th>
    <th>Epoch 100</th>
    <th>Epoch 150</th>
    <th>Epoch 200</th>
    <th>Epoch 250</th>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/7c94faa5-6540-4b91-8b01-a066045a7558" alt="Image 6" width="454"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/aa448978-ba9b-489d-936f-148f34e5b105" alt="Image 7" width="461"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/c428b4f2-d616-46bd-b633-23d851a7b168" alt="Image 8" width="522"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/572fb124-0f12-426c-baef-60d504aaac8f" alt="Image 9" width="462"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/3534bf17-6cba-4270-bf3f-a6f5e536ef68" alt="Image 10" width="443"></td>
  </tr>
</table>

We tested the model with the out-of-sample dataset of the mug. Although, we can clearly distinguish the handle of the mug, however, we could not segment any points till epoch ```181```. At epoch ```187```, the handle is wrongly segmented as the body. After epoch ```188``` till epoch ```250```, we segment our whole point cloud as the handle which is clearly wrong. 

<table>
  <tr>
    <th>Original</th>
    <th>Epoch 180</th>
    <th>Epoch 181</th>
    <th>Epoch 187</th>
    <th>Epoch 188</th>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/7418ee3a-d49d-4e18-ad0b-dbed733ea5a6" alt="Image 11" width="486"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/912dfaea-e536-4a1b-929c-2069ea5389ad" alt="Image 12" width="429"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/1b301675-4b1f-4604-9524-fbdfac4df272" alt="Image 13" width="344"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/6215bdc8-82ad-4c66-b6a6-aabe8c1b4776" alt="Image 14" width="449"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/3f0dd670-8c5f-4988-995f-a38f4380e615" alt="Image 15" width="427"></td>
  </tr>
</table>

Still no luck with the point cloud of the detergent. At epoch ```260``` we segmented half of the body of the object as the handle. This then decreases will epoch ```280```, however, we fail to segment the handle of the point cloud. 


<table>
  <tr>
    <th>Original</th>
    <th>Epoch 260</th>
    <th>Epoch 270</th>
    <th>Epoch 275</th>
    <th>Epoch 280</th>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/b194e70b-36b5-41cd-b3b1-4d19d29f5ab6" alt="Image 16" width="442"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/7945fd32-6c27-456c-9dc7-e4502d48884f" alt="Image 17" width="431"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/c7b8208c-75cb-4a9a-9767-a3d4a24df50e" alt="Image 18" width="391"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/4d94f7b1-d7eb-4df2-bef3-56d855370a53" alt="Image 19" width="411"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/f9cec748-c213-48a4-87be-635f6438bb54" alt="Image 20" width="390"></td>
  </tr>
</table>

At epoch ```240```, we could see we correctly segmented part of the handle. Alas, similar to the mug point cloud, we are segmenting the body of the object as the handle. 

<table>
  <tr>
    <th>Original</th>
    <th>Epoch 240</th>
    <th>Epoch 242</th>
    <th>Epoch 245</th>
    <th>Epoch 250</th>
  </tr>
  <tr>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/e737a4f3-023a-480d-877f-52f170df54c7" alt="Image 21" width="348"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/98c0e23f-162f-48e4-9b3b-d3a15367c948" alt="Image 22" width="340"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/e89cd972-b53c-4063-ac1d-f0643f919d7b" alt="Image 23" width="369"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/41867d34-b330-4c61-972f-1378a76201f7" alt="Image 24" width="369"></td>
    <td><img src="https://github.com/yudhisteer/Robotic-Grasping-Detection-with-PointNet/assets/59663734/91274274-7e07-4242-8191-c39c03ec067a" alt="Image 25" width="380"></td>
  </tr>
</table>

-------------------

## Conclusion
The PointNet was one of the first neural network architectures that could directly process raw point cloud data instead of transforming first into voxels as previous models would do. What makes PointNet powerful is its **permutation invariance** characteristic that can handle point clouds with varying order by aligning the point cloud into a canonical space. In this project, we used PointNet for both **Classification** and **Part-Segmentation**. We saw that the PointNet has a simple architecture - it comprises of **mini-PointNets** - and in turn, makes it computationally efficient. However,  PointNet is sensitive to noisy data and a limited dataset can be a major drawback. 

Collecting our own data with an app like **Polycam** is possible, but it can be time-consuming and might require multiple attempts for proper point cloud processing. We addressed this by augmenting our point cloud data to increase our dataset size. Although labeling using **Segments.ai** is user-friendly, it remains a manual effort that consumes time. For classification, only ```100 epochs``` were needed to achieve high accuracy on our test set. However, for part-segmentation, we trained our model for ```250 epochs```. During the evaluation of out-of-sample data, we encountered challenges in segmenting the handle from the body due to limited data. To improve this, we could consider a **transfer learning** approach, where we first train the model on the **ShapeNet** dataset, then use the learned weights to further train on our custom dataset, and finally, perform inference on our out-of-sample data. This approach could be listed as a further improvement to our project. 

Our goal to segment the handle of an object has proven conclusive when tested on the test dataset. However, for a zero shot segmentation on out-of-sample data, we still need to train our model on larger dataset. We showed how if we could scan an onject in 3D using LiDAR, we could run inference and detect the handle on the object. Our robot would then be able to grasp the object by the handle exactly how a human would do. 



-------------------

## References
[1] Medium. (n.d.). An In-Depth Look at PointNet. [Article]. [https://medium.com/@luis_gonzales/an-in-depth-look-at-pointnet-111d7efdaa1a](https://medium.com/@luis_gonzales/an-in-depth-look-at-pointnet-111d7efdaa1a)

[2] GitHub. (n.d.). PointNet Implementation. [Repository]. [https://github.com/luis-gonzales/pointnet_own](https://github.com/luis-gonzales/pointnet_own)

[3] Keras. (n.d.). PointNet Example. [Documentation]. [https://keras.io/examples/vision/pointnet/](https://keras.io/examples/vision/pointnet/)

[4] Medium. (n.d.). Understanding PointNet. [Article]. [https://medium.com/p/90398f880c9f](https://medium.com/p/90398f880c9f)

[5] Towards Data Science. (n.d.). 3D Deep Learning Python Tutorial: PointNet Data Preparation. [Article]. [https://towardsdatascience.com/3d-deep-learning-python-tutorial-pointnet-data-preparation-90398f880c9f](https://towardsdatascience.com/3d-deep-learning-python-tutorial-pointnet-data-preparation-90398f880c9f)

[6] Medium. (n.d.). Speaking Code: PointNet. [Article]. [https://medium.com/deem-blogs/speaking-code-pointnet-65b0a8ddb63f](https://medium.com/deem-blogs/speaking-code-pointnet-65b0a8ddb63f)

[7] arXiv. (n.d.). PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation. [Research Paper]. [https://arxiv.org/pdf/1506.02025.pdf](https://arxiv.org/pdf/1506.02025.pdf)

[8] Geek Culture. (n.d.). Understanding 3D Deep Learning with PointNet. [Article]. [https://medium.com/geekculture/understanding-3d-deep-learning-with-pointnet-fe5e95db4d2d](https://medium.com/geekculture/understanding-3d-deep-learning-with-pointnet-fe5e95db4d2d)

[9] Towards Data Science. (n.d.). Deep Learning on Point Clouds: Implementing PointNet in Google Colab. [Article]. [https://towardsdatascience.com/deep-learning-on-point-clouds-implementing-pointnet-in-google-colab-1fd65cd3a263](https://towardsdatascience.com/deep-learning-on-point-clouds-implementing-pointnet-in-google-colab-1fd65cd3a263)

[10] GitHub. (n.d.). PointNet Notebook. [Notebook]. [https://github.com/nikitakaraevv/pointnet/blob/master/nbs/PointNetClass.ipynb](https://github.com/nikitakaraevv/pointnet/blob/master/nbs/PointNetClass.ipynb)

[11] Medium. (n.d.). Introduction to PointNet. [Article]. [https://medium.com/@itberrios6/introduction-to-point-net-d23f43aa87d2](https://medium.com/@itberrios6/introduction-to-point-net-d23f43aa87d2)

[12] GitHub. (n.d.). PointNet PyTorch Implementation. [Repository]. [https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py](https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py)

[13] Medium. (n.d.). PointNet From Scratch. [Article]. [https://medium.com/@itberrios6/point-net-from-scratch-78935690e496](https://medium.com/@itberrios6/point-net-from-scratch-78935690e496)

[14] Medium. (n.d.). An In-Depth Look at PointNet. [Article]. [https://medium.com/@luis_gonzales/an-in-depth-look-at-pointnet-111d7efdaa1a](https://medium.com/@luis_gonzales/an-in-depth-look-at-pointnet-111d7efdaa1a)

[15] YouTube. (n.d.). PointNet Tutorial. [Video]. [https://www.youtube.com/watch?v=HIRj5pH2t-Y&t=427s&ab_channel=Lights%2CCamera%2CVision%21](https://www.youtube.com/watch?v=HIRj5pH2t-Y&t=427s&ab_channel=Lights%2CCamera%2CVision%21)

[16] Towards Data Science. (n.d.). A Comprehensive Introduction to Different Types of Convolutions in Deep Learning. [Article]. [https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215](https://towardsdatascience.com/a-comprehensive-introduction-to-different-types-of-convolutions-in-deep-learning-669281e58215)

