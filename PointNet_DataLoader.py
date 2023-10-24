import pathlib
import open3d as open3d
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import os
from torch.utils.data import DataLoader

# Function to check number of images in parent folder and subfolder
def check_directory(parent_folder):
    for filepath, directories, filenames in os.walk(parent_folder):
        print(f"There are {len(directories)} directories and {len(filenames)} files in '{filepath}'.")


# Make function to find classes in target directory
def find_classes(directory):
    """Finds the class folder names in a target directory.
    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        (list_of_class_names, dict(class_name: idx...))

    Example:
        find_classes("food_images/train")
        # >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

    # 3. Crearte a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx



# Simple point cloud coloring mapping
def read_pointnet_colors(seg_labels):
    map_label_to_rgb = {
        1: [0, 255, 0],
        2: [0, 0, 255]
    }
    colors = np.array([map_label_to_rgb[label] for label in seg_labels])
    return colors



def visualize_custom_point_cloud(points):
    test_cloud = open3d.geometry.PointCloud()
    test_cloud.points = open3d.utility.Vector3dVector(points)
    test_cloud.colors = open3d.utility.Vector3dVector(read_pointnet_colors(seg.numpy()))
    open3d.visualization.draw_geometries([test_cloud])



# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):

    # 2. Initialize with a targ_dir
    def __init__(self, targ_dir: str, classification: bool) -> None:
        # Get all points paths
        self.points = list(pathlib.Path(targ_dir).glob("*/*/*.pts"))
        #Get PCD paths
        self.pcd = list(pathlib.Path(targ_dir).glob("*/*/*.pcd"))
        # Get segmentation label
        self.seg = list(pathlib.Path(targ_dir).glob("*/*/*.seg"))
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)
        self.classification = classification


    # 4. Make function to load images
    def load_point_cloud(self, index: int, show:bool):
        "Opens an point cloud via a path and returns it."
        point_cloud_path = self.pcd[index]
        # print(point_cloud_path)
        # Read point cloud
        point_cloud = open3d.io.read_point_cloud(str(point_cloud_path))
        if show:
            open3d.visualization.draw_geometries([point_cloud])
        return point_cloud

    def load_segmentation(self, index: int):
        "Opens a segmentation label file via a path and returns it."
        seg_path = self.seg[index]
        with open(seg_path, "r") as seg_file:
            data = []
            for line in seg_file:
                # Split the line into values and convert them to integers
                values = [int(value) for value in line.strip().split()]
                data.extend(values)

        # Convert the data list to a PyTorch LongTensor
        seg_tensor = torch.LongTensor(data)
        # print(seg_tensor)
        return seg_tensor


    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.points)

    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int):
        "Returns one sample of data, data and label (X, y)."

        # Get the points
        point_cloud = self.load_point_cloud(index, show=False)
        # Get the points as a NumPy array
        points = np.asarray(point_cloud.points)
        # Convert NumPy array to PyTorch tensor
        points_tensor = torch.from_numpy(points).float()
        # print("Tensor shape:", points_tensor.shape)

        # Get segmentation label
        seg_tensor = self.load_segmentation(index)

        # Get class
        class_name  = self.pcd[index].parent.parent.name
        class_idx = self.class_to_idx[class_name]

        if self.classification:
            print("\n***---------Classfication------------***")
            return points_tensor, class_idx
        else:
            print("\n***----------Segmentation------------***")
            return points_tensor, seg_tensor, class_idx



if __name__ == "__main__":
    # Get the current directory
    current_directory = os.getcwd()

    # Set input directory
    cup_knife_pan = os.path.join(current_directory, 'data', 'cup_knife_pan')

    # # Check directories
    # check_directory(cup_knife_pan)

    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / "cup_knife_pan"

    # Setup train and testing paths
    train_dir = image_path / "train"
    test_dir = image_path / "test"


    classes, class_to_idx = find_classes(train_dir)
    print(classes, class_to_idx)
    print(len(classes))


    ### --------- CLASSIFICATION ------------ ####
    train_data_custom = ImageFolderCustom(targ_dir=train_dir, classification=True)
    test_data_custom = ImageFolderCustom(targ_dir=test_dir, classification=True)

    # print(len(train_data_custom), len(test_data_custom))
    # print(train_data_custom.classes)
    # print(train_data_custom.class_to_idx)

    # # visualize points
    # train_data_custom.load_point_cloud(index=2, show=False)
    #
    # # segmentation
    # train_data_custom.load_segmentation(index=2)
    #
    # points, class_idx = train_data_custom[1]
    #
    # print("\nClass id: ", class_idx)
    # print("Shape of point cloud: ", points.size())
    # print("Datatype of point cloud: ", points.type())
    #
    # print("\nPoint (x,y,z):")
    # print(points)


    # ### --------- SEGMENTATION ------------ ####
    # train_data_custom = ImageFolderCustom(targ_dir=train_dir, classification=False)
    # test_data_custom = ImageFolderCustom(targ_dir=test_dir, classification=False)
    #
    # print(len(train_data_custom), len(test_data_custom))
    # print(train_data_custom.classes)
    # print(train_data_custom.class_to_idx)
    #
    # # visualize points
    # train_data_custom.load_point_cloud(index=2, show=False)
    #
    # # segmentation
    # train_data_custom.load_segmentation(index=2)
    #
    # points, seg, class_idx = train_data_custom[1]
    #
    # print("\nClass id: ", class_idx)
    # print("Shape of point cloud: ", points.size())
    # print("Datatype of point cloud: ", points.type())
    # print("Shape of segmentation label: ", seg.size())
    # print("Data type of segmentation: ", seg.type())
    #
    # print("\nPoint (x,y,z):")
    # print(points)
    # print("\nSegmentation label:")
    # print(seg)
    # ## Visualize segmentation
    # visualize_custom_point_cloud(points)


















