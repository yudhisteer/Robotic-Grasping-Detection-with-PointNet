import os
import open3d as open3d
import numpy as np
from utils import visualization_draw_geometry, visualize_point_clouds
import copy
import random


def ransac(point_cloud, distance_threshold=0.33, ransac_n=3, num_iterations=100):
    """
    RANSAC-based plane segmentation for a point cloud.

    Parameters:
        point_cloud (open3d.geometry.PointCloud): The input point cloud.
        distance_threshold (float, optional): The maximum distance a point can be from the plane to be considered an inlier.
            Default is 0.33.
        ransac_n (int, optional): The number of points to randomly sample for each iteration of RANSAC. Default is 3.
        num_iterations (int, optional): The number of RANSAC iterations to perform. Default is 100.

    Returns:
        open3d.geometry.PointCloud, open3d.geometry.PointCloud: Two point clouds representing the inliers and outliers
        of the segmented plane, respectively.
    """
    # Perform plane segmentation using RANSAC
    plane_model, inliers = point_cloud.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n,
                                                     num_iterations=num_iterations)

    # Extract inlier and outlier point clouds
    inlier_cloud = point_cloud.select_by_index(inliers)
    outlier_cloud = point_cloud.select_by_index(inliers, invert=True)

    # Color the outlier cloud red and the inlier cloud blue
    outlier_cloud.paint_uniform_color([0.8, 0.2, 0.2])  # Red
    inlier_cloud.paint_uniform_color([0.25, 0.5, 0.75])  # Blue

    return outlier_cloud, inlier_cloud


def visualize_segmentation(outlier_cloud, inlier_cloud):
    # Convert Open3D PointCloud to numpy arrays
    outlier_points = np.asarray(outlier_cloud.points)
    inlier_points = np.asarray(inlier_cloud.points)

    # Create Open3D point cloud objects
    outlier_pcd = open3d.geometry.PointCloud()
    inlier_pcd = open3d.geometry.PointCloud()

    outlier_pcd.points = open3d.utility.Vector3dVector(outlier_points)
    inlier_pcd.points = open3d.utility.Vector3dVector(inlier_points)

    # Create visualizer
    visualizer = open3d.visualization.Visualizer()
    visualizer.create_window()

    visualizer.add_geometry(outlier_pcd)
    visualizer.add_geometry(inlier_pcd)

    outlier_pcd.paint_uniform_color([1.0, 0.0, 0.0])  # Red color for outliers
    inlier_pcd.paint_uniform_color([0.0, 0.0, 1.0])  # Blue color for inliers

    view_control = visualizer.get_view_control()
    view_control.set_lookat([0, 0, 0])  # Set the camera center
    view_control.set_up([0, 0, 1])  # Set the up direction
    view_control.set_front([1, 0, 0])  # Set the forward direction

    visualizer.run()
    visualizer.destroy_window()


def augmentation_noise(point_cloud, num_augmented_samples=5, noise_level=0.0005, save="augmented_point_cloud"):
    global output_folder

    for i in range(num_augmented_samples):
        # Clone original point cloud
        augmented_cloud = copy.deepcopy(point_cloud)

        # Apply noise to point coordinates
        noise = np.random.normal(0, noise_level, size=(len(augmented_cloud.points), 3))
        augmented_cloud.points = open3d.utility.Vector3dVector(np.asarray(augmented_cloud.points) + noise)

        # Generate a random color
        color = np.random.rand(3)

        # Assign  color to augmented point cloud
        augmented_cloud.colors = open3d.utility.Vector3dVector(np.tile(color, (len(augmented_cloud.points), 1)))

        # Visualize augmented point cloud
        open3d.visualization.draw_geometries([augmented_cloud], window_name=f"Augmented Point Cloud {i}")

        if save is not None:
            # Save  augmented point cloud
            output_ply_path = os.path.join(output_folder, f"{save}_{i}.ply")
            open3d.io.write_point_cloud(output_ply_path, augmented_cloud)

            print(f"Saved {save}_{i}.ply")


def augmentation_sampling(point_cloud, num_augmented_samples=5, min_points_to_sample=500, max_points_to_sample=1000, save="sampled_point_cloud"):
    for i in range(num_augmented_samples):
        # Generate a random number of points to sample within the defined range
        num_points_to_sample = np.random.randint(min_points_to_sample, max_points_to_sample + 1)
        print("num_points_to_sample: ", num_points_to_sample)

        # Randomly sample points
        if num_points_to_sample < len(point_cloud.points):
            sampled_indices = np.random.choice(len(point_cloud.points), num_points_to_sample, replace=False)
            sampled_points = np.asarray(point_cloud.points)[sampled_indices, :]

            # Create new point cloud with sampled points
            sampled_point_cloud = open3d.geometry.PointCloud()
            sampled_point_cloud.points = open3d.utility.Vector3dVector(sampled_points)

            # Visualize
            open3d.visualization.draw_geometries([sampled_point_cloud], window_name="Sampled Point Cloud")
        else:
            print("Number of points to sample exceeds the size of the original point cloud.")
            break

        if save is not None:
            # Save  augmented point cloud
            output_ply_path = os.path.join(output_folder, f"{save}_{i}.ply")
            open3d.io.write_point_cloud(output_ply_path, sampled_point_cloud)
            print(f"Saved {save}_{i}.ply")



def augmentation_deformation(point_cloud, num_augmented_samples=5, min_bending_angle=10.0, max_bending_angle=15.0):

    def generate_rotation_matrix(axis, angle):
        # Generate a 3D rotation matrix for a given axis and angle
        axis /= np.linalg.norm(axis)
        a = np.cos(angle / 2.0)
        b, c, d = -axis * np.sin(angle / 2.0)
        return np.array([
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (a * c + b * d)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (a * b + c * d), a * a + d * d - c * c - b * b]])

    for i in range(num_augmented_samples):
        # Clone the original point cloud
        deformed_cloud = copy.deepcopy(point_cloud)
        # Generate a random angle
        random_max_bending_angle_rad = np.random.randint(min_bending_angle, max_bending_angle + 1)

        # maximum bending angle in radians
        max_bending_angle_rad = np.radians(random_max_bending_angle_rad)
        print("\nMax_bending_angle_rad: ", max_bending_angle_rad)

        # Randomly select a point as the bending axis
        bending_axis_index = random.randint(0, len(deformed_cloud.points) - 1)
        bending_axis = deformed_cloud.points[bending_axis_index]

        # Apply bending deformation
        for i, point in enumerate(deformed_cloud.points):
            if i != bending_axis_index:
                # Generate a random rotation matrix for the bending angle
                bending_angle = random.uniform(-max_bending_angle_rad, max_bending_angle_rad)
                rotation_matrix = generate_rotation_matrix(bending_axis, bending_angle)

                # Apply the rotation to the point
                deformed_cloud.points[i] = np.dot(rotation_matrix, point)

        # Visualize
        open3d.visualization.draw_geometries([deformed_cloud], window_name="Sampled Point Cloud")





def augmentation_dropout(point_cloud, num_augmented_samples=5, max_dropout_prob=0.8):

    for i in range(num_augmented_samples):
        # Clone the original point cloud
        augmented_cloud = copy.deepcopy(point_cloud) #we put inside for loop so that we modify a new point cloud
        # Generate a random dropout_prob
        dropout_prob = np.random.uniform(0, max_dropout_prob)
        print("\nProbability dropout: ", dropout_prob)

        # Calculate the number of points to drop
        num_points = len(augmented_cloud.points)
        print("Original number of points: ", num_points)
        num_points_to_drop = int(num_points * dropout_prob)
        print("Number of points to drop: ", num_points_to_drop)

        # Randomly select points to drop
        if num_points_to_drop > 0:
            points_to_drop_indices = np.random.choice(num_points, num_points_to_drop, replace=False)
            points_to_keep = np.delete(np.asarray(augmented_cloud.points), points_to_drop_indices, axis=0)

            # Update the point cloud with the remaining points
            augmented_cloud.points = open3d.utility.Vector3dVector(points_to_keep)

        # Calculate the number of points to keep
        num_points = len(augmented_cloud.points)
        print("Number of points to keep: ", num_points)

        # Visualize
        open3d.visualization.draw_geometries([augmented_cloud], window_name="Augmented Point Cloud")







if __name__ == "__main__":
    # Get the current directory
    current_directory = os.getcwd()

    # Go back to the parent directory
    parent_directory = os.path.dirname(current_directory)

    # Set input directory
    point_cloud_folder = os.path.join(parent_directory, 'Data', 'Cup_PLY')
    output_folder = os.path.join(parent_directory, 'Data', 'Output')

    # Choose index
    index = 0

    # Get pcd file
    point_cloud_path = os.path.join(point_cloud_folder, os.listdir(point_cloud_folder)[index])
    point_cloud = open3d.io.read_point_cloud(point_cloud_path)

    # Separate points and Colors
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)

    # check point cloud
    print("\n Point Cloud shape: ")
    points = np.asarray(points)
    print(points.shape)
    print("\n Colors shape: ")
    colors = np.asarray(colors)
    print(colors.shape)

    ## visualize point cloud
    visualization_draw_geometry(point_cloud, background='black') # Dark background

    ### ------- SEGMENTATION
    outlier_cloud, inlier_cloud, = ransac(point_cloud, distance_threshold=0.005, ransac_n=3, num_iterations=100)
    print("\nOutlier shape: ")
    print(outlier_cloud)
    print("Inlier shape: ")
    print(inlier_cloud)
    # # ## Call the function with your outlier and inlier point clouds
    # visualize_segmentation(outlier_cloud, inlier_cloud)

    # # visualize point cloud
    visualization_draw_geometry(outlier_cloud, background='white') # Dark background
    # visualization_draw_geometry(inlier_cloud, background='white') # Dark background
    #
    # # Save the outlier point cloud to a PLY file in an output folder
    # open3d.io.write_point_cloud(os.path.join(output_folder, f'{index+1}.ply'), outlier_cloud)




    # ### ----- Data Augmentation: Noise
    # augmentation_noise(outlier_cloud, num_augmented_samples=5, noise_level=0.0025, save=None)



    # ### ----- Data Augmentation: Random Sampling
    # augmentation_sampling(outlier_cloud, num_augmented_samples=5, min_points_to_sample=1000, max_points_to_sample=5000,
    #                       save=None)


    ### ----- Data Augmentation: Random Deformation
    augmentation_deformation(outlier_cloud, num_augmented_samples=5, min_bending_angle=10.0, max_bending_angle=15.0)



    # ### ----- Data Augmentation: Random Dropout
    # augmentation_dropout(outlier_cloud, num_augmented_samples=5, max_dropout_prob=0.6)




