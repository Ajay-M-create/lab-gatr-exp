# %%
# Import required libraries
import numpy as np
import os
from scipy.spatial import ConvexHull
import math

# %%
# Import necessary libraries
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial import ConvexHull

# Visualize a convex hull
def visualize_convex_hull(points):
    # Create a new figure for 3D plotting
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Compute the convex hull of the points
    hull = ConvexHull(points)

    # Plot the points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=50, c='r', label="Points")

    # Plot the convex hull
    for simplex in hull.simplices:
        simplex_points = points[simplex]
        ax.plot_trisurf(simplex_points[:, 0], simplex_points[:, 1], simplex_points[:, 2], color='b', alpha=0.3)

    # Label the axes
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Show the plot
    plt.show()

# Generate 5 random points and compute their convex hull
def generate_random_convex_hull():
    # Generate 5 random 3D points in the range [0, 1]
    points = np.random.rand(5, 3)  # Ensures coordinates are between 0 and 1

    # Visualize the convex hull of the points
    visualize_convex_hull(points)

# Call the function to generate and visualize the convex hull
generate_random_convex_hull()

# %%
def save_point_cloud(points, filename):
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save the points to a .txt file
    np.savetxt(filename, points, fmt='%.6f', header="x y z", comments='')
    print(f"Saved point cloud to: {filename}")

# Function to generate and save 5000 random convex hulls with normalized coordinates
def generate_and_save_convex_hulls(num_hulls=5000, output_dir="./3d_point_cloud_dataset"):
    for i in range(num_hulls):
        # Generate 5 random 3D points in the range [0, 1]
        points = np.random.rand(5, 3)  # Ensures coordinates are between 0 and 1
        
        # Save all 5 points directly, without computing hull vertices
        filename = os.path.join(output_dir, f"convex_hull_{i}.txt")
        save_point_cloud(points, filename)
        
        # Optional: Compute and verify hull volume if needed
        hull = ConvexHull(points)
        volume = hull.volume
        if volume > 1:
            print(f"Hull {i} has volume {volume}, which exceeds 1.")

# Call the function to generate and save 5000 convex hulls
generate_and_save_convex_hulls(num_hulls=5000)