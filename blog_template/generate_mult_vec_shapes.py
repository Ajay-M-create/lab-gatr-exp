import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Define basis vector lengths
E1_LENGTH = 1    # Length of e1
E2_LENGTH = 2    # Length of e2
E3_LENGTH = 3    # Length of e3

def plot_basis_vectors(ax):
    """Plot and label the basis vectors with different lengths."""
    origin = np.array([0, 0, 0])
    colors = ['r', 'g', 'b']
    labels = ['e1', 'e2', 'e3']
    lengths = [E1_LENGTH, E2_LENGTH, E3_LENGTH]

    for i in range(3):
        vector = np.zeros(3)
        vector[i] = lengths[i]
        ax.quiver(*origin, *vector, color=colors[i], length=1, normalize=True, arrow_length_ratio=0.1)
        ax.text(*(vector + 0.1), labels[i], color=colors[i], fontsize=12)

def plot_vector(ax):
    """Plot a vector multivector as the sum of e1 and e2."""
    # Define the vector components (e1 + e2)
    x0, x1, x2 = E1_LENGTH, E2_LENGTH, 0  # e1 + e2
    vector = np.array([x0, x1, x2])
    
    # Plot the vector
    ax.quiver(0, 0, 0, x0, x1, x2, color='m', linewidth=2, label='Vector x = e1 + e2', arrow_length_ratio=0.1)
    
    # Plot basis vectors
    plot_basis_vectors(ax)
    
    # Set labels and limits
    ax.set_xlim([0, E1_LENGTH + E2_LENGTH + 1])
    ax.set_ylim([0, E1_LENGTH + E2_LENGTH + 1])
    ax.set_zlim([0, E3_LENGTH + 1])
    ax.set_xlabel('e1')
    ax.set_ylabel('e2')
    ax.set_zlabel('e3')
    ax.legend()
    ax.set_title(r'Vector Multivector' + '\n' + r'$x = (0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)$')
    ax.view_init(elev=20, azim=30)

def plot_bivector(ax):
    """Plot a bivector multivector as a parallelogram."""
    # Define two vectors that span the bivector (e1 and e2)
    v1 = np.array([E1_LENGTH, 0, 0])  # e1
    v2 = np.array([0, E2_LENGTH, 0])  # e2
    
    # Create vertices of the parallelogram
    origin = np.array([0, 0, 0])
    vertex1 = origin
    vertex2 = v1
    vertex3 = v1 + v2
    vertex4 = v2
    
    parallelogram = [vertex1, vertex2, vertex3, vertex4]

    # Create a polygon and add to plot
    poly = Poly3DCollection([parallelogram], alpha=0.5, facecolors='cyan')
    poly.set_edgecolor('k')
    ax.add_collection3d(poly)

    # Plot the vectors defining the bivector
    ax.quiver(*origin, *v1, color='r', linewidth=2, arrow_length_ratio=0.1, label='e1')
    ax.quiver(*origin, *v2, color='g', linewidth=2, arrow_length_ratio=0.1, label='e2')

    # Plot basis vectors
    plot_basis_vectors(ax)
    
    # Set labels and limits
    ax.set_xlim([0, E1_LENGTH + E2_LENGTH + 1])
    ax.set_ylim([0, E1_LENGTH + E2_LENGTH + 1])
    ax.set_zlim([0, E3_LENGTH + 1])
    ax.set_xlabel('e1')
    ax.set_ylabel('e2')
    ax.set_zlabel('e3')
    ax.legend()
    ax.set_title(r'Bivector Multivector' + '\n' + r'$x = (0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0)$')
    ax.view_init(elev=20, azim=30)

def plot_trivector(ax):
    """Plot a trivector multivector as a parallelepiped."""
    # Define three vectors that span the trivector (e1, e2, e3)
    v1 = np.array([E1_LENGTH, 0, 0])  # e1
    v2 = np.array([0, E2_LENGTH, 0])  # e2
    v3 = np.array([0, 0, E3_LENGTH])  # e3
    
    # Create vertices of the parallelepiped
    origin = np.array([0, 0, 0])
    vertices = [
        origin,
        v1,
        v2,
        v3,
        v1 + v2,
        v1 + v3,
        v2 + v3,
        v1 + v2 + v3
    ]
    
    # Define the sides of the parallelepiped
    faces = [
        [vertices[0], vertices[1], vertices[4], vertices[2]],
        [vertices[0], vertices[1], vertices[5], vertices[3]],
        [vertices[0], vertices[2], vertices[6], vertices[3]],
        [vertices[7], vertices[5], vertices[1], vertices[4]],
        [vertices[7], vertices[6], vertices[2], vertices[4]],
        [vertices[7], vertices[5], vertices[3], vertices[6]]
    ]
    
    # Create a collection for the faces
    poly = Poly3DCollection(faces, alpha=0.5, facecolors='orange')
    poly.set_edgecolor('k')
    ax.add_collection3d(poly)
    
    # Plot the vectors defining the trivector
    ax.quiver(*origin, *v1, color='r', linewidth=2, arrow_length_ratio=0.1, label='e1')
    ax.quiver(*origin, *v2, color='g', linewidth=2, arrow_length_ratio=0.1, label='e2')
    ax.quiver(*origin, *v3, color='b', linewidth=2, arrow_length_ratio=0.1, label='e3')

    # Plot basis vectors
    plot_basis_vectors(ax)
    
    # Set labels and limits
    ax.set_xlim([0, E1_LENGTH + E2_LENGTH + 1])
    ax.set_ylim([0, E1_LENGTH + E2_LENGTH + 1])
    ax.set_zlim([0, E3_LENGTH + 1])
    ax.set_xlabel('e1')
    ax.set_ylabel('e2')
    ax.set_zlabel('e3')
    ax.legend()
    ax.set_title(r'Trivector Multivector' + '\n' + r'$x = (0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0)$')
    ax.view_init(elev=20, azim=30)

def plot_point(ax):
    """Plot a point multivector as a single point in 3D space."""
    # Define point coordinates rho
    rho0, rho1, rho2 = E1_LENGTH, E2_LENGTH, E3_LENGTH  # Example coordinates
    point = np.array([rho0, rho1, rho2])
    
    # Plot the point
    ax.scatter(rho0, rho1, rho2, color='k', s=100, label='Point x')
    
    # Plot basis vectors
    plot_basis_vectors(ax)
    
    # Set labels and limits
    ax.set_xlim([0, E1_LENGTH + E2_LENGTH + 1])
    ax.set_ylim([0, E1_LENGTH + E2_LENGTH + 1])
    ax.set_zlim([0, E3_LENGTH + 1])
    ax.set_xlabel('e1')
    ax.set_ylabel('e2')
    ax.set_zlabel('e3')
    ax.legend()
    ax.set_title(r'Point Multivector' + '\n' + r'$x = (0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)$')
    ax.view_init(elev=20, azim=30)

def create_subplots():
    """Create a single figure with subplots for each multivector."""
    fig = plt.figure(figsize=(20, 15))

    # Create subplots
    ax1 = fig.add_subplot(221, projection='3d')
    plot_vector(ax1)

    ax2 = fig.add_subplot(222, projection='3d')
    plot_bivector(ax2)

    ax3 = fig.add_subplot(223, projection='3d')
    plot_trivector(ax3)

    ax4 = fig.add_subplot(224, projection='3d')
    plot_point(ax4)

    plt.tight_layout()
    plt.savefig('images/multivector_subplots.png')
    plt.close()

if __name__ == "__main__":
    create_subplots()
