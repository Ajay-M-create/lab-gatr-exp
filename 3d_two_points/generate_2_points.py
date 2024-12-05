import os
import random

def generate_1000_files():
    """
    Generates 1000 files, each containing two points in 3D space.
    The files are saved in the 'generated_files' directory with names test_1.txt to test_1000.txt.
    Each file follows the format:
    
    x y z
    <x1> <y1> <z1>
    <x2> <y2> <z2>
    """
    os.makedirs('generated_files', exist_ok=True)

    for i in range(1, 1001):
        filename = f'test_{i}.txt'
        with open(filename, 'w') as f:
            f.write("x y z\n")
            for _ in range(2):
                x = random.uniform(0, 1)
                y = random.uniform(0, 1)
                z = random.uniform(0, 1)
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

# Example usage:
generate_1000_files()
