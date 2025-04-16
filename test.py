import numpy as np
import matplotlib.pyplot as plt  


def create_matrix_A():
    A = np.arange(0,65536, 1)
            
    return np.reshape(A, (256,256)).T

# Task 2: Create Matrix B (256x256) with a circle of radius 40
def create_matrix_B():
    # Initialize a 256x256 matrix with zeros
    rows, cols = 256, 256
    B = np.zeros((rows, cols), dtype=np.uint8)  # Use uint8 for binary (0 or 1)
    
    # Center of the image
    center_x, center_y = 128, 128  # Since 256/2 = 128
    radius = 40
    
    # Fill the matrix: set pixels within the circle (radius 40) to 1
    for i in range(rows):
        for j in range(cols):
            # Calculate distance from the center
            distance = np.sqrt((i - center_x)**2 + (j - center_y)**2)
            if distance <= radius:
                B[i, j] = 1
    
    # Explicitly set the center to 1 (already covered by the circle, but per the problem statement)
    B[center_x, center_y] = 1
    
    return B

# Create the matrices
A = create_matrix_A()
B = create_matrix_B()
print(B)



# Visualize both matrices
plt.figure(figsize=(12, 5))

# Plot Matrix A
plt.subplot(1, 2, 1)
plt.imshow(A, cmap='viridis')
plt.title("Matrix A (256x256)")
plt.colorbar(label='Intensity (0-65535)')
plt.axis('off')

# Plot Matrix B
plt.subplot(1, 2, 2)
plt.imshow(B, cmap='gray')
plt.title("Matrix B (Circle of Radius 40)")
plt.axis('off')

plt.tight_layout()
plt.show()