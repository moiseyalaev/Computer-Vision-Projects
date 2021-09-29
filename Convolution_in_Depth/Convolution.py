# %%
# Explore convolution in depth through the "ascent" image from scipy
import cv2
import numpy as np
from scipy import misc

img = misc.ascent()

# %%
# show "ascent" => 2D Grey Scale img
import matplotlib.pyplot as plt
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(img)
plt.show()

# %%
# Store copy of img as np array so we can perform transformations (convolutions) on img
img_transformed = np.copy(i)
size_x = img_transformed.shape[0]
size_y = img_transformed.shape[1]

"""
Define 'filters' for convolution , all the digits
in the filter must either add up to 0 or 1. If not u
se weight to make the final product 0 or 1.
 """
weight  = 1

# filter = [ [0, 1, 0], [1, -4, 1], [0, 1, 0]]
filter = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]] # this filter looks like it detects edges
# filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

# %%
# Perform 2D Convolution

for x in range(1, size_x-1):
  for y in range(1, size_y-1):
      convolution = 0.0
      convolution = convolution + (i[x-1, y-1] * filter[0][0])
      convolution = convolution + (i[x, y-1] * filter[0][1])
      convolution = convolution + (i[x + 1, y-1] * filter[0][2])
      convolution = convolution + (i[x-1, y] * filter[1][0])
      convolution = convolution + (i[x, y] * filter[1][1])
      convolution = convolution + (i[x+1, y] * filter[1][2])
      convolution = convolution + (i[x-1, y+1] * filter[2][0])
      convolution = convolution + (i[x, y+1] * filter[2][1])
      convolution = convolution + (i[x+1, y+1] * filter[2][2])
      convolution = convolution * weight
      if(convolution < 0):
        convolution = 0
      if(convolution > 255):
        convolution = 255
      img_transformed[x, y] = convolution

# %%
# Plot the image to see effect of convolution
plt.gray()
plt.grid(False)
plt.imshow(img_transformed)
plt.axis('off')
plt.show()

print(img_transformed.shape)  # Notice the img size is 512x512

# %%
# Perform Pooling Layer

new_x = int(size_x / 2)
new_y = int(size_y / 2)
newImage = np.zeros((new_x, new_y))

for x in range(0, size_x, 2):
  for y in range(0, size_y, 2):
    pixels = []
    pixels.append(img_transformed[x, y])
    pixels.append(img_transformed[x+1, y])
    pixels.append(img_transformed[x, y+1])
    pixels.append(img_transformed[x+1, y+1])
    newImage[int(x/2),int(y/2)] = max(pixels)  # change max to min or average to see effect

# Plot the image.
plt.gray()
plt.grid(False)
plt.imshow(newImage)
# plt.axis('off')
plt.show()

#Note the size of the axes -- now 256 pixels instead of 512