"""
This code performs content-aware image resizing using seam carving algorithm.
It removes or adds the least important pixels in the image, minimizing the distortion of the important features.

The script takes 3 arguments:

'image' - path to input image
'save_path' - path to save the output image
'seams' - number of seams to remove (or add if negative value)

Usage:
python seam_carving.py <input_image> <output_image> <number_of_seams>
"""

import argparse
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage import filters
import numpy as np

def get_arguments():
    """
    Initialize the command line argument parser and add the required arguments.
    Return the parsed arguments.
    """
    # init parser
    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument('image', type=str)
    parser.add_argument('save_path', type=str)
    parser.add_argument('seams', type=int)
    # return parsed args
    return parser.parse_args()

def main(image_path, save_path, seams):
    """
    Load the input image from the given path and remove or add the specified number of seams.
    Save the output image to the given path.
    """
    image = imread(image_path)
    for _ in range(seams):
        image = run(image)
    imsave(save_path, image)  

def run(image):
    """
    Remove the lowest cost seam from the given image and return the updated image.
    """
    edges = filters.sobel(rgb2gray(image))
    costs = find_costs(edges)
    path = find_seam(costs, edges)
    return remove(image, path)

def find_costs(edges):
    """
    Compute the cumulative cost matrix for each pixel in the given image.
    Return the cost matrix.
    """
    costs = np.copy(edges)
    for i in range(1, len(edges)):
        for j in range(len(edges[0])):
            # find lowest cost path from previous row and update path costs
            if j == 0: # left edge
                cost = min([costs[i-1][j], costs[i-1][j+1]])
            elif j == len(edges[0]) - 1: # right edge
                cost = min([costs[i-1][j-1], costs[i-1][j]])
            else:  # middle
                cost = min([costs[i-1][j-1], costs[i-1][j], costs[i-1][j+1]])
            # update costs
            costs[i][j] += cost
    return costs

def find_seam(costs, edges):
    """
    Find the lowest cost seam in the given cost matrix and return the indices of pixels to be removed.
    """
    current_index = find_lowest(costs)
    seam = [current_index]
    for row in reversed(range(len(costs) - 1)):
        if current_index == 0:
            seam.append(np.argmin(costs[row][:2]) + seam[-1])
        elif current_index == len(costs[row]):
            seam.append(np.argmin(costs[row][-2:]) + seam[-1] -1)
        else:
            seam.append(np.argmin(costs[row][current_index-1:current_index+2]) + seam[-1] - 1)
        current_index = seam[-1]
    return seam

def find_lowest(costs):
    """
    Find the index of the pixel with the lowest cost in the last row of the given cost matrix.
    """
    lowest, lowest_index = float('inf'), 0
    for i in range(len(costs[-1])):
        if costs[-1][i] < lowest:
            lowest, lowest_index = costs[-1][i], i
    return lowest_index

def remove(image, path):
    """ 
    takes an input image and a path which specifies which pixels to remove from each row of the 
    image. It returns a new image with the specified pixels removed.
    """
    # remove one pixel at each index specified in path
    num_rows, num_col = image.shape[:2]
    new_image = np.zeros((num_rows, num_col - 1, 3))
    for row in range(num_rows):
        new_image[row][:] = np.concatenate((image[row][:path[row]], image[row][path[row]+1:]))
    return new_image


if __name__ == "__main__":
    # get arguments
    args = get_arguments()
    main(args.image, args.save_path, args.seams)

