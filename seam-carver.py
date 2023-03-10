import argparse
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import filters
import numpy as np

def get_arguments():
    # init parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('image')
    parser.add_argument('save_path')
    
    # return parsed args
    return parser.parse_args()

def main(image_path, path):
    image = imread(image_path)
    edges = filters.sobel(rgb2gray(image))
    

def seam(edges):
    # calculate seams
    cost = edges.copy()

    # construct array to track/store path origin
    origin = np.zeros(edges.shape)
    for i in range(len(origin[0])):
        origin[0][i] = i

    for i in range(1, len(edges)):
        for j in range(len(edges[0])):
            # find lowest cost path from previous row and update path origin
            # format: (value, i index of value, j index of value)  -  the indices are used to update origin
            if j == 0: # left edge
                prev = [(cost[i-1][j], i-1, j), (cost[i-1][j+1], i-1, j+1)]
                cost[i][j], i1, j1  = min(prev, key=lambda x: x[0])
            elif j == len(edges[0]): # right edge
                prev = [(cost[i-1][j-1], i-1, j-1), (cost[i-1][j], i-1, j)]
                cost[i][j], i1, j1 = min(prev, key=lambda x: x[0])
            else:  # middle
                prev = [(cost[i-1][j-1], i-1, j-1), (cost[i-1][j], i-1, j), (cost[i-1][j+1], i-1, j+1)]
                cost[i][j], i1, j1 = min(prev, key=lambda x: x[0])
            # update origin
            origin[i][j] = origin[i1][j1]

def remove(image):
    # remove
    new_image = None
    return new_image



if __name__ == "__main__":
    # get arguments
    args = get_arguments()
    main(args.image, args.save_path)
