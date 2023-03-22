import argparse
from skimage.io import imread, imsave
from skimage.color import rgb2gray
from skimage import filters
import numpy as np

def get_arguments():
    # init parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('image', type=str)
    parser.add_argument('save_path', type=str)
    parser.add_argument('seams', type=int)
    
    # return parsed args
    return parser.parse_args()

def main(image_path, save_path, seams):
    image = imread(image_path)
    for _ in range(seams):
        image = run(image)
    imsave(save_path, image)  

def run(image):
    # detect edges then find lowest cost seam and return image with seam removed
    edges = filters.sobel(rgb2gray(image))
    return remove(image, seam(edges))

def seam(edges):
    # find minimum seam and return path
    cost = edges.copy()

    # construct array to track/store path origin
    origin = np.zeros(edges.shape)
    for i in range(len(origin[0])):
        origin[0][i] = i

    for i in range(1, len(edges)):
        for j in range(len(edges[0])):
            print(i, j)
            # find lowest cost path from previous row and update path origin
            # format: (value, i index of value, j index of value)  -  the indices are used to update origin
            if j == 0: # left edge
                prev = [(cost[i-1][j], i-1, j), (cost[i-1][j+1], i-1, j+1)]
            elif j == len(edges[0]) - 1: # right edge
                print(j)
                prev = [(cost[i-1][j-1], i-1, j-1), (cost[i-1][j], i-1, j)]
            else:  # middle
                prev = [(cost[i-1][j-1], i-1, j-1), (cost[i-1][j], i-1, j), (cost[i-1][j+1], i-1, j+1)]
            # get lowest possible cost and its location
            c, i1, j1 = min(prev, key=lambda x: x[0])
            # update cost and origin
            cost[i][j] += c
            origin[i][j] = origin[i1][j1]

    # identify index of minimum seam's lowest point
    minimum, min_index = float('inf'), 0  
    for i in range(len(edges[-1])):
        if edges[-1][i] < minimum:
            minimum, min_index = edges[-1][i], i

    start = origin[-1][min_index]
    # track minimum seam back to origin and store path
    path, curr = [min_index], min_index
    for i in reversed(range(len(origin))-1):  
        if origin[i][curr] == start:
            path.append(curr)
        elif curr > 0 and origin[i][curr-1] == start:
            curr -= 1
            path.append(curr)
        else:  # if neither of the other two options were it, then the last one  has to be
            curr += 1
            path.append(curr)

    return path


def remove(image, path):
    # remove one pixel at each index specified in path
    num_rows, num_col = image.shape
    new_image = np.zeros((num_rows, num_col - 1))
    for row in range(num_rows):
        new_image[row][:] = np.concatenate((image[row][:path[row]], image[row][path[row]+1:]))
    return new_image


if __name__ == "__main__":
    # get arguments
    args = get_arguments()
    main(args.image, args.save_path, args.seams)
