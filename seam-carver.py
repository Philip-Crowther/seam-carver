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
    # detect edges then find lowest costs seam and return image with seam removed
    edges = filters.sobel(rgb2gray(image))
    costs = find_costs(edges)
    path = find_seam(costs, edges)
    return remove(image, path)

def find_costs(edges):
    # find minimum seam and return path
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
    # find final index of the lowest cost seam
    lowest, lowest_index = float('inf'), 0
    for i in range(len(costs[-1])):
        if costs[-1][i] < lowest:
            lowest, lowest_index = costs[-1][i], i
    return lowest_index

def remove(image, path):
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

