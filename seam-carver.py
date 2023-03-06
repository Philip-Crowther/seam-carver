import argparse

def get_arguments():
    # init parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('image')
    parser.add_argument('save_path')
    
    # return parsed args
    return parser.parse_args()

def main(image, path):
    pass

def weights(image):
    # calculate weight of each pixel
    weights = None
    return weights

def seams(weights):
    # calculate seams
    seams = None
    return seams

def lowest_cost_seam(seams):
    # find lowest cost seam
    seam = None

def remove(image):
    # remove seam
    new_image = None
    return new_image


if __name__ == "__main__":
    # get arguments
    args = get_arguments()
    main(args.image, args.save_path)
