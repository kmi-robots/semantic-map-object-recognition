import scipy.io
import os
import json
import argparse

"""
Trivial script to dump a JSON for 
object classes that have attributes 
annotated as Matlab files.
as released by ImageNet: http://image-net.org/download-attributes
"""

def main(path_to_mat):

    base = os.getcwd()

    mat_files = [os.path.join(base, path_to_mat, f) \
                 for f in os.listdir(os.path.join(base,path_to_mat))]

    for path in mat_files:

        mat = scipy.io.loadmat(path)

        print(mat)





if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('mat', help='Relative path to input Matlab files')
    args = parser.parse_args()


    main(args.mat)