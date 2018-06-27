import os
import argparse 


#In this case both sets were separate, no use of random split


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('trainpath', help="Path to filter image data")
    parser.add_argument('fullpath', help="Path to all images")
    parser.add_argument('fullpath2', help="Path no 2 (to all images)")
    parser.add_argument("testpath",  help="Path to test set")

    args =parser.parse_args()

    #####Hardcoded, Default names for Darknet
    outtrain = 'train.txt'
    outtest = 'test.txt'
    #########################################

    #Append mode for multiple runs if in different folders, change to 'w' otherwise
    with open(outtrain, 'a') as trainf, open(outtest, 'a') as testf:

        train_names = os.listdir(args.trainpath)
        

        test_names = os.listdir(args.testpath)

        for name in train_names:

            name = name[:-3] + 'jpg'     
        
            if os.path.isfile(os.path.join(args.fullpath, name)):
                
                trainf.write(os.path.join(args.fullpath, name)+'\n')
            else:

                trainf.write(os.path.join(args.fullpath2, name)+'\n')

        for name in test_names:

            testf.write(os.path.join(args.testpath, name)+'\n')
