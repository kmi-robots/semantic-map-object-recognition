import torch
from torch import optim
from torchvision import transforms
import os
import argparse
import sys
import pandas as pd


#custom classes and methods
import data_loaders
from models import NCCNet, ResSiamese, ResTwoBranch, KNet, NNet, ImprintedKNet #, ContrastiveLoss
from pytorchtools import EarlyStopping
from embedding_extractor import extract_embeddings
from train import train
from validate import validate
from test import test
from imprint import imprint


#-----------------------------------------------------------------------------------------#

#Hardcoded variables, stored by default in these locations, local to the repo -------------------------------------------------------------------------#
model_checkpoint = 'pt_results/kmish25/checkpoint.pt' #hardcoded in pytorchtools.py
path_to_train_data ='./data/processed/kmi_training.dat' #hardcoded in data_loaders.py ln. 205
#--- Simpler not to change -----------------------------------------------------------------------#


def main(args):

    """
    Expects two command line arguments or flags
    first one specifing which model to use, e.g., simple or NCC
    the second one indicating the dataset, e.g., MNIST or SNS2

    """


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_type = args.it
    path_to_input = args.path_to_test
    train_imgs = args.path_to_train
    path_to_train_embeds = args.emb
    #path_to_query_data = args.query

    do_learn = True if args.stage =='train' else False
    feature_extraction = True if args.transfer else False
    keep_embeddings = True if args.store_emb else False

    ResNet = True if args.resnet else False
    KNET = True if args.model == 'knet' else False
    NNET = True if args.model == 'nnet' else False
    imprinting = False if args.noimprint else True
    two_branch = True if args.twobranch else False
    STN = True if args.stn else False  # Whether to use Spatial Transformer module on input or not

    if STN:
        # Decrease learning rate
        lr = 0.00001

    if args.plots:

        from plot_results import gen_plots, bar_plot

    if ResNet:

        # Add hardcoded mean and variance values for torchvision pre-trained modules
        means = [0.485, 0.456, 0.406]
        stds = [0.229, 0.224, 0.225]

        mnist_trans = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.Grayscale(3),
                                          transforms.ToTensor(),
                                          transforms.Normalize(means, stds)])
    else:

        #simply standardize
        means = (0.5,)
        stds = (1.0,)
        mnist_trans = transforms.Compose([transforms.Resize((160, 60)),
                                    transforms.Grayscale(3),
                                    transforms.ToTensor(),
                                    transforms.Normalize(means, stds)])

    if args.ncc:

        #Supported in CPU-only for now
        device = torch.device('cpu')

        model = NCCNet().to(device)
        params_to_update = model.parameters()  # all params

        # Transformations applied to the images on training
        trans_train = transforms.Compose([
            transforms.Resize((160, 60)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)])

        # Transformations applied to the images on validation and at test time
        trans_val = transforms.Compose([
            transforms.Resize((160, 60)),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)])

    else:

        # Transformations applied to the images on training
        trans_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)])

        # Transformations applied to the images on validation and at test time
        trans_val = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(means, stds)])

        if two_branch:

            model = ResTwoBranch(feature_extraction=feature_extraction, stn=STN).to(device)


        elif NNET:

            model = NNet(feature_extraction=feature_extraction).to(device)

        elif KNET and not imprinting:

            model = KNet(feature_extraction=feature_extraction, num_classes=args.N).to(device)

        elif imprinting:
            model = ImprintedKNet(feature_extraction=feature_extraction, num_classes=args.N).to(device)


        else:

            model = ResSiamese(feature_extraction=feature_extraction, stn=STN).to(device)

        if feature_extraction:

            params_to_update = [param for param in model.parameters() if param.requires_grad]
            #only the last layers when doing feature extraction

        else:

            params_to_update = model.parameters()  # all params

    base_trans = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)])

    if not os.path.isdir('./data'):

        os.mkdir('data')


    if not os.path.isdir('./pt_results'):

        os.mkdir('pt_results')

    if not os.path.isdir('./pt_results/paper'):

        os.mkdir('pt_results/paper')


    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    if do_learn:  # training mode

        if args.mnist:

            train_loader = torch.utils.data.DataLoader(
              data_loaders.BalancedMNIST('./data', train=True, transform=mnist_trans, download=True), batch_size=args.batch,
               shuffle=True)


            val_loader = torch.utils.data.DataLoader(
                data_loaders.BalancedMNIST('./data', train=False, transform=mnist_trans, download=False), batch_size=args.batch,
                shuffle=False)


        else:


            train_loader = torch.utils.data.DataLoader(
               data_loaders.BalancedTriplets('./data', device, train=True, N=args.N, transform=base_trans, ResNet=ResNet, KNet=KNET), batch_size=args.batch,
               shuffle=True)

            val_loader = torch.utils.data.DataLoader(
                data_loaders.BalancedTriplets('./data', device, train=False, N=args.N, transform=base_trans, ResNet=ResNet, KNet=KNET), batch_size=args.batch,
                shuffle=False)

        optimizer = optim.SGD(params_to_update, lr=args.lr, momentum=args.momentum, weight_decay=args.wdecay )
        #optimizer = optim.Adam(params_to_update, lr=args.lr, weight_decay=weight_decay)

        epoch_train_metrics = [] #torch.empty((1,2))#((num_epochs, 2))
        epoch_val_metrics = [] #torch.empty_like(epoch_train_metrics)

        #Imprint weight of the model classifier first
        if imprinting:
            imprint(model, device, train_loader, num_classes=args.N)

        for epoch in range(args.epochs):

            if early_stopping.early_stop:
                print("Early stopping")

                break

            epoch_train_metrics.append(train(model, device, train_loader, epoch, optimizer, args.epochs, args.avg, knet=KNET,nnet=NNET))

            val_m = validate(model, device, val_loader, args.avg, knet=KNET,nnet=NNET)

            epoch_val_metrics.append(val_m)

            valid_loss = val_m[0]
            early_stopping(valid_loss, model)

        ## Plotting ##------------------------------------------------------------------#
        epoch_train_metrics = torch.stack(epoch_train_metrics, dim=0)
        epoch_val_metrics = torch.stack(epoch_val_metrics, dim=0)

        epoch_losses = torch.stack((epoch_train_metrics[:,0], epoch_val_metrics[:,0]), dim=1)
        epoch_accs = torch.stack((epoch_train_metrics[:,1], epoch_val_metrics[:,1]), dim=1)

        epoch_ps = torch.stack((epoch_train_metrics[:,2], epoch_val_metrics[:,2]), dim=1)
        epoch_rs = torch.stack((epoch_train_metrics[:,3], epoch_val_metrics[:,3]), dim=1)

        epoch_roc_auc = torch.stack((epoch_train_metrics[:,4], epoch_val_metrics[:,4]), dim=1)

        if args.plots:
            gen_plots(epoch_losses, epoch_accs, args.epochs, args.mnist, args.ncc)

            #Gen precision and recall plots
            gen_plots(epoch_ps, epoch_rs, args.epochs, args.mnist, args.ncc, precrec=True)
            #-------------------------------------------------------------------------------#

            #Gen ROC AUC score plot
            gen_plots(epoch_roc_auc, epoch_roc_auc, args.epochs, args.mnist, args.ncc, rocauc=True)
            #-------------------------------------------------------------------------------#

    else:

        #Test/Inference phase
        # Load training checkpoint and set model in test mode
        path_to_state = os.path.join(path_to_input.split('KMi_collection')[0],
                                     'kmish25/checkpoint_imprKNET_1prod.pt')  # checkpoint_imprKNET_1prod_DA_static
        model.load_state_dict(torch.load(path_to_state, map_location={'cuda:0': 'cpu'}))

        model.eval()

        keep_embeddings = False
        class_wise_res = None

        if args.it == 'camera':

            if args.stage=="explore" or args.stage == "only-segment": #Input data from sensor in real-time


                    # Init & start ROS node including a subscriber and a publisher
                    import rospy
                    from ROS_IO import ImageConverter

                    try:
                        rospy.init_node('image_converter') #, anonymous=True)

                    except Exception as e:
                        print(str(e))
                        sys.exit(0)

                    rate = rospy.Rate(10)


                    if args.stage=="only-segment":

                        via_project = {}
                        local_img_path = os.path.join(os.getcwd(), "object-recognition/data/KMi_collection/NW_activity_test")

                        if not os.path.isdir(local_img_path):

                            os.mkdir(local_img_path)
                            print("Created local image folder")

                        print("annotated test imgs will be saved under %s" % local_img_path)
                        project_name = "KMi_NW_activity_test"

                        via_project["_via_settings"]["core"]: {
                            "buffer_size": "18",
                            "filepath": {},
                            "default_filepath": local_img_path
                        }

                        via_project["_via_settings"]["project"] = {

                            "name": project_name

                        }

                        via_project["_via_img_metadata"] = {}

                        io = ImageConverter(path_to_input, args, model, device, base_trans,via_data=via_project )
                        io.start(path_to_input,args, model, device, base_trans, rate  )  #processing called inside the ROS node directly

                    else:
                        io = ImageConverter(path_to_input, args, model, device, base_trans)
                        io.start(path_to_input, args, model, device, base_trans, \
                                 rate)  # processing called inside the ROS node directly
                        #start with service data constantly true


            elif args.stage=="reason":

                #TODO code for loading a dataset with annotated bboxes, correct based on last saved rules and evaluate
                try:
                    rule_df = pd.read_pickle(os.getcwd()+'/data/extracted_rules.pkl')

                except FileNotFoundError:

                    print("Please re-run in explore mode first to generate a set of rules for reasoning")
                    sys.exit(0)

                try:
                    #under the hood same case as evaluating on ground truth but with validation against KB rules as only difference
                    input_type="json"
                    args.bboxes ="true"
                    class_wise_res = test(input_type, path_to_input, args, model, device, base_trans, \
                                          path_to_train_embeds=path_to_train_embeds, KBrules=rule_df)\


                except FileNotFoundError:

                    print("Please provide a valid path to VIA-annotated JSON file for test images")
                    sys.exit(0)

                pass

            else:

                print(str(args.stage)+" mode not supported for camera input, please choose other input time")
                sys.exit(0)

        else:

            class_wise_res = test(input_type, path_to_input, args, model, device, base_trans, \
                                  path_to_train_embeds=path_to_train_embeds)

        #Test plot grouped by class
        if class_wise_res is not None and args.plots:

            #Evaluation only available for data held out from ground truth
            #When evaluating on robot-collected data, rankings are logged anyways
            bar_plot(class_wise_res)


    if keep_embeddings:

        #Warning: available for custom set only, no MNIST
        extract_embeddings(model, model_checkpoint, path_to_train_data, train_imgs, \
                        device, path_to_train_embeds, transforms=base_trans)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #positional
    parser.add_argument('it', choices=['reference', 'pickled', 'json', 'camera'],
                        help='Input type at test time: can be one between reference, pickled or json (if environment data have been tagged)'
                             'Choose camera option for online testing on robot - requires ROS melodic')
    parser.add_argument('stage',choices=['train','test', 'baseline', 'explore', 'reason', 'only-segment'],
                        help='One between train, test or baseline (i.e., run baseline NN at test time)')

    parser.add_argument('path_to_train',
                        help='path to training data in anchor branch, i.e., taken from the environment. '
                             'Assumes shapenet for product is available in the local branch')

    parser.add_argument('path_to_test',
                        help='path to test data, in one of the three formats listed above')

    parser.add_argument('emb',
                        help='path where to store/retrieve the output embeddings')

    # optional boolean flags
    parser.add_argument('--resnet', default=True,
                        help='makes ResNet the building block of the CNN, True by default')

    parser.add_argument('--sem', choices=['concept-only', 'full', 'none'], default='full',
                        help='whether to add the semantic modules at inference time. It includes both ConceptNet and Visual Genome by default'
                             'Set to concept-only to discard Visual Genome or None to test without any semantics')

    parser.add_argument('--transfer', type=bool, default=False,
                        help='Defaults to False in all our reported trials. Change'
                        'to True if performing only transfer learning without fine-tuning'
                        )

    parser.add_argument('--store_emb', type=bool, default=True,
                        help='Defaults to True. Set to false if do not wish to store'
                             'the training embedddings locally'
                        )

    parser.add_argument('--noimprint', type=bool, default=False,
                        help='Defaults to False. '
                             'Set to through if running without weight imprinting')

    parser.add_argument('--twobranch', type=bool, default=False,
                        help='Defaults to False in all our reported trials. '
                             'Can be set to True to test a Siamese with weights learned independently on each branch')

    parser.add_argument('--plots', type=bool, default=False,
                        help='Optionally produces plot to check and val loss'
                    )

    parser.add_argument('--bboxes', choices=['true', 'segmented'], default='true',
                        help='Can be set to decide which bounding boxes to use: manually-annotated/ ground truth,'
                             'or produced via segmentation'
                        )

    #optional values to tweak params

    parser.add_argument('--model', choices=['knet, nnet', None], default='knet',
                        help='set to pick one between K-net or N-net or None. K-net is used by default')

    parser.add_argument('--N', type=int, choices=[10,20,25], default=25,
                        help='Number of object classes. Should be one between 10,20,25. Defaults to 25')

    parser.add_argument('--K', type=int, default=5,
                        help='Number of neighbours to consider for ranking at test time (KNN). Defaults to 5')

    parser.add_argument('--Kvoting', choices=['majority','discounted'], default='discounted',
                        help='How votes are computed for K>1. Defaults to discounted majority voting.')

    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size. Defaults to 16')


    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate. Defaults to 0.0001')

    parser.add_argument('--epochs', type=int, default=5000,
                        help='Number of training epochs. Defaults to 5000')

    parser.add_argument('--wdecay', type=float, default=0.00001,
                        help='Weight decay. Defaults to 0.00001')

    parser.add_argument('--patience', type=int, default=100,
                        help='Number of subsequent epochs of no improvement before early stop. Defaults to 100')

    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for SGD training. Defaults to 0.9')

    parser.add_argument('--avg', default='binary',
                        help='metric average type for training monitoring and plots. Please refer to'
                             ' the sklearn docs for all allowed values.')


    #The following args were used only for older trials

    parser.add_argument('--stn', type=bool, default=False,
                        help='Defaults to False in all our reported trials. '
                             'Can be set to True to add a Spatial Transformer Net before the ResNet module')

    parser.add_argument('--mnist', type=bool, default=False,
                        help='whether to test on the MNIST benchmark dataset')

    parser.add_argument('--ncc', type=bool, default=False,
                        help='whether to test with the NormXCorr architecture')

    parser.add_argument('--query', default='./data/shapenet20/test/',
                        help='Path to data used with support on test time in prior trials')


    main(parser.parse_args())

