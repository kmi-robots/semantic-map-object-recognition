import visdom
import torch

def gen_plots(epoch_losses, epoch_accs, epochs, MNIST, NCC, current_session= visdom.Visdom()):

    """
    Method to plot Loss v epoch and Accuracy v epoch
    on the current Visdom session passed
    :param current_session: Visdom session started in main method
    :param epoch_losses: NxM Tensor of N epochs over M series (e.g., M=2 for train and validation)
    :param epoch_accs: same as epoch_losses, but for accuracy values
    :param epochs: total number of epochs
    :return:
    """

    if MNIST & ( not NCC):

        title_l = 'Loss v. epoch - MNIST - Baseline'
        title_a = 'Accuracy v. epoch - MNIST - Baseline'

    elif MNIST & NCC:
        title_l = 'Loss v. epoch - MNIST - NCC'
        title_a = 'Accuracy v. epoch - MNIST - NCC'

    elif (not MNIST) & NCC:

        title_l = 'Loss v. epoch - SNS2 - NCC'
        title_a = 'Accuracy v. epoch - SNS2 - NCC'

    else:

        title_l = 'Loss v. epoch - SNS2 - Baseline'
        title_a = 'Accuracy v. epoch - SNS2 - Baseline'

    current_session.line(Y=epoch_losses, X= torch.Tensor(range(epoch_losses.shape[0])), opts={
        'showlegend': True,
        'title': title_l,
        'xlabel': 'Epoch no.',
        'ylabel': 'Loss',
        'legend': ['Training', 'Validation'],
        'xtickmin': 1,
        'xtickmax': epochs,
        'xtickstep':1,
        'ytickmin': 0.0,
        'ytickmax': 2.0
        })


    current_session.line(Y=epoch_accs, X= torch.Tensor(range(epoch_accs.shape[0])), opts={
        'showlegend': True,
        'title': title_a,
        'xlabel': 'Epoch no.',
        'ylabel': 'Accuracy [%]',
        'legend': ['Training', 'Validation'],
        'xtickmin': 1,
        'xtickmax': epochs,
        'xtickstep': 1,
        'ytickmin': 0,
        'ytickmax': 100
    }
             )