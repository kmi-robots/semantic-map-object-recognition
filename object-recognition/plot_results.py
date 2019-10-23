import visdom
import torch



def gen_plots(epoch_losses, epoch_accs, epochs, MNIST, NCC, current_session=visdom.Visdom(), precrec=False, rocauc=False):

    """
    Method to plot Loss v epoch and Accuracy v epoch
    on the current Visdom session passed
    Can be used for other metrics too
    :param current_session: Visdom session started in main method
    :param epoch_losses: NxM Tensor of N epochs over M series (e.g., M=2 for train and validation)
    :param epoch_accs: same as epoch_losses, but for accuracy values
    :param epochs: total number of epochs
    :param precrec: boolean flag for precision recall plots, defaults to False
    :param rocauc: boolean flag for roc auc score plots, defaults to False
    :return:
    """

    ymin1 = 0.0
    ymax1= 2.0

    if precrec:

        title_l_ = 'Precision'
        title_a_ = 'Recall'
        ymax1 = 1.0
        ymin2 = ymin1
        ymax2 = ymax1

    elif rocauc:

        title_l_ = 'ROC_AUC score'
        ymax1 = 1.0

        #Only one plot, differently from the other cases
        current_session.line(Y=epoch_losses, X=torch.Tensor(range(epoch_losses.shape[0])), opts={
            'showlegend': True,

            'title': title_l_,
            'xlabel': 'Epoch no.',
            'ylabel': title_l_,
            'legend': ['Training', 'Validation'],
            'xtickmin': 1,
            'xtickmax': epochs,
            'xtickstep': 50,
            'ytickmin': ymin1,
            'ytickmax': ymax1
        })

        return


    else:
        title_l_ = 'Loss'
        title_a_ = 'Accuracy [%]'
        ymin2 = 0
        ymax2 = 100


    if MNIST & ( not NCC):


        title_l = title_l_ + ' v. epoch - MNIST - Baseline'
        title_a = title_a_ + ' v. epoch - MNIST - Baseline'

    elif MNIST & NCC:

        title_l = title_l_ + ' v. epoch - MNIST - NCC'
        title_a = title_a_ + ' v. epoch - MNIST - NCC'

    elif (not MNIST) & NCC:

        title_l = title_l_ + ' v. epoch - SNS2 - NCC'
        title_a = title_a_ + 'v. epoch - SNS2 - NCC'

    else:

        title_l = title_l_ + ' v. epoch - SNS2 - Baseline'
        title_a = title_a_ + ' v. epoch - SNS2 - Baseline'

    current_session.line(Y=epoch_losses, X= torch.Tensor(range(epoch_losses.shape[0])), opts={
        'showlegend': True,

        'title': title_l,
        'xlabel': 'Epoch no.',
        'ylabel': title_l_,
        'legend': ['Training', 'Validation'],
        'xtickmin': 1,
        'xtickmax': epochs,
        'xtickstep':50,
        'ytickmin': ymin1,
        'ytickmax': ymax1
        })


    current_session.line(Y=epoch_accs, X= torch.Tensor(range(epoch_accs.shape[0])), opts={
        'showlegend': True,

        'title': title_a,
        'xlabel': 'Epoch no.',
        'ylabel': title_a_,
        'legend': ['Training', 'Validation'],
        'xtickmin': 1,
        'xtickmax': epochs,
        'xtickstep': 50,
        'ytickmin': ymin2,
        'ytickmax': ymax2
    }
             )


def bar_plot(class_wise_data,  current_session=visdom.Visdom()):

    class_serie, data_serie = list(class_wise_data)

    current_session.bar(
        X=data_serie,
        Y=class_serie,
        opts=dict(
            stacked=False,
            title='Class-wise Raking averages',
            xlabel='Top5-ranking Avg accuracy',
            xtickmin=0.0,
            xtickmax=1.0,
            xtickstep=0.1,

        )
    )
