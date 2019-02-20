import visdom

def gen_plots(epoch_losses, epoch_accs, current_session= visdom.Visdom()):

    """
    Method to plot Loss v epoch and Accuracy v epoch
    on the current Visdom session passed
    :param current_session: Visdom session started in main method
    :param epoch_losses: NxM Tensor of N epochs over M series (e.g., M=2 for train and validation)
    :param epoch_accs: same as epoch_losses, but for accuracy values
    :return:
    """

    # Metric over epoch plots        #Showing only int values on plot for xaxis

    current_session.line(Y=epoch_losses, opts={
        'showlegend': True,
        'title': 'Loss v. epoch',
        'xlabel': 'Epoch no.',
        'ylabel': 'Loss',
        'legend': ['Training', 'Validation'],
        'layoutopts': {'plotly': {'xaxis': {'tickformat': ',d'}}}
    }
             )

    current_session.line(Y=epoch_accs, opts={
        'showlegend': True,
        'title': 'Accuracy v. epoch',
        'xlabel': 'Epoch no.',
        'ylabel': 'Accuracy [%]',
        'legend': ['Training', 'Validation'],
        'layoutopts': {'plotly': {'xaxis': {'tickformat': ',d'}}}
    }
             )