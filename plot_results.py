from matplotlib import pyplot as plt
import pickle


def plot_results(results, dirname):
    plt.close('all')

    fig, axarr = plt.subplots(2, sharex=True, sharey=True)

    for k, result in results.items():
        axarr[0].plot(result.history['loss'], label=k, linewidth=1.5)
        axarr[1].plot(result.history['val_loss'], label=k, linewidth=1.5)

    axarr[1].set_xlabel('Epoch')
    axarr[0].set_ylabel('Training Error')
    axarr[1].set_ylabel('Validation Error')
    handles, labels = axarr[0].get_legend_handles_labels()
    lgd1 = plt.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1.2))
    plt.gcf().tight_layout()
    fig.subplots_adjust(hspace=0)
    plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
    axarr[0].grid()
    axarr[1].grid()
    plt.savefig('plots/' + dirname[:-1] + "_val.png", bbox_extra_artists=(lgd1, ), bbox_inches='tight')


def save_curve_data(results, filename):
    to_save = []
    for k, result in results.items():
        to_save.append(
            {"label": k, "train_loss_history": result.history['loss'], "val_loss_history": result.history['val_loss']})
    with open(filename, "wb") as f:
        pickle.dump(to_save, f)
