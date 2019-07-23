from matplotlib import pyplot as plt
import pickle
import keras
import seaborn as sns


LINEWIDTH=2.0
LINESTYLES=['-', '--', ':', '-.']

class MyException(Exception):
    pass


def plot_results(results, dirname, datatype="keras"):

    fig, axarr = plt.subplots(2, sharex=True, sharey=True)

    if datatype == "keras":
        i = 0
        for k, result in results.items():
            axarr[0].plot(result.history['loss'], label=k, linewidth=LINEWIDTH, linestyle=LINESTYLES[i])
            axarr[1].plot(result.history['val_loss'], label=k, linewidth=LINEWIDTH, linestyle=LINESTYLES[i])
            i += 1
            i = i % len(LINESTYLES)
    elif datatype == "saved":
        for i, result in enumerate(results):
            i = i % len(LINESTYLES)
            axarr[0].plot(result['train_loss_history'], label=result['label'], linewidth=LINEWIDTH, linestyle=LINESTYLES[i])
            axarr[1].plot(result['val_loss_history'], label=result['label'], linewidth=LINEWIDTH, linestyle=LINESTYLES[i])
    else:
        raise MyException("Illegal Datatype")
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


def plot_result(history, dirname):
    if isinstance(history, keras.callbacks.History):
        history=history.history
    sns.reset_orig()
    plt.clf()
    plt.style.use("seaborn-paper")
    fig = plt.figure(figsize=(10, 3), dpi=300, facecolor="white")
    # summarize history for loss
    ax = plt.subplot("121")
    ax.plot(history['loss'])
    ax.plot(history['val_loss'])
    ax.grid(True)
    ax.set_ylabel('Cross Entropy Error')
    ax.set_xlabel('Epoch')
    ax.set_ylim(0, 1)
    ax.legend(['training', 'validation'], loc='upper right')
    ax = plt.subplot("122")
    ax.plot(history['acc'])
    ax.plot(history['val_acc'])
    ax.set_ylabel('Accuracy')
    ax.grid(True)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Epoch')
    ax.legend(['training', 'validation'], loc='lower right')
    plt.savefig('plots/' + dirname + ".png", bbox_inches='tight')


def save_curve(history, modelname):
    with open(modelname+'_trainHistory', "wb") as f:
        pickle.dump(history, f)


def save_curve_data(results, filename):
    to_save = []
    for k, result in results.items():
        to_save.append(
            {"label": k, "train_loss_history": result.history['loss'], "val_loss_history": result.history['val_loss']})
    with open(filename, "wb") as f:
        pickle.dump(to_save, f)


if __name__=="__main__":
    print("Input Pickled Data Path: ")
    dirname = input()
    with open(dirname, "rb") as f:
        data = pickle.load(f)
    dirname = dirname.replace("/", "_")
    plot_results(data, dirname.replace("\/", "_"), datatype="saved")