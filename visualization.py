import cartopy.crs as ccrs
import cartopy.feature as cf
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc


def plot_oef_mode(EOF, PC, lon, lat):
    ii = 1
    color = 'RdBu_r'
    colorbarMin = -1
    colorbarMax = 1
    colorspace = 0.1
    level = np.arange(colorbarMin, colorbarMax + colorspace, colorspace)
    ax = plt.axes(projection=ccrs.cartopy.crs.PlateCarree(central_longitude=180))
    h = ax.contourf(lon, lat, EOF[ii, :, :], level, transform=ccrs.PlateCarree(), cmap=color, extend='both')
    cbar = plt.colorbar(h, orientation='horizontal', shrink=1, fraction=0.1, pad=0.1, aspect=40)
    cbar.ax.tick_params(labelsize=10)
    colorLabel = 'SST EOF [K]'
    cbar.set_label(label=colorLabel, fontsize=10)
    # Add in the coordinate system:
    long = np.arange(-180, 180, 45)  # spacing of 45 degrees
    latg = np.arange(-20, 40, 10)  # spacing of 15 degrees
    ax.set_xticks(long, crs=ccrs.PlateCarree())
    ax.set_yticks(latg, crs=ccrs.PlateCarree())
    ax.set_xticklabels(long, fontsize=8)
    ax.set_yticklabels(latg, fontsize=8)
    ax.set_ylabel('lat', fontsize=10)
    ax.set_xlabel('lon', fontsize=10)

    # Add in the continents
    # define the coastlines, the color (#000000) and the resolution (110m)
    feature1 = cf.NaturalEarthFeature(
        name='coastline', category='physical',
        scale='110m',
        edgecolor='#000000', facecolor='none')
    # define the land, the color (#AAAAAA) and the resolution (110m), mask the land, use for SST
    feature2 = cf.NaturalEarthFeature(
        name='land', category='physical',
        scale='110m',
        facecolor='#AAAAAA')

    ax.add_feature(feature2)

    # Set a title for your map:
    title = 'SST JAS EOF' + str(ii + 1)
    plt.title(title, fontsize=10, y=1.03)

    fig, axs = plt.subplots(1, figsize=plt.figaspect(0.15))
    plt.plot(PC[:, ii])


def plot_split_counts(train_y, val_y, test_y):
    ind = [0, 1, 2]
    names = ["train", "val", "test"]
    width = 0.75
    event_cnts = [np.unique(train_y[:, 1], return_counts=True)[1][1], np.unique(val_y[:, 1], return_counts=True)[1][1],
                  np.unique(test_y[:, 1], return_counts=True)[1][1]]
    nonevent_cnts = [np.unique(train_y[:, 1], return_counts=True)[1][0],
                     np.unique(val_y[:, 1], return_counts=True)[1][0],
                     np.unique(test_y[:, 1], return_counts=True)[1][0]]

    p1 = plt.barh(ind, event_cnts, width)
    p2 = plt.barh(ind, nonevent_cnts, width, left=event_cnts)

    plt.yticks(ind, names)
    plt.ylabel("data set")
    plt.xlabel("samples")
    plt.title("Train/Validation/Test Splits", fontsize=16)
    plt.legend(["Event", "Non-event"])


def plot_learning_curve(history):
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=plt.figaspect(0.25))
    ax1.plot(train_acc, label='Training Accuracy')
    ax1.plot(val_acc, label='Validation Accuracy')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1.1)
    ax1.legend()

    ax2.plot(train_loss, label='Training loss')
    ax2.plot(val_loss, label='Validation loss')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()


def plot_calibration_curve(calib_x, calib_y):
    plt.plot(calib_y, calib_x, marker='o', color="darkorange", label='LSTMatt')
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label='Best score')


def plot_roc_auc(model, test_X, test_y):
    yprob = model.predict(test_X)
    yprob = yprob[0:-20, 1]
    testy = test_y[0:-20, 1]
    lr_auc = roc_auc_score(testy, yprob)
    print(lr_auc)
    lr_fpr, lr_tpr, thredhs = roc_curve(testy, yprob)
    print(testy.shape, lr_fpr.shape, lr_tpr.shape, thredhs.shape)
    # plot the roc curve for the model
    plt.plot(lr_fpr, lr_tpr, marker='.', color="darkorange", label='LSTMatt')
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label='No Skill')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

    Y_predict = model.predict(test_X)
    yhat = np.argmax(Y_predict, axis=1)
    yhat = yhat[0:-20]
    lr_precision, lr_recall, _ = precision_recall_curve(testy, yprob)
    lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
    print(testy.shape, lr_recall.shape, lr_precision.shape)
    # summarize scores
    print('f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
    # plot the precision-recall curves
    no_skill = len(testy[testy == 1]) / len(testy)
    plt.plot(lr_recall, lr_precision, marker='.', color="darkorange", label='LSTMatt')
    plt.plot([0, 1], [no_skill, no_skill], color="navy", linestyle='--', label='No Skill')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()

def plot_weights(weights, test_X, ntimestep):
    val_weights = np.ndarray((len(test_X), ntimestep)) + np.nan
    for ii in range(len(test_X)):
        for j in range(ntimestep):
            val_weights[ii, j] = weights[ii][j][0]
    print(np.shape(val_weights))

    fig, axs = plt.subplots(1, figsize=plt.figaspect(0.15))
    for ii in range(len(test_X)):
        plt.plot(val_weights[ii, :])

    fig, axs = plt.subplots(1, figsize=plt.figaspect(0.15))
    plt.plot(np.nanmean(val_weights, axis=0), 'k')
