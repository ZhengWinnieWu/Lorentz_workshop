import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import resize
from tqdm import tqdm
import tensorflow as tf
import shap

#tf.compat.v1.disable_eager_execution()   # I remember innvestigate can be used without a problem when running this line
import innvestigate


############################
# RISE from DIANNA
############################

# Tthe original generate masks method from the official RISE repo https://github.com/eclique/RISE
# It creates super pixel masks that look like wavy blobs morphing into on another
def generate_masks(N, s, p1, model):
    cell_size = np.ceil(np.array(model.input_size) / s)
    up_size = (s + 1) * cell_size

    grid = np.random.rand(N, s, s) < p1
    grid = grid.astype('float32')

    masks = np.empty((N, *model.input_size))

    for i in tqdm(range(N), desc='Generating masks'):
        # Random shifts
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        # Linear upsampling and cropping
        masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                anti_aliasing=False)[x:x + model.input_size[0], y:y + model.input_size[1]]
    masks = masks.reshape(-1, *model.input_size, 1)
    return masks

# Mask 1 entire channel at a time, could be used for sequential input
def generate_channel_masks(N, n_masked_features, model, feature_axis):    
    masks = np.empty((N, *model.input_size))
    for i in range(N):
        n_features = model.input_size[feature_axis]
        mask_indices = np.random.randint(0, n_features, n_masked_features)
        mask = np.ones(model.input_size)
        mask[:,mask_indices] = 0
        masks[i] = mask
    return masks

# From the RISE repo 
# It masks the input using the given masks, runs the model for every masked input, and then adds the masks weighted by the model outputs
def explain(N, model, inp, masks, **kwargs):
    batch_size = kwargs.get('batch_size',100)
    # p1 is the same as p1 from generate_masks, fraction of input data to keep in each mask (Default: auto-tune this value)
    p1 = kwargs.get('p1',0.5)
    preds = []
    p1 = 0.5
    # Make sure multiplication is being done for correct axes
    masked = inp * masks
    for i in tqdm(range(0, len(masks), batch_size), desc='Explaining'):
        preds.append(model.predict(masked[i:min(i+batch_size, len(masks))]))
    preds = np.concatenate(preds)
    sal = preds.T.dot(masks.reshape(N, -1)).reshape(-1, *model.input_size)
    sal = sal / N / p1   # not sure whether for channel mask, need to divide p1 or not
    return sal

############################
# SHAP
############################
def explain_shap(model,X_data,N):
    # select a set of background examples to take an expectation over
    background = X_data[np.random.choice(X_data.shape[0], N, replace=False)]

    # explain predictions of the model on three images
    # can choose different explainer: DeepExplainer, KernelExplainer, etc.
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(X_data)
    
    # summary plot
#     f = plt.gcf()
#     shap.summary_plot(shap_values_mean, features=X_test, feature_names=feature_name, max_display = 5, show=False)
#     ax = plt.gca()
#     plt.show()
    
    return shap_values

############################
# LRP
############################
def explain_innvest(model,X_data,Y_data):
    # Creating an analyzer
    model_wo_sm = innvestigate.model_wo_softmax(tf_model)
    # there are different analyzer: input_t_gradient, deep_taylor, different LRP family
    # https://github.com/albermax/innvestigate
    gradient_analyzer = innvestigate.create_analyzer("input_t_gradient", model_wo_sm)
    
    # Only run for specific category, it is not neccessary
    for jj in range(len(X_data)):
        if Y_data[jj] == 0:
            continue
            
        y_pred = model.predict(X_train[jj:jj+1])   
         # for the right prediction of the selected category, can be changed
        if np.argmax(y_pred) == 1:   
            # Applying the analyzer
            analysis = gradient_analyzer.analyze(X_data[jj:jj+1])
            
    # Displaying the gradient
#     f = plt.gcf()
#     ax = plt.gca()
#     plt.imshow(np.nanmean(analysis_all,axis=0), cmap="coolwarm", interpolation="nearest")
#     plt.colorbar()
#     plt.show()
    
    return analysis