"""
Vis utilities. Code adapted from LOST: https://github.com/valeoai/LOST
"""
import cv2
import torch
import skimage.io
import numpy as np
import torch.nn as nn
from PIL import Image
import scipy
# import torchshow as ts
from skimage import exposure 
import matplotlib.pyplot as plt

def visualize_img(image, vis_folder, im_name):
    pltname = f"{vis_folder}/{im_name}"
    Image.fromarray(image).save(pltname)
    print(f"Original image saved at {pltname}.")

def visualize_predictions(img, pred, vis_folder, im_name, save=True):
    """
    Visualization of the predicted box and the corresponding seed patch.
    """
    image = np.copy(img)
    # Plot the box
    cv2.rectangle(
        image,
        (int(pred[0]), int(pred[1])),
        (int(pred[2]), int(pred[3])),
        (255, 0, 0), 3,
    )
    if save:
        pltname = f"{vis_folder}/{im_name}_TokenCut_pred.jpg"
        Image.fromarray(image).save(pltname)
        print(f"Predictions saved at {pltname}.")
    return image
  
def visualize_predictions_gt(img, pred, gt, vis_folder, im_name, dim, scales, save=True):
    """
    Visualization of the predicted box and the corresponding seed patch.
    """
    image = np.copy(img)
    # Plot the box
    cv2.rectangle(
        image,
        (int(pred[0]), int(pred[1])),
        (int(pred[2]), int(pred[3])),
        (255, 0, 0), 3,
    )
    # Plot the ground truth box
    if len(gt>1):
        for i in range(len(gt)):
            cv2.rectangle(
                image,
                (int(gt[i][0]), int(gt[i][1])),
                (int(gt[i][2]), int(gt[i][3])),
                (0, 0, 255), 3,
            )
    if save:
        pltname = f"{vis_folder}/{im_name}_TokenCut_BBOX.jpg"
        Image.fromarray(image).save(pltname)
        #print(f"Predictions saved at {pltname}.")
    return image

def visualize_eigvec(eigvec, scales=[16,16], dims=(14, 14), save=True):
    """
    Visualization of the second smallest eigvector
    """
    eigvec = eigvec[0,:, :, :].detach().cpu()
    for t, eigvec_t in enumerate(eigvec):
        for h, eigvect_h in enumerate(eigvec_t):    
            num_bins = 20

            # Compute histogram bins using torch.histc
            hist = torch.histc(eigvect_h, bins=num_bins)

            # Plot the histogram
            plt.bar(range(num_bins), hist)
            plt.xlabel('Bins')
            plt.ylabel('Frequency')
            plt.title('Histogram of Tensor Values')

            # Save the histogram plot to disk
            plt.savefig(f"Frame{t+1}-Head{h+1}_spatial_attn_hist.jpg")
            plt.close()
            # ts.save(eigvect_h.reshape(dims), f"Frame{t+1}-Head{h+1}_spatial_attn.jpg") 
            # print(f"Eigen attention saved at Frame{t+1}-Head{h+1}_spatial_attn.jpg.")
            eigvec_save = scipy.ndimage.zoom(eigvect_h.reshape(dims), scales, order=0, mode='nearest')
            if save:
                pltname = f"Frame{t+1}-Head{h+1}_spatial_attn.jpg"
                # ts.save(eigvec_save, pltname)
                plt.imsave(fname=pltname, arr=eigvec_save, cmap='cividis')
                print(f"Eigen attention saved at {pltname}.")


def inverse_normalize(tensors, mean, std):
    tensors = tensors[0].permute(1, 0, 2, 3)
    output = []
    for tensor in tensors:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        denormalized_tensor = tensor * std + mean

        # Convert to range [0, 255]
        denormalized_tensor *= 255.0
        output.append(denormalized_tensor)

    return output

# input = inverse_normalize(tensor=input, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

def visualize_heatmap(eigvec, frames, scales=[16,16], dims=(14, 14)):
    frames = inverse_normalize(frames, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    eigvec = eigvec[0,:, :, :].detach().cpu()
    for t, eigvec_t in enumerate(eigvec):
        frame = cv2.cvtColor(frames[t].permute(1, 2, 0).cpu().numpy().astype(np.uint8), cv2.COLOR_RGB2BGR)
        cam = scipy.ndimage.zoom(eigvec_t, zoom= [1,1], order=0, mode='nearest')
        map_img = exposure.rescale_intensity(cam, out_range=(0, 255))
        map_img = np.uint8(map_img)
        heatmap_img = cv2.applyColorMap(map_img, cv2.COLORMAP_JET)
        #merge map and frame
        fin = cv2.addWeighted(heatmap_img, 0.3, frame, 0.7, 0.5, dtype=cv2.CV_64F)

        #show result
        pltname = f"Frame{t+1}_spatial_attn.jpg"
        cv2.imwrite(pltname, fin)

        print(f"Heatmap saved at {pltname}.")