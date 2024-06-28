import numpy as np
from scipy import ndimage
from multiprocessing import Pool
from functools import partial

def ATF4_analysis(img, cutoff=0.95):
    '''
    Takes an image file and returns ...
    '''
    lipid_mean_list = []
    mean_ATF4_list = []
    lipid_percent_list = []
    
    #set threshold for lipids (you can either do this with a quantile or a threshold)
    #lipid_cutoff = threshold_isodata(test_img['image_data'][1])
    lipid_cutoff=np.quantile(img['image_data'][1], cutoff)

    #this code is creating a binary mask of the lipids
    lipids_mask = img['image_data'][1] > lipid_cutoff

    #this code is basically "clear outside' in imagej that clears the outside of all the lipids while keeping the pixel values
    lipids = np.where(img['image_data'][1] > lipid_cutoff, img['image_data'][1], 0)

    # Iterate over each mask
    for mask_label in range(1, img['masks'].max() + 1):

        # Calculate mean ATF4 for the current mask
        mean_ATF4 = ndimage.mean(img['image_data'][2], labels=img['masks'], index=mask_label)
        lipid_mean_per_cell=ndimage.mean(lipids, labels=img['masks'], index=mask_label)
        
        lipid_percent_area=ndimage.mean(lipids_mask, labels=img['masks'], index=mask_label)
        
        lipid_sum = ndimage.sum_labels(lipids, labels=img['masks'], index=mask_label)
        sum_ATF4 = ndimage.sum(img['image_data'][2], labels=img['masks'], index=mask_label)
        
        # Append mean ATF4 to the list for the current image
        mean_ATF4_list.append(mean_ATF4)
        lipid_mean_list.append(lipid_mean_per_cell)
        lipid_percent_list.append(lipid_percent_area)
        
    return mean_ATF4_list, lipid_mean_list, lipid_percent_list

def ATF4_parallel(img_dat, n_processes=48, progress_bar=lambda x, **progress_kwargs: x, **kwargs):
    p=Pool(processes=n_processes) # intialize multiprocessing pool (n_processes=number of cores in your CPU)

    ATF4_cutoff=partial(ATF4_analysis, **kwargs) # pass additional parameters to the analysis function (here it's cutoff)

    out=[x for x in progress_bar(p.imap(ATF4_cutoff, img_dat), total=len(img_dat), desc='doing ATF4 analysis')] # do multiprocessing and collect output
    p.close()
    return out