## Setup

Please download and extract the dataset from [here](https://drive.google.com/file/d/1VoSmRA9KIwaH56iluUuBEBwCbbq3x7Xt/view?usp=sharing).
After unzipping, set the appropiate path references in `dataset_location.py` file [here](https://github.com/learning3d/assignment2/blob/a1655b7b12abf9d3b12b2b7b40a64aaef3abb96d/dataset_location.py#L2)

Make sure you have installed the packages mentioned in `requirements.txt`.
This assignment will need the GPU version of pytorch.

## Reconstructing 3D from single view
This section will involve training a single view to 3D pipeline for voxels, point clouds and meshes.
Refer to the `save_freq` argument in `train_model.py` to save the model checkpoint quicker/slower. 

## Image to voxel grid
Run the file `python train_model.py --type 'vox'`, to train single view to voxel grid pipeline

After trained, visualize the input RGB, ground truth voxel grid and predicted voxel in `eval_model.py` file using:
`python eval_model.py --type 'vox' --load_checkpoint`


Model weights for Conv3DTranpose architecture:
https://drive.google.com/file/d/1ycAPZE3rY6P9y-hir6vj4YJ4Me_lzfko/view?usp=share_link

Model weights for Linear:
https://drive.google.com/file/d/1aS8SiJLmEBMar4kjsN32cRqougg3izTz/view?usp=sharing
