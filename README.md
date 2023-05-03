# Neural Approaches to 3D Scene Reconstruction

## NeRF
Results of reimplementing NeRF
![](vanilla_nerf/images/part_3_hr_n6_noembed_240.gif)

## GNT
Results on validation sets mentioned in our report can be found in this [Google Drive link](https://drive.google.com/drive/folders/1-AJg5LOle1OSBJgvEwQL2bzLiqaPDiin)

What we have implemented ourselves are:
1. The FeatureExtractor class implementing the UNet feature extractor (found at gnt/gnt/custom_feature_network.py)
2. The ShallowFeatureExtractor class implementing the fast one layer feature extractor (found at gnt/gnt/custom_feature_network.py)
3. The CustomTransformer2D class implementing the view transformer (found at gnt/gnt/transformer_network.py)
4. The Transformer class implementing the ray transformer (found at gnt/gnt/transformer_network.py)

