import torch
import torchvision

# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d


	# Option 1: Use binary cross entropy loss
	# m = torch.nn.Sigmoid()
	# loss = torch.nn.BCELoss()


	# Option 2: Use binary cross entropy loss and put five times more weight on postive than negative examples
	#loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(5, dtype=torch.float))
	#loss = loss(m(voxel_src), voxel_tgt)
	#loss = loss(voxel_src, voxel_tgt)

    # Option 3: Use focal loss
	loss = torchvision.ops.sigmoid_focal_loss(inputs=voxel_src, targets=voxel_tgt, alpha=0.6, gamma=1, reduction='sum')

	return loss
