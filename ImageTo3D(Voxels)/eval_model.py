import argparse
import time
import torch
from model import SingleViewto3D
from r2n2_custom import R2N2
from  pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
import dataset_location
import pytorch3d
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.ops import knn_points
import mcubes
import utils_vox
import matplotlib.pyplot as plt 
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
    look_at_view_transform
)
import imageio

def get_args_parser():
    parser = argparse.ArgumentParser('Singleto3D', add_help=False)
    parser.add_argument('--arch', default='resnet18', type=str)
    parser.add_argument('--max_iter', default=10000, type=str)
    parser.add_argument('--vis_freq', default=10000, type=str)
    parser.add_argument('--batch_size', default=1, type=str)
    parser.add_argument('--num_workers', default=0, type=str)
    parser.add_argument('--type', default='vox', choices=['vox', 'point', 'mesh'], type=str)
    parser.add_argument('--n_points', default=5000, type=int)
    parser.add_argument('--w_chamfer', default=1.0, type=float)
    parser.add_argument('--w_smooth', default=0.1, type=float)  
    parser.add_argument('--load_checkpoint', action='store_true')  
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    return parser

def preprocess(feed_dict, args):
    for k in ['images']:
        feed_dict[k] = feed_dict[k].to(args.device)

    images = feed_dict['images'].squeeze(1)
    mesh = feed_dict['mesh']
    if args.load_feat:
        images = torch.stack(feed_dict['feats']).to(args.device)

    return images, mesh

def save_plot(thresholds, avg_f1_score, args):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(thresholds, avg_f1_score, marker='o')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('F1-score')
    ax.set_title(f'Evaluation {args.type}')
    plt.savefig(f'eval_{args.type}', bbox_inches='tight')


def get_mesh_renderer(image_size=512, lights=None, device=None):
    """
    Returns a Pytorch3D Mesh Renderer.

    Args:
        image_size (int): The rendered image size.
        lights: A default Pytorch3D lights object.
        device (torch.device): The torch device to use (CPU or GPU). If not specified,
            will automatically use GPU if available, otherwise CPU.
    """
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer

def render_ground_truth_3d(voxels, index):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    m = torch.nn.Sigmoid()
    vertices_src, faces_src = mcubes.marching_cubes(m(voxels).detach().cpu().squeeze().numpy(), isovalue=0.5)
    vertices_src = torch.tensor(vertices_src).float()
    faces_src = torch.tensor(faces_src.astype(int))
    mesh_src = pytorch3d.structures.Meshes([vertices_src], [faces_src])
    # Get the renderer.
    renderer = get_mesh_renderer(image_size=256)

    center = torch.mean(vertices_src, dim=0)
    #print(center)
    vertices = vertices_src.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)

    faces = faces_src.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    color=[0.7, 0.7, 1]
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)
    meshes = mesh.extend(60)
    azim = torch.linspace(-180, 180, 60)

    R, T = look_at_view_transform(dist=40, azim=azim, at=(center.numpy(),))

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(device=device, R=R, T=T, fov=60)


    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    rend = renderer(meshes, cameras=cameras, lights=lights)
    # rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    rend = rend.cpu().numpy()[..., :3]  # (B, H, W, 4) -> (H, W, 3)

    my_images = []
    for i in range(60):
        my_images.append(rend[i])

    return rend[index], my_images

def evaluate_model(args):
    r2n2_dataset = R2N2("test", dataset_location.SHAPENET_PATH, dataset_location.R2N2_PATH, dataset_location.SPLITS_PATH, return_voxels=True, return_feats=args.load_feat)

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True)
    eval_loader = iter(loader)

    model =  SingleViewto3D(args)
    model.to(args.device)
    model.eval()

    start_iter = 0
    start_time = time.time()

    thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]

    avg_f1_score_05 = []
    avg_f1_score = []
    avg_p_score = []
    avg_r_score = []

    IOU = []


    if args.load_checkpoint:
        checkpoint = torch.load(f'checkpoint3_{args.type}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Succesfully loaded iter {start_iter}")
    
    print("Starting evaluating !")
    max_iter = len(eval_loader)
    for step in range(start_iter, max_iter):
        iter_start_time = time.time()

        read_start_time = time.time()

        feed_dict = next(eval_loader)

        images_gt, mesh_gt = preprocess(feed_dict, args)

        read_time = time.time() - read_start_time

        predictions = model(images_gt, args)

        voxels = feed_dict['voxels'].float()
        ground_truth_3d = voxels
        result = torch.sum(ground_truth_3d*(predictions.cpu()>1))/torch.sum(torch.max(ground_truth_3d, predictions.cpu()>1))
        IOU.append(result)


        #metrics = evaluate(predictions, mesh_gt, thresholds, args)
        if (step % args.vis_freq) == 0:
            # visualization block
            rend, my_images = render_ground_truth_3d(predictions[0][0], 50)
            rend2, _ = render_ground_truth_3d(ground_truth_3d[0][0], 35)
            #print(rend.shape)
            #print(images_gt[0].shape)
            plt.imsave(f'vis/predicted_{step}_{args.type}.png', rend)
            plt.imsave(f'vis/input_{step}_{args.type}.png', images_gt[0].cpu().numpy())
            plt.imsave(f'vis/gt_{step}_{args.type}.png', rend2)
            #imageio.mimsave(f'vis/predicted_{step}_{args.type}'+'.gif', my_images, fps=15)

      

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        print(step, "/", max_iter)
        # f1_05 = metrics['F1@0.050000']
        # avg_f1_score_05.append(f1_05)
        # avg_p_score.append(torch.tensor([metrics["Precision@%f" % t] for t in thresholds]))
        # avg_r_score.append(torch.tensor([metrics["Recall@%f" % t] for t in thresholds]))
        # avg_f1_score.append(torch.tensor([metrics["F1@%f" % t] for t in thresholds]))

        #print("[%4d/%4d]; ttime: %.0f (%.2f, %.2f); F1@0.05: %.3f; Avg F1@0.05: %.3f" % (step, max_iter, total_time, read_time, iter_time, f1_05, torch.tensor(avg_f1_score_05).mean()))
    

    #avg_f1_score = torch.stack(avg_f1_score).mean(0)

    avg_IOU_score = torch.stack(IOU).mean(0)

    print(avg_IOU_score)
    ##save_plot(thresholds, avg_f1_score,  args)
    print('Done!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Singleto3D', parents=[get_args_parser()])
    args = parser.parse_args()
    evaluate_model(args)
