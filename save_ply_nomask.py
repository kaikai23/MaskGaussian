import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel



def save_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        iter_folders = os.listdir(os.path.join(dataset.model_path, "point_cloud"))
        for iter_folder in iter_folders:
            gaussians.load_ply(os.path.join(dataset.model_path,
                                            "point_cloud",
                                            iter_folder,
                                            "point_cloud.ply"))
            iter, num = iter_folder.split("_")
            gaussians.save_ply_default(os.path.join(dataset.model_path,
                                                    "point_cloud",
                                                    iter+"_"+str(int(num)+1),
                                                    "point_cloud.ply"))


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Saving " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    save_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)