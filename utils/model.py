import os
import json

import torch
import numpy as np
import itertools

from model import E2ETTS, MultiPeriodDiscriminator, MultiScaleDiscriminator


def get_model(args, configs, device, train=False, ignore_layers=[]):
    (preprocess_config, model_config, train_config) = configs
    use_mpd = model_config["discriminator"]["use_mpd"]

    epoch = 1
    model = E2ETTS(preprocess_config, model_config, train_config).to(device)
    mpd = MultiPeriodDiscriminator(lrelu_slope=model_config["discriminator"]["lrelu_slope"]).to(device) if use_mpd else None
    msd = MultiScaleDiscriminator(lrelu_slope=model_config["discriminator"]["lrelu_slope"]).to(device)
    if args.restore_step:
        ckpt_path = os.path.join(
            train_config["path"]["ckpt_path"],
            "{}.pth.tar".format(args.restore_step),
        )
        ckpt = torch.load(ckpt_path, map_location=device)
        epoch = int(ckpt["epoch"])
        model_dict = ckpt['model']
        for ignore_layer in ignore_layers:
            model_dict = {k: v for k, v in model_dict.items()
                        if ignore_layer not in k}
            dummy_dict = model.state_dict()
            dummy_dict.update(model_dict)
            model_dict = dummy_dict
        model.load_state_dict(model_dict, strict=False)
        if use_mpd:
            mpd.load_state_dict(ckpt["mpd"])
        msd.load_state_dict(ckpt["msd"])

    if train:
        init_lr_M = train_config["optimizer"]["init_lr_M"]
        init_lr_D = train_config["optimizer"]["init_lr_D"]
        betas = train_config["optimizer"]["betas"]
        gamma = train_config["optimizer"]["gamma"]
        optM = torch.optim.AdamW(model.parameters(), lr=init_lr_M, betas=betas)
        if use_mpd:
            optD = torch.optim.AdamW(
                itertools.chain(msd.parameters(), mpd.parameters()), lr=init_lr_D, betas=betas)
        else:
            optD = torch.optim.AdamW(msd.parameters(), lr=init_lr_D, betas=betas)
        sdlM = torch.optim.lr_scheduler.ExponentialLR(optM, gamma=gamma)
        sdlD = torch.optim.lr_scheduler.ExponentialLR(optD, gamma=gamma)
        if args.restore_step:
            optM.load_state_dict(ckpt["optM"])
            optD.load_state_dict(ckpt["optD"])
            sdlM.load_state_dict(ckpt["sdlM"])
            sdlD.load_state_dict(ckpt["sdlD"])
        model.train()
        if use_mpd:
            mpd.train()
        msd.train()
        return model, mpd, msd, optM, optD, sdlM, sdlD, epoch

    model.eval()
    model.requires_grad_ = False
    return model


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param
