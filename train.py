import argparse
import os

import torch
import yaml
import numpy as np
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from torch.cuda import amp

from tqdm import tqdm

from utils.model import get_model, get_param_num
from utils.tools import get_configs_of, to_device, log, synth_one_sample
from model import E2ETTSLoss
from dataset import Dataset

import audio as Audio
from evaluate import evaluate

torch.backends.cudnn.benchmark = True


def train(rank, args, configs, batch_size, num_gpus):
    preprocess_config, model_config, train_config = configs
    use_mpd = model_config["discriminator"]["use_mpd"]

    if num_gpus > 1:
        init_process_group(
            backend=train_config["dist_config"]['dist_backend'],
            init_method=train_config["dist_config"]['dist_url'],
            world_size=train_config["dist_config"]['world_size'] * num_gpus,
            rank=rank,
        )
    device = torch.device('cuda:{:d}'.format(rank))

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, model_config, train_config, sort=True, drop_last=True
    )
    data_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=False,
        sampler=data_sampler,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    model, mpd, msd, optM, optD, sdlM, sdlD, epoch = get_model(args, configs, device, train=True,
                                                                ignore_layers=train_config["ignore_layers"])
    if num_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[rank]).to(device)
        if use_mpd:
            mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)
    # scaler = amp.GradScaler(enabled=args.use_amp)
    Loss = E2ETTSLoss(preprocess_config, model_config, train_config, device).to(device)

    # Training
    step = args.restore_step + 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]

    # Logging
    mel_fmax_loss = preprocess_config["preprocessing"]["mel"]["mel_fmax_loss"]
    STFT = Audio.stft.TorchSTFT(preprocess_config).to(device)

    # def model_update(models, step, loss, optimizer):
    #     # Backward
    #     scaler.scale(loss).backward()

    #     # Clipping gradients to avoid gradient explosion
    #     if step % grad_acc_step == 0:
    #         scaler.unscale_(optimizer)
    #         for model in models:
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

    #     # Update weights
    #     scaler.step(optimizer)
    #     scaler.update()
    #     optimizer.zero_grad()

    def model_update(models, step, loss, optimizer):
        # Backward
        loss.backward()

        # Clipping gradients to avoid gradient explosion
        for model in models:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

        # Update weights
        optimizer.step()
        optimizer.zero_grad()

    if rank == 0:
        n_model_params = get_param_num(model)
        print("Number of E2ETTS Parameters                  : {}".format(n_model_params))
        if use_mpd:
            n_mpd_params = get_param_num(mpd)
            print("          MultiPeriodDiscriminator Parameters: {}".format(n_mpd_params))
        n_msd_params = get_param_num(msd)
        print("          MultiScaleDiscriminator Parameters : {}".format(n_msd_params))
        print("          Total Parameters                   : {}\n".format(n_model_params + (n_mpd_params if use_mpd else 0) + n_msd_params))
        # Init logger
        for p in train_config["path"].values():
            os.makedirs(p, exist_ok=True)
        train_log_path = os.path.join(train_config["path"]["log_path"], "train")
        val_log_path = os.path.join(train_config["path"]["log_path"], "val")
        os.makedirs(train_log_path, exist_ok=True)
        os.makedirs(val_log_path, exist_ok=True)
        train_logger = SummaryWriter(train_log_path)
        val_logger = SummaryWriter(val_log_path)

        outer_bar = tqdm(total=total_step, desc="Training", position=0)
        outer_bar.n = args.restore_step
        outer_bar.update()

    train = True
    while train:
        if rank == 0:
            inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        if num_gpus > 1:
            data_sampler.set_epoch(epoch)
        for batchs in loader:
            if train == False:
                break
            for batch in batchs:
                batch = to_device(batch, device)

                #######################
                # Train Discriminator #
                #######################
                # with amp.autocast(args.use_amp):
                # Forward
                output = model(*(batch[2:]), step=step)
                y, y_g_hat = batch[6].unsqueeze(1), output[0]

                # MPD
                loss_disc_f = torch.zeros(1).to(device)
                if use_mpd:
                    y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
                    loss_disc_f, losses_disc_f_r, losses_disc_f_g = Loss.discriminator_loss(y_df_hat_r, y_df_hat_g)

                # MSD
                y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
                loss_disc_s, losses_disc_s_r, losses_disc_s_g = Loss.discriminator_loss(y_ds_hat_r, y_ds_hat_g)

                loss_disc_all = loss_disc_s + loss_disc_f

                model_update([msd, mpd] if use_mpd else [msd], step, loss_disc_all, optD)

                #######################
                # Train Generator #
                #######################
                # with amp.autocast(args.use_amp):
                # L1 Mel-Spectrogram Loss
                # loss_mel = Loss.spec_loss(y.squeeze(1), y_g_hat.squeeze(1)) * 45
                loss_mel = nn.functional.l1_loss(
                    STFT(y.squeeze(1), mel_fmax=mel_fmax_loss), STFT(y_g_hat.squeeze(1), mel_fmax=mel_fmax_loss)
                ) * 45

                # Upsampler
                loss_fm_f = torch.zeros(1).to(device)
                loss_gen_f = torch.zeros(1).to(device)
                if use_mpd:
                    y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
                    loss_fm_f = Loss.feature_loss(fmap_f_r, fmap_f_g)
                    loss_gen_f, losses_gen_f = Loss.generator_loss(y_df_hat_g)
                y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
                loss_fm_s = Loss.feature_loss(fmap_s_r, fmap_s_g)
                loss_gen_s, losses_gen_s = Loss.generator_loss(y_ds_hat_g)

                loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

                # Variance
                (
                    loss_var_all,
                    loss_pitch,
                    loss_energy,
                    loss_duration,
                    loss_ctc,
                    loss_bin,
                ) = Loss.variance_loss(batch, output, step=step)

                model_update([model], step, loss_gen_all + loss_var_all, optM)

                losses = (
                    loss_disc_all + loss_gen_all + loss_var_all, loss_gen_all, loss_var_all, loss_disc_s, loss_disc_f, loss_gen_s, loss_gen_f, loss_fm_s, loss_fm_f, loss_mel, loss_pitch, loss_energy, loss_duration, loss_ctc, loss_bin,
                )

                if rank == 0:
                    if step % log_step == 0:
                        losses_ = [sum(l.values()).item() if isinstance(l, dict) else l.item() for l in losses]
                        message1 = "Step {}/{}, ".format(step, total_step)
                        message2 = "Total Loss: {:.4f}, loss_gen_all: {:.4f}, loss_var_all: {:.4f}, loss_disc_s: {:.4f}, loss_disc_f: {:.4f}, loss_gen_s: {:.4f}, loss_gen_f: {:.4f}, loss_fm_s: {:.4f}, loss_fm_f: {:.4f}, loss_mel: {:.4f}, loss_pitch: {:.4f}, loss_energy: {:.4f}, loss_duration: {:.4f}, loss_ctc: {:.4f}, loss_bin: {:.4f}".format(
                            *losses_
                        )

                        with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                            f.write(message1 + message2 + "\n")

                        outer_bar.write(message1 + message2)

                        log(train_logger, step, losses=losses, lr=sdlM.get_last_lr()[-1])

                    if step % synth_step == 0:
                        figs, fig_attn, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                            batch,
                            output,
                            model_config,
                            preprocess_config,
                            STFT,
                        )
                        log(
                            train_logger,
                            step,
                            img=fig_attn,
                            tag="Training/attn",
                        )
                        log(
                            train_logger,
                            step,
                            figs=figs,
                            tag="Training",
                        )
                        sampling_rate = preprocess_config["preprocessing"]["audio"][
                            "sampling_rate"
                        ]
                        log(
                            train_logger,
                            step,
                            audio=wav_reconstruction,
                            sampling_rate=sampling_rate,
                            tag="Training/reconstructed",
                        )
                        log(
                            train_logger,
                            step,
                            audio=wav_prediction,
                            sampling_rate=sampling_rate,
                            tag="Training/synthesized",
                        )

                    if step % val_step == 0:
                        model.eval()
                        message = evaluate(device, model, mpd, msd, step, configs, val_logger, losses, STFT)
                        with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                            f.write(message + "\n")
                        outer_bar.write(message)

                        model.train()

                    if step % save_step == 0:
                        save_dict = {
                            "epoch": epoch,
                            "model": model.module.state_dict() if num_gpus > 1 else model.state_dict(),
                            "msd": msd.module.state_dict() if num_gpus > 1 else msd.state_dict(),
                            "optM": optM.state_dict(),
                            "optD": optD.state_dict(),
                            "sdlM": sdlM.state_dict(),
                            "sdlD": sdlD.state_dict(),
                        }
                        if use_mpd:
                            save_dict["mpd"] = mpd.module.state_dict() if num_gpus > 1 else mpd.state_dict()
                        torch.save(
                            save_dict,
                            os.path.join(
                                train_config["path"]["ckpt_path"],
                                "{}.pth.tar".format(step),
                            ),
                        )

                if step == total_step:
                    train = False
                    break
                step += 1
                if rank == 0:
                    outer_bar.update(1)

            if rank == 0:
                inner_bar.update(1)
        epoch += 1

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CPU training is not allowed."
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--path_tag", type=str, default="")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)
    path_tag = "_{}".format(args.path_tag) if args.path_tag != "" else args.path_tag
    train_config["path"]["ckpt_path"] = train_config["path"]["ckpt_path"]+"{}".format(path_tag)
    train_config["path"]["log_path"] = train_config["path"]["log_path"]+"{}".format(path_tag)
    train_config["path"]["result_path"] = train_config["path"]["result_path"]+"{}".format(path_tag)

    # Set Device
    torch.manual_seed(train_config["seed"])
    torch.cuda.manual_seed(train_config["seed"])
    num_gpus = torch.cuda.device_count()
    batch_size = int(train_config["optimizer"]["batch_size"] / num_gpus)
    
    # Log Configuration
    print("\n==================================== Training Configuration ====================================")
    print(' ---> Dataset:', args.dataset)
    print(' ---> Use MPD:', model_config["discriminator"]["use_mpd"])
    print(' ---> Automatic Mixed Precision:', args.use_amp)
    print(' ---> Number of used GPU:', num_gpus)
    print(' ---> Batch size per GPU:', batch_size)
    print(' ---> Batch size in total:', batch_size * num_gpus)
    print("=================================================================================================")
    print("Prepare training ...")

    if num_gpus > 1:
        mp.spawn(train, nprocs=num_gpus, args=(args, configs, batch_size, num_gpus))
    else:
        train(0, args, configs, batch_size, num_gpus)
