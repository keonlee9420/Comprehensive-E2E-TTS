import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model
from utils.tools import to_device, log, synth_one_sample
from model import E2ETTSLoss
from dataset import Dataset


def evaluate(device, model, mpd, msd, step, configs, logger=None, losses=None, STFT=None):
    preprocess_config, model_config, train_config = configs
    use_mpd = model_config["discriminator"]["use_mpd"]

    # Get dataset
    dataset = Dataset(
        "val.txt", preprocess_config, model_config, train_config, sort=False, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    # Get loss function
    mel_fmax_loss = preprocess_config["preprocessing"]["mel"]["mel_fmax_loss"]
    Loss = E2ETTSLoss(preprocess_config, model_config, train_config, device).to(device)

    # Evaluation
    loss_sums = [{k:0 for k in loss.keys()} if isinstance(loss, dict) else 0 for loss in losses]
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device)
            with torch.no_grad():
                ##########################
                # Evaluate Discriminator #
                ##########################
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

                #######################
                # Evaluate Generator #
                #######################
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

                losses = (
                    loss_disc_all + loss_gen_all + loss_var_all, loss_gen_all, loss_var_all, loss_disc_s, loss_disc_f, loss_gen_s, loss_gen_f, loss_fm_s, loss_fm_f, loss_mel, loss_pitch, loss_energy, loss_duration, loss_ctc, loss_bin,
                )

                for i in range(len(losses)):
                    if isinstance(losses[i], dict):
                        for k in loss_sums[i].keys():
                            loss_sums[i][k] += losses[i][k].item() * len(batch[0])
                    else:
                        loss_sums[i] += losses[i].item() * len(batch[0])

    loss_means = []
    loss_means_ = []
    for loss_sum in loss_sums:
        if isinstance(loss_sum, dict):
            loss_mean = {k:v / len(dataset) for k, v in loss_sum.items()}
            loss_means.append(loss_mean)
            loss_means_.append(sum(loss_mean.values()))
        else:
            loss_means.append(loss_sum / len(dataset))
            loss_means_.append(loss_sum / len(dataset))

    message = "Validation Step {}, Total Loss: {:.4f}, loss_gen_all: {:.4f}, loss_var_all: {:.4f}, loss_disc_s: {:.4f}, loss_disc_f: {:.4f}, loss_gen_s: {:.4f}, loss_gen_f: {:.4f}, loss_fm_s: {:.4f}, loss_fm_f: {:.4f}, loss_mel: {:.4f}, loss_pitch: {:.4f}, loss_energy: {:.4f}, loss_duration: {:.4f}, loss_ctc: {:.4f}, loss_bin: {:.4f}".format(
        *([step] + [l for l in loss_means_])
    )

    if logger is not None:
        figs, fig_attn, wav_reconstruction, wav_prediction, tag = synth_one_sample(
            batch,
            output,
            model_config,
            preprocess_config,
            STFT,
        )

        log(logger, step, losses=loss_means)
        log(
            logger,
            step,
            img=fig_attn,
            tag="Validation/attn",
        )
        log(
            logger,
            step,
            figs=figs,
            tag="Validation",
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            step,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/reconstructed",
        )
        log(
            logger,
            step,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/synthesized",
        )

    return message
