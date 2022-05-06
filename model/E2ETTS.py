import os
import json

import torch
import torch.nn as nn
from torch.nn import Conv1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, spectral_norm
import torch.nn.functional as F

from .modules import TextEncoder, Decoder, VarianceAdaptor, Upsampler
from utils.tools import get_mask_from_lengths, get_padding


class E2ETTS(nn.Module):
    """ End-to-End TTS """

    def __init__(self, preprocess_config, model_config, train_config):
        super(E2ETTS, self).__init__()
        self.model_config = model_config

        self.hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
        self.segment_length_up = preprocess_config["preprocessing"]["audio"]["segment_length"]
        self.segment_length = self.segment_length_up // self.hop_length

        self.encoder = TextEncoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config, train_config)
        self.decoder = Decoder(model_config)
        self.proj = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            model_config["transformer"]["decoder_hidden"],
        )
        self.upsampler = Upsampler(preprocess_config, model_config, train_config)

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            self.embedder_type = preprocess_config["preprocessing"]["speaker_embedder"]
            if self.embedder_type == "none":
                with open(
                    os.path.join(
                        preprocess_config["path"]["preprocessed_path"], "speakers.json"
                    ),
                    "r",
                ) as f:
                    n_speaker = len(json.load(f))
                self.speaker_emb = nn.Embedding(
                    n_speaker,
                    model_config["transformer"]["encoder_hidden"],
                )
            else:
                self.speaker_emb = nn.Linear(
                    model_config["external_speaker_dim"],
                    model_config["transformer"]["encoder_hidden"],
                )

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        audios=None,
        audio_lens=None,
        max_audio_len=None,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        seq_starts=None,
        attn_priors=None,
        spker_embeds=None,
        step=1,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
        cut=True
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        audio_masks = (
            get_mask_from_lengths(audio_lens, max_audio_len)
            if audio_lens is not None
            else None
        )
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        texts, text_embeds = self.encoder(texts, src_masks)

        speaker_embeds = None
        if self.speaker_emb is not None:
            if self.embedder_type == "none":
                speaker_embeds = self.speaker_emb(speakers) # [B, H]
            else:
                assert spker_embeds is not None, "Speaker embedding should not be None"
                speaker_embeds = self.speaker_emb(spker_embeds) # [B, H]

        (
            enc_outs,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
            p_predictions,
            e_predictions,
            attn_outs,
        ) = self.variance_adaptor(
            texts,
            text_embeds,
            src_lens,
            max_src_len,
            src_masks,
            mels,
            mel_lens,
            max_mel_len,
            mel_masks,
            p_targets,
            e_targets,
            seq_starts,
            attn_priors,
            speaker_embeds,
            step,
            p_control,
            e_control,
            d_control,
        )

        dec_outs, mel_masks = self.decoder(enc_outs, mel_masks)

        # Random Slicing
        if cut:
            dec_out_cuts = torch.zeros(
                dec_outs.shape[0], self.segment_length, dec_outs.shape[2], dtype=dec_outs.dtype, device=dec_outs.device)
            dec_out_cut_lengths = []
            for i, (dec_out_, seq_start_) in enumerate(zip(dec_outs, seq_starts)):
                dec_out_cut_length_ = self.segment_length + (mel_lens[i] - self.segment_length).clamp(None, 0)
                dec_out_cut_lengths.append(dec_out_cut_length_)
                cut_lower, cut_upper = seq_start_, seq_start_ + dec_out_cut_length_
                dec_out_cuts[i, :dec_out_cut_length_] = dec_out_[cut_lower:cut_upper, :]
            dec_out_cuts = self.proj(dec_out_cuts)
            dec_out_cut_lengths = torch.LongTensor(dec_out_cut_lengths).to(dec_outs.device)
            dec_out_cut_masks = get_mask_from_lengths(dec_out_cut_lengths, self.segment_length)
        else:
            dec_out_cuts = self.proj(dec_outs)
            dec_out_cut_lengths = mel_lens
            dec_out_cut_masks = mel_masks

        # Upsampling
        output = self.upsampler(dec_out_cuts.transpose(1, 2))
        output_masks = get_mask_from_lengths(dec_out_cut_lengths * self.hop_length, output.shape[-1])
        output = output.masked_fill(output_masks.unsqueeze(1), 0)

        return (
            output,
            dec_out_cuts,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            dec_out_cut_masks,
            src_lens,
            mel_lens,
            dec_out_cut_lengths,
            attn_outs,
        )


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False, lrelu_slope=0.1):
        super(DiscriminatorP, self).__init__()
        self.lrelu_slope = lrelu_slope
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, lrelu_slope=0.1):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2, lrelu_slope=lrelu_slope),
            DiscriminatorP(3, lrelu_slope=lrelu_slope),
            DiscriminatorP(5, lrelu_slope=lrelu_slope),
            DiscriminatorP(7, lrelu_slope=lrelu_slope),
            DiscriminatorP(11, lrelu_slope=lrelu_slope),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False, lrelu_slope=0.1):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.lrelu_slope = lrelu_slope
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, self.lrelu_slope)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self, lrelu_slope=0.1):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True, lrelu_slope=lrelu_slope),
            DiscriminatorS(lrelu_slope=lrelu_slope),
            DiscriminatorS(lrelu_slope=lrelu_slope),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
