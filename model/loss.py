import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import ssim
from text import sil_phonemes_ids
# import audio as Audio


class E2ETTSLoss(nn.Module):
    """ E2ETTS Loss """

    def __init__(self, preprocess_config, model_config, train_config, device):
        super(E2ETTSLoss, self).__init__()
        self.device = device
        self.loss_config = train_config["loss"]
        self.fft_sizes = train_config["loss"]["fft_sizes"]
        self.var_start_steps = train_config["step"]["var_start_steps"]
        self.pitch_config = preprocess_config["preprocessing"]["pitch"]
        self.use_pitch_embed = model_config["variance_embedding"]["use_pitch_embed"]
        self.use_energy_embed = model_config["variance_embedding"]["use_energy_embed"]
        self.binarization_loss_enable_steps = train_config["duration"]["binarization_loss_enable_steps"]
        self.binarization_loss_warmup_steps = train_config["duration"]["binarization_loss_warmup_steps"]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.sum_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()
        self.sil_ph_ids = sil_phonemes_ids()
        # self.fft_list = [
        #     Audio.stft.TorchSTFT(
        #         preprocess_config, n_fft=self.fft_sizes[i]).to(device)
        #     for i in range(len(self.fft_sizes))
        # ]

    def discriminator_loss(self, disc_real_outputs, disc_generated_outputs):
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1-dr)**2)
            g_loss = torch.mean(dg**2)
            loss += (r_loss + g_loss)
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses

    # def spec_loss(self, x_pred, x_tgt):
    #     return sum([self.mae_loss(fft(x_pred), fft(x_tgt)) for fft in self.fft_list])

    def feature_loss(self, fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss*2

    def generator_loss(self, disc_outputs):
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean((1-dg)**2)
            gen_losses.append(l)
            loss += l

        return loss, gen_losses

    def get_duration_loss(self, dur_pred, dur_gt, txt_tokens, losses):
        """
        :param dur_pred: [B, T], float, log scale
        :param txt_tokens: [B, T]
        :return:
        """
        dur_gt.requires_grad = False
        B, T = txt_tokens.shape
        nonpadding = self.src_masks.float()
        dur_gt = dur_gt.float() * nonpadding
        is_sil = torch.zeros_like(txt_tokens).bool()
        for p_id in self.sil_ph_ids:
            is_sil = is_sil | (txt_tokens == p_id)
        is_sil = is_sil.float()  # [B, T_txt]

        # phone duration loss
        if self.loss_config["dur_loss"] == "mse":
            losses["pdur"] = F.mse_loss(dur_pred, (dur_gt + 1).log(), reduction="none")
            losses["pdur"] = (losses["pdur"] * nonpadding).sum() / nonpadding.sum()
            dur_pred = (dur_pred.exp() - 1).clamp(min=0)
        elif self.loss_config["dur_loss"] == "mog":
            return NotImplementedError
        elif self.loss_config["dur_loss"] == "crf":
            # losses["pdur"] = -self.model.dur_predictor.crf(
            #     dur_pred, dur_gt.long().clamp(min=0, max=31), mask=nonpadding > 0, reduction="mean")
            return NotImplementedError
        losses["pdur"] = losses["pdur"] * self.loss_config["lambda_ph_dur"]

        # use linear scale for sent and word duration
        if self.loss_config["lambda_word_dur"] > 0:
            word_id = (is_sil.cumsum(-1) * (1 - is_sil)).long()
            if (word_id != 0.).any():
                word_dur_p = dur_pred.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_pred)[:, 1:]
                word_dur_g = dur_gt.new_zeros([B, word_id.max() + 1]).scatter_add(1, word_id, dur_gt)[:, 1:]
                wdur_loss = F.mse_loss((word_dur_p + 1).log(), (word_dur_g + 1).log(), reduction="none")
                word_nonpadding = (word_dur_g > 0).float()
                wdur_loss = (wdur_loss * word_nonpadding).sum() / word_nonpadding.sum()
                losses["wdur"] = wdur_loss * self.loss_config["lambda_word_dur"]
        if self.loss_config["lambda_sent_dur"] > 0:
            sent_dur_p = dur_pred.sum(-1)
            sent_dur_g = dur_gt.sum(-1)
            sdur_loss = F.mse_loss((sent_dur_p + 1).log(), (sent_dur_g + 1).log(), reduction="mean")
            losses["sdur"] = sdur_loss.mean() * self.loss_config["lambda_sent_dur"]
        return losses

    def get_pitch_loss(self, pitch_predictions, pitch_targets, losses):
        for _, pitch_target in pitch_targets.items():
            if pitch_target is not None:
                pitch_target.requires_grad = False
        f0 = pitch_targets["f0"]
        uv = pitch_targets["uv"]
        nonpadding = self.mel_masks.float()
        self.add_f0_loss(pitch_predictions["pitch_pred"], f0, uv, losses, nonpadding=nonpadding)
        return losses

    def add_f0_loss(self, p_pred, f0, uv, losses, nonpadding):
        assert p_pred[..., 0].shape == f0.shape
        if self.pitch_config["use_uv"]:
            assert p_pred[..., 1].shape == uv.shape
            losses["uv"] = (F.binary_cross_entropy_with_logits(
                p_pred[:, :, 1], uv, reduction="none") * nonpadding).sum() \
                           / nonpadding.sum() * self.loss_config["lambda_uv"]
            nonpadding = nonpadding * (uv == 0).float()

        f0_pred = p_pred[:, :, 0]
        if self.loss_config["pitch_loss"] in ["l1", "l2"]:
            pitch_loss_fn = F.l1_loss if self.loss_config["pitch_loss"] == "l1" else F.mse_loss
            losses["f0"] = (pitch_loss_fn(f0_pred, f0, reduction="none") * nonpadding).sum() \
                           / nonpadding.sum() * self.loss_config["lambda_f0"]
        elif self.loss_config["pitch_loss"] == "ssim":
            return NotImplementedError

    def get_energy_loss(self, energy_predictions, energy_targets):
        energy_targets.requires_grad = False
        energy_predictions = energy_predictions.masked_select(self.mel_masks)
        energy_targets = energy_targets.masked_select(self.mel_masks)
        energy_loss = F.l1_loss(energy_predictions, energy_targets)
        return energy_loss

    def get_init_losses(self, device):
        duration_loss = {
            "pdur": torch.zeros(1).to(device),
            "wdur": torch.zeros(1).to(device),
            "sdur": torch.zeros(1).to(device),
        }
        pitch_loss = {}
        if self.pitch_config["use_uv"]:
            pitch_loss["uv"] = torch.zeros(1).to(device)
        if self.loss_config["pitch_loss"] in ["l1", "l2"]:
            pitch_loss["f0"] = torch.zeros(1).to(device)
        energy_loss = torch.zeros(1).to(device)
        return duration_loss, pitch_loss, energy_loss

    def variance_loss(self, inputs, predictions, step):
        (
            texts,
            _, # src_lens,
            _, # max_src_len,
            _, # audios,
            _, # audio_lens,
            _, # max_audio_len,
            _, # mels,
            _, # mel_lens,
            _, # max_mel_len,
            pitch_data,
            energies,
            _, # seq_starts,
            attn_priors, # attn_priors,
            spker_embeds,
        ) = inputs[3:]
        (
            p_predictions,
            e_predictions,
            log_d_predictions,
            _, # d_rounded,
            src_masks,
            mel_masks,
            _, # dec_out_cut_masks,
            src_lens,
            mel_lens,
            _, # dec_out_cut_lengths,]
            attn_outs,
        ) = predictions[2:]
        self.src_masks = ~src_masks
        self.mel_masks = ~mel_masks[:, :mel_masks.shape[1]]
        self.mel_masks_fill = ~self.mel_masks
        attn_soft, attn_hard, attn_hard_dur, attn_logprob = attn_outs

        # Variances
        duration_loss, pitch_loss, energy_loss = self.get_init_losses(self.device)
        if step >= self.var_start_steps:
            duration_loss = self.get_duration_loss(log_d_predictions, attn_hard_dur, texts, duration_loss)
            if self.use_pitch_embed:
                pitch_loss = self.get_pitch_loss(p_predictions, pitch_data, pitch_loss)
            if self.use_energy_embed:
                energy_loss = self.get_energy_loss(e_predictions, energies)

        # Alignment
        ctc_loss = self.sum_loss(attn_logprob=attn_logprob, in_lens=src_lens, out_lens=mel_lens)
        if step < self.binarization_loss_enable_steps:
            bin_loss_weight = 0.
        else:
            bin_loss_weight = min((step-self.binarization_loss_enable_steps) / self.binarization_loss_warmup_steps, 1.0) * 1.0
        bin_loss = self.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft) * bin_loss_weight

        total_loss = (
            sum(duration_loss.values()) + sum(pitch_loss.values()) + energy_loss + ctc_loss + bin_loss
        )

        return (
            total_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            ctc_loss,
            bin_loss,
        )


class ForwardSumLoss(nn.Module):
    def __init__(self, blank_logprob=-1):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=3)
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob, in_lens, out_lens):
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = F.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid] + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: query_lens[bid], :, : key_lens[bid] + 1]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss += loss

        total_loss /= attn_logprob.shape[0]
        return total_loss


class BinLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hard_attention, soft_attention):
        log_sum = torch.log(torch.clamp(soft_attention[hard_attention == 1], min=1e-12)).sum()
        return -log_sum / hard_attention.sum()
