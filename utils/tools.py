import os
import json
import yaml

import torch
import torch.nn.functional as F
from torch.cuda import amp
import numpy as np
import matplotlib
matplotlib.use("Agg")
from scipy.io import wavfile
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

import audio as Audio
from utils.pitch_tools import denorm_f0


def get_configs_of(dataset):
    config_dir = os.path.join("./config", dataset)
    preprocess_config = yaml.load(open(
        os.path.join(config_dir, "preprocess.yaml"), "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(
        os.path.join(config_dir, "model.yaml"), "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(
        os.path.join(config_dir, "train.yaml"), "r"), Loader=yaml.FullLoader)
    return preprocess_config, model_config, train_config


def to_device(data, device):
    if len(data) == 18:
        (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            audios,
            audio_lens,
            max_audio_len,
            mels,
            mel_lens,
            max_mel_len,
            f0s,
            uvs,
            energies,
            seq_starts,
            attn_priors,
            spker_embeds,
        ) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        audios = torch.from_numpy(audios).float().to(device)
        audio_lens = torch.from_numpy(audio_lens).to(device)
        if mels is not None:
            mels = torch.from_numpy(mels).float().to(device)
        if mel_lens is not None:
            mel_lens = torch.from_numpy(mel_lens).to(device)
        f0s = torch.from_numpy(f0s).float().to(device)
        uvs = torch.from_numpy(uvs).float().to(device)
        energies = torch.from_numpy(energies).to(device)
        seq_starts = torch.from_numpy(seq_starts).long().to(device)
        if attn_priors is not None:
            attn_priors = torch.from_numpy(attn_priors).float().to(device)
        if spker_embeds is not None:
            spker_embeds = torch.from_numpy(spker_embeds).float().to(device)

        pitch_data = {
            "f0": f0s,
            "uv": uvs,
        }

        return [
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            audios,
            audio_lens,
            max_audio_len,
            mels,
            mel_lens,
            max_mel_len,
            pitch_data,
            energies,
            seq_starts,
            attn_priors,
            spker_embeds,
        ]

    if len(data) == 7:
        (ids, raw_texts, speakers, texts, src_lens, max_src_len, spker_embeds) = data

        speakers = torch.from_numpy(speakers).long().to(device)
        texts = torch.from_numpy(texts).long().to(device)
        src_lens = torch.from_numpy(src_lens).to(device)
        if spker_embeds is not None:
            spker_embeds = torch.from_numpy(spker_embeds).float().to(device)

        return (ids, raw_texts, speakers, texts, src_lens, max_src_len, spker_embeds)


def log(
    logger, step=1, losses=None, lr=None, fig=None, figs=None, img=None, audio=None, sampling_rate=22050, tag=""
):
    if losses is not None:
        logger.add_scalar("Loss/total_loss", losses[0], step)
        logger.add_scalar("Loss/loss_gen_all", losses[1], step)
        logger.add_scalar("Loss/loss_var_all", losses[2], step)
        logger.add_scalar("Loss/loss_disc_s", losses[3], step)
        logger.add_scalar("Loss/loss_disc_f", losses[4], step)
        logger.add_scalar("Loss/loss_gen_s", losses[5], step)
        logger.add_scalar("Loss/loss_gen_f", losses[6], step)
        logger.add_scalar("Loss/loss_fm_s", losses[7], step)
        logger.add_scalar("Loss/loss_fm_f", losses[8], step)
        logger.add_scalar("Loss/loss_mel", losses[9], step)
        for k, v in losses[10].items():
            logger.add_scalar("Loss/loss_{}".format(k), v, step)
        logger.add_scalar("Loss/loss_energy", losses[11], step)
        for k, v in losses[12].items():
            logger.add_scalar("Loss/loss_{}".format(k), v, step)
        logger.add_scalar("Loss/loss_ctc", losses[13], step)
        logger.add_scalar("Loss/loss_bin", losses[14], step)

    if lr is not None:
        logger.add_scalar("Training/learning_rate", lr, step)

    if fig is not None:
        logger.add_figure(tag, fig, step)

    if figs is not None:
        for k, v in figs.items():
            logger.add_figure("{}/{}".format(tag, k), v, step)

    if img is not None:
        logger.add_image(tag, img, step, dataformats='HWC')

    if audio is not None:
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            step,
            sample_rate=sampling_rate,
        )


def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(lengths.device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask


def expand(values, durations):
    out = list()
    for value, d in zip(values, durations):
        out += [value] * max(0, int(d))
    return np.array(out)


def synth_one_sample(targets, predictions, model_config, preprocess_config, STFT):

    pitch_config = preprocess_config["preprocessing"]["pitch"]
    pitch_type = pitch_config["pitch_type"]
    use_pitch_embed = model_config["variance_embedding"]["use_pitch_embed"]
    use_energy_embed = model_config["variance_embedding"]["use_energy_embed"]
    basename = targets[0][0]
    src_len = predictions[9][0].item()
    mel_len = predictions[10][0].item()

    attn_prior, attn_soft, attn_hard, attn_hard_dur, attn_logprob = targets[15], *predictions[12]
    attn_prior = attn_prior[0, :src_len, :mel_len].squeeze().detach().cpu().numpy() # text_len x mel_len
    attn_soft = attn_soft[0, 0, :mel_len, :src_len].detach().cpu().transpose(0, 1).numpy() # text_len x mel_len
    attn_hard = attn_hard[0, 0, :mel_len, :src_len].detach().cpu().transpose(0, 1).numpy() # text_len x mel_len
    fig_attn = plot_alignment(
        [
            attn_soft,
            attn_hard,
            attn_prior,
        ],
        ["Soft Attention", "Hard Attention", "Prior"]
    )

    figs = {}
    if use_pitch_embed:
        pitch_prediction, pitch_target = predictions[2], targets[12]
        f0 = pitch_target["f0"]
        f0 = denorm_f0(f0, pitch_target["uv"], pitch_config)
        uv_pred = pitch_prediction["pitch_pred"][:, :, 1] > 0
        pitch_pred = denorm_f0(pitch_prediction["pitch_pred"][:, :, 0], uv_pred, pitch_config)
        figs["f0"] = f0_to_figure(f0[0, :mel_len], None, pitch_pred[0, :mel_len])
    if use_energy_embed:
        energy_prediction = predictions[3][0, :mel_len].detach().cpu().numpy()
        energy_target = targets[13][0, :mel_len].detach().cpu().numpy()
        figs["energy"] = energy_to_figure(energy_target, energy_prediction)

    hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
    max_wav_value = preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    wav_target_len = targets[7][0].item()
    wav_target = targets[6][0, :wav_target_len]
    wav_prediction_len = predictions[11][0].item() * hop_length
    wav_prediction = predictions[0][0, 0, :wav_prediction_len]

    # mel_target = Audio.tools.get_mel_from_wav(
    #     wav_target.detach().cpu().numpy(), STFT)[0]
    # mel_prediction = Audio.tools.get_mel_from_wav(
    #     wav_prediction.detach().cpu().numpy(), STFT)[0]
    mel_target = STFT(wav_target.unsqueeze(0)).squeeze(0)
    mel_prediction = STFT(wav_prediction.unsqueeze(0)).squeeze(0)

    figs["mel"] = plot_mel(
        [
            mel_prediction.detach().cpu().numpy(),
            mel_target.detach().cpu().numpy(),
        ],
        ["Sampled Spectrogram", "Ground-Truth Spectrogram"],
    )

    wav_target = (wav_target * max_wav_value).detach().cpu().numpy().astype('int16')
    wav_prediction = (wav_prediction * max_wav_value).detach().cpu().numpy().astype('int16')

    return figs, fig_attn, wav_target, wav_prediction, basename


def synth_samples(targets, predictions, model_config, preprocess_config, path, args, STFT):

    hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
    max_wav_value = preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    multi_speaker = model_config["multi_speaker"]
    basenames = targets[0]
    wav_predictions = []
    for i in range(len(predictions[0])):
        basename = basenames[i]

        wav_prediction_len = predictions[11][i].item() * hop_length
        wav_prediction = predictions[0][i, 0, :wav_prediction_len]
        wav_prediction = wav_prediction * max_wav_value
        wav_predictions.append(wav_prediction.detach().cpu().numpy().astype('int16'))

        mel_prediction = STFT(wav_prediction.unsqueeze(0)).squeeze(0)

        fig_save_dir = os.path.join(
            path, str(args.restore_step), "{}_{}.png".format(basename, args.speaker_id)\
                if multi_speaker and args.mode == "single" else "{}.png".format(basename))
        fig = plot_mel(
            [
                mel_prediction.cpu().numpy(),
            ],
            ["Synthetized Spectrogram"],
        )
        plt.savefig(fig_save_dir)
        plt.close()

    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    for wav, basename in zip(wav_predictions, basenames):
        wavfile.write(os.path.join(
            path, str(args.restore_step), "{}_{}.wav".format(basename, args.speaker_id)\
                if multi_speaker and args.mode == "single" else "{}.wav".format(basename)),
            sampling_rate, wav)


def plot_mel(data, titles=None):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [None for i in range(len(data))]
    plt.tight_layout()

    for i in range(len(data)):
        mel = data[i]
        if isinstance(mel, torch.Tensor):
            mel = mel.detach().cpu().numpy()
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")

    return fig


def spec_to_figure(spec, vmin=None, vmax=None, filename=None):
    if isinstance(spec, torch.Tensor):
        spec = spec.detach().cpu().numpy()
    fig = plt.figure(figsize=(12, 6))
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    if filename is not None:
        plt.savefig(filename)
    return fig


def spec_f0_to_figure(spec, f0s, figsize=None, line_colors=['w', 'r', 'y', 'cyan', 'm', 'b', 'lime'], filename=None):
    max_y = spec.shape[1]
    if isinstance(spec, torch.Tensor):
        spec = spec.detach().cpu().numpy()
        f0s = {k: f0.detach().cpu().numpy() for k, f0 in f0s.items()}
    f0s = {k: f0 / 10 for k, f0 in f0s.items()}
    fig = plt.figure(figsize=(12, 6) if figsize is None else figsize)
    plt.pcolor(spec.T)
    for i, (k, f0) in enumerate(f0s.items()):
        plt.plot(f0.clip(0, max_y), label=k, c=line_colors[i], linewidth=1, alpha=0.8)
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    return fig


def f0_to_figure(f0_gt, f0_cwt=None, f0_pred=None):
    fig = plt.figure()
    if isinstance(f0_gt, torch.Tensor):
        f0_gt = f0_gt.detach().cpu().numpy()
    plt.plot(f0_gt, color="r", label="gt")
    if f0_cwt is not None:
        if isinstance(f0_cwt, torch.Tensor):
            f0_cwt = f0_cwt.detach().cpu().numpy()
        plt.plot(f0_cwt, color="b", label="cwt")
    if f0_pred is not None:
        if isinstance(f0_pred, torch.Tensor):
            f0_pred = f0_pred.detach().cpu().numpy()
        plt.plot(f0_pred, color="green", label="pred")
    plt.legend()
    return fig


def energy_to_figure(energy_gt, energy_pred=None):
    fig = plt.figure()
    if isinstance(energy_gt, torch.Tensor):
        energy_gt = energy_gt.detach().cpu().numpy()
    plt.plot(energy_gt, color="r", label="gt")
    if energy_pred is not None:
        if isinstance(energy_pred, torch.Tensor):
            energy_pred = energy_pred.detach().cpu().numpy()
        plt.plot(energy_pred, color="green", label="pred")
    plt.legend()
    return fig


def save_mel_and_audio(spectrogram, audio, sampling_rate, out_dir, basename, tag=None):
    # Save mel
    plt.imshow(spectrogram, origin='lower')
    plt.ylim(0, spectrogram.shape[0])
    plt.tick_params(labelsize='x-small',
                        left=False, labelleft=False)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '{}_{}.png'.format(basename, tag)\
            if tag is not None else '{}.png'.format(basename)), dpi=200)
    plt.close()

    # Save audio
    wavfile.write(
        os.path.join(out_dir, "{}_{}.wav".format(basename, tag)\
            if tag is not None else '{}.wav'.format(basename)),
        sampling_rate,
        audio.astype(np.int16),
    )


def plot_alignment(data, titles=None, save_dir=None):
    fig, axes = plt.subplots(len(data), 1, figsize=[6,4],dpi=300)
    plt.subplots_adjust(top = 0.9, bottom = 0.1, right = 0.95, left = 0.05)
    if titles is None:
        titles = [None for i in range(len(data))]

    for i in range(len(data)):
        im = data[i]
        axes[i].imshow(im, origin='lower')
        # axes[i].set_xlabel('Audio')
        # axes[i].set_ylabel('Text')
        axes[i].set_ylim(0, im.shape[0])
        axes[i].set_xlim(0, im.shape[1])
        axes[i].set_title(titles[i], fontsize='medium')
        axes[i].tick_params(labelsize='x-small')
        axes[i].set_anchor('W')
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    if save_dir is not None:
        plt.savefig(save_dir)
    plt.close()
    return data


def plot_embedding(out_dir, embedding, embedding_speaker_id, gender_dict, filename='embedding.png'):
    colors = 'r','b'
    labels = 'Female','Male'

    data_x = embedding
    data_y = np.array([gender_dict[spk_id] == 'M' for spk_id in embedding_speaker_id], dtype=np.int)
    tsne_model = TSNE(n_components=2, random_state=0, init='random')
    tsne_all_data = tsne_model.fit_transform(data_x)
    tsne_all_y_data = data_y

    plt.figure(figsize=(10,10))
    for i, (c, label) in enumerate(zip(colors, labels)):
        plt.scatter(tsne_all_data[tsne_all_y_data==i,0], tsne_all_data[tsne_all_y_data==i,1], c=c, label=label, alpha=0.5)

    plt.grid(True)
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def pad_1D(inputs, PAD=0):
    def pad_data(x, length, PAD):
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=PAD
        )
        return x_padded

    max_len = max((len(x) for x in inputs))
    padded = np.stack([pad_data(x, max_len, PAD) for x in inputs])

    return padded


def pad_2D(inputs, maxlen=None):
    def pad(x, max_len):
        PAD = 0
        if np.shape(x)[0] > max_len:
            raise ValueError("not max_len")

        s = np.shape(x)[1]
        x_padded = np.pad(
            x, (0, max_len - np.shape(x)[0]), mode="constant", constant_values=PAD
        )
        return x_padded[:, :s]

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[0] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])

    return output


def pad_3D(inputs, B, T, L):
    inputs_padded = np.zeros((B, T, L), dtype=np.float32)
    for i, input_ in enumerate(inputs):
        inputs_padded[i, :np.shape(input_)[0], :np.shape(input_)[1]] = input_
    return inputs_padded


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


def dur_to_mel2ph(dur, dur_padding=None, alpha=1.0):
    """
    Example (no batch dim version):
        1. dur = [2,2,3]
        2. token_idx = [[1],[2],[3]], dur_cumsum = [2,4,7], dur_cumsum_prev = [0,2,4]
        3. token_mask = [[1,1,0,0,0,0,0],
                            [0,0,1,1,0,0,0],
                            [0,0,0,0,1,1,1]]
        4. token_idx * token_mask = [[1,1,0,0,0,0,0],
                                        [0,0,2,2,0,0,0],
                                        [0,0,0,0,3,3,3]]
        5. (token_idx * token_mask).sum(0) = [1,1,2,2,3,3,3]

    :param dur: Batch of durations of each frame (B, T_txt)
    :param dur_padding: Batch of padding of each frame (B, T_txt)
    :param alpha: duration rescale coefficient
    :return:
        mel2ph (B, T_speech)
    """
    assert alpha > 0
    dur = torch.round(dur.float() * alpha).long()
    if dur_padding is not None:
        dur = dur * (1 - dur_padding.long())
    token_idx = torch.arange(1, dur.shape[1] + 1)[None, :, None].to(dur.device)
    dur_cumsum = torch.cumsum(dur, 1)
    dur_cumsum_prev = F.pad(dur_cumsum, [1, -1], mode="constant", value=0)

    pos_idx = torch.arange(dur.sum(-1).max())[None, None].to(dur.device)
    token_mask = (pos_idx >= dur_cumsum_prev[:, :, None]) & (pos_idx < dur_cumsum[:, :, None])
    mel2ph = (token_idx * token_mask.long()).sum(1)
    return mel2ph


def mel2ph_to_dur(mel2ph, T_txt, max_dur=None):
    B, _ = mel2ph.shape
    dur = mel2ph.new_zeros(B, T_txt + 1).scatter_add(1, mel2ph, torch.ones_like(mel2ph))
    dur = dur[:, 1:]
    if max_dur is not None:
        dur = dur.clamp(max=max_dur)
    return dur


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn"t know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
                   torch.cumsum(mask, dim=1).type_as(mask) * mask
           ).long() + padding_idx


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
        (_, channel, _, _) = img1.size()
        global window
        if window is None:
            window = create_window(window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
        return _ssim(img1, img2, window, window_size, channel, size_average)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)
