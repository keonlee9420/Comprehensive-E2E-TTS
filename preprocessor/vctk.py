import os
import random
import json

import re
import librosa
import torch
import numpy as np
from scipy.stats import betabinom
from sklearn.preprocessing import StandardScaler
from scipy.io.wavfile import read, write
from tqdm import tqdm
from pathlib import Path

from g2p_en import G2p
import audio as Audio
from text import text_to_sequence, grapheme_to_phoneme
from model import PreDefinedEmbedder
from utils.pitch_tools import get_pitch
from utils.tools import plot_embedding, spec_to_figure


class Preprocessor:
    def __init__(self, preprocess_config, model_config, train_config):
        random.seed(train_config['seed'])
        self.preprocess_config = preprocess_config
        self.dataset = preprocess_config["dataset"]
        self.in_dir = preprocess_config["path"]["corpus_path"]
        self.wav_tag = preprocess_config["path"]["wav_tag"]
        self.wav_dir = preprocess_config["path"]["wav_dir"]
        self.txt_dir = preprocess_config["path"]["txt_dir"]
        self.out_dir = preprocess_config["path"]["preprocessed_path"]
        self.val_size = preprocess_config["preprocessing"]["val_size"]
        self.n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        self.trim_top_db = preprocess_config["preprocessing"]["audio"]["trim_top_db"]
        self.filter_length = preprocess_config["preprocessing"]["stft"]["filter_length"]
        self.hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
        self.max_wav_value = preprocess_config["preprocessing"]["audio"]["max_wav_value"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.segment_length_up = preprocess_config["preprocessing"]["audio"]["segment_length"]
        self.segment_length = self.segment_length_up // self.hop_length
        self.beta_binomial_scaling_factor = preprocess_config["preprocessing"]["duration"]["beta_binomial_scaling_factor"]
        self.energy_normalization = preprocess_config["preprocessing"]["energy"]["normalization"]

        self.g2p = G2p()
        # self.STFT = Audio.stft.TacotronSTFT(
        #     preprocess_config["preprocessing"]["stft"]["filter_length"],
        #     preprocess_config["preprocessing"]["stft"]["hop_length"],
        #     preprocess_config["preprocessing"]["stft"]["win_length"],
        #     preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        #     preprocess_config["preprocessing"]["audio"]["sampling_rate"],
        #     preprocess_config["preprocessing"]["mel"]["mel_fmin"],
        #     preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        # )
        self.TorchSTFT = Audio.stft.TorchSTFT(preprocess_config)
        self.val_prior = self.val_prior_names(os.path.join(self.out_dir, "val.txt"))
        self.speaker_emb = None
        if preprocess_config["preprocessing"]["speaker_embedder"] != "none":
            self.speaker_emb = PreDefinedEmbedder(preprocess_config)

    def val_prior_names(self, val_prior_path):
        val_prior_names = set()
        if os.path.isfile(val_prior_path):
            print("Load pre-defined validation set...")
            with open(val_prior_path, "r", encoding="utf-8") as f:
                for m in f.readlines():
                    val_prior_names.add(m.split("|")[0])
            return list(val_prior_names)
        else:
            return None

    def _init_spker_embeds(self, spkers):
        spker_embeds = dict()
        for spker in spkers:
            spker_embeds[spker] = list()
        return spker_embeds

    def build_from_path(self):
        txt_dir = os.path.join(self.in_dir, self.txt_dir)
        wav_dir = os.path.join(self.in_dir, self.wav_dir)
        embedding_dir = os.path.join(self.out_dir, "spker_embed")
        os.makedirs((os.path.join(self.out_dir, "text")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "wav")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "mel")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "f0")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "energy")), exist_ok=True)
        os.makedirs((os.path.join(self.out_dir, "attn_prior")), exist_ok=True)
        os.makedirs(embedding_dir, exist_ok=True)

        print("Processing Data ...")
        # out = list()
        train = list()
        val = list()
        max_seq_len = 0
        mel_min = np.ones(self.n_mel_channels) * float('inf')
        mel_max = np.ones(self.n_mel_channels) * -float('inf')
        f0s = []
        energy_scaler = StandardScaler()
        n_wavs = 0

        def partial_fit(scaler, value):
            if len(value) > 0:
                scaler.partial_fit(value.reshape((-1, 1)))

        def compute_f0_stats(f0s):
            if len(f0s) > 0:
                f0s = np.concatenate(f0s, 0)
                f0s = f0s[f0s != 0]
                f0_mean = np.mean(f0s).item()
                f0_std = np.std(f0s).item()
            return (f0_mean, f0_std)

        def compute_energy_stats(energy_scaler, energy_dir="energy"):
            if self.energy_normalization:
                energy_mean = energy_scaler.mean_[0]
                energy_std = energy_scaler.scale_[0]
            else:
                # A numerical trick to avoid normalization...
                energy_mean = 0
                energy_std = 1

            energy_min, energy_max = self.normalize(
                os.path.join(self.out_dir, energy_dir), energy_mean, energy_std
            )
            return (energy_min, energy_max, energy_mean, energy_std)

        sub_dirs = os.listdir(txt_dir)
        if self.speaker_emb is not None:
            spker_embeds = self._init_spker_embeds(sub_dirs)

        skip_speakers = set()
        preprocessed_dirs = os.listdir(embedding_dir)
        for preprocessed_dir in preprocessed_dirs:
            skip_speakers.add(preprocessed_dir.split("-")[0])

        speakers = {}
        for spker_id, speaker in enumerate(tqdm(sub_dirs)):
            speakers[speaker] = spker_id
            for i, txt_name in enumerate(tqdm(os.listdir(os.path.join(txt_dir, speaker)))):

                basename = txt_name.split(".")[0]
                with open(os.path.join(txt_dir, speaker, txt_name), "r") as f:
                    text = f.readline().strip("\n")

                wav_path = os.path.join(os.path.join(wav_dir, speaker), "{}_{}.flac".format(basename, self.wav_tag))
                if not os.path.isfile(wav_path):
                    print("[Error] No flac file:{}".format(wav_path))
                    continue
                ret = self.process_utterance(text, wav_path, speaker, basename, skip_speakers)
                if ret is None:
                    continue
                else:
                    info, f0, energy, n_mel, n_wav, m_min, m_max, spker_embed = ret
                # out.append(info)

                if self.val_prior is not None:
                    if basename not in self.val_prior:
                        train.append(info)
                    else:
                        val.append(info)
                else:
                    if i == 0 or i == 1: 
                        val.append(info)
                    else:
                        train.append(info)

                if n_mel > max_seq_len:
                    max_seq_len = n_mel

                if len(f0) > 0:
                    f0s.append(f0)
                partial_fit(energy_scaler, energy)

                mel_min = np.minimum(mel_min, m_min)
                mel_max = np.maximum(mel_max, m_max)

                if self.speaker_emb is not None:
                    spker_embeds[speaker].append(spker_embed)

                n_wavs += n_wav

            # Calculate and save mean speaker embedding of this speaker
            if self.speaker_emb is not None and speaker not in skip_speakers:
                spker_embed_filename = '{}-spker_embed.npy'.format(speaker)
                np.save(os.path.join(self.out_dir, 'spker_embed', spker_embed_filename), \
                    np.mean(spker_embeds[speaker], axis=0), allow_pickle=False)

        print("Computing statistic quantities ...")
        f0s_stats = compute_f0_stats(f0s)

        # Perform normalization if needed
        energy_stats = compute_energy_stats(
            energy_scaler,
            energy_dir="energy",
        )

        # Save files
        with open(os.path.join(self.out_dir, "speakers.json"), "w") as f:
            f.write(json.dumps(speakers))

        with open(os.path.join(self.out_dir, "stats.json"), "w") as f:
            stats = {
                "f0": [float(var) for var in f0s_stats],
                "energy": [float(var) for var in energy_stats],
                "spec_min": mel_min.tolist(),
                "spec_max": mel_max.tolist(),
                "max_seq_len": max_seq_len,
            }
            f.write(json.dumps(stats))

        print(
            "Total time: {} hours".format(
                n_wavs / self.sampling_rate / 3600
            )
        )

        print("plot speaker embedding...")
        plot_embedding(
            self.out_dir, *self.load_embedding(embedding_dir),
            self.divide_speaker_by_gender(self.in_dir), filename="spker_embed_tsne.png"
        )

        # random.shuffle(out)
        # out = [r for r in out if r is not None]
        if self.val_prior is None:
            random.shuffle(train)
        train = [r for r in train if r is not None]
        val = [r for r in val if r is not None]

        # Write metadata
        with open(os.path.join(self.out_dir, "train.txt"), "w", encoding="utf-8") as f:
            for m in train:
                f.write(m + "\n")
        with open(os.path.join(self.out_dir, "val.txt"), "w", encoding="utf-8") as f:
            for m in val:
                f.write(m + "\n")

        return train, val #out

    def match_librosa_to_scipy(self, l_wave, nbits=16):
        """
        https://stackoverflow.com/questions/50062358/difference-between-load-of-librosa-and-read-of-scipy-io-wavfile
        """
        l_wave *= 2 ** (nbits - 1)
        return l_wave

    def load_wav(self, full_path):
        data, sampling_rate = librosa.load(full_path, self.sampling_rate)
        _, index = librosa.effects.trim(data, top_db=self.trim_top_db, frame_length=self.filter_length, hop_length=self.hop_length)
        data = data[index[0]:index[1]]
        data = self.match_librosa_to_scipy(data)
        duration = (index[1] - index[0]) / self.hop_length
        return data, sampling_rate, int(duration)

    def beta_binomial_prior_distribution(self, phoneme_count, mel_count, scaling_factor=1.0):
        P, M = phoneme_count, mel_count
        x = np.arange(0, P)
        mel_text_probs = []
        for i in range(1, M+1):
            a, b = scaling_factor*i, scaling_factor*(M+1-i)
            rv = betabinom(P, a, b)
            mel_i_prob = rv.pmf(x)
            mel_text_probs.append(mel_i_prob)
        return np.array(mel_text_probs)

    def process_utterance(self, raw_text, wav_path, speaker, basename, skip_speakers):
        text_filename = "{}-text-{}.npy".format(speaker, basename)
        wav_filename = "{}-wav-{}.wav".format(speaker, basename)
        mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        f0_filename = "{}-f0-{}.npy".format(speaker, basename)
        energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        attn_prior_filename = "{}-attn_prior-{}.npy".format(speaker, basename)

        # Preprocess text
        phone = grapheme_to_phoneme(raw_text, self.g2p)
        phones = "{" + "}{".join(phone) + "}"
        phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
        phones = phones.replace("}{", " ")
        if not os.path.isfile(os.path.join(self.out_dir, "text", text_filename)):
            text = np.array(text_to_sequence(phones, self.cleaners))
            np.save(
                os.path.join(self.out_dir, "text", text_filename),
                text,
            )
        else:
            text = np.load(os.path.join(self.out_dir, "text", text_filename))

        # Load and process wav files
        if not os.path.isfile(os.path.join(self.out_dir, "wav", wav_filename)):

            # _, wav = self.load_audio(wav_path)
            wav, sampling_rate, duration = self.load_wav(wav_path)

            # wav = wav / max(abs(wav)) * self.max_wav_value
            wav = wav / self.max_wav_value
            wav = librosa.util.normalize(wav) * 0.95
            write(
                os.path.join(self.out_dir, "wav", wav_filename),
                self.sampling_rate,
                wav.astype(np.float32),
            )
        else:
            wav, _ = librosa.load(os.path.join(self.out_dir, "wav", wav_filename), self.sampling_rate)

        # Load mel-spectrogram
        energy = None
        if not os.path.isfile(os.path.join(self.out_dir, "mel", mel_filename)):

            # Compute mel-scale spectrogram
            # mel_spectrogram, _ = Audio.tools.get_mel_from_wav(wav, self.STFT)
            mel_spectrogram, energy = [x.squeeze(0).numpy() for x in self.TorchSTFT(torch.from_numpy(wav).float().unsqueeze(0), return_energy=True)]
            mel_spectrogram = mel_spectrogram[:, : duration]
            energy = energy[: duration]

            np.save(
                os.path.join(self.out_dir, "mel", mel_filename),
                mel_spectrogram.T,
            )
            np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)
        else:
            mel_spectrogram = np.load(os.path.join(self.out_dir, "mel", mel_filename)).T
            energy = np.load(os.path.join(self.out_dir, "energy", energy_filename))
        # spec_to_figure(mel_spectrogram.T, filename="./spec_{}.png".format(basename))

        # Load f0
        if not os.path.isfile(os.path.join(self.out_dir, "f0", f0_filename)):
            f0, _ = self.get_pitch(wav, mel_spectrogram.T)
            f0 = f0[: duration]

            np.save(os.path.join(self.out_dir, "f0", f0_filename), f0)
        else:
            f0 = np.load(os.path.join(self.out_dir, "f0", f0_filename))

        # Calculate attention prior
        if not os.path.isfile(os.path.join(self.out_dir, "attn_prior", attn_prior_filename)):
            attn_prior = self.beta_binomial_prior_distribution(
                mel_spectrogram.shape[1],
                len(text),
                self.beta_binomial_scaling_factor,
            )
            np.save(os.path.join(self.out_dir, "attn_prior", attn_prior_filename), attn_prior)
        else:
            attn_prior = np.load(os.path.join(self.out_dir, "attn_prior", attn_prior_filename))

        # Speaker embedding
        spker_embed=None
        if self.speaker_emb is not None and speaker not in skip_speakers:
            if wav is not None:
                spker_embed = self.speaker_emb(wav)
            else:
                wav_raw, wav = self.load_audio(wav_path)
                spker_embed = self.speaker_emb(wav)

        return (
            "|".join([basename, speaker, phones, raw_text]),
            f0,
            self.remove_outlier(energy),
            mel_spectrogram.shape[1],
            len(wav),
            np.min(mel_spectrogram, axis=1),
            np.max(mel_spectrogram, axis=1),
            spker_embed,
        )

    def get_pitch(self, wav, mel):
        f0, pitch_coarse = get_pitch(wav, mel, self.preprocess_config)
        return f0, pitch_coarse

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def normalize(self, in_dir, mean, std):
        max_value = np.finfo(np.float64).min
        min_value = np.finfo(np.float64).max
        for filename in os.listdir(in_dir):
            filename = os.path.join(in_dir, filename)
            values = (np.load(filename) - mean) / std
            np.save(filename, values)

            max_value = max(max_value, max(values))
            min_value = min(min_value, min(values))

        return min_value, max_value

    def divide_speaker_by_gender(self, in_dir, speaker_path="speaker-info.txt"):
        speakers = dict()
        with open(os.path.join(in_dir, speaker_path), encoding='utf-8') as f:
            for line in tqdm(f):
                if "ID" in line: continue
                parts = [p.strip() for p in re.sub(' +', ' ',(line.strip())).split(' ')]
                spk_id, gender = parts[0], parts[2]
                speakers[str(spk_id)] = gender
        return speakers

    def load_embedding(self, embedding_dir):
        embedding_path_list = [_ for _ in Path(embedding_dir).rglob('*.npy')]
        embedding = None
        embedding_speaker_id = list()
        # Gather data
        for path in tqdm(embedding_path_list):
            embedding = np.concatenate((embedding, np.load(path)), axis=0) \
                                            if embedding is not None else np.load(path)
            embedding_speaker_id.append(str(str(path).split('/')[-1].split('-')[0]))
        return embedding, embedding_speaker_id
