import json
import math
import os

import librosa
from scipy.io.wavfile import read
import numpy as np
from torch.utils.data import Dataset

from text import text_to_sequence
from utils.pitch_tools import norm_interp_f0
from utils.tools import pad_1D, pad_2D, pad_3D


class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, model_config, train_config, sort=False, drop_last=False
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocess_config = preprocess_config
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        self.max_wav_value = preprocess_config["preprocessing"]["audio"]["max_wav_value"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.batch_size = train_config["optimizer"]["batch_size"]
        self.hop_length = preprocess_config["preprocessing"]["stft"]["hop_length"]
        self.segment_length_up = preprocess_config["preprocessing"]["audio"]["segment_length"]
        self.segment_length = self.segment_length_up // self.hop_length
        self.load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'

        self.basename, self.speaker, self.raw_text = self.process_meta(
            filename
        )
        with open(os.path.join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last

    def __len__(self):
        return len(self.raw_text)

    # def load_audio_to_torch(self, audio_path):
    #     audio, sample_rate = librosa.load(audio_path)
    #     return audio.squeeze(), sample_rate

    def load_wav(self, full_path):
        sampling_rate, data = read(full_path)
        return data, sampling_rate

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone_path = os.path.join(
            self.preprocessed_path,
            "text",
            "{}-text-{}.npy".format(speaker, basename),
        )
        phone = np.load(phone_path)
        audio_path = os.path.join(
            self.preprocessed_path,
            "wav",
            "{}-wav-{}.wav".format(speaker, basename)
        )
        audio, sampling_rate = self.load_wav(audio_path)
        assert sampling_rate == self.sampling_rate
        # audio, sampling_rate = self.load_audio_to_torch(audio_path)
        # audio = audio / self.max_wav_value
        # audio = librosa.util.normalize(audio) * 0.95
        # if sampling_rate != self.sampling_rate:
        #     raise ValueError("{} SR doesn't match target {} SR".format(
        #         sampling_rate, self.sampling_rate))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-mel-{}.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        f0_path = os.path.join(
            self.preprocessed_path,
            "f0",
            "{}-f0-{}.npy".format(speaker, basename),
        )
        f0 = np.load(f0_path)
        f0, uv = norm_interp_f0(f0, self.preprocess_config["preprocessing"]["pitch"])
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-energy-{}.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        attn_prior_path = os.path.join(
            self.preprocessed_path,
            "attn_prior",
            "{}-attn_prior-{}.npy".format(speaker, basename),
        )
        attn_prior = np.load(attn_prior_path)
        spker_embed = np.load(os.path.join(
            self.preprocessed_path,
            "spker_embed",
            "{}-spker_embed.npy".format(speaker),
        )) if self.load_spker_embed else None

        # Random Slicing
        seq_start = 0
        max_seq_start = mel.shape[0] - self.segment_length
        if max_seq_start > 0:
            seq_start = np.random.randint(0, max_seq_start) * self.hop_length
        audio = audio[seq_start:seq_start+self.segment_length_up]

        sample = {
            "id": basename,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "audio": audio,
            "mel": mel,
            "f0": f0,
            "uv": uv,
            "energy": energy,
            "seq_start": seq_start // self.hop_length,
            "attn_prior": attn_prior,
            "spker_embed": spker_embed,
        }

        return sample

    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            raw_text = []
            for line in f.readlines():
                n, s, _, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                raw_text.append(r)
            return name, speaker, raw_text

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        audios = [data[idx]["audio"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        f0s = [data[idx]["f0"] for idx in idxs]
        uvs = [data[idx]["uv"] for idx in idxs]
        energies = [data[idx]["energy"] for idx in idxs]
        seq_starts = [data[idx]["seq_start"] for idx in idxs]
        attn_priors = [data[idx]["attn_prior"] for idx in idxs]
        spker_embeds = np.concatenate(np.array([data[idx]["spker_embed"] for idx in idxs]), axis=0) \
            if self.load_spker_embed else None

        text_lens = np.array([text.shape[0] for text in texts])
        audio_lens = np.array([audio.shape[0] for audio in audios])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        speakers = np.array(speakers)
        texts = pad_1D(texts)
        audios = pad_1D(audios)
        mels = pad_2D(mels)
        f0s = pad_1D(f0s)
        uvs = pad_1D(uvs)
        energies = pad_1D(energies)
        attn_priors = pad_3D(attn_priors, len(idxs), max(text_lens), max(mel_lens))
        seq_starts = np.array(seq_starts)

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            text_lens,
            max(text_lens),
            audios,
            audio_lens,
            max(audio_lens),
            mels,
            mel_lens,
            max(mel_lens),
            f0s,
            uvs,
            energies,
            seq_starts,
            attn_priors,
            spker_embeds,
        )

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))

        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config, model_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.load_spker_embed = model_config["multi_speaker"] \
            and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'

        self.basename, self.speaker, self.raw_text = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.raw_text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone_path = os.path.join(
            self.preprocessed_path,
            "text",
            "{}-text-{}.npy".format(speaker, basename),
        )
        phone = np.load(phone_path)
        spker_embed = np.load(os.path.join(
            self.preprocessed_path,
            "spker_embed",
            "{}-spker_embed.npy".format(speaker),
        )) if self.load_spker_embed else None

        return (basename, speaker_id, phone, raw_text, spker_embed)

    def process_meta(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            name = []
            speaker = []
            raw_text = []
            for line in f.readlines():
                n, s, _, r = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                raw_text.append(r)
            return name, speaker, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])
        spker_embeds = np.concatenate(np.array([d[4] for d in data]), axis=0) \
            if self.load_spker_embed else None

        texts = pad_1D(texts)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens), spker_embeds
