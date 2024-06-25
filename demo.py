#!/usr/bin/env python
# coding: utf-8

# ## Setup
#
# Includes:
#
# - How to generate a placeholder dataset if you haven't already, just the basics to run "training" e2e on a tiny dataset
# - How to download a dataset from OpenSLR

# ### Imports & paths

# In[14]:


# imports
import math
import wave
import struct
import os
import urllib.request
import tarfile
from audiolm_pytorch import SoundStream, SoundStreamTrainer, HubertWithKmeans, SemanticTransformer, SemanticTransformerTrainer, HubertWithKmeans, CoarseTransformer, CoarseTransformerWrapper, CoarseTransformerTrainer, FineTransformer, FineTransformerWrapper, FineTransformerTrainer, AudioLM
from torch import nn
import torch
import torchaudio


# define all dataset paths, checkpoints, etc
dataset_folder = "placeholder_dataset"
soundstream_ckpt = "results/soundstream.8.pt" # this can change depending on number of steps
hubert_ckpt = 'hubert/hubert_base_ls960.pt'
hubert_quantizer = f'hubert/hubert_base_ls960_L9_km500.bin' # listed in row "HuBERT Base (~95M params)", column Quantizer


# ### Data

# In[15]:


'''
# Placeholder data generation
def get_sinewave(freq=440.0, duration_ms=200, volume=1.0, sample_rate=44100.0):
  # code adapted from https://stackoverflow.com/a/33913403
  audio = []
  num_samples = duration_ms * (sample_rate / 1000.0)
  for x in range(int(num_samples)):
    audio.append(volume * math.sin(2 * math.pi * freq * (x / sample_rate)))
  return audio

def save_wav(file_name, audio, sample_rate=44100.0):
  # Open up a wav file
  wav_file=wave.open(file_name,"w")
  # wav params
  nchannels = 1
  sampwidth = 2
  # 44100 is the industry standard sample rate - CD quality.  If you need to
  # save on file size you can adjust it downwards. The stanard for low quality
  # is 8000 or 8kHz.
  nframes = len(audio)
  comptype = "NONE"
  compname = "not compressed"
  wav_file.setparams((nchannels, sampwidth, sample_rate, nframes, comptype, compname))
  # WAV files here are using short, 16 bit, signed integers for the
  # sample size.  So we multiply the floating point data we have by 32767, the
  # maximum value for a short integer.  NOTE: It is theortically possible to
  # use the floating point -1.0 to 1.0 data directly in a WAV file but not
  # obvious how to do that using the wave module in python.
  for sample in audio:
      wav_file.writeframes(struct.pack('h', int( sample * 32767.0 )))
  wav_file.close()
  return

def make_placeholder_dataset():
  # Make a placeholder dataset with a few .wav files that you can "train" on, just to verify things work e2e
  if os.path.isdir(dataset_folder):
    return
  os.makedirs(dataset_folder)
  save_wav(f"{dataset_folder}/example.wav", get_sinewave())
  save_wav(f"{dataset_folder}/example2.wav", get_sinewave(duration_ms=500))
  os.makedirs(f"{dataset_folder}/subdirectory")
  save_wav(f"{dataset_folder}/subdirectory/example.wav", get_sinewave(freq=330.0))

make_placeholder_dataset()
'''


# In[16]:


# Get actual dataset. Uncomment this if you want to try training on real data

# full dataset: https://www.openslr.org/12
# We'll use https://us.openslr.org/resources/12/dev-clean.tar.gz development set, "clean" speech.
# We *should* train on, well, training, but this is just to demo running things end-to-end at all so I just picked a small clean set.

# url = "https://us.openslr.org/resources/12/dev-clean.tar.gz"
# filename = "dev-clean"
# filename_targz = filename + ".tar.gz"
# if not os.path.isfile(filename_targz):
#   urllib.request.urlretrieve(url, filename_targz)
# if not os.path.isdir(filename):
#   # open file
#   with tarfile.open(filename_targz) as t:
#     t.extractall(filename)


# ## Training
#
# Now that we have a dataset, we can train AudioLM.
#
# **Note**: do NOT type "y" to overwrite previous experiments/ checkpoints when running through the cells here unless you're ready to the entire results folder! Otherwise you will end up erasing things (e.g. you train SoundStream first, and if you choose "overwrite" then you lose the SoundStream checkpoint when you then train SemanticTransformer).

# ### SoundStream

# In[17]:


soundstream = SoundStream(
    codebook_size = 1024,
    rq_num_quantizers = 8,
)
soundstream.compile()

trainer = SoundStreamTrainer(
    soundstream,
    folder = dataset_folder,
    batch_size = 1,
    grad_accum_every = 8,         # effective batch size of 32
    data_max_length = 320 * 32,
    save_results_every = 2,
    save_model_every = 4,
    num_train_steps = 9,
    lr=1e-3,
).cuda()
# NOTE: I changed num_train_steps to 9 (aka 8 + 1) from 10000 to make things go faster for demo purposes
# adjusting save_*_every variables for the same reason

trainer.train()


# ### SemanticTransformer

# In[18]:

"""

# hubert checkpoints can be downloaded at
# https://github.com/facebookresearch/fairseq/tree/main/examples/hubert
if not os.path.isdir("hubert"):
  os.makedirs("hubert")
if not os.path.isfile(hubert_ckpt):
  hubert_ckpt_download = f"https://dl.fbaipublicfiles.com/{hubert_ckpt}"
  urllib.request.urlretrieve(hubert_ckpt_download, f"./{hubert_ckpt}")
if not os.path.isfile(hubert_quantizer):
  hubert_quantizer_download = f"https://dl.fbaipublicfiles.com/{hubert_quantizer}"
  urllib.request.urlretrieve(hubert_quantizer_download, f"./{hubert_quantizer}")

wav2vec = HubertWithKmeans(
    checkpoint_path = f'./{hubert_ckpt}',
    kmeans_path = f'./{hubert_quantizer}'
)

semantic_transformer = SemanticTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    dim = 1024,
    depth = 6
).cuda()


trainer = SemanticTransformerTrainer(
    transformer = semantic_transformer,
    wav2vec = wav2vec,
    folder = dataset_folder,
    batch_size = 1,
    data_max_length = 320 * 32,
    num_train_steps = 1
)

trainer.train()


# ### CoarseTransformer

# In[19]:


wav2vec = HubertWithKmeans(
    checkpoint_path = f'./{hubert_ckpt}',
    kmeans_path = f'./{hubert_quantizer}'
)

soundstream = SoundStream(
    codebook_size = 1024,
    rq_num_quantizers = 8,
)

soundstream.load(f"./{soundstream_ckpt}")

coarse_transformer = CoarseTransformer(
    num_semantic_tokens = wav2vec.codebook_size,
    codebook_size = 1024,
    num_coarse_quantizers = 3,
    dim = 512,
    depth = 6
)

trainer = CoarseTransformerTrainer(
    transformer = coarse_transformer,
    codec = soundstream,
    wav2vec = wav2vec,
    folder = dataset_folder,
    batch_size = 1,
    data_max_length = 320 * 32,
    save_results_every = 2,
    save_model_every = 4,
    num_train_steps = 9
)
# NOTE: I changed num_train_steps to 9 (aka 8 + 1) from 10000 to make things go faster for demo purposes
# adjusting save_*_every variables for the same reason

trainer.train()


# ### FineTransformer

# In[20]:


soundstream = SoundStream(
    codebook_size = 1024,
    rq_num_quantizers = 8,
)

soundstream.load(f"./{soundstream_ckpt}")

fine_transformer = FineTransformer(
    num_coarse_quantizers = 3,
    num_fine_quantizers = 5,
    codebook_size = 1024,
    dim = 512,
    depth = 6
)

trainer = FineTransformerTrainer(
    transformer = fine_transformer,
    codec = soundstream,
    folder = dataset_folder,
    batch_size = 1,
    data_max_length = 320 * 32,
    num_train_steps = 9
)
# NOTE: I changed num_train_steps to 9 (aka 8 + 1) from 10000 to make things go faster for demo purposes
# adjusting save_*_every variables for the same reason

trainer.train()


# ## Inference

# In[21]:


# Everything together
audiolm = AudioLM(
    wav2vec = wav2vec,
    codec = soundstream,
    semantic_transformer = semantic_transformer,
    coarse_transformer = coarse_transformer,
    fine_transformer = fine_transformer
)

generated_wav = audiolm(batch_size = 1)


# In[22]:


output_path = "out.wav"
sample_rate = 44100
torchaudio.save(output_path, generated_wav.cpu(), sample_rate)


# In[22]:



"""
