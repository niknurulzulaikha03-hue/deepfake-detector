import librosa
import os
from moviepy.editor import VideoFileClip
import numpy as np

def extract_mfcc(video_path):

    clip = VideoFileClip(video_path)

    if clip.audio is None:
        return None

    audio_path = "temp_audio.wav"

    clip.audio.write_audiofile(audio_path, verbose=False, logger=None)

    y, sr = librosa.load(audio_path)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    os.remove(audio_path)

    return np.mean(mfcc.T, axis=0)
