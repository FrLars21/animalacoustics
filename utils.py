DB_PATH = "database.db"
AUDIO_FOLDER = "uploads/"

import torch
from transformers import ClapModel, ClapProcessor

# Load embedding model
model = ClapModel.from_pretrained("davidrrobinson/BioLingual")
processor = ClapProcessor.from_pretrained("davidrrobinson/BioLingual")

import io
import librosa
import soundfile as sf
import base64

import time
import os
from math import ceil

from sqlite_utils import Database

def timeit_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time} seconds")
        return result
    return wrapper

@timeit_decorator
def embed_str(str: str):
    """
    Embeds a str with BioLingual.
    """
    inputs = processor(text=str, return_tensors="pt", padding=True)
    with torch.no_grad():
        text_embed = model.get_text_features(**inputs)
    return text_embed.numpy()[0].astype(np.float32)

def vector_search(query:str, k:int):

    db = Database(DB_PATH)

    query_embedding = embed_str([query])

    return(
        db.query("""
        SELECT
            q.audio_chunk_id,
            q.distance,
            c.recording_id,
            c.start,
            c.end,
            r.filename,
            r.datetime,
            d.name as dataset
        FROM (
            SELECT
                audio_chunk_id,
                distance
            FROM vec_biolingual
            WHERE embedding MATCH :query_embedding
                and k = :k
        ) q
        LEFT JOIN audio_chunks c ON q.audio_chunk_id = c.id
        LEFT JOIN recordings r ON c.recording_id = r.id
        LEFT JOIN datasets d ON r.dataset_id = d.id
        """, {"query_embedding": query_embedding, "k": k})
    )

def get_audio_duration(file_path):
    return sf.info(file_path).duration

def is_fully_embedded(file_path, duration):
    file_name = os.path.basename(file_path)
    expected_chunks = ceil(duration / 10)  # 10-second chunks

    db = Database("database.db")
    actual_chunks = db.execute(
        "SELECT COUNT(*) FROM biolingual_embedding WHERE input_file = ?",
        [file_name]
    ).fetchone()[0]
    return actual_chunks in (expected_chunks, expected_chunks - 1) # allow for not perfectly divisble by 10 length

import numpy as np

def amplify_audio(numpy_array, factor=1.5, normalize=True):
    """
    Amplifies the audio in the numpy array by the given factor.

    Parameters:
    numpy_array (numpy.ndarray): The input audio signal.
    factor (float): The amplification factor (default is 1.5).
    normalize (bool): If True, normalizes the output to the range [-1, 1] after amplification.
                      If False, clips the values in the range [-1, 1] (default is False).

    Returns:
    numpy.ndarray: The amplified audio signal.
    """
    # Check for invalid factor values
    if factor <= 0:
        raise ValueError("Amplification factor must be greater than zero.")

    # Amplify the audio by the factor
    amplified_audio = numpy_array * factor

    # Normalize or clip to the range [-1, 1]
    if normalize:
        max_value = np.max(np.abs(amplified_audio))
        if max_value > 1:
            amplified_audio = amplified_audio / max_value
    else:
        amplified_audio = np.clip(amplified_audio, -1, 1)

    return amplified_audio

def numpy_to_flac(numpy_array, sample_rate=44100):
    # Create an in-memory bytes buffer
    buffer = io.BytesIO()

    # Write the numpy array to the buffer as a FLAC file
    sf.write(buffer, numpy_array, sample_rate, format='FLAC')

    # Move the buffer's position to the start
    buffer.seek(0)

    return buffer

def buffer_to_base64(buffer):
    # Read the buffer's content and encode it to base64
    audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return audio_base64

@timeit_decorator
def retrieve_audio(file_name, start_time, duration):
    audio_clip, sr = librosa.load(path=f'{AUDIO_FOLDER}{file_name}', sr=48000, mono=True, offset=start_time, duration=duration)
    return audio_clip, sr

def load_audio(file_name, start_time, end_time):
    duration = end_time - start_time
    audio_clip, sr = retrieve_audio(file_name, start_time, duration)

    # Make audio louder
    audio_clip_louder = amplify_audio(audio_clip, factor=8)

    # Convert NumPy array to FLAC buffer
    flac_buffer = numpy_to_flac(audio_clip_louder, sr)

    # Convert buffer to base64 string
    audio_base64 = buffer_to_base64(flac_buffer)

    return audio_base64

def convert_seconds(seconds):
    return "%02d:%02d" % (seconds // 60, seconds % 60)
