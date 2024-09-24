AUDIO_FOLDER = "uploads/"

import io
import librosa
import soundfile as sf
import base64

import time
import os
from math import ceil
import numpy as np
import torch
from fastlite import *
import sqlite_vec
from fastcore.parallel import threaded
from tqdm import tqdm
from transformers import ClapModel, ClapProcessor

import functools

def timeit_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def is_fully_embedded(file_name, duration):
    """Check if a recording has been fully embedded."""
    expected_chunks = ceil(duration / 10) # 10-second chunks

    db = database("database.db")
    actual_chunks = db.execute(
        "SELECT COUNT(*) FROM vec_biolingual WHERE input_file = ?",
        [file_name]
    ).fetchone()[0]
    return actual_chunks in (expected_chunks, expected_chunks - 1) # allow for not perfectly divisble by 10 length

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
def load_and_preprocess_audio(file_path, target_sr=48000):
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)

        return audio, sr
    except Exception as e:
        print(f"Error loading audio file: {e}")
        raise

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

def embed_audio(db, model, processor, recording_id, audio, sample_rate, batch_size=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    chunk_size = sample_rate * 10 # 10 seconds per chunk
    chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

    with tqdm(total=len(chunks), desc="Processing audio") as pbar:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            inputs = processor(audios=batch, return_tensors="pt", sampling_rate=sample_rate).to(device)

            with torch.no_grad():
                audio_embed = model.get_audio_features(**inputs)

            for j, embed in enumerate(audio_embed):
                chunk_index = i + j
                start_time = chunk_index * 10
                end_time = min((chunk_index + 1) * 10, len(audio) / sample_rate)
                
                new_chunk = db.t.audio_chunks.insert({"recording_id": recording_id, "start": start_time, "end": end_time})
                db.execute("insert into vec_biolingual (audio_chunk_id, embedding) values (?, ?)", [new_chunk["id"], embed.cpu().numpy().astype(np.float32)])

            pbar.update(len(batch))

@threaded
def process_recording(recording_id: int, input_folder: str = "uploads"):
    db = database("database.db")
    db.conn.enable_load_extension(True)
    db.conn.load_extension(sqlite_vec.loadable_path())
    db.conn.enable_load_extension(False)
    
    recordings = db.t.recordings
    recording = recordings[recording_id]
    
    print("Loading model...")
    try:
        model = ClapModel.from_pretrained("davidrrobinson/BioLingual")
        processor = ClapProcessor.from_pretrained("davidrrobinson/BioLingual", clean_up_tokenization_spaces=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    audio_path = os.path.join(input_folder, recording["filename"])
    print(f"Loading and preprocessing audio file {recording['filename']}...")
    start_time = time.time()
    try:
        audio, sample_rate = load_and_preprocess_audio(audio_path)
    except Exception as e:
        print(f"Failed to load and preprocess {recording['filename']}: {e}")
        return
    print(f"Audio file {recording['filename']} loaded and preprocessed in {time.time() - start_time:.2f} seconds")

    print(f"Processing audio file {recording['filename']}...")
    start_time = time.time()
    try:
        embed_audio(db, model, processor, recording_id, audio, sample_rate)
    except Exception as e:
        print(f"Failed to embed {recording['filename']}: {e}")
        return
    print(f"Audio file {recording['filename']} embedded in {time.time() - start_time:.2f} seconds")

    recording["status"] = "processed"
    recordings.update(recording)
    print("status set to processed in db :)")