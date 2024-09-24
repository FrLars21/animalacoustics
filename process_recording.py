import os
import time
from tqdm import tqdm
import numpy as np
import librosa
import torch
from transformers import ClapModel, ClapProcessor

import sqlite_vec
from fastcore.parallel import threaded
from fastlite import *

def load_and_preprocess_audio(file_path, target_sr=48000):
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=True)

        return audio, sr
    except Exception as e:
        print(f"Error loading audio file: {e}")
        raise

def insert_embedding(db, recording_id, start_time, end_time, embedding):
    audio_chunks  = db.t.audio_chunks
    new_chunk = audio_chunks.insert({"recording_id": recording_id, "start": start_time, "end": end_time})
    #vec_biolingual.insert({"audio_chunk_id": new_chunk["id"], "embedding": embedding})
    db.execute("insert into vec_biolingual (audio_chunk_id, embedding) values (?, ?)", [new_chunk["id"], embedding])

def process_audio(db, model, processor, recording_id, audio, sample_rate, batch_size=25):
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
                insert_embedding(db, recording_id, start_time, end_time, embed.cpu().numpy().astype(np.float32))

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
        processor = ClapProcessor.from_pretrained("davidrrobinson/BioLingual")
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
    #try:
    process_audio(db, model, processor, recording_id, audio, sample_rate)
    #except Exception as e:
    #    print(f"Failed to embed {recording['filename']}: {e}")
    #    return
    print(f"Audio file {recording['filename']} embedded in {time.time() - start_time:.2f} seconds")

    recording["status"] = "processed"
    recordings.update(recording)
    print("status set to processed in db :)")