from sqlite_utils import Database

# sqlite-vec needs to be installed before running this script!
# sqlite-utils install sqlite-utils-sqlite-vec
# UPDATE: i actually think this is done automatically when pip install requirements.txt

def setup_database():
    db = Database("database.db")

    # Create tables if they don't exist
    db["datasets"].create({
        "id": int,
        "name": str,
        "description": str
    }, pk="id", if_not_exists=True, not_null={"name"})

    db["recordings"].create({
        "id": int,
        "dataset_id": int,
        "filename": str,
        "duration": int,
        "datetime": str,
        "status": str # convert this to enum when moving closer to prod
    }, pk="id", if_not_exists=True, foreign_keys=[
        ("dataset_id", "datasets", "id")
    ], not_null={"dataset_id", "filename", "duration", "datetime", "status"})

    db["audio_chunks"].create({
        "id": int,
        "recording_id": int,
        "start": int,
        "end": float
    }, pk="id", if_not_exists=True, foreign_keys=[
        ("recording_id", "recordings", "id")
    ], not_null={"recording_id", "start", "end"})

    # Check installation
    sqlite_version = db.execute("SELECT sqlite_version()").fetchone()[0]
    print(f"SQLite version: {sqlite_version}")
    sqlite_vec_version = db.execute("SELECT vec_version()").fetchone()[0]
    print(f"sqlite-vec version: {sqlite_vec_version}")

    # Create vector index for BioLingual embeddings
    db.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_biolingual USING vec0(
        audio_chunk_id INTEGER PRIMARY KEY,
        embedding FLOAT[512]
    )
    """)

    # Create triggers for cascading deletes
    db.execute("""
    CREATE TRIGGER IF NOT EXISTS delete_dataset_cascade
    AFTER DELETE ON datasets
    BEGIN
        DELETE FROM recordings WHERE dataset_id = OLD.id;
    END
    """)

    db.execute("""
    CREATE TRIGGER IF NOT EXISTS delete_recording_cascade
    AFTER DELETE ON recordings
    BEGIN
        DELETE FROM audio_chunks WHERE recording_id = OLD.id;
    END
    """)

    # Create trigger to keep vec_biolingual in sync with audio_chunks
    db.execute("""
    CREATE TRIGGER IF NOT EXISTS after_delete_audio_chunk 
    AFTER DELETE ON audio_chunks
    BEGIN
        DELETE FROM vec_biolingual WHERE audio_chunk_id = OLD.id;
    END
    """)

    print("Vector index and triggers created successfully.")
    print("Database setup completed successfully.")

if __name__ == "__main__":
    setup_database()