# animalacoustics
Explore massive bioacoustic datasets.

## TODO (v0.1)

- Add foreign keys and recursive triggers on delete (also to vec_biolingual index)
- Keep uploaded files in sync with recordings (e.g. delete them when a recording is deleted)
- Bulk processing/embedding of recordings
- Test on webserver

## TODO (v0.2)

- Query by 'internal' audio clip.
- Show datasets on a map.
- Better logging and error handling when processing recordings.
- Notes/bookmarks for audio chunks.

## TODO (v0.3)

- 'Load more'-button for search results.
- Query by 'external' audio clip.
- Vector arithmetic.
- Visualize embeddings in 2D.

## Setup sqlite-vec

Install via sqlite-utils cli:

```bash
sqlite-utils install sqlite-utils-sqlite-vec
```

Check installation:

```bash
sqlite-utils memory 'select vec_version()'
```

Create the vector index with the proper embedding dimension for the BioLingual model:

```bash
sqlite-utils database.db 'create virtual table vec_biolingual using vec0(
  audio_chunk_id integer primary key,
  embedding float[512]
);'
```

Data is inserted with a normal `INSERT INTO` statement. The `audio_chunk_id` should point to a numeric `id` from the `audio_chunks` table to actually link the embedding to a specific audio chunk.

Setup a trigger to keep the `vec_biolingual` table in sync with the `audio_chunks` table on row deletion:

```bash
sqlite-utils database.db 'create trigger after_delete_audio_chunk after delete on audio_chunks begin
  delete from vec_biolingual where audio_chunk_id = old.id;
end;'
```

The `vec_biolingual` table can be queried with a KNN-style query:

```bash
-- KNN-style query: the 20 closest headlines to the input embedding `?`
select
	rowid,
	distance
from vec_biolingual
where embedding match ?
  and k = 20;
```

In practice, the input embedding is generated on-the-fly with the BioLingual model and passed to the query as a float array. This embedding will usually represent a text query.