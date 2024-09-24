# animalacoustics
Explore massive bioacoustic datasets.

## TODO (v0.1)

- ~~Add foreign keys and recursive triggers on delete (also to vec_biolingual index)~~
- Bulk processing/embedding of recordings
- Keep uploaded files in sync with recordings (e.g. delete them when a recording is deleted)
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

## Installation

Clone the repo and run `pip install -r requirements.txt`.

Serve the FastHTML app via uvicorn. The database will be initialized automatically on the first run.

```bash
python app.py
```