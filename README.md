# animalacoustics
Explore massive bioacoustic datasets.

## TODO (v0.1)

- ~~Add foreign keys and recursive triggers on delete (also to vec_biolingual index)~~
- ~~Keep uploaded files in sync with recordings (e.g. delete them when a recording is deleted)~~
- ~~Test on webserver~~

## TODO (v0.2)

- ~~Query by 'internal' audio clip.~~
- Show BirdNET detections in a table below a full recording.
- Show datasets on a map.
- Notes/bookmarks for audio chunks.
- 'Load more'-button for search results (instead of n-results)
- Better logging and error handling when processing recordings.
- 'Predict with BirdNET' button on recordings.

## TODO (v0.3)

- Query by 'external' audio clip.
- Vector arithmetic.
- Visualize embeddings in 2D.
- Bulk processing/embedding of recordings
- More robust UI
- Better exports to relevant formats (Training BirdNET classifiers, Raven Pro selection tables, etc.)
- Keyboard shortcuts for common actions.

## Installation

Clone the repo and run `pip install -r requirements.txt`.

Serve the FastHTML app via uvicorn. The database will be initialized automatically on the first run.

```bash
python app.py
```