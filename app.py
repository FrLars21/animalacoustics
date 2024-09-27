from datetime import datetime, timedelta
import re
import os
from queue import Queue
import threading
import time
from fasthtml.common import *
from shad4fast import *
from lucide_fasthtml import Lucide
import sqlite_vec
import soundfile as sf
from utils import chunk_and_embed_recording, load_audio
from components import ApplicationShell, ProcessButton, DropzoneUploader
from config import load_config
import aiofiles

# Load configuration
config = load_config()

db = database("database.db")
db.conn.enable_load_extension(True)
db.conn.load_extension(sqlite_vec.loadable_path())
db.conn.enable_load_extension(False)

# print("SQLite version:", db.execute("SELECT sqlite_version()").fetchone())
# print("vec_version:", db.execute("select vec_version()").fetchone())

datasets, recordings, audio_chunks = db.t.datasets, db.t.recordings, db.t.audio_chunks
if datasets not in db.t:
    datasets.create(id=int, name=str, description=str, pk="id")
    recordings.create(id=int, dataset_id=int, filename=str, duration=int, datetime=str, status=str, pk="id", foreign_keys=[("dataset_id", "datasets", "id")])
    audio_chunks.create(id=int, recording_id=int, start=int, end=float, pk="id", foreign_keys=[("recording_id", "recordings", "id")])

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

    db.execute("""
        CREATE TRIGGER IF NOT EXISTS after_delete_audio_chunk 
        AFTER DELETE ON audio_chunks
        BEGIN
            DELETE FROM vec_biolingual WHERE audio_chunk_id = OLD.id;
        END
    """)

Dataset, Recording, AudioChunk = datasets.dataclass(), recordings.dataclass(), audio_chunks.dataclass()

# Load ML models
import torch
import numpy as np

from transformers import ClapModel, ClapProcessor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ClapModel.from_pretrained("davidrrobinson/BioLingual").to(device)
processor = ClapProcessor.from_pretrained("davidrrobinson/BioLingual", clean_up_tokenization_spaces=True)

# Task queue and related variables
task_queue = Queue()
processing_thread = None
is_processing = False

def process_queue():
    global is_processing
    while True:
        if not task_queue.empty():
            is_processing = True
            recording_id = task_queue.get()
            chunk_and_embed_recording(recording_id, model=model, processor=processor)
            task_queue.task_done()
        else:
            is_processing = False
            time.sleep(1)  # Wait a bit before checking the queue again

def ensure_processing_thread():
    global processing_thread
    if processing_thread is None or not processing_thread.is_alive():
        processing_thread = threading.Thread(target=process_queue, daemon=True)
        processing_thread.start()

#app, rt = fast_app(
#    pico=False,
#    hdrs=(ShadHead(tw_cdn=True, theme_handle=True),Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js"),),
#)

app = FastHTML(hdrs=(ShadHead(tw_cdn=True, theme_handle=True),))
rt = app.route

### DATASETS ###

@rt('/datasets')
def get():
    return ApplicationShell(
        Div(
            H2("Datasets", cls="text-2xl font-bold tracking-tight"),
            P("A dataset is a the union between a single recorder, recorded at a specific location, and a specific time period. Think of it as one SD card/battery cycle.", cls="text-muted-foreground"),
            cls="space-y-0.5"
        ),
        Separator(cls="my-6 h-[1px]"),
        P("Click on a dataset in the sidebar to manage its recordings, or add a click the button to add a new one."),
        active_link_id="sidebar-datasets-link"
    )

@rt('/datasets/{dataset_id:int}')
def get(dataset_id:int):
    try:
        dataset = datasets[dataset_id]
    except Exception as e:
        return ApplicationShell(P(f"Dataset with id {dataset_id} not found."), active_link_id="sidebar-datasets-link")

    return ApplicationShell(
        Tabs(
            TabsList(
                TabsTrigger("Metadata", value="metadata"),
                TabsTrigger("Recordings", value="recordings"),
            ),
            TabsContent(value="metadata")(
                Section(cls='space-y-6 lg:max-w-2xl')(
                    Form(hx_put="/update-dataset", hx_target=f"#dataset-link-{dataset.id}", cls="space-y-8")(
                        Hidden(name="id", value=dataset.id),
                        Div(
                            Label('Dataset name', htmlFor="name"),
                            Input(placeholder='Tofte Syd #1', id='name', name='name', value=dataset.name, required=True),
                            P('This is the public display name of your dataset. Set it arbitrarily to something meaningful.', id='name-description', cls='text-[0.8rem] text-muted-foreground'),
                            cls='space-y-2'
                        ),
                        Div(
                            Label('Dataset description', htmlFor='description'),
                            Textarea(dataset.description, placeholder='For example used to elaborate on recording location or for notes on sound quality.', name='description', id='description'),
                            P('For brief notes on the dataset.', id='description-description', cls='text-[0.8rem] text-muted-foreground'),
                            cls='space-y-2'
                        ),
                        Button('Update metadata', type='submit'),
                    )
                ),
                Section(cls='space-y-6 mt-12 lg:max-w-2xl')(
                    Div(cls="space-y-0.5")(
                        H2('Delete dataset', cls='text-lg font-medium'),
                        P('Delete the dataset and all its recordings. Warning: this action cannot be undone.', cls="text-sm text-muted-foreground"),
                    ),
                    Div(data_orientation='horizontal', role='none', cls='shrink-0 bg-border h-[1px] w-full lg:max-w-2xl'),
                    Button('Delete dataset', variant="destructive", hx_delete=f"/delete-dataset", hx_vals=f'{{"dataset_id":{dataset.id}}}', hx_confirm="Are you sure you want to delete this dataset? This action cannot be undone."),
                ),
            ),
            TabsContent(value="recordings")(
                Section(cls='space-y-6 mt-8 lg:max-w-4xl')(
                    Div(cls="space-y-1")(
                        H2("Manage Recordings", cls="text-2xl font-semibold tracking-tight"),
                        P("Upload, manage and process field recordings.", cls="text-sm text-muted-foreground"),
                    ),
                    Div(cls="rounded-md border")(
                        Table(
                            TableHeader(
                                TableRow(
                                    #TableHead(Checkbox(id="select-all", name="select-all")), 
                                    *[TableHead(header) for header in ["Filename", "Starts at", "Duration", "Process", "Delete"]]
                                )
                            ),
                            TableBody(
                                *[recording for recording in recordings(where="dataset_id=?", where_args=[dataset.id], order_by="datetime ASC")],
                                id="file-rows"
                            ),
                        ),
                    ),
                ),
                Section(cls="space-y-6 mt-12 lg:max-w-2xl")(
                    Div(cls="space-y-1")(
                        H2("Upload Files", cls="text-2xl font-semibold tracking-tight"),
                        P("Upload one or more audio files to add to this dataset.", cls="text-sm text-muted-foreground"),
                    ),
                    DropzoneUploader(dataset.id),
                ),
            ),
            standard=True,
            cls="w-full",
        ),
        active_link_id=dataset_id
    )

@rt('/new-dataset')
def post():
    dataset = datasets.insert(name="Untitled dataset")
    return Redirect(f"/datasets/{dataset.id}")

@rt('/update-dataset')
def put(dataset: Dataset):
    new_dataset = datasets.update(dataset)
    return Button(new_dataset.name, variant="secondary", cls="w-full !justify-start", id=f"dataset-link-{new_dataset.id}", hx_swap_oob="true")

@rt('/delete-dataset')
def delete(dataset_id: int):
    datasets.delete(dataset_id)
    return Redirect('/datasets')

### RECORDINGS ###
@patch
def __ft__(self:Recording):
    return TableRow(
        #TableCell(Checkbox(id="terms", name="terms", value="agree", checked=False)),
        TableCell(A(self.filename, href=f"/recording/{self.id}", cls="hover:underline")),
        TableCell(
            Div(Lucide("calendar-fold", size=16), datetime.fromisoformat(self.datetime).strftime("%d %b. %Y"), cls="flex gap-1 items-center"),
            Div(Lucide("clock", size=16), datetime.fromisoformat(self.datetime).strftime("%H:%M"), cls="flex gap-1 items-center")
        ),
        TableCell(f"{(self.duration // 60) + (1 if self.duration % 60 > 30 else 0)} min"),
        TableCell(ProcessButton(self.id, self.status)),
        TableCell(
            Button(
                Lucide("eraser", size=16), "Delete",
                variant="destructive", size="sm", cls="gap-2",
                hx_delete=f"/delete-recording", hx_vals=f'{{"recording_id":{self.id}}}',
                hx_target="closest tr", hx_swap="delete",
                hx_confirm="Are you sure you want to delete this recording? This action cannot be undone."
            )
        ),
    )

parent_dir_path = os.path.dirname(os.path.realpath(__file__))
reg_re_param("audioext", "mp3|wav|ogg|flac")
@rt("/{fname:path}.{ext:audioext}")
def get(fname:str, ext:str): return FileResponse(os.path.join(parent_dir_path, config['uploads_path'], fname + "." + ext))

@rt('/recording/{recording_id:int}')
def get(recording_id: int, t:int=0):
    recording = recordings[recording_id]
    return ApplicationShell(
        Div(
            H2("Recording", cls="text-2xl font-bold tracking-tight"),
            P("Listen to a full length recording.", cls="text-muted-foreground"),
            cls="space-y-0.5"
        ),
        Separator(cls="my-6 h-[1px]"),
        Div(
            H3(recording.filename, cls="text-xl font-semibold"),
            Audio(src=f"/{config['uploads_path']}/{recording.filename}#t={t}", controls=True, cls="w-full"),
            cls="space-y-4"
        ),
        active_link_id=recording.dataset_id
    )

@rt('/upload-recording')
async def post(request):
    form = await request.form()
    file, status_id, dataset_id = form.get('file'), form.get('statusId'), form.get('dataset_id')
    filename = file.filename

    if not re.match(r'^[A-Za-z0-9]+_\d{8}_\d{6}(_.*)?\.flac$', filename):
        return "", Span('Invalid filename format', cls='text-xs font-medium text-red-600', id=status_id, hx_swap_oob="true")

    if recordings(where="filename = ?", where_args=[filename]):
        return "", Span('File already exists', cls='text-xs font-medium text-red-600', id=status_id, hx_swap_oob="true")

    file_path = os.path.join(config['uploads_path'], filename)

    try:
        # Asynchronously save the file
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()  # read async
            await out_file.write(content)  # write async

        with sf.SoundFile(file_path) as audio_file:
            duration_seconds = len(audio_file) // audio_file.samplerate

        date_time_parts = filename[:-5].split('_')[1:3]
        datetime_str = datetime.strptime('_'.join(date_time_parts), "%Y%m%d_%H%M%S").isoformat(timespec='seconds')

        uploaded_file = recordings.insert({
            'dataset_id': dataset_id,
            'filename': filename,
            'duration': duration_seconds,
            'datetime': datetime_str,
            'status': 'unprocessed'
        })
        return uploaded_file, Span('Uploaded', cls='text-xs font-medium text-green-600', id=status_id, hx_swap_oob="true")
    except Exception as e:
        if os.path.exists(file_path): os.remove(file_path)
        return "", Span(f'Error: {str(e)}', cls='text-xs font-medium text-red-600', id=status_id, hx_swap_oob="true")

@rt('/delete-recording')
def delete(recording_id: int):
    recording = recordings[recording_id]
    file_path = os.path.join(config['uploads_path'], recording.filename)

    try:
        recordings.delete(recording_id)
    except Exception as e:
        return e

    if os.path.exists(file_path):
        os.remove(file_path)

    return ""

@rt('/process/{recording_id:int}')
def post(recording_id: int):
    recording = recordings[recording_id]
    recording.status = "processing"
    recordings.update(recording)

    # add processing to task queue
    task_queue.put(recording_id)
    ensure_processing_thread()
    print(f"Added recording {recording_id} to task queue")

    return ProcessButton(recording_id, recording.status)

""" async def status_generator(recording_id: int):
    while True:
        recording = recordings[recording_id]
        if recording.status != "processing":
            yield sse_message(ProcessButton(recording_id, recording.status))
            break
        await asyncio.sleep(5)  # Check every 5 seconds

@rt("/status-stream/{recording_id:int}")
async def get(recording_id: int):
    return EventStream(status_generator(recording_id)) """

## SEARCH

@rt('/')
def get():
    return ApplicationShell(
        Section(cls="space-y-2 my-8")(
            H1("Search Recordings", cls="text-lg font-medium"),
            P("Search for recordings by filename, date, or duration.", cls="text-sm text-muted-foreground")
        ),
        Section(cls="mb-8")(
            Form(id="search-form", hx_get="/search", hx_target="#search-results", hx_disabled_elt="#search-button")(
                Div(cls="flex gap-2")(
                    Input(type="text", name="query", placeholder="Search recordings...", cls="w-full"),
                    Select(
                        SelectTrigger(SelectValue(placeholder="No. results"), cls="!w-[140px] gap-2"),
                        SelectContent(
                            SelectGroup(
                                SelectLabel("No. results"),
                                *[SelectItem(str(n), value=n) for n in (5, 10, 20, 50)],
                            ), id="k"
                        ),
                        standard=True, id="k", name="k", cls="shrink-0"
                    ),
                    Button(Lucide("search", size=16), "Search", cls="shrink-0 gap-2", id="search-button"),
                )
            )
        ),
        Section(
            Table(TableHeader(TableRow(*[TableHead(col) for col in ["#", "Audio", "Date", "Time", "Dataset"]])), TableBody(id="search-results"))
        ),
        active_link_id="sidebar-search-link"
    )

@rt('/search')
def get(query:str, k:int):
    if not k: k = 5

    def embed_str(str: str):
        inputs = processor(text=str, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_embed = model.get_text_features(**inputs)
        return text_embed.numpy()[0].astype(np.float32)

    def vector_search(query:str, k:int):
        query_embedding = embed_str([query])

        return(
            db.q("""
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
                    WHERE embedding MATCH ?
                        and k = ?
                ) q
                LEFT JOIN audio_chunks c ON q.audio_chunk_id = c.id
                LEFT JOIN recordings r ON c.recording_id = r.id
                LEFT JOIN datasets d ON r.dataset_id = d.id
            """, [query_embedding, k])
        )

    matches = vector_search(query, k=k)

    table_rows = [TableRow(
        TableCell(i+1),
        TableCell(Audio(src=f"data:audio/flac;base64,{load_audio(r['filename'], r['start'], r['end'])}", type="audio/flac", controls=True)),
        TableCell(Div(Lucide("calendar-fold", size=16), (datetime.fromisoformat(r["datetime"])).strftime("%d %b. %Y"), cls="flex gap-1 items-center")),
        TableCell(Div(Lucide("clock", size=16), (datetime.fromisoformat(r["datetime"]) + timedelta(seconds=r["start"])).strftime("%H:%M:%S"), cls="flex gap-1 items-center")),
        TableCell(Div(r["dataset"], A("Listen in context", href=f"/recording/{r['recording_id']}?t={r['start']}", cls="text-blue-500"), cls="flex flex-col"))
    ) for i, r in enumerate(matches)]

    return table_rows

@rt("/stats")
def get_stats():
    import psutil
    process = psutil.Process()
    return {
        "cpu_percent": process.cpu_percent(),
        "memory_usage": process.memory_info().rss / 1024 / 1024,  # MB
        "thread_count": threading.active_count()
    }

serve()