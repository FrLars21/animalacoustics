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
from components import ApplicationShell, ProcessButton, DropzoneUploader, HeadingBlock, DropdownMenu, DropdownMenuItem, DropdownMenuLabel
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

floating_ui_script = Script("""
    import {autoUpdate, computePosition, offset} from 'https://cdn.jsdelivr.net/npm/@floating-ui/dom@1.6.10/+esm';

    document.addEventListener('DOMContentLoaded', () => {
        const triggers = document.querySelectorAll('[id^="dropdown-trigger-"]');
        const menus = document.querySelectorAll('[id^="dropdown-menu-"]');

        // Move all menus to the body
        menus.forEach(menu => {
            document.body.appendChild(menu);
        });

        let activeMenu = null;
        let cleanup = null;

        function showMenu(triggerId) {
            const menuId = triggerId.replace('trigger', 'menu');
            const trigger = document.getElementById(triggerId);
            const menu = document.getElementById(menuId);

            if (activeMenu) {
                hideMenu();
            }

            menu.style.display = 'block';
            activeMenu = menu;

            function updatePosition() {
                computePosition(trigger, menu, {
                    placement: 'bottom-end',
                    middleware: [offset(5)]
                }).then(({x, y}) => {
                    Object.assign(menu.style, {
                        left: `${x}px`,
                        top: `${y}px`,
                    });
                });
            }

            cleanup = autoUpdate(trigger, menu, updatePosition);
        }

        function hideMenu() {
            if (activeMenu) {
                activeMenu.style.display = 'none';
                if (cleanup) {
                    cleanup();
                    cleanup = null;
                }
                activeMenu = null;
            }
        }

        triggers.forEach(trigger => {
            trigger.addEventListener('click', (e) => {
                e.stopPropagation();
                const triggerId = trigger.id;
                if (activeMenu && activeMenu.id === triggerId.replace('trigger', 'menu')) {
                    hideMenu();
                } else {
                    showMenu(triggerId);
                }
            });
        });

        // Close menu when clicking outside
        document.addEventListener('click', hideMenu);
    });
""", type="module")

app = FastHTML(hdrs=(ShadHead(tw_cdn=True),floating_ui_script,))
rt = app.route

### DATASETS ###

@rt('/datasets')
def get():
    return ApplicationShell(
        HeadingBlock("Datasets", "A dataset is a the union between a single recorder, recorded at a specific location, and a specific time period. Think of it as one SD card/battery cycle.", cls="max-w-2xl"),
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
                Section(cls="space-y-6 mt-12 lg:max-w-2xl")(
                    HeadingBlock("Delete dataset", "Delete the dataset and all its recordings. Warning: this action cannot be undone."),
                    Button("Delete dataset", variant="destructive", hx_delete=f"/delete-dataset", hx_vals=f'{{"dataset_id":{dataset.id}}}', hx_confirm="Are you sure you want to delete this dataset? This action cannot be undone."),
                ),
            ),
            TabsContent(value="recordings")(
                Section(cls='space-y-6 mt-8 lg:max-w-4xl')(
                    HeadingBlock("Manage Recordings", "Upload, manage and process field recordings."),
                    Div(cls="rounded-md border")(
                        Table(
                            TableHeader(
                                TableRow(*[TableHead(header) for header in ["Filename", "Starts at", "Duration", "Process", ""]])
                            ),
                            TableBody(
                                *[recording for recording in recordings(where="dataset_id=?", where_args=[dataset.id], order_by="datetime ASC")],
                                id="file-rows"
                            ),
                        ),
                    ),
                ),
                Section(cls="space-y-6 mt-12 lg:max-w-2xl")(
                    HeadingBlock("Upload Files", "Upload one or more audio files to add to this dataset."),
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
    return TableRow(id=f"recording-row-{self.id}")(
        TableCell(A(self.filename, href=f"/recording/{self.id}", cls="hover:underline")),
        TableCell(
            Div(Lucide("calendar-fold", size=16), datetime.fromisoformat(self.datetime).strftime("%d %b. %Y"), cls="flex gap-1 items-center"),
            Div(Lucide("clock", size=16), datetime.fromisoformat(self.datetime).strftime("%H:%M"), cls="flex gap-1 items-center")
        ),
        TableCell(f"{(self.duration // 60) + (1 if self.duration % 60 > 30 else 0)} min"),
        TableCell(ProcessButton(self.id, self.status)),
        TableCell(
            DropdownMenu(
                DropdownMenuLabel("Actions"),
                Separator(cls="-mx-1 my-1 h-px bg-muted"),
                DropdownMenuItem(A("Listen to recording", href=f"/recording/{self.id}", target="_blank"), icon="external-link"),
                Separator(cls="-mx-1 my-1 h-px bg-muted"),
                DropdownMenuItem(
                    "Process again", 
                    icon="repeat-2",
                    disabled=True
                ),
                Separator(cls="-mx-1 my-1 h-px bg-muted"),
                DropdownMenuItem(
                    "Delete recording", 
                    icon="trash", 
                    text_color="destructive",
                    hx_delete=f"/delete-recording", 
                    hx_vals=f'{{"recording_id":{self.id}}}',
                    hx_target=f"#recording-row-{self.id}", 
                    hx_swap="delete",
                    hx_confirm="Are you sure you want to delete this recording? This action cannot be undone."
                ),
                id=self.id
            )
        )
    )

parent_dir_path = os.path.dirname(os.path.realpath(__file__))
reg_re_param("audioext", "mp3|wav|ogg|flac")
@rt("/recording/{fname:path}.{ext:audioext}")
def get(fname:str, ext:str): return FileResponse(os.path.join(config['uploads_path'], fname.lstrip('/recording/') + "." + ext))

@rt('/recording/{recording_id:int}')
def get(recording_id: int, t:int=0):
    recording = recordings[recording_id]
    return ApplicationShell(    
        HeadingBlock("Recording", "Listen to a full length recording.", cls="max-w-2xl"),
        Separator(cls="my-6 h-[1px]"),
        Div(
            H3(recording.filename, cls="text-xl font-semibold"),
            Audio(src=f"{recording.filename}#t={t}", controls=True, cls="w-full"),
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

## SEARCH

@rt('/')
def get():
    return ApplicationShell(
        Div(cls="space-y-4 mb-6")(
            H2("Semantic Search", cls="text-2xl font-semibold tracking-tight"),
            Div(cls="text-sm text-muted-foreground space-y-2")(
                P("Find 10-second audio chunks from your archive that are most similar to either:"),
                Ul(cls="list-disc list-inside")(
                    Li(
                        "A natural language description/keywords of the audio you're looking for ",
                        A("(see this link for inspiration)", href="https://huggingface.co/datasets/davidrrobinson/AnimalSpeak", target="_blank", cls="underline")
                    ),
                    Li("An audio clip of the sound you want to find (tag the audio chunk id the search query with @id, e.g. @123),"),
                    Li("The mean embedding of n audio clips of the sound you want to find (tag more than one audio chunks, e.g: @123 @456).")
                )
            )
        ),
        Section(cls="max-w-xl mb-8")(
            Form(id="search-form", cls="flex gap-2", hx_get="/search", hx_target="#search-results", hx_disabled_elt="#search-button")(
                Input(type="text", name="query", placeholder="Search recordings...", cls="w-full"),
                Button(Lucide("search", size=16), "Search", cls="shrink-0 gap-2", id="search-button"),
            )
        ),
        Section(
            Table(TableHeader(TableRow(*[TableHead(col) for col in ["#", "Audio", "Date", "Time", "ID", "Dataset"]])), TableBody(id="search-results")),
            Button("Load more", id="load-more-btn", cls="hidden", hx_swap_oob="true")
        ),
        active_link_id="sidebar-search-link"
    )

@rt('/search')
def get(query:str, page:int=1, pooling:str="mean"):
    per_page = 5

    def embed_str(str: str):
        inputs = processor(text=str, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_embed = model.get_text_features(**inputs)
        return text_embed.numpy()[0].astype(np.float32)

    # Check if the query contains audio chunk IDs
    chunk_ids = re.findall(r'@(\d+)', query)
    
    if chunk_ids:
        # Join the chunk_ids into a comma-separated string
        chunk_ids_str = ','.join(chunk_ids)
        # Fetch embeddings for all mentioned audio chunk IDs
        embeddings = db.t.vec_biolingual(where="audio_chunk_id IN (" + chunk_ids_str + ")")
        
        if embeddings:
            # Convert embeddings to numpy arrays
            embedding_arrays = [np.frombuffer(emb['embedding'], dtype=np.float32) for emb in embeddings]
            
            # Perform pooling based on the specified method
            if pooling == "mean":
                query_embedding = np.mean(embedding_arrays, axis=0)
            elif pooling == "max":
                query_embedding = np.max(embedding_arrays, axis=0)
            elif pooling == "median":
                query_embedding = np.median(embedding_arrays, axis=0)
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")
        else:
            # If no embeddings found, return an empty result
            return []
    else:
        # If it's not audio chunk IDs, embed the query string as before
        query_embedding = embed_str([query])

    def vector_search(qemb:np.array, k:int, page:int, per_page:int):
        offset = (page - 1) * per_page
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
                LIMIT ? OFFSET ?
            """, [qemb, k, per_page, offset])
        )

    k = page * per_page
    matches = vector_search(query_embedding, k=k, page=page, per_page=per_page)

    table_rows = [TableRow(
        TableCell(i+1),
        TableCell(Audio(src=f"data:audio/flac;base64,{load_audio(r['filename'], r['start'], r['end'])}", type="audio/flac", controls=True)),
        TableCell(Div(Lucide("calendar-fold", size=16), (datetime.fromisoformat(r["datetime"])).strftime("%d %b. %Y"), cls="flex gap-1 items-center")),
        TableCell(Div(Lucide("clock", size=16), (datetime.fromisoformat(r["datetime"]) + timedelta(seconds=r["start"])).strftime("%H:%M:%S"), cls="flex gap-1 items-center")),
        TableCell(r['audio_chunk_id']),
        TableCell(Div(r["dataset"], A("Listen in context", href=f"/recording/{r['recording_id']}?t={r['start']}", cls="text-blue-500"), cls="flex flex-col"))
    ) for i, r in enumerate(matches, start=(page-1)*per_page)]

    load_more_button = Button(
        "Load more",
        id="load-more-btn",
        cls="mt-2",
        hx_get=f"/search?query={query}&page={page + 1}",
        hx_target="#search-results",
        hx_disabled_elt="#load-more-btn",
        hx_swap="beforeend",
        hx_swap_oob="true"
    )

    return table_rows, load_more_button

@rt('/admin')
def get():
    checks = [
        ("Audio Chunks", "Check for correct amount of audio chunks per recording"),
        ("Vec Biolingual Rows", "Check for correct amount of vec_biolingual rows per recording"),
        ("Orphaned Audio Files", "Check for audio files on disc without a database 'recordings' row"),
        ("Missing Audio Files", "Check for database 'recordings' rows without a file on disc"),
    ]
    
    return ApplicationShell(
        HeadingBlock("Admin Dashboard", "Validate data and monitor the application"),
        Separator(cls="my-6"),
        Section(cls="space-y-8")(
            *[Card(
                CardHeader(
                    CardTitle(title),
                    CardDescription(description)
                ),
                CardContent(
                    Div(cls="space-y-4")(
                        Button("Run Check", hx_post=f"/admin/check-{title.lower().replace(' ', '-')}", hx_target=f"#{title.lower().replace(' ', '-')}-result"),
                        Div(id=f"{title.lower().replace(' ', '-')}-result", cls="mt-2")
                    )
                ),
                standard=True
            ) for title, description in checks],
            Card(
                CardHeader(
                    CardTitle("App Monitoring"),
                    CardDescription("Monitor application performance and status")
                ),
                CardContent(P("TBD: Monitoring features here.")),
                standard=True
            )
        ),
        active_link_id="sidebar-admin-link"
    )

def create_issue_table(issues, headers):
    return Table(
        TableHeader(TableRow(*[TableHead(header) for header in headers])),
        TableBody(*[
            TableRow(*[
                TableCell(str(issue[key]) if isinstance(issue, dict) else str(issue))
                for key in (issue.keys() if isinstance(issue, dict) else headers)
            ]) for issue in issues
        ]),
        cls="w-full mt-4"
    )

def create_alert(result, headers):
    if result['issues']:
        issue_table = create_issue_table(result['issues'], headers)
        return Alert(
            AlertTitle(result['title']),
            AlertDescription(Div(P(result['description']), issue_table)),
            variant="destructive"
        )
    return Alert(
        AlertTitle(result['title']),
        AlertDescription(result['description']),
        variant="default"
    )

@rt('/admin/check-audio-chunks')
def post():
    return create_alert(check_audio_chunks(), ["ID", "Filename", "Duration", "Actual Chunks", "Expected Chunks"])

@rt('/admin/check-vec-biolingual-rows')
def post():
    return create_alert(check_vec_biolingual_rows(), ["ID", "Filename", "Vec Count", "Chunk Count"])

@rt('/admin/check-orphaned-audio-files')
def post():
    return create_alert(check_orphaned_audio_files(), ["Filename"])

@rt('/admin/check-missing-audio-files')
def post():
    return create_alert(check_missing_audio_files(), ["Filename"])

def check_audio_chunks():
    # there seems to be a bug where recording with a duration
    # that is exactly divisible by 10 is given a full extra chunk
    # when embedding. We account for by adding 1 to the expected
    # chunk count here. But the error should be fixed in the
    # embedding function eventually.
    issues = db.q("""
        SELECT r.id, r.filename, r.duration, COUNT(ac.id) as chunk_count, 
               CASE 
                   WHEN r.duration % 10 = 0 THEN (r.duration / 10) + 1
                   ELSE CEIL(r.duration / 10.0)
               END as expected_chunks
        FROM recordings r
        LEFT JOIN audio_chunks ac ON r.id = ac.recording_id
        GROUP BY r.id
        HAVING chunk_count != expected_chunks AND chunk_count > 0
    """)
    return {
        'title': "Audio Chunks Check",
        'description': f"Found {len(issues)} recordings with incorrect number of audio chunks." if issues else "All recordings have the correct number of audio chunks.",
        'issues': issues
    }

def check_vec_biolingual_rows():
    issues = db.q("""
        SELECT r.id, r.filename, COUNT(vb.audio_chunk_id) as vec_count, 
               COUNT(ac.id) as chunk_count
        FROM recordings r
        LEFT JOIN audio_chunks ac ON r.id = ac.recording_id
        LEFT JOIN vec_biolingual vb ON ac.id = vb.audio_chunk_id
        GROUP BY r.id
        HAVING vec_count != chunk_count
    """)
    return {
        'title': "Vec Biolingual Rows Check",
        'description': f"Found {len(issues)} recordings with mismatched vec_biolingual rows." if issues else "All recordings have the correct number of vec_biolingual rows.",
        'issues': issues
    }

def check_orphaned_audio_files():
    config = load_config()
    uploads_path = config['uploads_path']
    db_files = set(r.filename for r in db.t.recordings())
    disk_files = set(f for f in os.listdir(uploads_path) if f.endswith('.flac'))
    orphaned_files = disk_files - db_files
    return {
        'title': "Orphaned Audio Files Check",
        'description': f"Found {len(orphaned_files)} audio files on disk without corresponding database entries." if orphaned_files else "No orphaned audio files found.",
        'issues': [{'Filename': f} for f in orphaned_files]
    }

def check_missing_audio_files():
    config = load_config()
    uploads_path = config['uploads_path']
    db_files = set(r.filename for r in db.t.recordings())
    disk_files = set(f for f in os.listdir(uploads_path) if f.endswith('.flac'))
    missing_files = db_files - disk_files
    return {
        'title': "Missing Audio Files Check",
        'description': f"Found {len(missing_files)} database entries without corresponding audio files on disk." if missing_files else "No missing audio files found.",
        'issues': [{'Filename': f} for f in missing_files]
    }

serve()