from datetime import datetime, timedelta
from fasthtml.common import *
from fastlite import *
from shad4fast import *
from lucide_fasthtml import Lucide
import re
import soundfile as sf
import os
import asyncio

from process_recording import process_recording
from utils import load_audio, convert_seconds, vector_search

from components import ApplicationShell, DropzoneUploader, SpectrogramPlayer

db = Database("database.db")
datasets, recordings, audio_chunks = db.t.datasets, db.t.recordings, db.t.audio_chunks
if datasets not in db.t:
    datasets.create(id=int, name=str, description=str, pk='id')
    recordings.create(id=int, dataset_id=int, filename=str, duration=int, datetime=str, status=str, pk='id')
    audio_chunks.create(id=int, recording_id=int, start=int, end=float, pk='id')
Dataset, Recording, AudioChunk = datasets.dataclass(), recordings.dataclass(), audio_chunks.dataclass()

app, rt = fast_app(
    pico=False,
    hdrs=(ShadHead(tw_cdn=True, theme_handle=True),Script(src="https://unpkg.com/htmx-ext-sse@2.2.1/sse.js"),),
)

def ProcessButton(recording_id: int, status: str):
    return Button(
        Lucide(
            "play" if status == "unprocessed" else
            "refresh-cw" if status == "processed" else
            "loader-circle",
            size=16,
            cls="animate-spin" if status == "processing" else ""
        ),
        "Reprocess" if status == "processed" else
        "Process" if status == "unprocessed" else
        "Processing",
        variant="outline",
        size="sm",
        cls="gap-2",
        disabled=status == "processing",
        id=f"process-button-{recording_id}",
        hx_post=f"/process/{recording_id}",
        hx_vals=f'{{"recording_id":{recording_id}}}',
        hx_target="this",
        hx_swap="outerHTML",
        hx_ext="sse" if status == "processing" else None,
        sse_connect=f"/status-stream/{recording_id}" if status == "processing" else None,
        sse_swap="message"
    )

@patch
def __ft__(self:Recording):
    return TableRow(
        TableCell(
            Checkbox(
                id="terms",
                name="terms",
                value="agree",
                checked=False,
            )
        ),
        TableCell(A(self.filename, href=f"/recording/{self.id}", cls="hover:underline")),
        TableCell(
            Div(Lucide("calendar-fold", size=16), datetime.fromisoformat(self.datetime).strftime("%d %b. %Y"), cls="flex gap-1 items-center"),
            Div(Lucide("clock", size=16), datetime.fromisoformat(self.datetime).strftime("%H:%M"), cls="flex gap-1 items-center")
        ),
        TableCell(f"{(self.duration // 60) + (1 if self.duration % 60 > 30 else 0)} min"),
        TableCell(ProcessButton(self.id, self.status)),
        TableCell(
            Button(
                Lucide(
                    "eraser",
                    size=16,
                ),
                "Delete",
                variant="destructive",
                size="sm",
                cls="gap-2",
                disabled=self.status == "processing",
                hx_delete=f"/delete-recording",
                hx_vals=f'{{"recording_id":{self.id}}}',
                hx_target="closest tr",
                hx_swap="delete",
                hx_confirm="Are you sure you want to delete this recording? This action cannot be undone."
            )
        ),
    )

@patch
def __ft__(self:Dataset):
    return(
        Tabs(
            TabsList(
                TabsTrigger("Metadata", value="metadata"),
                TabsTrigger("Recordings", value="recordings"),
            ),
            TabsContent(value="metadata")(
                Section(cls='space-y-6 lg:max-w-2xl')(
                    Form(hx_put="/update-dataset", target_id="dataset", cls="space-y-8")(
                        Hidden(name="id", value=self.id),
                        Div(
                            Label('Dataset name', htmlFor="name"),
                            Input(placeholder='Tofte Syd #1', id='name', name='name', value=self.name, required=True),
                            P('This is the public display name of your dataset. Set it arbitrarily to something meaningful.', id='name-description', cls='text-[0.8rem] text-muted-foreground'),
                            cls='space-y-2'
                        ),
                        Div(
                            Label('Dataset description', htmlFor='description'),
                            Textarea(self.description, placeholder='For example used to elaborate on recording location or for notes on sound quality.', name='description', id='description'),
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
                    Button('Delete dataset', variant="destructive", hx_delete=f"/delete-dataset", hx_vals=f'{{"dataset_id":{self.id}}}', hx_confirm="Are you sure you want to delete this dataset? This action cannot be undone."),
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
                                TableRow(TableHead(Checkbox(id="select-all", name="select-all")), *[TableHead(header) for header in ["Filename", "Starts at", "Duration", "Process", "Delete"]])
                            ),
                            TableBody(
                                *[recording for recording in recordings(where="dataset_id=?", where_args=[self.id])],
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
                    DropzoneUploader(self.id),
                ),
            ),
            standard=True,
            cls="w-full",
        ),
    )

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
        dataset = P(f"Dataset with id {dataset_id} not found.")

    return ApplicationShell(
        Div(dataset, id="dataset", cls='flex-1'),
        active_link_id=dataset_id
    )

@rt('/new-dataset')
def post():
    dataset = datasets.insert(name="Untitled dataset")
    return Redirect(f"/datasets/{dataset.id}")

@rt('/update-dataset')
def put(dataset: Dataset):
    new_dataset = datasets.update(dataset)
    return new_dataset, Button(new_dataset.name, variant="secondary", cls="w-full !justify-start", id=f"dataset-link-{new_dataset.id}", hx_swap_oob="true")

@rt('/delete-dataset')
def delete(dataset_id: int):
    datasets.delete(dataset_id)
    return Redirect('/datasets')

### RECORDINGS ###

# Serve .flac files in the uploads directory
@app.get("/uploads/{fname}.flac")
def serve_flac(request, fname: str):
    file_path = f'uploads/{fname}.flac'
    file_size = os.path.getsize(file_path)
    range_header = request.headers.get('Range', None)
    
    if range_header:
        byte1, byte2 = 0, None
        m = re.search(r'(\d+)-(\d*)', range_header)
        if m:
            byte1, byte2 = int(m.group(1)), m.group(2)
            if byte2:
                byte2 = int(byte2)
        
        length = (byte2 or file_size - 1) - byte1 + 1
        with open(file_path, 'rb') as f:
            f.seek(byte1)
            data = f.read(length)
        
        headers = {
            'Content-Range': f'bytes {byte1}-{byte1 + length - 1}/{file_size}',
            'Accept-Ranges': 'bytes',
            'Content-Length': str(length),
            'Content-Type': 'audio/flac',
        }
        return Response(data, status_code=206, headers=headers)
    
    return FileResponse(file_path, media_type='audio/flac')

@rt('/recording/{recording_id:int}')
def get(recording_id: int):
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
            #AudioPlayer(f"/uploads/{recording.filename}"),
            SpectrogramPlayer(f"/uploads/{recording.filename}"),
            cls="space-y-4"
        ),
        active_link_id=recording.dataset_id
    )

@rt('/upload-recording')
async def post(request):
    form = await request.form()
    file = form.get('file')
    statusId = form.get('statusId')
    dataset_id = form.get('dataset_id')
    filename = file.filename
        
    # Validate filename format
    if not re.match(r'^[A-Za-z0-9]+_\d{8}_\d{6}\.flac$', filename):
        return "", Span(f'Invalid filename format', cls='text-xs font-medium text-red-600', id=statusId, hx_swap_oob="true")

    # Check if the filename already exists in the 'recordings' table
    existing_file = recordings(where="filename = ?", where_args=[filename])
    if existing_file:
        return "", Span(f'File already exists', cls='text-xs font-medium text-red-600', id=statusId, hx_swap_oob="true")

    # Extract information and save file
    recorder_id, date, start_time = filename[:-5].split('_')
    datetime_obj = datetime.strptime(f"{date}_{start_time}", "%Y%m%d_%H%M%S")
    datetime_str = datetime_obj.isoformat(timespec='seconds')
    file_path = os.path.join('uploads', filename)
    
    # Manually save the file
    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Get audio duration
    with sf.SoundFile(file_path) as audio_file:
        duration_seconds = len(audio_file) // audio_file.samplerate

    # Insert into recordings table
    try:
        uploaded_file = recordings.insert({
            'dataset_id': dataset_id,
            'filename': filename,
            'duration': duration_seconds,
            'datetime': datetime_str,
            'status': 'unprocessed'
        })
        return uploaded_file, Span(f'Uploaded', cls='text-xs font-medium text-green-600', id=statusId, hx_swap_oob="true")
    except Exception as e:
        return "", Span(f'Database insert error: {str(e)}', cls='text-xs font-medium text-red-600', id=statusId, hx_swap_oob="true")

@rt('/delete-recording')
def delete(recording_id: int):
    recordings.delete(recording_id)
    return ""

@rt('/process/{recording_id:int}')
def post(recording_id: int):
    recording = recordings[recording_id]
    recording.status = "processing"
    recordings.update(recording)

    # start processing in a new thread
    process_recording(recording_id)

    return ProcessButton(recording_id, recording.status)

async def status_generator(recording_id: int):
    while True:
        recording = recordings[recording_id]
        if recording.status != "processing":
            yield sse_message(ProcessButton(recording_id, recording.status))
            break
        await asyncio.sleep(5)  # Check every 5 seconds

@rt("/status-stream/{recording_id:int}")
async def get(recording_id: int):
    return EventStream(status_generator(recording_id))

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
                        SelectTrigger(
                            SelectValue(placeholder="No. results"),
                            cls="!w-[140px] gap-2"
                        ),
                        SelectContent(
                            SelectGroup(
                                SelectLabel("No. results"),
                                SelectItem("5", value=5),
                                SelectItem("10", value=10),
                                SelectItem("20", value=20),
                                SelectItem("50", value=50),
                            ),
                            id="k",
                        ),
                        standard=True,
                        id="k",
                        name="k",
                        cls="shrink-0"
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
    matches = vector_search(query, k=k)
    #print(list(matches))
    #print("heyoa")

    table_rows = [TableRow(
        TableCell(i+1),
        TableCell(Audio(src=f"data:audio/flac;base64,{load_audio(r['filename'], r['start'], r['end'])}", type="audio/flac", controls=True)),
        TableCell(Div(Lucide("calendar-fold", size=16), (datetime.fromisoformat(r["datetime"])).strftime("%d %b. %Y"), cls="flex gap-1 items-center")),
        TableCell(Div(Lucide("clock", size=16), (datetime.fromisoformat(r["datetime"]) + timedelta(seconds=r["start"])).strftime("%H:%M:%S"), cls="flex gap-1 items-center")),
        TableCell(r["dataset"])
    ) for i, r in enumerate(matches)]

    return table_rows


serve()