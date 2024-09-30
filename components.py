from fasthtml.common import *
from shad4fast import *
from lucide_fasthtml import Lucide

db = database("database.db")
datasets = db.t.datasets
Dataset = datasets.dataclass()

def HeadingBlock(title: str, subtitle: str, **kwargs):
    cls_value = kwargs.pop('cls', '') + ' space-y-1'
    return Div(cls=cls_value, **kwargs)(
        H2(title, cls="text-2xl font-semibold tracking-tight"),
        P(subtitle, cls="text-sm text-muted-foreground")
    )

def Navbar():
    return Header(cls="container mx-auto flex justify-between items-center py-2 border-b h-10")(
        A("AnimalAcoustics", href="/", cls="font-semibold")
    )

def Sidebar(active_link_id):
    return Aside(cls="pb-16 pt-4 space-y-4 border-r self-start sticky h-screen overflow-y-auto top-0")(
        Header(cls="px-3 mb-8")(
            H2("🦜", Span("AnimalAcoustics", cls="ml-2 bg-clip-text text-transparent bg-gradient-to-r from-green-400 to-yellow-500"), cls="px-4 text-lg font-bold tracking-tight"),
        ),
        Section(cls="px-3 py-2")(
            H2("Library", cls="mb-2 px-4 text-lg font-semibold tracking-tight"),
            Nav(cls='flex flex-col space-y-1')(
                A(Button(Lucide("binoculars", size=16), Span("Semantic Search", cls="truncate"), variant="secondary" if "sidebar-search-link" == active_link_id else "ghost", cls="w-full !justify-start gap-2"), href="/"),
                A(Button(Lucide("audio-lines", size=16), Span("Datasets", cls="truncate"), variant="secondary" if "sidebar-datasets-link" == active_link_id else "ghost", cls="w-full !justify-start gap-2"), href="/datasets"),
                A(Button(Lucide("settings", size=16), Span("Admin", cls="truncate"), variant="secondary" if "sidebar-admin-link" == active_link_id else "ghost", cls="w-full !justify-start gap-2"), href="/admin"),
            ),
        ),
        Section(cls="px-3 py-2")(
            H2("Datasets", cls="mb-2 px-4 text-lg font-semibold tracking-tight"),
            Nav(cls="flex space-x-2 lg:flex-col lg:space-x-0 lg:space-y-1")(
                *[A(Button(Lucide("list-music", size=16), Span(ds.name, cls="truncate"), variant="secondary" if ds.id == active_link_id else "ghost", cls="!justify-start gap-2 w-full", id=f"dataset-link-{ds.id}", title=ds.name), href=f"/datasets/{ds.id}") for ds in datasets()],
            ),
            Button(Lucide("circle-plus", size=16), 'New dataset', variant="outline", size="sm", cls="gap-2 mt-2 mx-4", hx_post="/new-dataset"),
        ),
    )

def ApplicationShell(*args, active_link_id):
    return (
        #Navbar(),
        Div(cls="grid lg:grid-cols-5")(
            Sidebar(active_link_id=active_link_id),
            Div(cls="col-span-4")(
                Div(cls="h-full px-4 py-6 lg:px-8")(*args)
            )
        )
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
        # hx_ext="sse" if status == "processing" else None,
        # sse_connect=f"/status-stream/{recording_id}" if status == "processing" else None,
        # sse_swap="message"
    )

def DropzoneUploader(dataset_id):
    return Div(id='upload-zone', cls='max-w-md')(
        Div(id='dropzone', cls='border-2 border-dashed border-gray-300 rounded-lg p-8 text-center cursor-pointer hover:border-blue-500 transition duration-300')(
            P('Drag & drop .flac files here or click to select', cls='text-gray-600'),
            Input(type='hidden', id='dataset-id', value=dataset_id),
            Input(type='file', 
                  id='file-input', 
                  name='files[]', 
                  multiple='true', 
                  accept='.flac', 
                  cls='hidden')
        ),
        Div(id='file-list', cls='mt-4'),
        Script("""
            document.addEventListener('DOMContentLoaded', function() {
                const dropzone = document.getElementById('dropzone');
                const fileInput = document.getElementById('file-input');
                const fileList = document.getElementById('file-list');
                const datasetId = document.getElementById('dataset-id').value;
                const MAX_CONCURRENT_UPLOADS = 3;
                let activeUploads = 0;
                let uploadQueue = [];

                function processQueue() {
                    while (activeUploads < MAX_CONCURRENT_UPLOADS && uploadQueue.length > 0) {
                        const {file, statusId} = uploadQueue.shift();
                        uploadFile(file, statusId);
                        activeUploads++;
                    }
                }

                function handleFiles(files) {
                    Array.from(files).filter(file => file.name.endsWith('.flac')).forEach(file => {
                        const statusId = 'status-' + Math.random().toString(36).substr(2, 9);
                        const fileItem = createFileItem(file, statusId, 'Queued');
                        fileList.appendChild(fileItem);
                        uploadQueue.push({file, statusId});
                    });
                    processQueue();
                }

                function uploadFile(file, statusId) {
                    const fileItem = document.getElementById(`container-${statusId}`);
                    updateStatus(fileItem, 'Uploading');

                    htmx.on(fileItem, 'htmx:xhr:progress', function(evt) {
                        updateProgress(fileItem, evt.detail.loaded / evt.detail.total * 100);
                    });

                    const formData = new FormData();
                    formData.append('file', file);
                    formData.append('statusId', statusId);
                    formData.append('dataset_id', datasetId);

                    htmx.ajax('POST', '/upload-recording', {
                        target: '#file-rows',
                        swap: 'beforeend',
                        values: formData,
                        source: fileItem,
                    }).then(() => {
                        updateProgress(fileItem, 100);
                        activeUploads--;
                        processQueue();
                    }).catch(() => {
                        updateStatus(fileItem, 'Error');
                        activeUploads--;
                        processQueue();
                    });
                }

                function createFileItem(file, statusId, initialStatus) {
                    const fileItem = document.createElement('div');
                    fileItem.className = 'mt-4 p-4 bg-gray-50 rounded-lg';
                    fileItem.id = `container-${statusId}`;
                    fileItem.setAttribute('hx-encoding', 'multipart/form-data');
                    fileItem.innerHTML = `
                        <div class="flex justify-between items-center mb-2">
                            <span class="text-sm font-medium text-gray-900">${file.name}</span>
                            <span class="text-xs font-medium text-gray-500 status" hx-swap-oob="true" id="${statusId}">${initialStatus}</span>
                        </div>
                        <div class="w-full bg-gray-200 rounded-full h-2.5">
                            <div class="bg-blue-600 h-2.5 rounded-full progress-bar" style="width: 0%"></div>
                        </div>
                    `;
                    return fileItem;
                }

                function updateProgress(fileItem, percent) {
                    const progressBar = fileItem.querySelector('.progress-bar');
                    progressBar.style.width = `${percent}%`;
                }

                function updateStatus(fileItem, status) {
                    fileItem.querySelector('.status').textContent = status;
                }

                dropzone.addEventListener('click', () => fileInput.click());
                fileInput.addEventListener('change', (e) => handleFiles(e.target.files));
                dropzone.addEventListener('dragover', (e) => {
                    e.preventDefault();
                    dropzone.classList.add('border-blue-500');
                });
                dropzone.addEventListener('dragleave', () => {
                    dropzone.classList.remove('border-blue-500');
                });
                dropzone.addEventListener('drop', (e) => {
                    e.preventDefault();
                    dropzone.classList.remove('border-blue-500');
                    handleFiles(e.dataTransfer.files);
                });
            });
        """)
    )