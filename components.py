from fastlite import *
from fasthtml.common import *
from shad4fast import *
from lucide_fasthtml import Lucide

db = Database("database.db")
datasets = db.t.datasets
Dataset = datasets.dataclass()

def ThemeToggle(variant="outline", cls=None, **kwargs):
    return Button(
        Lucide("sun", cls="dark:flex hidden", size=16),
        Lucide("moon", cls="dark:hidden", size=16),
        variant=variant,
        size="icon",
        cls=f"theme-toggle {cls}",
        **kwargs,
    )

def Navbar():
    return Header(cls="container mx-auto flex justify-between items-center py-2 border-b border-border")(
        A("AnimalAcoustics", href="/", cls="font-semibold"),
        ThemeToggle()
    )

def Sidebar(active_link_id):
    return Aside(cls='lg:w-1/5')(
        Div(cls="mb-6")(
            H2("Library", cls="mb-2 px-4 text-lg font-semibold tracking-tight"),
            Nav(cls='flex space-x-2 lg:flex-col lg:space-x-0 lg:space-y-1')(
                A(Button(Lucide("binoculars", size=16), Span("Search", cls="truncate"), variant="secondary" if "sidebar-search-link" == active_link_id else "ghost", cls="w-full !justify-start gap-2"), href="/"),
                A(Button(Lucide("audio-lines", size=16), Span("Datasets", cls="truncate"), variant="secondary" if "sidebar-datasets-link" == active_link_id else "ghost", cls="w-full !justify-start gap-2"), href="/datasets"),
            ),
        ),
        Div(cls="mb-2")(
            H2("Datasets", cls="mb-2 px-4 text-lg font-semibold tracking-tight"),
            Nav(cls='flex space-x-2 lg:flex-col lg:space-x-0 lg:space-y-1')(
                *[A(Button(Lucide("list-music", size=16), Span(ds.name, cls="truncate"), variant="secondary" if ds.id == active_link_id else "ghost", cls="!justify-start gap-2 w-full", id=f"dataset-link-{ds.id}", title=ds.name), href=f"/datasets/{ds.id}") for ds in datasets()],
            ),
        ),
        Button(Lucide("circle-plus", size=16), 'New dataset', variant="outline", size="sm", cls="gap-2", hx_post="/new-dataset"),
    )

def ApplicationShell(*args, active_link_id, **kwargs):
    return (
        Navbar(), 
        Div(cls="container mx-auto my-8")(
            Div(cls="flex flex-col space-y-8 lg:flex-row lg:space-x-12 lg:space-y-0")(
                Sidebar(active_link_id=active_link_id),
                Div(cls="flex-1")(
                    *args
                ),
            ),
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
        hx_ext="sse" if status == "processing" else None,
        sse_connect=f"/status-stream/{recording_id}" if status == "processing" else None,
        sse_swap="message"
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
               
                function handleFiles(fileList) {
                    const files = Array.from(fileList).filter(file => file.name.endsWith('.flac'));
                    files.forEach(uploadFile);
                }

                function uploadFile(file) {
                    const statusId = 'status-' + Math.random().toString(36).substr(2, 9);
                    const fileItem = createFileItem(file, statusId);
                    fileList.appendChild(fileItem);
                    newFile = document.getElementById(`container-${statusId}`);
               
                    // add a click listener to the given div
                    htmx.on(fileItem, 'htmx:xhr:progress', function(evt){
                        //console.log(evt);
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
                        //fileItem.querySelector('.status').textContent = 'Checking..';
                        updateProgress(fileItem, 100);
                    }).catch(() => {
                        fileItem.querySelector('.status').textContent = 'Error';
                    });
                }

                function createFileItem(file, statusId) {
                    const fileItem = document.createElement('div');
                    fileItem.className = 'mt-4 p-4 bg-gray-50 rounded-lg';
                    fileItem.id = `container-${statusId}`;
                    fileItem.setAttribute('hx-encoding', 'multipart/form-data');
                    fileItem.innerHTML = `
                        <div
                            class="flex justify-between items-center mb-2" 
                        ">
                            <span class="text-sm font-medium text-gray-900">${file.name}</span>
                            <span class="text-xs font-medium text-gray-500 status" hx-swap-oob="true" id="${statusId}">Uploading...</span>
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
            });
        """)
    )

def AudioPlayer(fpath):
    return (
        Audio(
            id="audioPlayer",
            controls=True,
            src=fpath,
            cls="w-full"
        ),
        Button(Lucide("play", size=16), id="playBtn"),
        Button(Lucide("pause", size=16), id="pauseBtn"),
        Script("""
            const audioPlayer = document.getElementById('audioPlayer');
            const playBtn = document.getElementById('playBtn');
            const pauseBtn = document.getElementById('pauseBtn');

            playBtn.addEventListener('click', () => {
                audioPlayer.play();
            });

            pauseBtn.addEventListener('click', () => {
                audioPlayer.pause();
            });
        """)
    )

def SpectrogramPlayer(fpath):
    """
    Script(
        const audio = document.getElementById('audio');
        const spectrogram = document.getElementById('spectrogram');
        const playPauseBtn = document.getElementById('playPauseBtn');
        const seekBar = document.getElementById('seekBar');
        const timeDisplay = document.getElementById('timeDisplay');

        let isPlaying = false;

        // Set up Web Audio API
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        const audioContext = new AudioContext();
        const source = audioContext.createMediaElementSource(audio);
            
        // Create analyser
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        source.connect(analyser);
        analyser.connect(audioContext.destination);

        // Canvas setup
        const canvasCtx = spectrogram.getContext('2d');
        const WIDTH = spectrogram.width;
        const HEIGHT = spectrogram.height;
        let drawWidth = 2; // Width of each frequency slice
        let currentX = 0;

        // Initialize canvas
        canvasCtx.fillStyle = 'rgb(0, 0, 0)';
        canvasCtx.fillRect(0, 0, WIDTH, HEIGHT);

        // Play/Pause functionality
        playPauseBtn.addEventListener('click', function() {
            if (isPlaying) {
                audio.pause();
                playPauseBtn.textContent = 'Play';
                isPlaying = false;
            } else {
                audioContext.resume(); // Required for some browsers
                audio.play();
                playPauseBtn.textContent = 'Pause';
                isPlaying = true;
                requestAnimationFrame(drawSpectrogram);
            }
        });

        // Update seek bar as the audio plays
        audio.addEventListener('timeupdate', function() {
            const current = audio.currentTime;
            const duration = audio.duration;
            seekBar.value = (current / duration) * 100;
            updateTimeDisplay(current, duration);
        });

        // Seek functionality
        seekBar.addEventListener('input', function() {
            const duration = audio.duration;
            audio.currentTime = (seekBar.value / 100) * duration;
        });

        // Update time display
        function updateTimeDisplay(current, duration) {
            const currentMinutes = Math.floor(current / 60);
            const currentSeconds = Math.floor(current % 60);
            const durationMinutes = Math.floor(duration / 60);
            const durationSeconds = Math.floor(duration % 60);
            timeDisplay.textContent = 
                `${currentMinutes}:${currentSeconds.toString().padStart(2, '0')} / ` +
                `${durationMinutes}:${durationSeconds.toString().padStart(2, '0')}`;
        }

        // Initialize time display once metadata is loaded
        audio.addEventListener('loadedmetadata', function() {
            updateTimeDisplay(0, audio.duration);
        });

        // Spectrogram rendering
        function drawSpectrogram() {
            if (!isPlaying) return;

            requestAnimationFrame(drawSpectrogram);
            analyser.getByteFrequencyData(dataArray);

            // Shift the canvas to the left
            const imageData = canvasCtx.getImageData(drawWidth, 0, WIDTH - drawWidth, HEIGHT);
            canvasCtx.fillStyle = 'rgb(0, 0, 0)';
            canvasCtx.fillRect(0, 0, WIDTH, HEIGHT);
            canvasCtx.putImageData(imageData, 0, 0);

            // Draw the new frequency data as the rightmost column
            for (let y = 0; y < HEIGHT; y++) {
                const frequencyIndex = Math.floor((y / HEIGHT) * bufferLength);
                const value = dataArray[frequencyIndex];
                const color = `rgb(${value}, ${value}, ${value})`; // Grayscale, can be customized
                canvasCtx.fillStyle = color;
                canvasCtx.fillRect(WIDTH - drawWidth, HEIGHT - y, drawWidth, 1);
            }
        }
    )
    """
    return (
        Div(cls='audio-player space-y-4')(
            Canvas(id='spectrogram', width='800', height='200', cls='w-full h-48 bg-gray-900 rounded-lg'),
            Div(cls='controls flex items-center space-x-4')(
                Button('Play', id='playPauseBtn', cls='px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600'),
                Input(type='range', id='seekBar', min='0', value='0', cls='flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer'),
                Span('0:00 / 0:00', id='timeDisplay', cls='text-gray-500')
            ),
            Audio(id='audio', src=fpath, cls='hidden')
        ),
        Script("""
            import WaveSurfer from 'https://unpkg.com/wavesurfer.js';
            import Spectrogram from 'https://unpkg.com/wavesurfer.js/dist/plugins/spectrogram.js';

            const wavesurfer = WaveSurfer.create({
                container: '#waveform',
                waveColor: 'violet',
                progressColor: 'purple',
                plugins: [
                    Spectrogram.create({
                        container: '#spectrogram'
                    })
                ]
            });

            document.getElementById('file-input').addEventListener('change', function(event) {
                const file = event.target.files[0];
                const url = URL.createObjectURL(file);
                wavesurfer.load(url);
            });
        """)
    )