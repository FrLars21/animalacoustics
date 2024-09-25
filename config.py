# Default configuration
config = {
    "uploads_path": "uploads"
}

def load_config(file_path=None):
    """
    Load configuration from a file if provided, otherwise use default.
    """
    global config
    if file_path:
        # Here you can add logic to load from a file if needed
        pass
    return config