import os
from server import app, initialize_runtime

initialize_runtime(os.environ.get('DEVICE', 'cuda:0'))
