import backend.ipv4_patch  # force IPv4
from backend import _ipv4_only  # Force IPv4 early

from backend.app_factory import create_app
app = create_app()

