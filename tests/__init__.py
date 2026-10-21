import os
import tempfile

# Every test in a Relay directory of its own: nothing a test runs writes to the user's
# (~/.config/relay: the harnesses installed, the conversations noted).
os.environ.setdefault("RELAY_HOME", tempfile.mkdtemp(prefix="relay-tests-"))
