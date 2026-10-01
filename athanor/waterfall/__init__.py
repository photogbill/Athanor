"""The Waterfall — a model's choices, recorded token by token.

A recording holds, for every token a model wrote, the distribution it was
choosing from (the k most probable tokens, the probability left over, the
entropy), the token that was chosen and where it ranked, and when. The
player (``athanor.gui``) draws it the way an RF waterfall draws a band: time
down the screen, candidates across, brightness for probability.

    from athanor.waterfall import attach, read
    with attach(llm) as rec:                       # a llama_cpp.Llama
        out = llm.create_chat_completion(messages)
    path = rec.save(folder, finish_reason=out["choices"][0]["finish_reason"])
    r = read(path)
    r.candidates(0)                                # what it chose from, first token

The format is open and needs only numpy to read: docs/formats/recording.md.
"""

from .attach import RecorderUnavailable, attach, check_binding
from .fileformat import DEFAULT_K, FLAG_CONTROL, FLAG_EOG, FLAG_NONFINITE, FORMAT_VERSION
from .fileformat import RecordingError, step_dtype
from .reading import Recording, list_recordings, read
from .recorder import Recorder


def default_folder():
    """Where recordings go unless told otherwise: the host's data folder."""
    from ..host import get_host
    return get_host().data_dir() / "recordings"


__all__ = ["attach", "check_binding", "RecorderUnavailable", "Recorder", "Recording", "read",
           "list_recordings", "default_folder", "step_dtype", "RecordingError",
           "FORMAT_VERSION", "DEFAULT_K", "FLAG_EOG", "FLAG_CONTROL", "FLAG_NONFINITE"]
