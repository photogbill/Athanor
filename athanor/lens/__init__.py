"""The lens — every layer's output, read as the words it would say (M1, M30).

    python -m athanor record model.gguf --prompt "…" --tap residual
    python -m athanor lens  recordings/<that recording>.athrec-meta

    from athanor import lens, waterfall
    lens.run(path)                       # writes <stem>.athrec-lens beside the recording
    waterfall.read(path).lens.depth      # decision depth, a layer number per token

Three parts: ``unembed`` (the model's final norm and output matrix, decoded
once from the GGUF and cached under the model's fingerprint), ``run`` (the
pass over a recording's residual stream) and ``reading`` (the track read
back; ``fileformat`` is its layout). Needs numpy and the model file once;
never llama.cpp.
"""

from .fileformat import FLAG_NO_RESIDUAL, FLAG_TOP_DISAGREES, LENS_META_SUFFIX, LENS_SUFFIX
from .reading import LensData, read
from .run import DEFAULT_K, parse_layers, run
from .unembed import LensUnavailable, Unembedding, build as build_unembedding, fingerprint
from .unembed import for_recording as unembedding_for, load as load_unembedding

__all__ = ["run", "read", "LensData", "LensUnavailable", "Unembedding", "build_unembedding",
           "load_unembedding", "unembedding_for", "fingerprint", "parse_layers", "DEFAULT_K",
           "LENS_SUFFIX", "LENS_META_SUFFIX", "FLAG_NO_RESIDUAL", "FLAG_TOP_DISAGREES"]
