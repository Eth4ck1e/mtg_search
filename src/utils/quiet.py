"""Suppress known-benign upstream warnings from model-loading libraries.

Import this module (side-effect on import) at the top of any script that loads
a HuggingFace model via sentence-transformers.

Two known-benign warnings are silenced here — neither is actionable in our code:

1. ``huggingface_hub`` — "You are sending unauthenticated requests to the HF Hub".
   Soft nag suggesting HF_TOKEN. Model is cached locally; the check works fine
   without auth. Set HF_TOKEN in .env if you want faster downloads for future
   model swaps.

2. ``transformers`` — "Detected the usage of `get_extended_attention_mask`".
   Nomic Embed's custom pooling code uses a transformers API deprecated in
   v5.12+. Waiting on Nomic to ship an updated model checkpoint. Not our code.

Both are emitted via each library's own logger (not Python's ``warnings`` module),
so the fix is to lower those loggers' verbosity — not ``warnings.filterwarnings``.

If either warning becomes a real signal (e.g., transformers actually removes
the API and the model breaks), remove the filter here so the failure surfaces.
"""

from __future__ import annotations

import logging

# transformers has its own logging module; set_verbosity_error() suppresses
# WARNING-level messages including the get_extended_attention_mask deprecation.
try:
    import transformers

    transformers.logging.set_verbosity_error()
except ImportError:
    pass

# huggingface_hub emits the unauthenticated-requests notice via its own logger.
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
