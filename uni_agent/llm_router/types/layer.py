"""Cache layer constants — backend-agnostic canonical layer names.

Backend-specific medium strings (e.g. vLLM's ``"GPU"``/``"cpu"``) are mapped to
these constants at each backend's decoder boundary. Downstream store and
strategy layers reference cache layers via these constants — never raw backend
strings.
"""

from __future__ import annotations

from enum import Enum


class Layer(str, Enum):
    """Canonical cache-layer names (backend-agnostic).

    Inherits ``str`` so members interoperate with plain strings: YAML-loaded
    ``layer_weights`` keys (``"gpu"``) index the same dict slot as ``Layer.GPU``,
    and ``Layer.GPU == "gpu"`` holds for set/validation comparisons.
    """

    GPU = "gpu"  # GPU — local reverse index
    CPU = "cpu"  # CPU (e.g. mooncake L2)
    SSD = "ssd"  # SSD
