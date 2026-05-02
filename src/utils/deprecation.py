from __future__ import annotations

import warnings


def warn_legacy_entrypoint(script_name: str, replacement: str) -> None:
    warnings.warn(
        f"{script_name} is a deprecated legacy entrypoint. Use {replacement} instead.",
        DeprecationWarning,
        stacklevel=2,
    )
