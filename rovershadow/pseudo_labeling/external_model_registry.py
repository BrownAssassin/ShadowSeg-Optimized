"""Registry and resolver for external shadow model checkpoints."""

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ExternalCheckpointProvider:
    """Description of one external-checkpoint provider endpoint."""

    provider: str
    model_key: str
    url: str
    filename: str
    sha256: str | None = None


def _default_provider_registry() -> list[ExternalCheckpointProvider]:
    """Build provider list in priority order (A->B->C)."""
    mtmt_url = os.environ.get(
        "ROVERSHADOW_MTMT_URL",
        "https://example.invalid/mtmt-net-shadow/mtmt_shadow.pth",
    )
    dhan_url = os.environ.get(
        "ROVERSHADOW_DHAN_URL",
        "https://example.invalid/dhan-shadow/dhan_shadow.pth",
    )
    mirror_url = os.environ.get(
        "ROVERSHADOW_SHADOW_MIRROR_URL",
        "https://example.invalid/rovershadow/mirror_shadow.pth",
    )
    return [
        ExternalCheckpointProvider(
            provider="provider_a_mtmt_official",
            model_key="mtmt",
            url=mtmt_url,
            filename="mtmt_shadow.pth",
            sha256=os.environ.get("ROVERSHADOW_MTMT_SHA256"),
        ),
        ExternalCheckpointProvider(
            provider="provider_b_dhan_official",
            model_key="dhan",
            url=dhan_url,
            filename="dhan_shadow.pth",
            sha256=os.environ.get("ROVERSHADOW_DHAN_SHA256"),
        ),
        ExternalCheckpointProvider(
            provider="provider_c_project_mirror",
            model_key="mirror",
            url=mirror_url,
            filename="mirror_shadow.pth",
            sha256=os.environ.get("ROVERSHADOW_SHADOW_MIRROR_SHA256"),
        ),
    ]


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return SHA256 checksum for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksum(path: Path, expected_sha256: str | None) -> bool:
    """Return True when checksum matches or when no checksum is provided."""
    if expected_sha256 is None:
        return True
    actual = _sha256_file(path)
    return actual.lower() == expected_sha256.lower()


def _download_file(url: str, out_path: Path, timeout_sec: int) -> None:
    """Download URL to path with bounded timeout and atomic rename."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with urllib.request.urlopen(url, timeout=timeout_sec) as response:
        with tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    tmp_path.replace(out_path)


def _iter_candidates(
    providers: Iterable[ExternalCheckpointProvider],
    external_model: str,
) -> Iterable[ExternalCheckpointProvider]:
    """Yield providers filtered by model preference while preserving order."""
    if external_model == "auto":
        yield from providers
        return
    for provider in providers:
        if provider.model_key == external_model:
            yield provider


def resolve_external_checkpoint(
    cache_dir: Path,
    external_model: str = "auto",
    explicit_checkpoint: Path | None = None,
    timeout_sec: int = 20,
    simulate_download_failure: bool = False,
) -> tuple[Path | None, str]:
    """Resolve an external checkpoint path.

    Returns:
        tuple[path|None, str]: checkpoint path if available and status label.
    """
    if explicit_checkpoint is not None:
        if not explicit_checkpoint.is_file():
            raise FileNotFoundError(f"External checkpoint not found: {explicit_checkpoint}")
        return explicit_checkpoint, "explicit"

    providers = _default_provider_registry()
    for provider in _iter_candidates(providers, external_model):
        out_path = cache_dir / provider.filename
        if out_path.is_file() and _verify_checksum(out_path, provider.sha256):
            return out_path, f"cache:{provider.provider}"

        if simulate_download_failure:
            print(f"[WARN] Simulating download failure for {provider.provider}")
            continue

        try:
            print(f"[INFO] Downloading external checkpoint from {provider.provider} ...")
            _download_file(provider.url, out_path, timeout_sec=timeout_sec)
            if not _verify_checksum(out_path, provider.sha256):
                print(f"[WARN] SHA256 mismatch for {provider.provider}; skipping checkpoint.")
                out_path.unlink(missing_ok=True)
                continue
            return out_path, f"download:{provider.provider}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"[WARN] Failed provider {provider.provider}: {exc}")
            continue

    return None, "unavailable"
