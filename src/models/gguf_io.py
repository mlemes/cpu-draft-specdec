"""Minimal GGUF v3 reader, writer, and inspector (Python stdlib only)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import BinaryIO, Callable, Iterable, Sequence

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

# Metadata value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

_SCALAR_STRUCTS: dict[int, struct.Struct] = {
    GGUF_TYPE_UINT8: struct.Struct("<B"),
    GGUF_TYPE_INT8: struct.Struct("<b"),
    GGUF_TYPE_UINT16: struct.Struct("<H"),
    GGUF_TYPE_INT16: struct.Struct("<h"),
    GGUF_TYPE_UINT32: struct.Struct("<I"),
    GGUF_TYPE_INT32: struct.Struct("<i"),
    GGUF_TYPE_FLOAT32: struct.Struct("<f"),
    GGUF_TYPE_BOOL: struct.Struct("<B"),
    GGUF_TYPE_UINT64: struct.Struct("<Q"),
    GGUF_TYPE_INT64: struct.Struct("<q"),
    GGUF_TYPE_FLOAT64: struct.Struct("<d"),
}

_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")
_HEADER_STRUCT = struct.Struct("<4sIQQ")

# ggml tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q8_0 = 8

GGML_TYPE_NAMES: dict[int, str] = {
    GGML_TYPE_F32: "F32",
    GGML_TYPE_F16: "F16",
    GGML_TYPE_Q4_0: "Q4_0",
    GGML_TYPE_Q8_0: "Q8_0",
}

GGML_NAME_TO_ID: dict[str, int] = {v: k for k, v in GGML_TYPE_NAMES.items()}

# (block_size_elements, block_nbytes)
_TYPE_LAYOUT: dict[str, tuple[int, int]] = {
    "F32": (1, 4),
    "F16": (1, 2),
    "Q4_0": (32, 18),
    "Q8_0": (32, 34),
}


@dataclass(frozen=True)
class TensorInfo:
    """Metadata for a single tensor in a GGUF file."""

    name: str
    shape: tuple[int, ...]
    ggml_type: str
    offset: int
    nbytes: int


@dataclass(frozen=True)
class GGUFFile:
    """Parsed GGUF v3 header, metadata, and tensor table."""

    version: int
    alignment: int
    metadata: dict[str, object]
    tensors: list[TensorInfo]
    data_offset: int


def _align_up(offset: int, alignment: int) -> int:
    return ((offset + alignment - 1) // alignment) * alignment


def tensor_nbytes(shape: Sequence[int], ggml_type: str | int) -> int:
    """Return byte size of a tensor in ggml shape order (ne0 first)."""
    if isinstance(ggml_type, int):
        if ggml_type not in GGML_TYPE_NAMES:
            raise ValueError(f"Unsupported ggml type ID: {ggml_type}")
        type_name = GGML_TYPE_NAMES[ggml_type]
    else:
        type_name = ggml_type.upper()

    if type_name not in _TYPE_LAYOUT:
        raise ValueError(f"Unsupported ggml type: {ggml_type!r}")

    if not shape:
        raise ValueError("Tensor shape must be non-empty")
    for dim in shape:
        if dim < 0:
            raise ValueError(f"Negative dimension in shape: {shape}")

    block_elems, block_bytes = _TYPE_LAYOUT[type_name]
    if block_elems > 1 and shape[0] % block_elems != 0:
        raise ValueError(
            f"Quantized type {type_name} requires ne0 ({shape[0]}) "
            f"to be a multiple of {block_elems}"
        )

    total_elems = math.prod(shape)
    return (total_elems // block_elems) * block_bytes


def _read_exact(f: BinaryIO, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise ValueError(f"Unexpected EOF: expected {n} bytes, got {len(data)}")
    return data


def _read_string(f: BinaryIO) -> str:
    (length,) = _U64.unpack(_read_exact(f, 8))
    raw = _read_exact(f, length)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Invalid UTF-8 string in GGUF: {exc}") from exc


def _read_value(f: BinaryIO, vtype: int) -> object:
    if vtype == GGUF_TYPE_STRING:
        return _read_string(f)
    if vtype == GGUF_TYPE_BOOL:
        (raw_bool,) = _SCALAR_STRUCTS[GGUF_TYPE_BOOL].unpack(_read_exact(f, 1))
        return bool(raw_bool)
    if vtype in _SCALAR_STRUCTS:
        st = _SCALAR_STRUCTS[vtype]
        (val,) = st.unpack(_read_exact(f, st.size))
        return val
    if vtype == GGUF_TYPE_ARRAY:
        (elem_type,) = _U32.unpack(_read_exact(f, 4))
        (count,) = _U64.unpack(_read_exact(f, 8))
        return [_read_value(f, elem_type) for _ in range(count)]
    raise ValueError(f"Unknown GGUF metadata value type: {vtype}")


def read_gguf(path: str | os.PathLike[str]) -> GGUFFile:
    """Read a GGUF v3 file header, metadata, and tensor info."""
    file_path = Path(path)
    with file_path.open("rb") as f:
        magic, version, tensor_count, kv_count = _HEADER_STRUCT.unpack(
            _read_exact(f, _HEADER_STRUCT.size)
        )
        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {magic!r}")
        if version != GGUF_VERSION:
            raise ValueError(f"Unsupported GGUF version: {version} (expected 3)")

        metadata: dict[str, object] = {}
        for _ in range(kv_count):
            key = _read_string(f)
            (vtype,) = _U32.unpack(_read_exact(f, 4))
            metadata[key] = _read_value(f, vtype)

        raw_align = metadata.get("general.alignment", DEFAULT_ALIGNMENT)
        if not isinstance(raw_align, int) or isinstance(raw_align, bool) or raw_align <= 0:
            raise ValueError(f"Invalid general.alignment: {raw_align!r}")
        alignment = raw_align

        raw_tensors: list[tuple[str, tuple[int, ...], str, int, int]] = []
        for _ in range(tensor_count):
            name = _read_string(f)
            (n_dims,) = _U32.unpack(_read_exact(f, 4))
            dims = tuple(_U64.unpack(_read_exact(f, 8))[0] for _ in range(n_dims))
            (type_id,) = _U32.unpack(_read_exact(f, 4))
            (rel_offset,) = _U64.unpack(_read_exact(f, 8))
            type_name = GGML_TYPE_NAMES.get(type_id, f"UNKNOWN_{type_id}")
            if type_name.startswith("UNKNOWN_"):
                nbytes = 0
            else:
                nbytes = tensor_nbytes(dims, type_name)
            raw_tensors.append((name, dims, type_name, rel_offset, nbytes))

        data_offset = _align_up(f.tell(), alignment)

    tensors = [
        TensorInfo(
            name=name,
            shape=dims,
            ggml_type=type_name,
            offset=data_offset + rel_offset,
            nbytes=nbytes,
        )
        for name, dims, type_name, rel_offset, nbytes in raw_tensors
    ]
    return GGUFFile(
        version=version,
        alignment=alignment,
        metadata=metadata,
        tensors=tensors,
        data_offset=data_offset,
    )


def _pack_string(s: str) -> bytes:
    encoded = s.encode("utf-8")
    return _U64.pack(len(encoded)) + encoded


def _write_value(f: BinaryIO, vtype: int, value: object) -> None:
    if vtype == GGUF_TYPE_STRING:
        if not isinstance(value, str):
            raise TypeError(f"Expected str for STRING, got {type(value).__name__}")
        f.write(_pack_string(value))
    elif vtype == GGUF_TYPE_BOOL:
        f.write(_SCALAR_STRUCTS[GGUF_TYPE_BOOL].pack(1 if value else 0))
    elif vtype in _SCALAR_STRUCTS:
        f.write(_SCALAR_STRUCTS[vtype].pack(value))
    elif vtype == GGUF_TYPE_ARRAY:
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError("ARRAY value must be (elem_type, sequence)")
        elem_type, items = value
        f.write(_U32.pack(elem_type))
        f.write(_U64.pack(len(items)))
        for item in items:
            _write_value(f, elem_type, item)
    else:
        raise ValueError(f"Unsupported GGUF value type: {vtype}")


TensorProducer = bytes | bytearray | memoryview | Callable[[BinaryIO], None]


def write_gguf(
    path: str | os.PathLike[str],
    metadata: Iterable[tuple[str, int, object]],
    tensors: Sequence[tuple[str, tuple[int, ...], str, TensorProducer]],
    *,
    alignment: int = DEFAULT_ALIGNMENT,
) -> None:
    """Write a GGUF v3 file, streaming tensor data to disk."""
    meta_list = list(metadata)
    rel_offset = 0
    tensor_entries: list[tuple[str, tuple[int, ...], int, int, int, TensorProducer]] = []

    for name, shape, ggml_type, producer in tensors:
        type_name = ggml_type.upper()
        if type_name not in GGML_NAME_TO_ID:
            raise ValueError(f"Unsupported ggml type: {ggml_type!r}")
        nbytes = tensor_nbytes(shape, type_name)
        rel_offset = _align_up(rel_offset, alignment)
        tensor_entries.append(
            (name, shape, GGML_NAME_TO_ID[type_name], rel_offset, nbytes, producer)
        )
        rel_offset += nbytes

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as f:
        f.write(
            _HEADER_STRUCT.pack(
                GGUF_MAGIC, GGUF_VERSION, len(tensor_entries), len(meta_list)
            )
        )
        for key, vtype, value in meta_list:
            f.write(_pack_string(key))
            f.write(_U32.pack(vtype))
            _write_value(f, vtype, value)

        for name, shape, type_id, t_rel_offset, _, _ in tensor_entries:
            f.write(_pack_string(name))
            f.write(_U32.pack(len(shape)))
            for dim in shape:
                f.write(_U64.pack(dim))
            f.write(_U32.pack(type_id))
            f.write(_U64.pack(t_rel_offset))

        data_start = _align_up(f.tell(), alignment)
        pad = data_start - f.tell()
        if pad:
            f.write(b"\x00" * pad)

        for _, _, _, t_rel_offset, nbytes, producer in tensor_entries:
            target_pos = data_start + t_rel_offset
            cur_pos = f.tell()
            if cur_pos < target_pos:
                f.write(b"\x00" * (target_pos - cur_pos))
            start_write = f.tell()
            if callable(producer):
                producer(f)
            else:
                f.write(producer)
            written = f.tell() - start_write
            if written != nbytes:
                raise ValueError(
                    f"Tensor producer wrote {written} bytes, expected {nbytes}"
                )


def _format_inspect(gguf: GGUFFile) -> dict[str, object]:
    meta_out: dict[str, object] = {}
    for k, v in gguf.metadata.items():
        if isinstance(v, list):
            meta_out[k] = {"array_len": len(v)}
        else:
            meta_out[k] = v

    tensors_out = [
        {
            "name": t.name,
            "shape": list(t.shape),
            "type": t.ggml_type,
            "offset": t.offset,
            "nbytes": t.nbytes,
        }
        for t in gguf.tensors
    ]
    total_bytes = sum(t.nbytes for t in gguf.tensors)
    return {
        "version": gguf.version,
        "alignment": gguf.alignment,
        "metadata": meta_out,
        "tensors": tensors_out,
        "total_tensor_bytes": total_bytes,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="models.gguf_io")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_p = sub.add_parser("inspect", help="Inspect a GGUF file")
    inspect_p.add_argument("file", help="Path to GGUF file")

    args = parser.parse_args(argv)
    if args.command == "inspect":
        try:
            gguf = read_gguf(args.file)
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(_format_inspect(gguf), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
