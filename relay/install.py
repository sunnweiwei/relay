"""`relay install <harness>`: point a harness at Relay by editing its own config file.

A harness profile lists settings. An `endpoint` setting wraps whatever upstream the harness
is configured with (or its default): the config gets `<relay>/up/<id><original path>` and
Relay forwards `/up/<id>/...` to the original origin, so any provider or gateway the user
had keeps working. Previous values are recorded in Relay's state file, so
`relay uninstall` puts the harness back exactly as it was.
"""

from __future__ import annotations

import functools
import json
import os
import re
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Setting:
    path: Path  # the harness's config file: .json, .yaml, .toml or .env
    key: tuple[str, ...]  # nested key (YAML list items as "[name=value]"); a single key for .env
    value: Any = None  # a literal value to write, or ...
    endpoint: str | None = None  # ... the default upstream to wrap when the key is unset


def state_file() -> Path:
    return Path(os.getenv("RELAY_HOME", "~/.config/relay")).expanduser() / "installed.json"


def install(name: str, relay_url: str, settings: list[Setting]) -> None:
    state = _load()
    if name in state:
        raise ValueError(f"{name} is already installed; run `relay uninstall {name}` first")
    records = []
    for index, setting in enumerate(settings):
        record: dict[str, Any] = {"path": str(setting.path), "key": list(setting.key)}

        def value(previous: Any, setting: Setting = setting, record: dict[str, Any] = record) -> Any:
            if setting.endpoint is None:
                return setting.value
            original = urlsplit(previous if isinstance(previous, str) and previous else setting.endpoint)
            record["mount"] = f"/up/{name}-{index}"
            record["origin"] = f"{original.scheme}://{original.netloc}"
            return f"{relay_url.rstrip('/')}{record['mount']}{original.path.rstrip('/')}"

        record["previous"] = _update(setting.path, setting.key, value)
        records.append(record)
    state[name] = records
    _save(state)


def uninstall(name: str) -> list[Path]:
    state = _load()
    records = state.pop(name, None)
    if records is None:
        raise ValueError(f"{name} is not installed")
    for record in reversed(records):
        edit(Path(record["path"]), tuple(record["key"]), record["previous"])
    _save(state)
    return [Path(record["path"]) for record in records]


def mounts() -> dict[str, str]:
    """Installed mounts, e.g. {"/up/codex-0": "https://chatgpt.com"}."""

    path = state_file()
    if not path.exists():
        return {}
    stat = path.stat()
    return _mounts(path, (stat.st_mtime_ns, stat.st_size))


@functools.lru_cache(maxsize=4)
def _mounts(path: Path, stamp: tuple[int, int]) -> dict[str, str]:
    state = json.loads(path.read_text())
    return {r["mount"]: r["origin"] for records in state.values() for r in records if "mount" in r}


def edit(path: Path, key: tuple[str, ...], value: Any) -> Any:
    """Set one setting (remove it when `value` is None) and return its previous value."""

    return _update(path, key, lambda _: value)


def read(path: Path, key: tuple[str, ...]) -> Any:
    """The current value of one setting (None when unset)."""

    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return _apply(text, path.suffix, key, lambda previous: previous)[1]


def _update(path: Path, key: tuple[str, ...], value: Callable[[Any], Any]) -> Any:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    text, previous = _apply(text, path.suffix, key, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return previous


def _apply(text: str, suffix: str, key: tuple[str, ...], value: Callable[[Any], Any]) -> tuple[str, Any]:
    if suffix == ".json":
        data = json.loads(text) if text.strip() else {}
        steps = _json_walk(data, key[:-1])
        node = steps[-1][2] if steps else data
        previous = node.pop(key[-1], None)
        if (new := value(previous)) is not None:
            node[key[-1]] = new
        for parent, part, child in reversed(steps):  # drop containers left empty
            if child in ({}, []) or (part.startswith("[") and child == _selected(part)):
                parent.remove(child) if isinstance(parent, list) else parent.pop(part)
        text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    elif suffix in {".yaml", ".yml"}:
        lines = text.splitlines()
        previous = _yaml_update(lines, key, value)
        text = "\n".join(lines) + "\n"
    else:  # line-oriented formats: TOML (optionally inside a [table]) and .env
        toml = suffix == ".toml"
        lines = text.splitlines()
        *table, name = key
        start, end = _toml_region(lines, table) if toml else (0, len(lines))
        pattern = re.compile(rf"^\s*{re.escape(name)}\s*=")
        index = next((i for i in range(start, end) if pattern.match(lines[i])), None)
        previous = None
        if index is not None:
            previous = _scalar(lines.pop(index).split("=", 1)[1].strip(), "toml" if toml else "env")
        if (new := value(previous)) is not None:
            line = f"{name} = {_toml(new)}" if toml else f"{name}={new}"
            lines.insert(end if index is None else index, line)
        text = "\n".join(lines) + "\n"
    return text, previous


def _json_walk(data: dict[str, Any], parts: tuple[str, ...]) -> list[tuple[Any, str, Any]]:
    """(parent, part, child) along `parts`, creating what is missing; "[name=value]" selects a list item."""

    steps, node = [], data
    for part, after in zip(parts, (*parts[1:], "")):
        if part.startswith("["):
            child = next((item for item in node if item == {**item, **_selected(part)}), None)
            if child is None:
                child = _selected(part)
                node.append(child)
        else:
            child = node.setdefault(part, [] if after.startswith("[") else {})
        steps.append((node, part, child))
        node = child
    return steps


def _selected(part: str) -> dict[str, str]:
    name, value = part[1:-1].split("=", 1)
    return {name: value}


def yaml_keys(path: Path, key: tuple[str, ...]) -> list[str]:
    """Names of the mapping keys at `key` in a YAML file (empty when absent)."""

    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    block = _yaml_block(lines, key, create=False)
    if block is None:
        return []
    start, end, column = block
    child = _yaml_child_column(lines, start, end, column)
    return [_yaml_key(lines[i]) for i in range(start, end) if _yaml_column(lines[i]) == child]


def _yaml_update(lines: list[str], key: tuple[str, ...], value: Callable[[Any], Any]) -> Any:
    """Edit one block-style YAML scalar in place, keeping comments and custom tags.
    A key part "[name=value]" selects the list item whose `name` is `value`."""

    *parents, leaf = key
    block = _yaml_block(lines, tuple(parents), create=True)
    start, end, column = block
    child = _yaml_child_column(lines, start, end, column)
    index = next((i for i in range(start, end) if _yaml_column(lines[i]) == child and _yaml_key(lines[i]) == leaf), None)
    previous = None if index is None else _scalar(lines[index].split(":", 1)[1].strip(), "yaml")
    new = value(previous)
    if index is not None:
        prefix = lines[index][:child]  # keeps a "- " list marker
        if new is not None:
            lines[index] = f"{prefix}{leaf}: {_yaml_scalar(new)}"
        elif prefix.strip():
            raise ValueError(f"cannot remove {leaf!r}: it starts a YAML list item")
        else:
            del lines[index]
            _yaml_prune(lines, tuple(parents))
    elif new is not None:
        lines.insert(end, f"{' ' * child}{leaf}: {_yaml_scalar(new)}")
    else:
        _yaml_prune(lines, tuple(parents))
    return previous


def _yaml_prune(lines: list[str], key: tuple[str, ...]) -> None:
    """Remove the containers along `key` that are left empty (a list item holding only its selector)."""

    for depth in range(len(key), 0, -1):
        block = _yaml_block(lines, key[:depth], create=False)
        if block is None:
            continue
        start, end, _ = block
        item = key[depth - 1].startswith("[")
        first = start if item else start - 1  # the "- name: value" or "key:" line
        if any(_yaml_column(lines[i]) is not None for i in range(first + 1, end)):
            return
        if not item and not lines[first].rstrip().endswith(":"):
            return  # an inline value, not a container Relay created
        del lines[first:end]


def _yaml_scalar(value: Any) -> str:
    """A plain scalar when YAML reads it back as the same string, JSON (valid YAML) otherwise."""

    words = {"true", "false", "yes", "no", "on", "off", "y", "n", "null"}
    plain = isinstance(value, str) and re.fullmatch(r"[A-Za-z_/][\w./:@%+~-]*", value) and value.lower() not in words
    return value if plain else json.dumps(value)


def _yaml_block(lines: list[str], key: tuple[str, ...], create: bool) -> tuple[int, int, int] | None:
    """(start, end, column): the lines below `key`, whose keys sit right of `column`."""

    start, end, column = 0, len(lines), -1
    for part in key:
        if part.startswith("["):
            name, wanted = part[1:-1].split("=", 1)
            index = next((i for i in range(start, end) if lines[i].lstrip().startswith("- ")
                          and _yaml_indent(lines[i]) > column and _yaml_key(lines[i]) == name
                          and str(_scalar(lines[i].split(":", 1)[1].strip(), "yaml")) == wanted), None)
            if index is None:
                if not create:
                    return None
                items = [_yaml_indent(lines[i]) for i in range(start, end) if lines[i].lstrip().startswith("- ")]
                index = end
                lines.insert(index, f"{' ' * (items[0] if items else max(column, 0))}- {name}: {wanted}")
            column = _yaml_indent(lines[index])  # the item's keys start on the "- " line
            start, end = index, _yaml_end(lines, index, column)
            continue
        child = _yaml_child_column(lines, start, end, column)
        index = next((i for i in range(start, end) if _yaml_column(lines[i]) == child and _yaml_key(lines[i]) == part), None)
        if index is None:
            if not create:
                return None
            lines.insert(end, f"{' ' * child}{part}:")
            index = end
        start, end, column = index + 1, _yaml_end(lines, index, child), child
    return start, end, column


def _yaml_child_column(lines: list[str], start: int, end: int, column: int) -> int:
    columns = [c for i in range(start, end) if (c := _yaml_column(lines[i])) is not None and c > column]
    return min(columns) if columns else column + 2 if column >= 0 else 0


def _yaml_end(lines: list[str], index: int, column: int) -> int:
    for i in range(index + 1, len(lines)):
        if lines[i].strip() and not lines[i].lstrip().startswith("#") and _yaml_indent(lines[i]) <= column:
            return i
    return len(lines)


def _yaml_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _yaml_column(line: str) -> int | None:
    """Column of the mapping key on a line ("- key:" counts the key), else None."""

    rest, column = line.lstrip(" "), _yaml_indent(line)
    while rest.startswith("- "):
        rest, column = rest[2:], column + 2
    return column if re.match(r"[^\s:#'\"][^:]*:(\s|$)|['\"][^'\"]+['\"]:(\s|$)", rest) else None


def _yaml_key(line: str) -> str:
    rest = line.lstrip(" ")
    while rest.startswith("- "):
        rest = rest[2:]
    return rest.split(":", 1)[0].strip().strip("'\"")


def _toml_region(lines: list[str], table: list[str]) -> tuple[int, int]:
    """Line range holding a table's keys (top level: before the first header).
    A missing table is appended to `lines`."""

    headers = [i for i, line in enumerate(lines) if line.lstrip().startswith("[")]
    if not table:
        return 0, headers[0] if headers else len(lines)
    header = f"[{'.'.join(table)}]"
    start = next((i + 1 for i in headers if lines[i].strip() == header), None)
    if start is None:
        lines += ["", header]
        return len(lines), len(lines)
    return start, next((i for i in headers if i >= start), len(lines))


def _toml(value: Any) -> str:
    """A value as TOML writes it on one line (JSON's spelling for scalars)."""

    if isinstance(value, dict):
        return "{ " + ", ".join(f"{key} = {_toml(item)}" for key, item in value.items()) + " }"
    if isinstance(value, list):
        return "[" + ", ".join(_toml(item) for item in value) + "]"
    return json.dumps(value)


def _scalar(raw: str, kind: str) -> Any:
    if kind == "toml":
        return tomllib.loads(f"v = {raw}")["v"]
    if kind in {"yaml", "env"}:  # quoted, or bare with an optional inline comment
        if raw[:1] == '"':
            return json.loads(raw[: raw.rindex('"') + 1])
        if raw[:1] == "'":
            return raw[1 : raw.rindex("'")]
        return raw.split(" #", 1)[0].strip()  # drop an inline comment
    return raw


def _load() -> dict[str, Any]:
    path = state_file()
    return json.loads(path.read_text()) if path.exists() else {}


def _save(state: dict[str, Any]) -> None:
    path = state_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n")
