"""YAML inheritance and explicit configuration references, shared by all entry points."""

import copy
import re
from pathlib import Path

import yaml

REFERENCE = re.compile(r"\$\{([a-zA-Z_][a-zA-Z_0-9.]*)\}")


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def read_yaml(path, *, overrides=None, keys=None):
    def inherit(filename, stack):
        filename = Path(filename).resolve()
        if filename in stack:
            raise ValueError(f"Configuration inheritance cycle: {filename}")
        raw = yaml.safe_load(filename.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"Expected a YAML mapping: {filename}")
        parent = raw.pop("extends", None)
        if parent is None:
            return raw
        if not isinstance(parent, str) or not parent:
            raise ValueError("extends must be one YAML path, relative to this file")
        return merge(inherit(filename.parent / parent, {*stack, filename}), raw)

    raw = merge(inherit(path, set()), overrides or {})
    cache, active = {}, set()

    def resolve(location):
        if location in active:
            raise ValueError(f"Configuration reference cycle: {'.'.join(map(str, location))}")
        if location in cache:
            return copy.deepcopy(cache[location])
        value = raw
        try:
            for part in location:
                value = value[part]
        except (KeyError, TypeError, IndexError):
            raise ValueError(
                f"Unknown configuration reference: {'.'.join(map(str, location))}"
            ) from None
        active.add(location)
        if isinstance(value, dict):
            result = {key: resolve((*location, key)) for key in value}
        elif isinstance(value, list):
            result = [resolve((*location, i)) for i in range(len(value))]
        elif isinstance(value, str):
            exact = REFERENCE.fullmatch(value)
            if exact:
                result = resolve(tuple(exact[1].split(".")))
            else:

                def replace(match):
                    target = resolve(tuple(match[1].split(".")))
                    if not isinstance(target, (str, int, float, bool)):
                        raise ValueError("Only scalar references can be embedded in strings")
                    return str(target)

                result = REFERENCE.sub(replace, value)
        else:
            result = value
        active.remove(location)
        cache[location] = result
        return copy.deepcopy(result)

    return {key: resolve((key,)) for key in raw if keys is None or key in keys}
