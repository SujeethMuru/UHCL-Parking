"""Save local artifacts without destroying the previous successful version."""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile

import cv2


def stamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S_%f')


def write_artifact(path, content):
    """Replace atomically; use a fresh name if an existing file is locked."""
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix='.parking-',delete=False) as file:
        temporary=Path(file.name)
        file.write(content)
    try:
        try:
            os.replace(temporary,path)
            return path
        except PermissionError:
            # Photo viewers can hold Windows files open against replacement.
            fallback=path.with_name(f'{path.stem}.{stamp()}{path.suffix}')
            if path.name.endswith('.local.json'):
                fallback=path.with_name(f'{path.name[:-11]}.{stamp()}.local.json')
            os.replace(temporary,fallback)
            return fallback
    finally:
        if temporary.exists():
            temporary.unlink()


def save_image(path,frame):
    ok,encoded=cv2.imencode(Path(path).suffix,frame)
    if not ok:
        raise RuntimeError(f'Could not encode image for {path}')
    return write_artifact(path,encoded.tobytes())


def save_calibration(frame,spaces,config,output,metadata=None):
    """Persist recovery outlines first; avoid rewriting identical reference data."""
    config,output=Path(config),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    config.parent.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(frame.tobytes()).hexdigest()
    legacy=output/'reference.png'
    reference=None
    if legacy.exists():
        previous=cv2.imread(str(legacy))
        if previous is not None and hashlib.sha256(previous.tobytes()).hexdigest()==digest:
            reference=legacy
    if reference is None:
        reference=output/f'reference-{digest[:16]}.png'
        if not reference.exists():
            reference=save_image(reference,frame)
    payload={'image_sha256':digest,'reference_image':str(reference.resolve()),'spaces':spaces}
    if metadata is not None:
        payload['video']=metadata
    data=json.dumps(payload,indent=2).encode('utf-8')
    # A separate recovery copy survives a main-config write/replace failure.
    recovery=config.with_name(f'{config.stem.replace(".local", "")}.recovery.{stamp()}.local.json')
    write_artifact(recovery,data)
    if config.exists():
        backup=config.with_name(f'{config.stem.replace(".local", "")}.{stamp()}.local.json')
        try:
            write_artifact(backup,config.read_bytes())
        except OSError as exc:
            print(f'Previous-config backup failed: {exc}. New outlines have a recovery copy in {recovery}.')
    try:
        saved=write_artifact(config,data)
    except OSError as exc:
        print(f'Could not update {config}: {exc}. Your outlines are saved in {recovery}.')
        return recovery
    if saved!=config:
        print(f'Configuration was locked. Saved your edits to {saved}; use --config "{saved}" next time.')
    return saved
