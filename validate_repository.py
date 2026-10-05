"""Validate local documentation links, image provenance and saved metric consistency."""
from pathlib import Path
import hashlib
import json
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent


def main():
    errors = []
    for path in ROOT.rglob('*.md'):
        if '.git' in path.parts:
            continue
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            target = target.strip('<>').split('#')[0]
            if not target or re.match(r'^[a-z]+:', target):
                continue
            if not (path.parent / unquote(target)).exists():
                errors.append(f'{path.relative_to(ROOT)}: missing {target}')
    for path in (ROOT/'results').rglob('run_summary.json'):
        data = json.loads(path.read_text(encoding='utf-8'))
        if 'player_detections_by_team' in data:
            if sum(data['player_detections_by_team'].values()) != data['player_detections_total']:
                errors.append(f'{path}: team counts do not sum to total')
        if data.get('ball_visible_frames', 0) > data['frames_processed']:
            errors.append(f'{path}: ball visibility exceeds frame count')
        if 'ball_measured_frames' in data and data['ball_visible_frames'] != data['ball_measured_frames'] + data['ball_predicted_visible_frames']:
            errors.append(f'{path}: measured/predicted counts disagree')
    for manifest in (ROOT/'docs'/'images').rglob('*manifest.json'):
        data = json.loads(manifest.read_text(encoding='utf-8'))
        records = data if isinstance(data, list) else data.get('frames', [])
        for record in records:
            path = manifest.parent / record['file']
            if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                errors.append(f'{manifest}: image hash mismatch for {path.name}')
    if errors:
        raise SystemExit('\n'.join(errors))
    print('Documentation links, image hashes, and saved metric totals are valid.')


if __name__ == '__main__':
    main()
