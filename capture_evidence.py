"""Extract verifiable video frames and a labeled contact sheet; never invent detections."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont


def read_frame(path: Path, index: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Frame {index} unavailable in {path}")
    return frame, fps, count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-video', type=Path, required=True)
    parser.add_argument('--annotated-video', type=Path, required=True)
    parser.add_argument('--classical-video', type=Path)
    parser.add_argument('--frames', type=int, nargs='+', default=[60, 159, 240])
    parser.add_argument('--annotated-start-frame', type=int, default=0, help='Absolute source index corresponding to annotated frame 0')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--label', default='Supplied final hybrid output')
    args = parser.parse_args()
    if any(i < args.annotated_start_frame for i in args.frames):
        parser.error('Selected frames must be within the annotated segment.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, panels = [], []
    streams = [('input', args.input_video), ('hybrid', args.annotated_video)]
    if args.classical_video:
        streams.insert(1, ('classical', args.classical_video))
    for idx in args.frames:
        for kind, path in streams:
            local = idx if kind == 'input' else idx - args.annotated_start_frame
            frame, fps, count = read_frame(path, local)
            name = f'{kind}_frame{idx}.png'
            dest = args.output_dir / name
            if not cv2.imwrite(str(dest), frame):
                raise SystemExit(f'Could not save {dest}')
            records.append({'file': name, 'source_video': path.name, 'source_output_frame_index': local,
                            'absolute_input_frame_index': idx, 'source_fps': fps, 'source_frame_count': count,
                            'extraction': 'OpenCV VideoCapture seek/read; unmodified decoded frame',
                            'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()})
            im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            im.thumbnail((640, 360))
            tile = Image.new('RGB', (640, 402), '#101b27')
            tile.paste(im, ((640-im.width)//2, 42+(360-im.height)//2))
            draw = ImageDraw.Draw(tile)
            draw.text((14, 13), f'{kind.upper()}  |  frame {idx}  |  {idx / fps:.2f} s', fill='white', font=ImageFont.load_default(size=19))
            panels.append(tile)
    sheet = Image.new('RGB', (640 * len(streams), 402 * len(args.frames)), '#101b27')
    for n, tile in enumerate(panels):
        sheet.paste(tile, ((n % len(streams))*640, (n // len(streams))*402))
    sheet.save(args.output_dir / 'comparison.png', optimize=True)
    (args.output_dir/'capture_manifest.json').write_text(json.dumps({'label': args.label, 'indexing': 'zero-based', 'frames': records}, indent=2)+'\n', encoding='utf-8')
    print(f'Captured {len(records)} original decoded frames and comparison.png')


if __name__ == '__main__':
    main()
