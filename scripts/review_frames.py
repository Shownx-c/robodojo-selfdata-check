#!/usr/bin/env python3
"""Export three cameras at a flagged RGB index and its temporal neighbors."""
import argparse
from pathlib import Path

import cv2
import h5py
from PIL import Image, ImageDraw

from check_hdf5 import CAMERAS, jpeg_buffer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hdf5', type=Path, required=True)
    parser.add_argument('--frame', type=int, required=True, help='Zero-based RGB frame, NOT dense control row')
    parser.add_argument('--radius', type=int, default=1)
    parser.add_argument('--color', choices=('rgb-direct', 'standard-jpeg', 'unknown'), default='unknown')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.frame < 0 or not 0 <= args.radius <= 5:
        parser.error('Need frame >= 0 and radius between 0 and 5')
    if args.out.exists():
        parser.error('Output exists; choose a fresh filename')
    if args.out.resolve().is_relative_to(args.hdf5.resolve().parent):
        parser.error('Write evidence outside the raw episode directory')
    rules = ('rgb-direct', 'standard-jpeg') if args.color == 'unknown' else (args.color,)
    indices = list(range(args.frame - args.radius, args.frame + args.radius + 1))
    width, height, label_height, header = 320, 240, 40, 62
    canvas = Image.new('RGB', (width * len(indices), header + (height + label_height) * 3 * len(rules)), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), f'{args.hdf5.name} | RGB center={args.frame} | {args.color}', fill='black')
    note = ('Unknown color: both interpretations shown; neither is validated by this image.'
            if args.color == 'unknown' else 'Color interpretation selected from the source contract: ' + args.color)
    draw.text((8, 24), note, fill='black')
    draw.text((8, 42), 'Columns: temporal neighbors. Rows: head, left wrist, right wrist.', fill='black')
    with h5py.File(args.hdf5, 'r') as f:
        for rule_id, rule in enumerate(rules):
            for cam_id, cam in enumerate(CAMERAS):
                name = next((k for k in (f'vision/{cam}/colors', f'rgb/{cam}') if k in f), None)
                for col, index in enumerate(indices):
                    x, y = col * width, header + (rule_id * 3 + cam_id) * (height + label_height)
                    control = int(f['rgb_step_index'][index]) if 'rgb_step_index' in f and 0 <= index < len(f['rgb_step_index']) else '?'
                    draw.text((x + 4, y + 2), f'{cam} | RGB={index} | control={control}', fill='red' if index == args.frame else 'black')
                    draw.text((x + 4, y + 18), rule + (' (candidate)' if args.color == 'unknown' else ''), fill='black')
                    image = None
                    if name and 0 <= index < len(f[name]):
                        buf = jpeg_buffer(f[name][index])
                        image = cv2.imdecode(buf, cv2.IMREAD_COLOR) if len(buf) else None
                    if image is None:
                        draw.text((x + 8, y + label_height + 20), 'Missing / out of range / decode failed', fill='red')
                    else:
                        if rule == 'standard-jpeg':
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        thumb = Image.fromarray(image)
                        thumb.thumbnail((width, height))
                        canvas.paste(thumb, (x, y + label_height))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(args.out.resolve())


if __name__ == '__main__':
    main()
