"""Build a comparison CSV and static chart from the curated run summaries."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = [
    ('Supplied classical safe_v8', 'historical/classical_safe_v8'),
    ('Supplied hybrid YOLO11s', 'historical/hybrid_yolo11s_highres'),
    ('Verified classical', 'verified/classical_10sec'),
    ('Verified hybrid YOLO11s', 'verified/hybrid_10sec'),
]


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    rows = []
    for label, directory in RUNS:
        path = ROOT/'results'/directory/'run_summary.json'
        s = json.loads(path.read_text(encoding='utf-8'))
        rows.append({'run': label, 'source': f'{directory}/run_summary.json',
                     'frames': s['frames_processed'], 'detections': s['player_detections_total'],
                     'red': s['player_detections_by_team']['team_red'],
                     'light': s['player_detections_by_team']['team_light'],
                     'other': s['player_detections_by_team']['team_other'],
                     'track_ids': s['tracks_total'], 'ball_visible_frames': s['ball_visible_frames']})
    with (ROOT/'results'/'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, (ax, tx) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={'width_ratios': [2.7, 1]}, layout='constrained')
    fig.suptitle('Soccer analysis: output counts over 300 frames', fontsize=16, fontweight='bold')
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for key, label, color in [('red','Red team','#c84944'),('light','Light team','#6997bf'),('other','Other','#c39938')]:
        values = np.array([r[key] for r in rows])
        ax.barh(y, values, left=left, label=label, color=color, height=.6)
        left += values
    ax.set_yticks(y, [r['run'] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel('Accumulated accepted player detections')
    ax.set_xlim(0, 5200)
    for idx, row in enumerate(rows):
        ax.text(row['detections']+75, idx, f"{row['detections']:,}", va='center', fontsize=10)
    ax.legend(loc='center', bbox_to_anchor=(.67, .5), ncol=3, frameon=False, fontsize=9)
    tx.barh(y, [r['track_ids'] for r in rows], color='#526d58', height=.6)
    tx.set_yticks(y, ['']*len(rows))
    tx.invert_yaxis()
    tx.set_xlim(0, 65)
    tx.set_xlabel('Created player track IDs')
    for idx, row in enumerate(rows):
        tx.text(row['track_ids']+1, idx, str(row['track_ids']), va='center', fontsize=10)
    for axes in (ax,tx):
        axes.spines[['top','right']].set_visible(False)
        axes.grid(axis='x', alpha=.15)
        axes.set_axisbelow(True)
    fig.supxlabel('Output availability and IDs are not ground-truth recall or identity accuracy.', fontsize=10)
    fig.savefig(ROOT/'docs'/'images'/'results_comparison.png', dpi=180)
    plt.close(fig)
    print('Wrote results/comparison.csv and docs/images/results_comparison.png')


if __name__ == '__main__':
    main()
