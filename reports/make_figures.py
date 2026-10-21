#!/usr/bin/env python3
"""Regenerate reports/figures/dataset_*.png and async_timeline.png from the training server data.

  python3 reports/make_figures.py --project /home/work/vln_project/vla_ws_restore/vla_project
"""
import argparse
import collections
import glob
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIG = Path(__file__).resolve().parent / 'figures'
CMD_COLORS = {'follow': '#4C78A8', 'turn_left': '#F58518', 'turn_right': '#54A24B',
              'safe_stop': '#B279A2', 'stop': '#9D9D9D'}
SPLIT_COLORS = {'train': '#4C78A8', 'val': '#E45756'}


def load_rows(project):
    rows = []
    for split in ('train', 'val'):
        path = project / f'outputs/s2_train_data_reviewed_landmark/conversations_{split}.json'
        rows += json.loads(path.read_text())
    return rows


def episode_paths(rows):
    dirs = {}
    for r in rows:
        if r.get('source_image'):
            dirs[r['episode']] = (Path(r['source_image']).parents[2], r['split'])
    out = {}
    for ep, (d, split) in dirs.items():
        poses = []
        for f in sorted(glob.glob(str(d / 'anno/*.json.gz'))):
            a = json.load(gzip.open(f, 'rt'))
            poses.append((a['x'], a['y'], a['theta']))
        out[ep] = (np.array(poses), split)
    return out


def ego_aligned(poses):
    # start at origin, initial heading pointing up
    xy = poses[:, :2] - poses[0, :2]
    th = poses[0, 2]
    c, s = np.cos(-th + np.pi / 2), np.sin(-th + np.pi / 2)
    return xy @ np.array([[c, s], [-s, c]])


def dataset_figures(project):
    rows = load_rows(project)
    paths = episode_paths(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, parking in zip(axes, (False, True)):
        for ep, (poses, split) in sorted(paths.items()):
            if (int(ep.split('data')[1]) >= 900) != parking:
                continue
            xy = ego_aligned(poses)
            ax.plot(xy[:, 0], xy[:, 1], color=SPLIT_COLORS[split], lw=1.6 if split == 'val' else 1.0,
                    alpha=0.9 if split == 'val' else 0.55)
            ax.plot(xy[-1, 0], xy[-1, 1], 'o', ms=3, color=SPLIT_COLORS[split])
        ax.set_aspect('equal')
        ax.grid(alpha=0.3)
        ax.set_xlabel('x [m] (start heading = up)')
        ax.set_ylabel('y [m]')
        ax.set_title('Parking (park_forward), 16 episodes' if parking else 'Driving, 13 episodes')
    handles = [plt.Line2D([], [], color=c, lw=2, label=f'{s}') for s, c in SPLIT_COLORS.items()]
    axes[0].legend(handles=handles, loc='best')
    fig.suptitle('VLA-E2E episode trajectories (ego pose, aligned to start)')
    fig.tight_layout()
    fig.savefig(FIG / 'dataset_trajectories.png', dpi=110)
    plt.close(fig)

    counts = collections.Counter((r['split'], r['category']) for r in rows)
    cmds = list(CMD_COLORS)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(cmds))
    for i, split in enumerate(('train', 'val')):
        vals = [counts[(split, c)] for c in cmds]
        bars = ax.bar(x + (i - 0.5) * 0.38, vals, 0.38, color=SPLIT_COLORS[split], label=split)
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks(x, cmds)
    ax.set_ylabel('S2 samples')
    ax.set_title('S2 samples per command (train 4,711 / val 2,571)')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / 'dataset_commands.png', dpi=110)
    plt.close(fig)


def async_timeline(project, episode='undistort_orin_pc_data11'):
    path = project / f'outputs/shadow_test_async_20261020/final_gpu/closed_loop_{episode}.jsonl'
    recs = [json.loads(line) for line in path.open()]
    t = np.arange(len(recs)) * 0.5
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(12, 6.2), sharex=True,
                                        gridspec_kw={'height_ratios': [1, 2, 2]})
    for i, r in enumerate(recs):
        ax0.axvspan(t[i], t[i] + 0.5, color=CMD_COLORS.get(r['command'], '#ccc'), lw=0)
    ax0.set_yticks([])
    ax0.set_ylabel('command')
    ax0.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c, label=k) for k, c in CMD_COLORS.items()
                        if any(r['command'] == k for r in recs)], ncol=5, loc='upper right', fontsize=8)
    ok = np.array([r['control_decision'] == 'trajectory' for r in recs])
    rtt = np.array([r['rtt_ms'] for r in recs])
    ax1.scatter(t[ok], rtt[ok], s=8, color='#4C78A8', label='trajectory')
    ax1.scatter(t[~ok], rtt[~ok], s=12, color='#E45756', marker='x', label='fail-safe STOP')
    ax1.axhline(500, color='k', ls='--', lw=1)
    ax1.text(t[-1], 510, '2 Hz budget 500 ms', ha='right', fontsize=8)
    ax1.set_ylabel('response RTT [ms]')
    ax1.set_ylim(0, 900)
    ax1.legend(loc='upper left', fontsize=8)
    age = np.array([r.get('goal_age_ms') or np.nan for r in recs]) / 1000
    ax2.plot(t, age, '.-', ms=3, lw=0.8, color='#54A24B')
    ax2.axhline(2.25, color='#E45756', ls='--', lw=1)
    ax2.text(t[-1], 2.3, 'MAX_GOAL_AGE_S 2.25', ha='right', fontsize=8, color='#E45756')
    ax2.set_ylabel('S2 goal age [s]')
    ax2.set_xlabel('time [s] (2 Hz replay)')
    fig.suptitle(f'Async runtime, 2 Hz real-time replay: {episode.replace("undistort_orin_pc_", "")}')
    fig.tight_layout()
    fig.savefig(FIG / 'async_timeline.png', dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, default=Path('/home/work/vln_project/vla_ws_restore/vla_project'))
    args = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    dataset_figures(args.project)
    async_timeline(args.project)


if __name__ == '__main__':
    main()
