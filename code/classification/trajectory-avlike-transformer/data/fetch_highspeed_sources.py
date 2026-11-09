"""Fetch public sources and NGSIM leader rows for the >50 mph experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request
import zipfile

import numpy as np
import pandas as pd

CATS_COMMIT = 'c680d560976a5b78afe485e54416730e97d3a5cb'
NGSIM = 'https://data.transportation.gov/resource/8ect-6jqj.csv'
COLS = 'location,vehicle_id,frame_id,global_time,local_x,local_y,v_length,v_class,v_vel,v_acc,lane_id,preceding,space_headway'


def query_url(where):
    return NGSIM + '?' + urllib.parse.urlencode({
        '$where': where, '$select': COLS, '$order': 'location,global_time,vehicle_id', '$limit': 50000})


def fetch(url, dest, records):
    if not dest.exists():
        req = urllib.request.Request(url, headers={'User-Agent': 'trajectory-research/1.0'})
        with urllib.request.urlopen(req, timeout=55) as response:
            data = response.read()
        dest.write_bytes(data)
    records.append(dict(file=dest.name, url=url, bytes=dest.stat().st_size,
                        sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
    print(f'FETCH {dest.name} {dest.stat().st_size} bytes', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='data/external_raw_20261010')
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    fetch(f'https://codeload.github.com/CATS-Lab/Filed-Experiment-Data-ACC_Data/zip/{CATS_COMMIT}',
          root/'cats_original.zip', records)
    # Extract only under the destination; no archive paths may escape it.
    with zipfile.ZipFile(root/'cats_original.zip') as archive:
        for entry in archive.infolist():
            target = (root / entry.filename).resolve()
            if not target.is_relative_to(root.resolve()):
                raise ValueError('Unsafe archive member')
        archive.extractall(root)
    fetch('https://ndownloader.figshare.com/files/59770106', root/'vanderbilt_step2.csv', records)
    fetch('https://api.figshare.com/v2/articles/26339512/versions/4', root/'ultra_metadata_v4.json', records)
    micro = ('https://raw.githubusercontent.com/microSIM-ACC/ICE/main/'
             '2-Vehicle%20ACC%20Car%20Following%20Experiments%20(CCF%2C%20Same%20Desired%20Speed).zip')
    fetch(micro, root/'microsim_original.zip', records)
    where = "location in ('i-80','us-101') AND v_class=2 AND v_vel>73.33333333333333"
    fetch(query_url(where), root/'ngsim_fast.csv', records)
    data = pd.read_csv(root/'ngsim_fast.csv').drop_duplicates()
    if len(data) >= 50000:
        raise ValueError('NGSIM query may be truncated; implement pagination before using it')
    if data.duplicated(['location','global_time','vehicle_id']).any():
        raise ValueError('Conflicting NGSIM states')
    data['recording_key'] = data.global_time - data.frame_id * 100
    data = data.sort_values(['location','recording_key','vehicle_id','global_time'])
    groups = data.groupby(['location','recording_key','vehicle_id'])
    boundary = ((groups.global_time.diff() != 100) | (groups.preceding.diff() != 0)
                | (groups.lane_id.diff() != 0))
    data['segment'] = boundary.cumsum()
    candidates = data[data.preceding > 0].groupby('segment').filter(lambda g: len(g) >= 40)
    clauses = []
    for _, group in candidates.groupby('segment'):
        row = group.iloc[0]
        clauses.append(f"(location='{row.location}' AND vehicle_id={int(row.preceding)} "
                       f'AND global_time between {int(group.global_time.min())} and {int(group.global_time.max())})')
    leaders = []
    for start in range(0, len(clauses), 6):
        path = root / f'ngsim_leaders_{start//6:02d}.csv'
        fetch(query_url(' OR '.join(clauses[start:start+6])), path, records)
        block = pd.read_csv(path)
        if len(block) >= 50000:
            raise ValueError('Leader query may be truncated')
        leaders.append(block)
    pd.concat(leaders).drop_duplicates().to_csv(root/'ngsim_leaders.csv', index=False)
    (root/'downloads.json').write_text(json.dumps(dict(
        files=records, cats_commit=CATS_COMMIT,
        optional_unavailable=[
            dict(url='https://vanderbilt.app.box.com/v/accData', observed_status=404,
                 effect='Arizona 2018 original not available from official link; excluded'),
            dict(url='https://vanderbilt.app.box.com/s/76m0vmlpzoj2059p047yt0jkx3mtyxdp', observed_status=404,
                 effect='Tennessee original unavailable; using explicitly identified Ultra-AV step2 release')]), indent=2))
    print(f'DONE {len(records)} downloads; NGSIM candidate target rows={len(candidates)}', flush=True)


if __name__ == '__main__':
    main()
