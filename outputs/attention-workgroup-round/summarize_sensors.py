"""Align ADLX observations with the generation part of each measured request."""
import argparse
import csv
import json
from pathlib import Path
from statistics import median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('samples', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    raw = Path(__file__).with_name('gpu-sensors.csv')
    with raw.open(encoding='utf-8-sig', newline='') as stream:
        observations = list(csv.DictReader(stream))
    rows = json.loads(args.samples.read_text(encoding='utf-8'))
    result = []
    for request in rows:
        first = request['started_unix'] + request['ttft_s']
        end = request['ended_unix']
        points = [p for p in observations if first <= int(p['unix_ms']) / 1000 <= end]
        sensors = {}
        for name in ['gpu_clock_mhz', 'gpu_temperature_c', 'gpu_usage_percent', 'gpu_power_w']:
            values = [float(p[name]) for p in points if p[name]]
            if values:
                sensors[name] = dict(min=min(values), median=median(values), max=max(values))
        result.append(dict(case=request['case'], trial=request['trial'], context=request['context'],
                           start_unix=first, end_unix=end, count=len(points), sensors=sensors))
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
