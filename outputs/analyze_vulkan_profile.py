"""Summarize Vulkan timing windows; fused/concurrent groups remain indivisible."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def category(op):
    if 'FLASH_ATTN_EXT' in op:
        return 'attention'
    if 'm=248320 ' in op and 'MUL_MAT' in op:
        return 'vocabulary_projection'
    if 'MUL_MAT_ID' in op:
        return 'expert_matmul_groups'
    if 'GATED_DELTA_NET' in op or 'SSM_CONV' in op:
        return 'recurrent_groups'
    if 'MUL_MAT' in op:
        return 'other_matmul_groups'
    return 'other_groups'


def parse(text):
    windows = []
    for block in text.split('Vulkan Timings:')[1:]:
        total = re.search(r'Total time: ([0-9.e+]+) us', block)
        if not total:
            continue
        rows = []
        for line in block.splitlines():
            match = re.match(r'(.+?): (\d+) x ([0-9.e+]+) us = ([0-9.e+]+) us',line)
            if match:
                rows.append({'op':match[1], 'count':int(match[2]),
                    'average_us':float(match[3]), 'total_us':float(match[4]),
                    'category':category(match[1])})
        groups = defaultdict(float)
        for row in rows:
            groups[row['category']] += row['total_us']
        batch_sizes = [int(n) for row in rows for n in re.findall(r'batch=(\d+)',row['op'])]
        query_sizes = [int(n) for row in rows for n in re.findall(r'q\(256,(\d+),16,1\)',row['op'])]
        kv_sizes = [int(n) for row in rows for n in re.findall(r'k\(256,(\d+),2,1\)',row['op'])]
        family = 'main-like' if any('GATED_DELTA_NET' in row['op'] for row in rows) else 'draft-like'
        mixed = max(batch_sizes+query_sizes+[0]) > 2
        total_us = float(total[1])
        windows.append({'index':len(windows),'family_inferred':family,
            'contains_prefill_or_more_than_two_query_tokens':mixed,
            'max_logged_kv_length':max(kv_sizes+[0]), 'total_us':total_us,
            'categories':[{'category':k,'total_us':v,'percent_of_window':100*v/total_us}
                for k,v in sorted(groups.items(),key=lambda item:item[1],reverse=True)],
            'rows':sorted(rows,key=lambda row:row['total_us'],reverse=True)})
    return windows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log',type=Path,required=True)
    p.add_argument('--start-bytes',type=int,default=0)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    data=args.log.read_bytes()
    encoding='utf-16-le' if data.startswith(b'\xff\xfe') else 'utf-8-sig'
    text=data[args.start_bytes:].decode(encoding,errors='replace')
    windows=parse(text)
    args.out.write_text(json.dumps({'source':str(args.log.resolve()),
        'start_bytes':args.start_bytes,
        'notes':['Percentages are of recorded GPU timing windows, not end-to-end wall time.',
            'Fused/concurrent groups are categorized by a contained operation; member costs cannot be separated.',
            'Main/draft families are inferred from Gated DeltaNet presence, not explicit logger labels.'],
        'windows':windows},indent=2),encoding='utf-8')
    for window in windows:
        if not window['contains_prefill_or_more_than_two_query_tokens']:
            print(json.dumps({k:v for k,v in window.items() if k!='rows'}))


if __name__=='__main__':
    main()
