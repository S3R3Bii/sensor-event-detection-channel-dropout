"""Read the released scores and describe the frozen policy; no fitting or tuning."""

import csv
import hashlib
import json
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'evidence'
SCENARIOS = ('clean', 'brief', 'sustained')


def read_rows():
    with (EVIDENCE / 'test_predictions.csv').open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 16_200
    assert len({(r['window_id'], r['model'], r['scenario']) for r in rows}) == len(rows)
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row['model'] == 'cnn':
            grouped[row['scenario']].append(row)
    result = {}
    for name in SCENARIOS:
        sample = grouped[name]
        assert len(sample) == 1800
        assert len({r['device'] for r in sample}) == 30
        issued_wrong = deferred_wrong = issued_correct = deferred_correct = 0
        kept_fn = kept_fp = deferred_fn = deferred_fp = 0
        for row in sample:
            y, p = int(row['label']), float(row['probability'])
            keep = row['retained'] == 'True'
            assert keep == (max(p, 1-p) >= float(row['confidence_threshold']))
            predicted = int(p >= .5)
            wrong = y != predicted
            issued_wrong += keep and wrong
            deferred_wrong += not keep and wrong
            issued_correct += keep and not wrong
            deferred_correct += not keep and not wrong
            kept_fn += keep and y == 1 and wrong
            kept_fp += keep and y == 0 and wrong
            deferred_fn += not keep and y == 1 and wrong
            deferred_fp += not keep and y == 0 and wrong
        total = len(sample)
        deferred = deferred_wrong + deferred_correct
        full_wrong = issued_wrong + deferred_wrong
        crossover = Fraction(deferred_wrong, deferred)
        result[name] = {
            'windows': total, 'events': sum(int(r['label']) for r in sample),
            'issued_correct': issued_correct, 'issued_wrong': issued_wrong,
            'deferred_correct_if_issued': deferred_correct, 'deferred_wrong_if_issued': deferred_wrong,
            'issued_false_negatives': kept_fn, 'issued_false_positives': kept_fp,
            'deferred_false_negatives_if_issued': deferred_fn, 'deferred_false_positives_if_issued': deferred_fp,
            'deferred': deferred, 'all_issued_wrong': full_wrong,
            'coverage': 1-deferred/total, 'error_if_all_issued': full_wrong/total,
            'error_among_issued': issued_wrong/(total-deferred),
            'break_even_cost_per_deferral': float(crossover),
            'break_even_exact': str(crossover),
            'cost_at_original_weight': (issued_wrong+.18*deferred)/total,
            'deferrals_per_1000_windows_if_all_this_scenario': 1000*deferred/total,
        }
    return result


def main():
    result = {
        'scope': 'Post-hoc description of saved held-out scores and the unchanged frozen policy. No new threshold selection, training, field outcomes, or human review outcomes.',
        'inputs_sha256': {name: hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest()
                          for name in ('test_predictions.csv', 'results.json', 'validation_frozen.json')},
        'cases': summarize(read_rows()),
    }
    (ROOT / 'analysis' / 'fixed_policy_summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps(result['cases']['sustained'], indent=2))


if __name__ == '__main__':
    main()
