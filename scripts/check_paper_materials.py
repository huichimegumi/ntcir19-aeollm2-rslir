#!/usr/bin/env python3
"""Check the committed paper CSV snapshots; this does not rerun inference."""
from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'results' / 'paper'
TABLES = {
    3: ('table3_dimension_isolation.csv', 4),
    4: ('table4_order_sensitivity.csv', 6),
    5: ('table5_order_symmetrization_accuracy.csv', 6),
    6: ('table6_deepseek_osodt.csv', 6),
}


def read_csv(path: Path, expected_rows: int) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f'{path}: missing or duplicate columns')
        rows = list(reader)
    if len(rows) != expected_rows or any(None in row or not all(row.values()) for row in rows):
        raise ValueError(f'{path}: expected {expected_rows} complete rows')
    for row in rows:
        for key, value in row.items():
            if key not in {'model', 'method'} and not Decimal(value).is_finite():
                raise ValueError(f'{path}: invalid number in {key}')
    return rows


def near(actual: Decimal, expected: Decimal, source: str) -> None:
    if abs(actual - expected) > Decimal('0.0001'):
        raise ValueError(f'{source}: {actual} differs from {expected}')


def check() -> None:
    rows = {number: read_csv(ROOT / filename, count) for number, (filename, count) in TABLES.items()}
    for row in rows[3]:
        near(Decimal(row['delta_accuracy']), Decimal(row['odt_accuracy']) - Decimal(row['joint_accuracy']),
             f"Table 3 {row['model']} delta")
    for row in rows[5]:
        for sym in ('s1', 's2'):
            for fixed in ('original', 'reverse'):
                near(Decimal(row[f'{sym}_minus_fixed_{fixed}']),
                     Decimal(row[f'osodt_{sym}_accuracy']) - Decimal(row[f'fixed_{fixed}_mean_accuracy']),
                     f"Table 5 {row['model']} {sym}/{fixed}")
    expected_grid = {(model, method) for model in ('DeepSeek-v4-Flash', 'DeepSeek-v4-Pro')
                     for method in ('Original', 'Reverse', 'OSODT')}
    if {(row['model'], row['method']) for row in rows[6]} != expected_grid:
        raise ValueError('Table 6 model/method grid is incomplete')
    formal = read_csv(ROOT / 'formal' / 'official_metrics.csv', 3)
    indexed = {row['method']: row for row in formal}
    labels = ('DeepSeek-v3.2', 'RSLIR (OSODT)', 'Delta (RSLIR - DeepSeek-v3.2)')
    if set(indexed) != set(labels):
        raise ValueError('Table 9 method rows are incomplete')
    for column in formal[0]:
        if column == 'method':
            continue
        near(Decimal(indexed[labels[2]][column]),
             Decimal(indexed[labels[1]][column]) - Decimal(indexed[labels[0]][column]),
             f'Table 9 {column} delta')


if __name__ == '__main__':
    check()
    print('Paper CSV snapshots passed structural and arithmetic checks (Tables 3–6 and 9).')
