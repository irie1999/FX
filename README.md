# FX アルゴリズムトレード / バックテスト

シンプルな FX 取引アルゴリズム（SMA クロス + RSI フィルタ）と、ベクトル化バックテスト環境。

## 構成

```
src/fx/
  data.py       # CSV / 合成データのロード
  strategy.py   # シグナル生成 (SMA cross + RSI)
  backtest.py   # ベクトル化バックテスト (スプレッド考慮)
  metrics.py    # 損益・シャープ・最大DD・勝率
  main.py       # CLI エントリーポイント
tests/          # pytest テスト
```

## セットアップ

```bash
pip install -r requirements.txt
```

## 使い方

合成データでサンプル実行:

```bash
python -m fx.main --synthetic --bars 5000
```

CSV を使う場合 (列: `timestamp,open,high,low,close`):

```bash
python -m fx.main --csv path/to/usdjpy_1h.csv
```

主要オプション:

- `--fast` 短期 SMA 期間 (default 20)
- `--slow` 長期 SMA 期間 (default 50)
- `--rsi` RSI 期間 (default 14)
- `--rsi-upper` / `--rsi-lower` RSI フィルタの閾値
- `--spread` スプレッド (price units)
- `--size` 1 トレードの枚数

## テスト

```bash
PYTHONPATH=src pytest -q
```
