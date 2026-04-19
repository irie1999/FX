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

HistData.com の Generic ASCII M1 をそのまま読み込む場合 (ファイル移動不要):

```bash
# 単一ファイル
python -m fx.main --histdata "C:/Users/you/Downloads/DAT_ASCII_USDJPY_M1_2024.csv"

# 複数ファイル / glob / ディレクトリ指定もOK
python -m fx.main --histdata "C:/Users/you/Downloads/DAT_ASCII_USDJPY_M1_*.csv" --resample 1h

# 1 分足を 5 分足 / 1 時間足 / 日足にリサンプル
python -m fx.main --histdata "C:/Users/you/Downloads/" --resample 1h
```

HistData のタイムスタンプは EST（DST なし、UTC-5）なので内部で UTC に変換されます。

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
