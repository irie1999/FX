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

### HistData を自動ダウンロード

ブラウザで手動 DL しなくても、`tools/fetch_histdata.py` がまとめて取得します（stdlib のみ、追加 pip install 不要）。

```bash
# 完了済みの年を年次 ZIP で取得
python tools/fetch_histdata.py --pair USDJPY --years 2023 2024

# 今年分は月単位で取得
python tools/fetch_histdata.py --pair USDJPY --months 2026-01 2026-02 2026-03

# そのままバックテストへ
python -m fx.main --histdata "data/raw/DAT_ASCII_USDJPY_M1_*.csv" --resample 1h
```

`--out` で保存先を変えられます。既に同名 CSV があればスキップします。

## HTML レポート出力

バックテスト結果を自己完結型の HTML レポートとして書き出せます（チャートは base64 PNG で埋め込み、外部リソース依存なし）。

```bash
# デフォルトの保存先 results/report.html、終わったら既定ブラウザで自動オープン
python -m fx.main --synthetic --bars 5000 --html

# 保存先を指定
python -m fx.main --histdata "data/raw/DAT_ASCII_USDJPY_M1_*.csv" --resample 1h \
    --html results/usdjpy_1h.html --title "USDJPY 1h backtest"

# ブラウザを開きたくない場合
python -m fx.main --synthetic --html --no-open
```

レポートに含まれる内容:
- パフォーマンス指標（Total Return / Sharpe / Max DD / 勝率 / PF ほか）
- パラメータ・期間・スプレッド設定
- エクイティカーブとドローダウン
- 価格チャート（SMA ・売買シグナルのマーカー付き）
- トレード別 PnL の棒グラフ
- トレード一覧（先頭 50 件）

主要オプション:

- `--fast` 短期 SMA 期間 (default 20)
- `--slow` 長期 SMA 期間 (default 50)
- `--rsi` RSI 期間 (default 14)
- `--rsi-upper` / `--rsi-lower` RSI フィルタの閾値
- `--stop-atr N` ATR ベースのストップロス (N × ATR 距離で自動決済)
- `--spread` スプレッド (price units)
- `--size` 1 トレードの枚数
- `--start YYYY-MM-DD` / `--end YYYY-MM-DD` バックテスト期間の絞り込み
  例: `--start 2024-01-01 --end 2024-12-31` で 2024 年のみ検証

## テスト

```bash
PYTHONPATH=src pytest -q
```
