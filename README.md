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

## グリッドサーチ

`tools/grid_search.py` でパラメータ空間を総当たりし、PF / Sharpe / MAR などでランキング。

```bash
python tools/grid_search.py --histdata "data/raw/DAT_ASCII_USDJPY_M1_*.csv" \
    --resample 1d --equity 500000 \
    --fast 10,15,20,30 --slow 30,50,75,100 \
    --rsi-period 10,14,21 \
    --stop-atr none,1.0,1.25,1.5,2.0 \
    --adx-threshold 0,20,25,30 \
    --top 30 --sort-by pf \
    --html results/grid.html --csv-out results/grid.csv
```

- カンマ区切りで各軸の候補値を指定
- `--stop-atr` に `none` を含めると「ストップ無効」も比較対象
- `--sort-by` は `pf` / `sharpe` / `cagr` / `mar` / `net_profit` から選択
- `--html` で上位 N 件をダークテーマ HTML で表示（`--no-open` でブラウザ起動抑止）
- `--csv-out` で全組み合わせの結果を CSV 保存（分析・可視化に利用）
- `fast >= slow` や `rsi_lower >= rsi_upper` の組み合わせは自動除外

## ウォークフォワード検証

`tools/walk_forward.py` は「各窓で再最適化 → 次の窓で OOS 検証」を繰り返し、**カーブフィットを見抜く**ための仕組みです。

```bash
python tools/walk_forward.py --histdata "data/raw/DAT_ASCII_USDJPY_M1_*.csv" \
    --resample 1d --equity 500000 \
    --train 540 --test 180 --step 180 \
    --fast 10,15,20,30 --slow 50,75,100 \
    --rsi-period 14,21 --rsi-upper 60,70 --rsi-lower 30 \
    --stop-atr 1.0,1.25,1.5,2.0 \
    --sort-by pf --html results/wfa.html
```

- `--train` / `--test` / `--step` は **バー数**（`--resample` 後の本数）
- 各窓: train 範囲でグリッドサーチ → ベストパラメータを test で OOS 検証
- 全 test 区間を繋いで **OOS 資産推移** と集計指標を算出
- HTML レポートには窓ごとの詳細 + パラメータ安定性（どのパラメータが何回選ばれたか）も含む
- 判定: **OOS PF > 1.5 なら本物**、OOS PF < 1.0 ならカーブフィット

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
- `--adx-threshold N` ADX > N の時だけエントリー (トレンド強度フィルタ、0 = 無効)
- `--adx-period N` ADX 算出期間 (default 14)
- `--currency SYM` HTML に表示する通貨記号 (default ¥、EURUSD 等は `$` 推奨)
- `--spread` スプレッド (price units)
- `--size` 1 トレードの枚数
- `--start YYYY-MM-DD` / `--end YYYY-MM-DD` バックテスト期間の絞り込み
  例: `--start 2024-01-01 --end 2024-12-31` で 2024 年のみ検証

## テスト

```bash
PYTHONPATH=src pytest -q
```
