# ロボット動作の再実行

動画の編集はCPUで実行済みです。以下の物理再生と新規モデル推論は、今回のWindows作業環境では実行していません。配列の整合性と制御フローのテストは実施していますが、シミュレータ・GPUまで含む動作確認済みとは扱いません。

## 外部環境

Linux / Python 3.10の独立した環境で、以下の公開依存を用意してください。講座の限定公開評価コードは必要ありません。

| 依存 | 収録時の参照版 |
|---|---|
| LIBERO-plus | `4976dc30028e805ff8094b55501d532c48fec182` |
| LIBERO（基本アセット） | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| LIBERO-plus assets | `Sylvest/LIBERO-plus` の `assets.zip`、revision `dd2bd61b7d9a6fef1abc52d606e983b41886a149` |
| 物理・環境 | numpy 1.26.4、mujoco 3.7.0、robosuite 1.4.0、gym 0.25.2、bddl 3.6.0 |
| 録画 | opencv-python-headless 4.11.0.86 |

導入は [LIBERO-plus](https://github.com/sylvestf/LIBERO-plus/tree/4976dc30028e805ff8094b55501d532c48fec182) と [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) の手順を参照してください。GL/EGL、ImageMagick/Wand、SciPy、scikit-imageなどの依存も必要です。パッケージの一部だけを動画編集用venvに追加する方式は避け、シミュレータ用環境を分けます。

LIBERO-plusの `libero/libero/assets` に拡張アセットを配置し、LIBEROの基本アセットも用意します。取得物は `external/` に置くとGit対象外になります。外部アセット自体はこのリポジトリに同梱しません。

以下ではリポジトリのルートを作業ディレクトリにし、シェル変数 `PAI_LIBERO_PLUS` と `PAI_LIBERO_BASE` に、それぞれ取得したリポジトリの絶対パスを設定した状態を想定します。

```sh
export PAI_LIBERO_PLUS=/path/to/LIBERO-plus
export PAI_LIBERO_BASE=/path/to/LIBERO
export LIBERO_CONFIG_PATH="$PWD/tmp/libero-config"
export PYTHONPATH="$PAI_LIBERO_PLUS:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl

python - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ['PAI_LIBERO_PLUS']) / 'libero/libero'
b = Path(os.environ['PAI_LIBERO_BASE']) / 'libero/libero'
for required in (p / 'bddl_files', p / 'assets', b / 'assets'):
    if not required.is_dir():
        raise SystemExit(f'Missing required directory: {required}')
config_dir = Path(os.environ['LIBERO_CONFIG_PATH'])
config_dir.mkdir(parents=True, exist_ok=True)
config = dict(benchmark_root=str(p), bddl_files=str(p/'bddl_files'),
              init_states=str(p/'init_files'), datasets=str(p/'datasets'),
              assets=str(b/'assets'))
# JSON is also valid YAML. Keep this config inside the ignored local tmp directory.
(config_dir/'config.yaml').write_text(json.dumps(config, indent=2), encoding='utf-8')
PY

python -m pip install -e .
python -m pai_demo --help
```

`LIBERO_CONFIG_PATH` をローカルに指定するため、既存のホームディレクトリのLIBERO設定を書き換える必要はありません。

## 1. 保存した行動を物理再生する

```sh
python -m pai_demo replay \
  --episode data/episodes/diag_t02_s00 \
  --bddl-root "$PAI_LIBERO_PLUS/libero/libero/bddl_files" \
  --output runs/replay-milk
```

`--output` は未作成のフォルダを指定します。`--no-video` を追加すると録画を省略しますが、この実装では観測カメラ用の描画環境は必要です。

コードはBDDLのSHA-256を確認し、初期状態を読み込み、記録と同じ10ステップの安定化処理を行ってから全保存行動を実行します。特定の3例には収録時と同じ事前resetも行います。出力は次のとおりです。

- `agentview.mp4` / `wrist.mp4`：今回の再生映像
- `trajectory.npz`：今回の行動と関節軌道
- `report.json`：初期状態・関節軌道・目標判定の一致結果

一致しない場合は `replay_matches=false` と終了コード2になります。異なる物理エンジン・アセット・初期化の結果を、元の動作が再現できたものとして扱いません。元の非対象物変位判定は出自情報として残しますが、この小さなランタイムで競技の衝突判定や総合スコアを再計算しません。

## 2. 新しくAIを推論する

[OpenVLA-OFTの公開導入手順](https://github.com/moojink/openvla-oft/blob/e4287e94541f459edc4feabc4e181f537cd569a8/SETUP.md) に従ったGPU環境と、2視点・8次元proprio・7次元actionのL1回帰ヘッドを備えた互換モデルを用意します。LoRA adapterだけではなく、マージ済みモデルとaction head / proprio projectorが必要です。

公開コード参照版は `e4287e94541f459edc4feabc4e181f537cd569a8` です。公開OFTソースを `PYTHONPATH` に追加し、次のように実行します。

```sh
export PAI_OFT_ROOT=/path/to/openvla-oft
export PYTHONPATH="$PAI_OFT_ROOT:$PYTHONPATH"
python -m pai_demo infer \
  --episode data/episodes/diag_t02_s00 \
  --bddl-root "$PAI_LIBERO_PLUS/libero/libero/bddl_files" \
  --checkpoint /path/to/local/merged-model \
  --unnorm-key YOUR_DATASET_STATISTICS_KEY \
  --max-steps 300 \
  --output runs/new-inference-milk
```

`YOUR_DATASET_STATISTICS_KEY` は、その重みの `dataset_statistics.json` にある正規化キーに置き換えます。モデルはローカルファイルから読み込み、重みや設定ファイルは書き換えません。

この推論入口は公開OFT APIに対する小さなアダプターです。収録時の独自モデル配備・transformers環境を丸ごと移したものではありません。同一の `mixed_000500` 重みは同梱しておらず、別の公開ベースモデルや環境を指定した新規推論が、既存録画と同じ行動になる保証はありません。

## 軽量検証

```sh
python scripts/check_runtime.py
```

これにより選定データの読み取り・ハッシュ・形状に加え、ダミー環境で初期化順序、保存行動の全件再生、軌道不一致の検出、新規推論の終了条件、不正行動の拒否を確認できます。これは実機やGPUでの試験の代わりではありません。
