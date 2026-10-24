# 外部ソフトウェアと記録の出典

このリポジトリは、追加学習済みOpenVLA-OFTによる選定済みのロボット実行記録と、それを扱うコードを収録しています。モデル推論と物理シミュレーションには以下の外部ソフトウェア・アセットを使用します。

> **注記：** 外部ライブラリのソース、シーンアセット、モデル重みは同梱していません。ライブラリの導入やアセット・重みの取得には、各提供元の配布物が別途必要です。参照版と導入手順は [再実行手順](docs/reproduction.md) を参照してください。

## 外部依存

| 依存先 | 役割 | 参照先 |
|---|---|---|
| OpenVLA-OFT | 画像・言語・ロボット状態からの行動推論 | [GitHub](https://github.com/moojink/openvla-oft) |
| OpenVLA-OFT+ | 追加学習の出発点となった公開モデル | [Hugging Face](https://huggingface.co/Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata) |
| LIBERO-plus | 録画時に使ったシミュレータと公開タスク | [GitHub](https://github.com/sylvestf/LIBERO-plus) |
| LIBERO | 基盤となるロボットシミュレーションとアセット | [GitHub](https://github.com/Lifelong-Robot-Learning/LIBERO) |
| LIBERO-plus assets | 拡張シーンのアセット | [Hugging Face](https://huggingface.co/datasets/Sylvest/LIBERO-plus) |
| NumPy | 動作配列の読み取り・検証 | [公式サイト](https://numpy.org/) |
| FFmpeg / imageio-ffmpeg | 映像の連結・画面配置・出力 | [FFmpeg](https://ffmpeg.org/) / [imageio-ffmpeg](https://github.com/imageio/imageio-ffmpeg) |
| Pillow | 検査用画像の作成 | [公式サイト](https://python-pillow.github.io/) |

外部依存には、それぞれの提供元のライセンス・利用条件が適用されます。本リポジトリは、第三者のソフトウェア・モデル・アセットに独自のライセンスを付与するものではありません。

## 映像と推論の関係

`media/source/` は、追加学習済み OpenVLA-OFT `mixed_000500` がシミュレータ内のロボットを制御した実行記録の選抜です。動画生成AIによる映像や手動操作の録画ではありません。`data/episodes/` の行動は、その実行時に記録されたものです。

保存行動の再生は、新たなAI推論ではありません。新規推論には、外部の推論コードと別途用意したモデル重みが必要です。収録時の追加学習済み重みは同梱していないため、公開ベースモデルを用意しても同じ動作になる保証はありません。

各記録の指示・実行条件・成功判定・ハッシュは `data/manifest.json` と [記録の出自](docs/provenance.md) に記載しています。学習経緯は [モデルの説明](docs/model.md) を参照してください。
