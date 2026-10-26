# 数据获取与运行

## 获取数据

项目已有的 `../Data/wikipedia`、`../Data/mooc`、`../Data/lastfm` 和 `../Data/uci` 含有可直接读取的 `ml_<数据集>.csv`、`ml_<数据集>.npy`、`ml_<数据集>_node.npy`。新增数据时，将这三个文件放在 `../Data/<数据集>/`。

原始数据下载地址：[Wikipedia](https://snap.stanford.edu/jodie/wikipedia.csv)、[Reddit](https://snap.stanford.edu/jodie/reddit.csv)、[MOOC](https://snap.stanford.edu/jodie/mooc.csv)、[LastFM](https://snap.stanford.edu/jodie/lastfm.csv)、[UCI](https://zenodo.org/records/7213796/files/uci.zip?download=1)、[Contact](https://zenodo.org/records/7213796/files/Contacts.zip?download=1)。对 JODIE 格式的原始 CSV，可在 `SAPrompt/code` 目录执行：

```bash
mkdir -p processed
cp ../Data/wikipedia/wikipedia.csv processed/wikipedia.csv
python utils/preprocess_data.py --data wikipedia --bipartite
mkdir -p ../Data/wikipedia
cp processed/ml_wikipedia.csv processed/ml_wikipedia.npy processed/ml_wikipedia_node.npy ../Data/wikipedia/
```

将 `wikipedia` 替换为相应的 JODIE 数据集名。UCI 和 Contact 需整理为相同的三个处理后文件，且事件 CSV 包含 `u,i,ts,label,idx` 列，节点和边索引分别对应两个 NPY 文件的行。

## 运行

在 `SAPrompt/code` 目录执行：

```bash
python -m pip install -r requirements.txt
python pre-training.py --data wikipedia --pretrain-epochs 20
python run.py adapt --data wikipedia --task link --epochs 20
python run.py adapt --data wikipedia --task node --epochs 20
```

预训练权重默认保存到 `results/tgn_wikipedia.pt`；运行 `adapt` 时可以通过 `--checkpoint` 指定其他 `THPrompt_Code` TGN 权重。结果保存在 `results/saprompt_<任务>_<数据集>.json`。通过 `--data-root` 指定其他数据目录，通过 `--device cpu` 在 CPU 上运行。
