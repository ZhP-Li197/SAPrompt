# Data and Running

## Data

The included `../Data` folder has processed Wikipedia, MOOC, LastFM, and UCI data. For another dataset, place `ml_<name>.csv`, `ml_<name>.npy`, and `ml_<name>_node.npy` in `../Data/<name>/`.

Raw data: [Wikipedia](https://snap.stanford.edu/jodie/wikipedia.csv), [Reddit](https://snap.stanford.edu/jodie/reddit.csv), [MOOC](https://snap.stanford.edu/jodie/mooc.csv), [LastFM](https://snap.stanford.edu/jodie/lastfm.csv), [UCI](https://zenodo.org/records/7213796/files/uci.zip?download=1), [Contact](https://zenodo.org/records/7213796/files/Contacts.zip?download=1).

To preprocess a JODIE CSV from the `code` directory:

```bash
mkdir -p processed ../Data/wikipedia
cp ../Data/wikipedia/wikipedia.csv processed/wikipedia.csv
python utils/preprocess_data.py --data wikipedia --bipartite
cp processed/ml_wikipedia* ../Data/wikipedia/
```

## Run

From the `code` directory:

```bash
python -m pip install -r requirements.txt
python pre-training.py --data wikipedia
python run.py adapt --data wikipedia --task link
python run.py adapt --data wikipedia --task node
```

Outputs are saved in `results/`. Use `--data` for another dataset and `--checkpoint` to load a pre-trained TGN model.
