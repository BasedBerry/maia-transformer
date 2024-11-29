# Maia2: A Unified Model for Human-AI Alignment in Chess

The official implementation of the NeurIPS 2024 paper **Maia-2** [[preprint](https://arxiv.org/abs/2409.20553)]. This work was led by [CSSLab](https://csslab.cs.toronto.edu/) at the University of Toronto.

## Abstract
There are an increasing number of domains in which artificial intelligence (AI) systems both surpass human ability and accurately model human behavior. This introduces the possibility of algorithmically-informed teaching in these domains through more relatable AI partners and deeper insights into human decision-making. Critical to achieving this goal, however, is coherently modeling human behavior at various skill levels. Chess is an ideal model system for conducting research into this kind of human-AI alignment, with its rich history as a pivotal testbed for AI research, mature superhuman AI systems like AlphaZero, and precise measurements of skill via chess rating systems. Previous work in modeling human decision-making in chess uses completely independent models to capture human style at different skill levels, meaning they lack coherence in their ability to adapt to the full spectrum of human improvement and are ultimately limited in their effectiveness as AI partners and teaching tools. In this work, we propose a unified modeling approach for human-AI alignment in chess that coherently captures human style across different skill levels and directly captures how people improve. Recognizing the complex, non-linear nature of human learning, we introduce a skill-aware attention mechanism to dynamically integrate players’ strengths with encoded chess positions, enabling our model to be sensitive to evolving player skill. Our experimental results demonstrate that this unified framework significantly enhances the alignment between AI and human players across a diverse range of expertise levels, paving the way for deeper insights into human decision-making and AI-guided teaching tools.

## Requirements

```sh
chess==1.10.0
einops==0.8.0
gdown==5.2.0
numpy==2.1.3
pandas==2.2.3
pyzstd==0.15.9
Requests==2.32.3
torch==2.4.0
tqdm==4.65.0
```

The version requirements may not be very strict, but the above configuration should work.

## Installation

```sh
pip install maia2
```

## Quick Start: Batch Inference

```python
from maia2 import model, dataset, inference
```

You can load a model for `"rapid"` or `"blitz"` games with either CPU or GPU.

```python
maia2_model = model.from_pretrained(type="rapid", device="gpu")
```

Load a pre-defined example test dataset for demonstration.

```python
data = dataset.load_example_test_dataset()
```

Batch Inference
- `batch_size=1024`: Set the batch size for inference.
- `num_workers=4`: Use multiple worker threads for data loading and processing.
- `verbose=1`: Show the progress bar during the inference process.

```python
data, acc = inference.inference_batch(data, maia2_model, verbose=1, batch_size=1024, num_workers=4)
print(acc)
```

`data` will be updated in-place to include inference results.


## Position-wise Inference

We use the same example test dataset for demonstration.
```python
prepared = inference.prepare()
```

Once the prepapration is done, you can easily run inference position by position:
```python
for fen, move, elo_self, elo_oppo in data.values[:10]:
    move_probs, win_prob = inference.inference_each(maia2_model, prepared, fen, elo_self, elo_oppo)
    print(f"Move: {move}, Predicted: {move_probs}, Win Prob: {win_prob}")
    print(f"Correct: {max(move_probs, key=move_probs.get) == move}")
```

Try to tweak the skill level (ELO) of the activce player `elo_self` and opponent play `elo_oppo`! You may find it insightful for some positions.


## Training

### Download data from [Lichess Database](https://database.lichess.org/)

Please download the game data of the time period you would like to train on in `.pgn.zst` format. Data decompressing is handled by `maia2`, so you don't need to decompress these files before training.

### Training with our default settings

Please modify `your_data_root` to where you store the downloaded lichess data. It will take around 1 week to finish training 1 epoch with 2\*A100 and 16\*CPUs.

```python
from maia2 import train
import argparse

def parse_args(args=None):

    parser = argparse.ArgumentParser()

    # Supporting Arguments
    parser.add_argument('--data_root', default=your_data_root, type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--verbose', default=1, type=int)
    parser.add_argument('--max_epochs', default=1, type=int)
    parser.add_argument('--max_ply', default=300, type=int)
    parser.add_argument('--clock_threshold', default=30, type=int)
    parser.add_argument('--chunk_size', default=20000, type=str)
    parser.add_argument('--start_year', default=2018, type=int)
    parser.add_argument('--start_month', default=5, type=int)
    parser.add_argument('--end_year', default=2023, type=int)
    parser.add_argument('--end_month', default=11, type=int)
    parser.add_argument('--from_checkpoint', default=False, type=bool)
    parser.add_argument('--checkpoint_epoch', default=0, type=int)
    parser.add_argument('--checkpoint_year', default=2018, type=int)
    parser.add_argument('--checkpoint_month', default=5, type=int)
    parser.add_argument('--num_cpu_left', default=8, type=int)
    parser.add_argument('--queue_length', default=2, type=int)

    # Tunable Arguments
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--wd', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=8192, type=int)
    parser.add_argument('--first_n_moves', default=10, type=int)
    parser.add_argument('--last_n_moves', default=10, type=int)
    parser.add_argument('--dim_cnn', default=256, type=int)
    parser.add_argument('--dim_vit', default=1024, type=int)
    parser.add_argument('--num_blocks_cnn', default=5, type=int)
    parser.add_argument('--num_blocks_vit', default=2, type=int)
    parser.add_argument('--input_channels', default=18, type=int)
    parser.add_argument('--vit_length', default=8, type=int)
    parser.add_argument('--elo_dim', default=128, type=int)
    parser.add_argument('--side_info', default=True, type=bool)
    parser.add_argument('--side_info_coefficient', default=1, type=float)
    parser.add_argument('--value', default=True, type=bool)
    parser.add_argument('--value_coefficient', default=1, type=float)
    parser.add_argument('--max_games_per_elo_range', default=20, type=int)

    return parser.parse_args(args)

cfg = parse_args()
train.run(cfg)
```

If you would like to restore training from a checkpoint, please modify the `from_checkpoint`, `checkpoint_year`, and `checkpoint_month` to indicate the initialization you need.


## Citation

If you find our code or pre-trained models useful, please cite the arxiv version for now as follows:

```bibtex
@article{tang2024maia,
  title={Maia-2: A Unified Model for Human-AI Alignment in Chess},
  author={Tang, Zhenwei and Jiao, Difan and McIlroy-Young, Reid and Kleinberg, Jon and Sen, Siddhartha and Anderson, Ashton},
  journal={arXiv preprint arXiv:2409.20553},
  year={2024}
}
```

We will update the citation infomation to the official version once NeurIPS 2024 Proceedings are published.

## Contact

If you have any questions or suggestions, please feel free to contact us via email: josephtang@cs.toronto.edu.

## License

This project is licensed under the [MIT License](LICENSE).
