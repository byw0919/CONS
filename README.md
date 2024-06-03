# Cautiously-Optimistic Knowledge Sharing for Cooperative Multi-Agent Reinforcement Learning

The official code base of [**C**autiously-**O**ptimistic k**N**owledge **S**haring **(CONS)** (AAAI 2024)](https://ojs.aaai.org/index.php/AAAI/article/view/29677)

## Installation

### Dependicies
- gym==0.18.0
- matplotlib==3.5.2 
- numpy==1.19.4 
- ray==0.8.6 
- six==1.16.0 
- torch==1.12.0

Clone this repository and install:
```
pip install -r requirements.txt
```

## Traning
**Patient Gold Miner (PGM)**

* PGM-6ag: PGM-6ag-v0
```
python main.py --alg cons --env_name ma_gym:PGM-6ag-v0 --max_episodes 80000
```
* PGM-3ag: PGM-3ag-v0
```
python main.py --alg cons --env_name ma_gym:PGM-3ag-v0 --max_episodes 100000
```

**Find the Treasure (FT)**

* Find the Treasure: FindTreasure-v0

```
python main.py --alg cons --env_name ma_gym:FindTreasure-v0 --max_episodes 50000 --reuse_network True --individual_rewards False --c_ep 30000 --c_w 0.4
```

**Cleanup**

```
python main_cleanup.py --max_episodes 80000 --c_w 0.4
```
## Evaluating
some trained models are in `model`

**PGM-6ag**
```
python main.py --alg cons --env_name ma_gym:PGM-6ag-v0 --pkl_dir model/PGM-6ag/cons/ --load_model --evaluate --render
```
**PGM-3ag**
```
python main.py --alg cons --env_name ma_gym:PGM-3ag-v0 --pkl_dir model/PGM-3ag/cons/ --load_model --evaluate --render
```
**Find the Treasure**
```
python main.py --alg cons --env_name ma_gym:FindTreasure-v0 --reuse_network True --individual_rewards False --pkl_dir model/FindTreasure/cons/ --load_model --evaluate --render
```
**Cleanup**
```
python main_cleanup.py --pkl_dir model/Cleanup/cons/ --load_model --evaluate --render
```

## Citation
Please cite our AAAI paper if you use this repository in your publications:

```
@inproceedings{ba2024cautiously,
  title={Cautiously-Optimistic Knowledge Sharing for Cooperative Multi-Agent Reinforcement Learning},
  author={Ba, Yanwen and Liu, Xuan and Chen, Xinning and Wang, Hao and Xu, Yang and Li, Kenli and Zhang, Shigeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={16},
  pages={17299--17307},
  year={2024}
}
```

## Aknowledgement
* [starry-sky6688](https://github.com/starry-sky6688) for [MARL-Algorithms](https://github.com/starry-sky6688/MARL-Algorithms)
  * I implement CONS and [AdHoc_TD](https://www.ifaamas.org/AAMAS/aamas2017/proceedings/pdfs/p1100.pdf) based on `MARL-Algorithms`
* [Anurag Koul](https://github.com/koulanurag) for [ma-gym](https://github.com/koulanurag/ma-gym)
  * I implement `patient_gold_miner` and `find_treasure` based on `ma-gym`
* [Jiachen Yang](https://github.com/011235813/lio) and [Eugene Vinitsky](https://github.com/eugenevinitsky/sequential_social_dilemma_games) for cleanup environment

