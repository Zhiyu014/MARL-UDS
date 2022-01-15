# MARL-UDS
Source code, data and figures for the MARL-enabled real-time control of urban drainage system.

The SWMM model [Astlingen](https://github.com/open-toolbox/SWMM-Astlingen) is the case area. Thanks to Congcong Sun and other contributers for developing this model.

## 4.1 Synthenic Rainfall
- All the codes and SWMM inp files for Uncontrolled, BC and EFD under synthenic rainfall.
- `model` contains the parameters of trained agents and the training episode history.
- `train` contains files of the training events.
- `test` contains the testing models, results data and figures included in the paper.

## 4.2 Historical rainfall
- All the codes and SWMM inp files for control strategies under historical rainfall.
- Rainfall series data from [Astlingen](https://github.com/open-toolbox/SWMM-Astlingen).
- `model` contains the parameters of trained agents and the training episode history.
- `train` contains files of the training events.
- `test` contains the testing models, results data and figures included in the paper.

## 5.1 IQL
- Codes and data for independent Q-learning.
- Figures of the comparison between IQL & VDN included in the paper

## SI_D_Reward_shaping
- Parameters of the trained agents by the 3 reward functions.
- Figures of the Supplementart Information D.
