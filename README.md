# MARL-UDS
Source code, data and figures for the MARL-enabled real-time control of urban drainage system.

The SWMM model [Astlingen](https://github.com/open-toolbox/SWMM-Astlingen) is the case area. Thanks to Congcong Sun and other contributers for developing this model.

- `model` contains the parameters of trained agents and the training episode history in DQN, IQL and VDN.
- `train` contains files of the training events.
- `test` contains the results data and figures included in the paper.
- `Astlingen.inp` The template SWMM inp file.
- `Astlingen_SWMM.inp` The SWMM inp file for BC.
- `1Astlingen-Erft1.txt`` The rainfall series forked from [Astlingen](https://github.com/open-toolbox/SWMM-Astlingen), and `rainfiles.csv` merge all the 4 series.
- `rains_2h.csv` The extracted rainfall events.
- Codes of the algorithms, agent, environment and tests included.
