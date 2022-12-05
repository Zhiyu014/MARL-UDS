# MARL-UDS
Source code, data and figures for the MARL-enabled real-time control of urban drainage systems.

## UDS Environments
1. **Astlingen**: A benchmark model of a combined sewer system with 6 storage tanks and 4 controllable outflow orifices. The SWMM model [Astlingen](https://github.com/open-toolbox/SWMM-Astlingen) is the case area. Thanks to Dr. Sun and other contributers for developing this model.

2. **chaohu**: A real-case model of a combined sewer system with 3 pump stations and forebays. Papers using this model:
    - Liao, Z., Gu, X., Xie, J., Wang, X., & Chen, J. (2019). An integrated assessment of drainage system reconstruction based on a drainage networkmodel. Environmental Science and Pollution Research, 26(26), 26563–26576. https://www.researchgate.net/deref/https%3A%2F%2Fdoi.org%2F10.1007%2Fs11356-019-05280-1
    - Zhi, G., Liao, Z., Tian, W., Wang, X., & Chen, J. (2019). A 3D dynamic visualization method coupled with an urban drainage model. Journal ofHydrology, 577, 123988. https://www.researchgate.net/deref/https%3A%2F%2Fdoi.org%2F10.1016%2Fj.jhydrol.2019.123988
    - Tian, W., Liao, Z., Zhi, G., Zhang, Z.&Wang, X., 2022b. Combined Sewer Overflow and Flooding Mitigation Through a Reliable Real-Time Control Based on Multi-Reinforcement Learning and Model Predictive Control. Water Resources Research, 58(7): e2021WR030703. https://doi.org/10.1029/2021WR030703
    - Tian, W., Liao, Z., Zhang, Z., Wu, H.&Xin, K., 2022a. Flooding and Overflow Mitigation Using Deep Reinforcement Learning Based on Koopman Operator of Urban Drainage Systems. Water Resources Research, 58(7): e2021WR030939. https://doi.org/10.1029/2021WR030939

## Algorithms
1. DQN (D3QN)
2. IQL
3. VDN
4. QMIX

## Requirements:
- tensorflow >= 2.3
- pyswmm >= 0.6.2
- pystorms >= 1.0.0
- swmm-api == 0.2.0.18
- pymoo == 0.6.0
