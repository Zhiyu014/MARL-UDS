# MARL-UDS
Source code, data and figures for the MARL-enabled real-time control of urban drainage systems. The published article refer to [Towards coordinated and robust real-time control: a decentralized approach for combined sewer overflow and urban flooding reduction based on multi-agent reinforcement learning](https://authors.elsevier.com/c/1gI5s9pi-WQFd).

## UDS Environments
1. **Astlingen**: A benchmark SWMM model [Astlingen](https://github.com/open-toolbox/SWMM-Astlingen) of a combined sewer system with 6 storage tanks and 4 controllable outflow orifices. Thanks to Dr. Sun and other contributors for developing this model.
    - Sun C, Lorenz Svensen J, Borup M, Puig V, Cembrano G, Vezzaro L. An MPC-Enabled SWMM Implementation of the Astlingen RTC Benchmarking Network. Water. 2020; 12(4):1034. https://doi.org/10.3390/w12041034
    - Schütze, M.; Lange, M.; Pabst, M.; Haas, U. Astlingen—A benchmark for real time control (RTC). Water Sci. Technol. 2017, 2, 552–560. https://doi.org/10.2166/wst.2018.172

2. **Chaohu**: A real-case model of a combined sewer system with 3 pump stations and forebays. Papers using this model:
    - Liao, Z., Gu, X., Xie, J., Wang, X., & Chen, J. (2019). An integrated assessment of drainage system reconstruction based on a drainage networkmodel. Environmental Science and Pollution Research, 26(26), 26563–26576. https://doi.org/10.1007/s11356-019-05280-1
    - Zhi, G., Liao, Z., Tian, W., Wang, X., & Chen, J. (2019). A 3D dynamic visualization method coupled with an urban drainage model. Journal ofHydrology, 577, 123988. https://doi.org/10.1016/j.jhydrol.2019.123988
    - Tian, W., Liao, Z., Zhi, G., Zhang, Z.&Wang, X., 2022b. Combined Sewer Overflow and Flooding Mitigation Through a Reliable Real-Time Control Based on Multi-Reinforcement Learning and Model Predictive Control. Water Resources Research, 58(7): e2021WR030703. https://doi.org/10.1029/2021WR030703
    - Tian, W., Liao, Z., Zhang, Z., Wu, H.&Xin, K., 2022a. Flooding and Overflow Mitigation Using Deep Reinforcement Learning Based on Koopman Operator of Urban Drainage Systems. Water Resources Research, 58(7): e2021WR030939. https://doi.org/10.1029/2021WR030939

## Algorithms
1. DQN: Deep Q-learning (double & dueling network)
2. IQL: [Independent Q-learning](https://arxiv.org/abs/1511.08779)
3. VDN: [Value Decomposition Network](https://arxiv.org/abs/1706.05296)
4. QMIX: [Monotonic Value Function Factorisation](https://arxiv.org/abs/2003.08839)
5. A2C&MAA2C(demo): [advantage actor-critic](https://arxiv.org/abs/1602.01783)
6. PPO&MAPPO(demo): 
    - [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
    - [MAPPO](https://arxiv.org/abs/2103.01955)

## Requirements:
- tensorflow >= 2.3
- tensorflow_probability >= 0.11.1
- spektral == 1.2.0
- pyswmm >= 0.6.2
- pystorms >= 1.0.0
- swmm-api == 0.2.0.18
- pymoo == 0.6.0
