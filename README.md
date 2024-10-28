# SHINE: Social Homology Identification for Navigation in Crowded Environments


# Improving robot navigation in crowded environments using intrinsic rewards (ICRA 2023)

# [Paper](https://arxiv.org/pdf/2404.16705) || [Video](https://www.youtube.com/watch?v=pOzRyWBk7MI)

## Abstract
Navigating mobile robots in social environments remains a challenging task due to the intricacies of human-robot interactions. Most of the motion planners designed for crowded and dynamic environments focus on choosing the best velocity to reach the goal while avoiding collisions, but do not explicitly consider the high-level navigation behavior (avoiding through the left or right side, letting others pass or passing before others, etc.). In this work, we present a novel motion planner that incorporates topology distinct paths representing diverse navigation strategies around humans. The planner selects the topology class that imitates human behavior the best using a deep neural network model trained on real-world human motion data, ensuring socially intelligent and contextually aware navigation. Our system refines the chosen path through an optimization-based local planner in real time, ensuring seamless adherence to desired social behaviors. In this way, we decouple perception and local planning from the decision-making process. We evaluate the prediction accuracy of the network with real-world data. In addition, we assess the navigation capabilities in both simulation and a real-world platform, comparing it with other state-of-the-art planners. We demonstrate that our planner exhibits socially desirable behaviors and shows a smooth and remarkable performance.


## Code
To be done...

## Citation
If you use this work in your own research or wish to refer to the paper's results, please use the following BibTeX entries.
```bibtex
@article{martinez2024shine,
  title={SHINE: Social Homology Identification for Navigation in Crowded Environments},
  author={Martinez-Baselga, Diego and de Groot, Oscar and Knoedler, Luzia and Riazuelo, Luis and Alonso-Mora, Javier and Montano, Luis},
  journal={arXiv preprint arXiv:2404.16705},
  year={2024}
}
