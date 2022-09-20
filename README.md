# PixAdapt: A novel approach to adaptive image encryption

This repository contains the code for our paper on [PixAdapt: A novel approach to adaptive image encryption](https://doi.org/10.1016/j.chaos.2022.112628). 

## Abstract

Image encryption using genetic approach is a recent and advanced technique which has grabbed attention in recent years. Currently, most image encryption algorithms (using genetic approach) use a static set of parameters for image encryption without considering the features representative of the image. In this study, an innovative adaptive image encryption algorithm – PixAdapt is developed. The process of image encryption is being re-engineered in a way to calculate the fitness of encrypted image using UACI and adapting the respective parameters using genetic hill climb or simulated annealing. Pseudorandom numbers have been generated using the linear feedback shift register and chaos-based maps such as the Logistic map, Rossler map, Henon map and Tent map. PixAdapt algorithm also uses confusion and diffusion process to ensure that plain text image and cipher text image are completely un-related. The use of metaheuristic search techniques for optimization of image encryption parameters has been implemented for the first time. The results obtained show that the genetic hill climb algorithm encrypts the various images giving the most optimal value of UACI. The algorithm has been tested for fitness improvement, parameter evolution, statistical analysis, and quality of encryption. PixAdapt is not only unique but has proven the encryption parameter UACI to be an appropriate fitness function to encrypt an image efficiently.

If you use any of the code or refer to this paper, please cite using:

```
@article{TULI2022112628,
title = {PixAdapt: A novel approach to adaptive image encryption},
journal = {Chaos, Solitons & Fractals},
volume = {164},
pages = {112628},
year = {2022},
issn = {0960-0779},
doi = {https://doi.org/10.1016/j.chaos.2022.112628},
url = {https://www.sciencedirect.com/science/article/pii/S0960077922008098},
author = {Rohan Tuli and Hitesh Narayan Soneji and Prathamesh Churi}}
publisher = {Elsevier}
```

## License
This project is licensed under the GNU General Public License v3.0. See LICENSE for more details
