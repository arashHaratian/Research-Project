# RankMe for Detecting Mode-Drop in Generative AI

To build the singularity container, please use:

```sh
singularity build pipeline.sif pipeline.def
```

> **YOU MAY NEED TO HAVE THE `CelebA` DATASET IN YOUR DIRECTORY.**



In order to reproduce the plots and the results, please use:

```sh
singularity run pipeline.sif
```


## Reference

[1] Quentin Garrido, Randall Balestriero, Laurent Najman, and Yann Lecun. Rankme: Assessing the downstream performance of pretrained self-supervised representations by their rank, 2023