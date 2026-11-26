![LOGO](https://github.com/DeepWave-KAUST/DIIP_FWI/blob/main/asset/Figure1.png)

Reproducible material for **Diffusion Image Prior for Enhanced Full Waveform Inversion - Omar M. Saad, Hao Zhang and Tariq Alkhalifah**

# Project structure
This repository is organized as follows:

* :open_file_folder: **Model**: folder containing the diffusion model and the pretrained weights (uploaded to Zenodo https://zenodo.org/records/21432155);
* :open_file_folder: **asset**: folder containing logo;
* :open_file_folder: **data**: folder containing data (uploaded to the restricted area);
* :open_file_folder: **utils**: folder containing the utilites

## Notebooks
The following notebooks are provided:

- :open_file_folder: **DIIP_FWI_Syn1**: script for FWI using DIIP under a first synthetic scenario;
- :open_file_folder: **DIIP_FWI_Syn2**: script for FWI using DIIP under a second synthetic scenario;
- :open_file_folder: **DIIP_Viking**:   script for multi-scale FWI using DIIP for Viking field data;


## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. 

Remember to always activate the environment by typing:
```
conda activate DIIP
```

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce RTX A6000 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.

## Cite us 
```bibtex
@article{saad2026diffusion,
  author  = {Saad, Omar M. and Zhang, Hao and Alkhalifah, Tariq},
  title   = {Diffusion Image Prior for Enhanced Full Waveform Inversion},
  journal = {Journal of Geophysical Research: Machine Learning and Computation},
  year    = {2026},
  volume  = {3},
  number  = {6},
  pages   = {e2026JH001602},
  doi     = {10.1029/2026JH001602}
}

