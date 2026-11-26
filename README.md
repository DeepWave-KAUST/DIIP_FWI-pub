![LOGO](https://github.com/DeepWave-KAUST/DIIP_FWI/blob/main/asset/Figure1.png)

Reproducible material for **Diffusion Image Prior for Enhanced Full Waveform Inversion - Omar M. Saad, Hao Zhang and Tariq Alkhalifah**

[Click here](https://kaust.sharepoint.com/:f:/r/sites/M365_Deepwave_Documents/Shared%20Documents/Restricted%20Area/REPORTS/DW0131?csf=1&web=1&e=YLP9ah) to access the Project Report. Authentication to the _Restricted Area_ filespace is required.

# Project structure
This repository is organized as follows:

* :open_file_folder: **Model**: folder containing the diffusion model and the pretrained weights (uploaded to the restricted area);
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
DW0131 - Omar et al. (2026) Repository for DW0131: Diffusion Image Prior for Enhanced Full Waveform Inversion.

