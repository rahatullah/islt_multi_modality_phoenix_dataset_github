# Sign Language Transformers (CVPR'20)

This repo contains the training and evaluation code for the paper [Sign Language Transformers: Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation on Indian Sign Language](pending_link). 

This code is based on [Joey NMT](https://github.com/joeynmt/joeynmt) but modified to realize joint continuous sign language recognition and translation. For text-to-text translation experiments, you can use the original Joey NMT framework.
 
## Requirements
* It is recommended to use Windows 10+ due to better GPU shared memory and support for DirectML

* Download the feature files using the `data/download.sh` script.

* Create a conda or python virtual environment. Use these commands for conda environment creation:
    `conda create --name slt_cuda python="3.10"`

* Activate the environment. For conda:
    `conda activate slt_cuda`

* Create a conda python 3.10 environment and execute the following commands on the environment terminal.

    `conda install numpy scipy cython numba pandas tensorboard matplotlib tqdm pyyaml -y`
    `conda install cuda -c nvidia`
    `pip install opencv-python wget mediapipe seaborn scikit-learn cupy`
    `conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y`
    `pip install tensorflow[and-cuda] torchtext==0.6.0 portalocker openpyxl progress jupyterlab notebook voila`


## Usage
* for CUDA version
  `set CUDA_LAUNCH_BLOCKING=1`
  `python -m signjoey train configs/sign.yaml`


## Alternate Usage

* Activate the environment and run this command:

    `./Execute_all.sh`

! Note that the default data directory is `./data`. If you download them to somewhere else, you need to update the `data_path` parameters in your config file.   
## ToDo:

- [X] *Initial code release.*
- [X] (Nice to have) - Guide to set up conda environment included.
- [X] Ported the model to DirectML, code now compatible with AMD, Intel and Nvidia GPUs.
- [ ] *Release image features for ISL dataset.*
- [ ] Share extensive qualitative and quantitative results & config files to generate them.


## Reference

Please cite the paper below if you use this code in your research:

    @inproceedings{camgoz2020sign,
      author = {Praveen Kumar and Rina Damdoo and Rahul Gogoi},
      title = {Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation on Indian Sign language},
      booktitle = {pending},
      year = {2023}
    }

## Acknowledgements
<sub>This work was funded by the SNSF Sinergia project "Scalable Multimodal Sign Language Technology for Sign Language Learning and Assessment" (SMILE) grant agreement number CRSII2 160811 and the European Union’s Horizon2020 research and innovation programme under grant agreement no. 762021 (Content4All). This work reflects only the author’s view and the Commission is not responsible for any use that may be made of the information it contains. We would also like to thank NVIDIA Corporation for their GPU grant. </sub>


gfhdhfhhhdhf
dsshsdfhsdhdh