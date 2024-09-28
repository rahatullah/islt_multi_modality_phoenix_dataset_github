conda install numpy pandas tensorboard matplotlib tqdm pyyaml -y &&
pip install opencv-python wget torchvision &&
conda install pytorch cpuonly -c pytorch -y &&
pip install torch-directml tensorflow-cpu==2.10 tensorflow-directml-plugin torchtext==0.6.0 portalocker openpyxl progress jupyterlab notebook voila