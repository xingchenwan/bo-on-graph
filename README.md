# Create virtual env & install dependencies
```
conda create -n graph
conda install networkx numpy pandas matplotlib seaborn jupyterlab
conda install botorch -c pytorch -c gpytorch -c conda-forge
conda activate graph
```