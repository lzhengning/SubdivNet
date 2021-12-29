# Subdivision-based Mesh Convolutional Networks

The implementation of `SubdivNet` in our paper,

[Subdivion-based Mesh Convolutional Networks](https://arxiv.org/abs/2106.02285)

![teaser](teaser.jpg)


## News
* This paper was accepted by ACM Transactions on Graphics. 
* Improvided implementation and more documentation.

## Features
* Provides implementations of mesh classification and segmentation on various datasets.
* Provides ready-to-use datasets, pretrained models, training and evaluation scripts.
* Supports a batch of meshes with different number of faces.

## Requirements
* python3.7+
* CUDA 10.1+
* [Jittor](https://github.com/Jittor/jittor)

To install other python requirements:

```setup
pip install -r requirements.txt
```

## Fetch Data
This repo provides training scripts for classification and segementation, 
on the following datasets,

- shrec11-split10
- shrec11-split16
- cubes
- manifold40 (based on ModelNet40)
- humanbody
- coseg-aliens

To download the preprocessed data, run

```
sh scripts/<DATASET_NAME>/get_data.sh
```

> The `Manfold40` dataset (before remeshed, without subdivision connectivity) can be downloaded via [this link](https://cloud.tsinghua.edu.cn/f/2a292c598af94265a0b8/?dl=1). This version cannot be used as inputs of SubdivNet. To train SubdivNet, run scripts/manifold40/get_data.sh

## Training
To train the model(s) in the paper, run this command:

```
sh scripts/<DATASET_NAME>/train.sh
```

To speed up training, you can use multiple gpus. First install `OpenMPI`: 

```
sudo apt install openmpi-bin openmpi-common libopenmpi-dev
```

Then run the following command,

```
CUDA_VISIBLE_DEVICES="2,3" mpirun -np 2 sh scripts/<DATASET_NAME>/train.sh
```

## Evaluation

To evaluate the model on a dataset, run:

```
sh scripts/<DATASET_NAME>/test.sh
```

The pretrained weights are provided. Run the following command to download them.

```
sh scripts/<DATASET_NAME>/get_pretrained.sh
```

## Visualize
After testing the segmentation network, there will be colored shapes in a `results` directory. Use your favorite 3D viewer to check them.

## How to apply SubdivNet to your own data
SubdivNet cannot be directly applied to any meshes, because 
To create your own data with subdivision sequence connectivity, you may use the provided
tool that implements the MAPS algorithm. You may also refer to [NeuralSubdivision](https://github.com/HTDerekLiu/neuralSubdiv), as they provide a MATLAB scripts for remeshing.

To run our implemented MAPS algorithm, first install the following python dependecies,

```
triangle
pymeshlab
shapely
sortedcollections
networkx
rtree
```

Then see `datagen_maps.py` and modify the configurations to remesh your 3D shapes for subdivision connectivity.

## Cite
Please cite our paper if you use this code in your own work:

```
@misc{hu2021subdivisionbased,
      title={Subdivision-Based Mesh Convolution Networks}, 
      author={Shi-Min Hu and Zheng-Ning Liu and Meng-Hao Guo and Jun-Xiong Cai and Jiahui Huang and Tai-Jiang Mu and Ralph R. Martin},
      year={2021},
      eprint={2106.02285},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
