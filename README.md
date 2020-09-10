# FruitRecognition
a simple fruit recognition system

[![](https://img.shields.io/badge/Python-3.7-yellow)](https://www.python.org/)
[![](https://img.shields.io/badge/PyTorch-1.3.1-brightgreen)](https://github.com/pytorch/pytorch)
[![](https://img.shields.io/badge/Numpy-1.15.1-red)](https://github.com/numpy/numpy/)
[![](https://img.shields.io/badge/Cv2-4.1.2-blue)](https://github.com/opencv/opencv)
[![](https://img.shields.io/badge/CUDA-8.0-orange)](https://developer.nvidia.com/cuda-downloads)

## Usage

### 1.prepare dataset

The folder structure is as follows
```
├── data
│   ├── dataset # train datasets
│   |── FIDS30  # original datasets
│   |── ...
│
├── outputs # generated images
│
├── train.py # training code
├── utils.py
├──...
```

### 2.preprocess cartoon images

```bash
cd ./data/
python transform_data.py
```

### 3.train

```bash
python main.py --cartoon_name your_cartoon_image # yon can see more arguments in config.py
```