# Face-Keypoints

## How to Use

- Clone this repo to your machine ```git clone https://github.com/Rajrup/Face-Keypoints.git```
- Install the python dependencies (see below)
- Download the checkpoints(see below)
- Double check the model paths
- Ready to run ```pipeline.py```

## Checkpoint Preparation

- Find all the models: [Google Drive](https://drive.google.com/drive/folders/1oMfRxOJ9pDtRz1v9r7AwdiXc7toTkQTq?usp=sharing)
- Extract the Tensorflow model files into ```models/``` folder.
  - Face: Path will look this - ```./models/frozen_inference_graph_face.pb```
  - PRNet: Path will look this - ```./models/PRNet/net-data/256_256_resfcn256_weight.data-00000-of-00001```

## Requirements
This code has been tested in ```Python 3.7```.
See ```requirements.txt``` for python packages.

```bash
pip install -r requirements.txt
```

## Running Pipeline

- Tensorflow Pipeline:

    ```python
    python pipeline
    ```

- Tensorflow Serving Pipeline: TODO

## Components

One module's output will go to the next one

- Video Reader
- Face Detection ([Face](https://github.com/yeephycho/tensorflow-face-detection)
- Key Point Extraction ([PRNet](https://github.com/YadiraF/PRNet))
