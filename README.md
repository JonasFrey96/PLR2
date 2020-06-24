# TrackThis

![](doc/TrackThis%20Kalman%20Filter.png)

## Instructions

### General

This is the base setup to performe experiments for 6D object detection.
The main file to run and evaluate your network can be found in `tools/lightning.py`

In `tools/lightning.py` implement your training procedure.
Two config files have to be passed into this file the enviroment `env` and experiment `exp` file.

- `env` defines your local setup and global paths outside of the repository.
- `exp` contains all hyperparameters and training steup needed to train your model.

Both files are passed into the lightning module should be stored in the `yaml/'-folder.

### Installation:

Prerequests:

- Cuda Version: 10.2

Install conda env:

```
cd PLR2
conda env create -f environment.yml
```

Install KNN (not tested):

```
conda activate track
cd PLR2/lib/knn
python setup.py build
cp -r lib.linux-x86_64-3.7/* ./
```

Setting up global variables:
Got to:
`yaml/env/env_ws.yml`
and edit the following pathts:
TODO

### Lightning Module:

#### Dataloaders:

Can be configured for training, validation and testing.
The test data is a complete hold-out-data and should only be used before we publish our paper.

The validation data is used to tune hyperparameters.

The configuration of the dataloader looks as follows:

```
d_val:
  name: "ycb"
  objects: 21
  num_points: 1000
  num_pt_mesh_small: 500
  num_pt_mesh_large: 2300
  obj_list_fil: null
  obj_list_sym:
    - 12
    - 15
    - 18
    - 19
    - 20
  batch_list_cfg:
    sequence_names: null #either null or what name should be in the sequence for example to only plot hard interactions
    seq_length: 3
    fixed_length: true
    sub_sample: 1
    mode: "test" #dense_fusion_test
    add_syn_to_train: false
  noise_cfg:
    status: false
    noise_trans: 0.0
  output_cfg:
    force_one_object_visible: true
    status: false
    refine: false
    add_depth_image: false
    visu:
      status: true
      return_img: true
```

**batch_list_cfg:** configures the data that is used:

5 Options exists:

- dense_fusion_test: (auto sequence length of 1, exact same data as for DenseFusion)
- dense_fusion_train: (auto sequence length of 1, exact same data as for DenseFusion)
- test: (sequence data can be generated length > 1, same sequences as for DenseFusion selected)
- train: (sequence data can be generated length > 1, same sequences as for DenseFusion selected (only real data used) )
- train_inc_syn: (real and synthetic data used)

**output_cfg:** specifies what is returned by the dataloader

**Important:**
In the lightning module the validation and training data are seperated via **_sklearn.model_selection.train_test_split_** in `def train_dataloader(self):` Since everything is seeded with 42 the train and validation split is reproducable. In the curent validation setuo the validation data constists out of synthetic and real data.

#### Logging:

Each run is stored in a seperated folder.
The folder is configured in: `exp_ws` _model_path_

self.hparams are automatically stored as a yaml. In our case we add the env and exp to reproduce exactly our experiment.

Logging is done via Tensorboard:
When using VS-Code this automatically does all the port forwarding for you:
`tensorboard --logdir=./ --host=0.0.0.0`

Have a look at the PyTorch lightning Tensorboard integration.
Simply return in `def validation_epoch_end` or `def validation_step` a **'log': tensorboard_logs** (tensorboard_logs contains a dict with all metrices that should be logged)

#### Visu:

Images can be added via self.logger.experiment which is a TensorBoard Summary writer.

#### ModelCheckpoints and Loading:

Check PytorchLightning Documentation,

### Extending the Network:

Add modules to the `src` folder or files to `lib`.

## License

Licensed under the [MIT License](LICENSE)
