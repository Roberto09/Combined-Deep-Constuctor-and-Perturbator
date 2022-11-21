# Combining Constructive and Perturbative Deep Learning Algorithms for the Capacitated Vehicle Routing Problem

This is the official implementation code for the paper "Combining Deep Constructor and Perturbative Deep Learning Algorithms for the Capacitated Vehicle Routing Problem". Here you will find a way to test the models from the paper since we include their weights here. Additionally, you will be able to train your own models and test them.

## Before testing or training the CDCP
* We used conda to manage the environment used to train and test the CDCP. We include a "./env.yml" file with the packages of this environment. You can follow this [guide](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) to install the environment. To run any of the code in this repo, we strongly recommend installing and activating this environment.
* Currently you can only train or test if your computer or server has a CUDA device.

## Testing
Our models and the ones you train using "./train_models.py" will be stored, along with training data, inside the "./models" directory. Here you will already find the models we trained and used for the paper.

To test a model, you must run a command with the following structure:

```bash
$ python ./test_models.py <GRAPH_SIZE> <INSTANCES> <BATCH_SIZE_ADM> <MOD_DIR>
```
where:

* GRAPH_SIZE: this is the number of locations in the VRP, can only be {20, 50, 100}.
* INSTANCES: this is the number of VRP random instances which will be solved with the CDCP.
* BATCH_SIZE_ADM: this is simply how many instances will be solved at the time using the ADM.
* MOD_DIR: this argument is optional, it points to the model directory. If it is not included then we will use by default the model from the paper trained for the given graph size. 

example:
```bash
$ python ./test_models.py 20 10240 1024 ./models/model_20_paper
```

Testing will generate a set of solutions using CDCP. Such solutions will be stored in <MOD_DIR>/sampled_vrps_and_sols/for_evaluation/cdcp_paths.pkl. In general, after training and testing, the <MOD_DIR> inside the "./models" directory should look like this:

```
<MOD_DIR>
├── adm.pt
├── lsh.pt
└── sampled_vrps_and_sols
    ├── for_evaluation
    │   ├── batches_coords.npy
    │   ├── adm_paths.npy
    │   └── cdcp_paths.pkl
    └── for_training
        ├── batches_coords.npy
        └── adm_paths.npy
```

## Training
You can train your own models as well, for this you must run a command with the following structure:
```bash
$ python ./train_models.py <GRAPH_SIZE> <MEM_EFFICIENT_ADM> <TRAIN_STEPS>
```
where:
* GRAPH_SIZE: this is the number of locations in the VRP, can only be {20, 50, 100}.
* MEM_EFFICIENT_ADM: this is just a boolean flag to note wether to use the memory efficient version (discussed in the paper) of the ADM.
* TRAIN_STEPS: this is the number of steps the LSH will train for. A recommendation is to use as much (or a bit more) as the paper suggests for each graph size.

example:
```bash
$ python ./train_models.py 20 False 600
```

Training will generate a directory under ./models which will initially look like this:
```
./model_20_paper
├── adm.pt
├── lsh.pt
└── sampled_vrps_and_sols
    └── for_training
        ├── batches_coords.npy
        └── adm_paths.npy
```


## Some notes about training and testing:
* You can always just run the commands above in the background by using nohup. Example:
    ```bash
    $ nohup python ./test_models.py 20 10240 1024 ./models/model_20_paper &> testing_output.out &
    ```

    or

    ```bash
    $ nohup python ./train_models.py 20 False &> training_output.out &
    ```
