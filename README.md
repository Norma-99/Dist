# Dist

## Execution commands 

To execute a shell script in Sert 
Note: you can add some parameters (-q, --constraint, --mem)

```bash
sbatch job_name.sh
```

To execute a desired experiment using the architecture

```bash
python3 -m differential_privacy --config configs/desired_dir/model.json
```

To generate a Feed Forward Neural Network model

```bash
python3 scripts/generate_ffnn.py --layer_size 32 32 32 > configs/desired_dir/model.json
```

To generate an LSTM Neural Network model

```bash
python3 scripts/generate_lstm.py --hidden_count 3 --layer_size 32 32 32 > configs/desired_dir/model.json
```

## Project folders

In the config folder you will find the different configurations for each network to execute. The config files contain the number of fog nodes, the number of iterations, epochs, name of the results folder etc... You can also find the model generated with the above command. 

The preprocessing files are found in the dist/ directory. All datasets (.pickle) generated from the preprocessing are stored in the datasets/ folder. 
In the dist folder it can also be found the entire models without the distributed architecture applied. 

It can be found two different jobs, one to execute Neural Networks without the distributed framework (job_test.sh) and another that runs the distributed framewowrk (job.sh). 
Note: In order to this files to work all the environment must be set as in the script. 

In the results/ folder all the results of the executions can be found. 

The last folder is the differential_privacy/ folder. This folder is where the distributed architecture is stored. 
The only script that you will need to touch is neural_network.py to adapt your network to the one in the architecture. The rest is set and ready to go. 


