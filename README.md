# OLX Click Predict
_Hybrid Recurrent Neural Network architectures_

Author: Dimitris Michailidis.  
Conducted during my internship between April and June 2019.  
For Questions please contact: jimichailidis@gmail.com

### Setup
Create a virtual environment with conda/pip and then run:  
`conda install --file requirements.txt` or  
`pip install -r requirements.txt`

Then you need to install the local modules using pip:  
`pip install -e .` (note: the dot should be included)

#### Process:
1.  Then run the following script:  
`python scripts/filter_dataset.py`  
This will create a new folder data/olx/clicks/clean that contains:
    - Sessions of length >= 2.
    - Only click interactions, not impressions.
    - Items that are clicked more than 10 times in total.

2. Run the preparation script to create train and test sets:  
`python scripts/prepare_train.py`

3. Run the prepare_attributes script to append attributes from the json files to the datasets.  
`python experiments/olx_clicks/prepare_attributes.py`
4. Run the prepare_vehicles script to create the final train/test datasets.  
`python scripts/prepare_vehicles.py`

### How to run train/evaluation
_Check out figures/Code diagram.pdf for a diagram of the modules._
The training and evaluation of the models is run via one of the following files:

- `./experiments/olx_clicks/run.py` for the hybrid RNN models.
- `./experiments/baselines/run.py` forthe non-RNN baselines.

The Args class under experiments/olx_clicks/args.py controls the inputs/configurations of the model.  
Take a look at it for details of each configuration.

A sample dataset can be run to ensure that everything works as intended. To run it change the following configurations:  
~~~~
self.data_folder = 'data/olx_train/clicks/samples/'
self.train_path = 'train.csv'
self.valid_path = 'test.csv'
self.test_path = 'test.csv'
self.batch_size = 10
~~~~

### Hybrid combinations using Args
_detailed information about each mode can be found under helpers/enums.py  _
Hybrid models can be configured using 4 enums in args.py: [location_mode, parallel_mode, latent_mode, combination_mode]
  
- **parallel_mode**: activates the parallel mode in 3 different stages. If not NONE, it also requires the location_mode and the combination_mode to be set accordingly. Note: parallel_mode cannot be active (not NONE) when latent_mode is active. Only one of them can work at the same time. 
- **latent_mode**: activates the latent model in 4 different modes. If not NONE, it also requires the location_mode to be not NONE. It does not need the combination_mode to be set. Note: latent_mode cannot be active (not NONE) when parallel_mode is active. Only one of them can work at the same time.
- **location_mode**: controls how the location information is being represented. Valid for both latent and parallel modes. Cannot be NONE when parallel_mode or location_mode is active. It can also be used on its own without parallel/latent mode. If yes, it will act as base model which uses only location to predict the next items.
- **combination_mode**: controls how the parallel models are is only relevant when parallel_mode is not NONE. Other than that its value does not matter.

Examples:  
The baseGRU model configuration:
~~~
self.location_mode = LocationMode.NONE
self.parallel_mode = ParallelMode.NONE
self.latent_mode = LatentMode.NONE
self.combination_mode = CombinationMode.WEIGHTED_SUM # does not matter
~~~

The best performing paralllel configuration (Decoder weighted sum combination):
~~~
self.location_mode = LocationMode.FULLCONCAT
self.parallel_mode = ParallelMode.DECODER
self.latent_mode = LatentMode.NONE
self.combination_mode = CombinationMode.WEIGHTED_SUM
~~~

The best performing latent configuration:
~~~
self.location_mode = LocationMode.FULLCONCAT
self.parallel_mode = ParallelMode.NONE
self.latent_mode = LatentMode.LATENTAPPEND
self.combination_mode = CombinationMode.WEIGHTED_SUM
~~~

### Training Logs
Each time you run run.py two files are created under logs:

- <current_datetime>_train: logs the error, mrr, recall for each session at each step.
- <current_datetime>_sessions: logs the rank of the next item for each session at each step.
