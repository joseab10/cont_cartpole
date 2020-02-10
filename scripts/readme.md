# Deep RL in CartPole Environment

The purpose of this project is to test and compare different RL algorithms on OpenAI's CartPole environment.

The main scripts are for:

## Training

The script `train_agent.py` is the go-to program for setting up, training and testing an RL agent.

The script will run the training and test functions for each agent a specified number of times to get better statistics,
 save (using _pickle_) the model of the agent after each run, and save the statistics and their plots after all runs
  have taken place.

### Syntax
It's syntax is as follows

    train_agent.py --exp <experiment_file>

where:

- **`experiment_file`** _(str)_ _[*, def="experiments_example.json"]_: is the path to a json file including a list of the desired experiments to be run, each with it's own
configuration.
<br>An example can be found in `experiments_example.json`.
<br>**Note**: Not every single parameter needs to be configured, as a matter of fact, everything has already some
default values and only those things that need to be changed have to be included.
For the _defaults_, make sure to check the dictionaries in `parameters.py`. Be sure to also check the
`experiments_min.json` file for that purpose.

The objective of the experiment files is to keep track of all the configuration settings, while easily associating them
with their corresponding models, stats and plots. For that purpose, it is encouraged to not edit them after a successful
run, but rather create a new file for further testing.


## Testing

Even though `train_agent.py` already tests an agent's performance in a deterministic way, it is usually the case that
this is done for statistical purposes and without rendering the environment to allow for a quicker simulation. For that
 reason, the script `test_agent.py` can be used to actually see the agent in action.


### Syntax

It's syntax is as follows:
 
    test_agent.py --file <model_file> [--ts <time_steps>] [--inist <initial_state>] [--inirnd <initial_noise>]
    [--rew <reward_function>] [--sw <smoothing_window>]`

where:

- **`model_file`** _(str)_ : is the path to a pickle file containing an agent's saved state.

- **`time_steps`** _(int)_ _[Opt, def=5]_ : is the number of maximum steps that an episode should last.

- **`initial_state`** _(str)_ _[Opt, def="none"]_ : is the initial state of the environment.See the dictionary
 `_para_init_states_dict` in `parameters.py` to include more possible values.

    - _**"none"**_ : cart at _x=0_ with no velocity, pendulum pointing downwards with no angular velocity.
    
    - _**"up"**_ : cart at _x=0_ with no velocity, pendulum pointing upwards with no angular velocity.
    
- **`initial_noise`** _(str)_ _[Opt, def="none"]_ : is the initial uniform random noise to add to the initial state.
 To add more possible values, check `_para_init_noises_dict` in `parameters.py`.

    - _**"none"**_ : random uniform noise of ±0.05 is added to all state variables.
    
    - _**"det"**_ : no noise is added to the initial state, hence it is a deterministic initial position.
    
    - _**"360"**_ : random uniform noise of ±0.05 is added to all state variables except for the angular position, to
     which ±π is added. _I.e._: the pendulum can start at any possible angle.
    
- **`reward_function`** _(str)_ _[Opt, def="none"]_ : is the reward function to be used by the environment. In order to
 add more possible values, check `_para_reward_func_dict` in `parameters.py`.
 
    - _**"none"**_ : The original sparse reward function will be used.
        - _r(x,v,θ,ω)_ _=_ _-1_ if _x_ _<_ _-1_ or _x_ _>_ _1_, _1_ if _-0.1_ _≤_ _θ_ _≤_ _0.1_, else _0_
    
    - _**"info"**_ : A smoother, less sparse, and more _"informative"_ reward function will be used:
        - _r(x,v,θ,ω)_ _=_ _50/ts_ _·_ cos _(θ)³_ if _-1_ _≤_ _x_ _≤_ _1_, else _-100_ 
       
- **`smoothing_window`** _(int)_ _[Opt, def=10]_ : the smoothing window over which the statistics will be averaged for plotting
 purposes.
 
 
 ## Plotting
 
 The script `plot_data.py` can be used to generate plots from saved statistical data with different settings as the ones
 generated automatically by the `train_agent.py` script.
 
 
 ### Syntax
 
 It's syntax is as follows:
 
    plot_data.py --file <stats_file> [--nosh] [--save] [--dir <plot_path>] [--exp <experiment_name>] [--runs] [--nagg]
    [--smw <smoothing_window>]
    
where:

- **`stats_file`** _(str)_ : is the path to a saved statistics file to be plotted.

- **`--nosh`** : Flag for not displaying/showing the plots.

- **`--save`** : Flag for saving the plots as `*.png` image files.

    - **`plot_path`** _(str)_ _[Opt, def="../save/plots"]_ : Path for saving the plots.
    
    - **`experiment_name`** _(str)_ _[Opt, def="plot_stats"]_ : Name for the files to be saved. This will be combined
    with other information such as _train/test_, time stamp, etc. to form the complete file name.

- **`--runs`** : Flag for plotting individual run learning curves in a single figure.

- **`--nagg`** : Flag for not plotting aggregated statistics such as _min_, _max_, _stdev_ and _mean_.

- **`smoothing_window`** _(int)_ _[Opt, def=10]_ : the smoothing window over which the statistics will be averaged for plotting
 purposes.