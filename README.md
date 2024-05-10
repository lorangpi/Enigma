# To install this repository:
1. git clone the main repository
2. Install the dependencies for the Enigma repository (pip install -r requirements.txt)
3. Navigate the main repository and git clone https://github.com/lorangpi/robosuite_fetch.git
4. Rename the cloned repository "robosuite" (mv robosuite_fetch robosuite)
5. cd robosuite
6. pip install -e .

# To Test a PDDL Domain and PDDL Problem file:
Metric-FF-v2.1/./ff -o ./PDDL/Domains/domain_blonet.pddl -f ./PDDL/Problems/problem_blonet.pddl -s 0

or

./fast-downward.py [options] <domain.pddl> <problem.pddl>

# To record demonstrations on the Hanoi Problem enviornment:
```
python record_demos_automation.py --split-action --seed 0 --name automated_demos
```
The `--split-action` argument splits the demonstration into four action steps: reach_pick, pick, reach_drop, and drop. This will generate a `data/demo_seed_0/automated_demos` directory that contains the recorded demonstrations. 

# To train action step polcies
first, convert the recorded demonstrations into HuggingFace trajectories
```
python data_preprocessing.py --data_dir data/demo_seed_0/automated_demos
```
Then launch either SQIL training or GAIL training
```
python SQIL_training.py --data_dir data/demo_seed_0/automated_demos -action reach_pick
```
This will launch policy learning for the reach_pick action step using the recorded demos. For sequential training, checkout the `sequential_learning` branch and repeat the above steps.

# To test the learned policies
first replace the paths to trained policies in `Executor_RL` with the paths to the learned policies, then:
```
python execution.py
```

# To learn a PDDL representation from these demonstrations:
python graph_to_pddl.py --results_dir <results_directory>
