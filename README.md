# To install this repository:
1. git clone the main repository
2. Navigate the main repository and git clone https://github.com/lorangpi/robosuite_fetch.git
3. Rename the cloned repository "robosuite" (mv robosuite_fetch robosuite)
4. cd robosuite
5. pip install -e .

# To Test a PDDL Domain and PDDL Problem file:
Metric-FF-v2.1/./ff -o ./PDDL/Domains/domain_blonet.pddl -f ./PDDL/Problems/problem_blonet.pddl -s 0

or

./fast-downward.py [options] <domain.pddl> <problem.pddl>

# To record demonstrations on the Hanoi Problem enviornment:
python record_demos.py

# To learn a PDDL representation from these demonstrations:
python graph_to_pddl.py --results_dir <results_directory>