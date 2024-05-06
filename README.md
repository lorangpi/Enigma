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
python record_demos_automation.py

# To learn a PDDL representation from these demonstrations:
python graph_to_pddl.py --results_dir <results_directory>