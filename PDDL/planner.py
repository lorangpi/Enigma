import os
import copy
import subprocess

pddl_dir = "./PDDL"
domain_dir = "Domains"
problem_dir = "Problems"

def add_predicates_to_pddl(pddl_name, init_predicates):
    pddl_file_path = pddl_dir + os.sep + problem_dir + os.sep + pddl_name
    with open(pddl_file_path, 'r') as file:
        lines = file.readlines()

    init_index = lines.index('(:init \n')
    for predicate, value in init_predicates.items():
        if value:
            # first convert the predicate of the form "p1(o1,o1)" to "p1 o1 o1"
            predicate = predicate.replace('(', ' ').replace(')', ' ').replace(',', ' ')
            # then add the predicate to the init section
            lines.insert(init_index + 1, f'({predicate})\n')

    # define new problem file path with the end file being named as "problem_dummy.pddl" (os.sep is used to handle the path separator)
    problem_path = pddl_dir + os.sep + problem_dir + os.sep + "problem_dummy.pddl"

    # overwrite the new problem file
    with open(problem_path, 'w') as file:
        file.writelines(lines)


def call_planner(domain, problem, structure="pddl"):
    '''
        Given a domain and a problem file
        This function return the ffmetric Planner output.
        In the action format
    '''
    domain_path = pddl_dir + os.sep + domain_dir + os.sep + domain + ".pddl"
    problem_path = pddl_dir + os.sep + problem_dir + os.sep + problem + ".pddl"
    if structure == "pddl":
        run_script = f"Metric-FF-v2.1/./ff -o {domain_path} -f {problem_path} -s 0"
        output = subprocess.getoutput(run_script)
        #print("Output = ", output)
        if "unsolvable" in output or "goal can be simplified to FALSE" in output:
            return False, False
        try:
            output = output.split('ff: found legal plan as follows\n')[1]
            output = output.split('\ntime spent:')[0]
            # Remove empty lines
            output = os.linesep.join([s for s in output.splitlines() if s])
        except Exception as e:
            print("The planner failed because of: {}.\nThe output of the planner was:\n{}".format(e, output))

        plan, game_action_set = _output_to_plan(output, structure=structure)
        return plan, game_action_set

def _output_to_plan(output, structure):
    '''
    Helper function to perform regex on the output from the planner.
    ### I/P: Takes in the ffmetric output and
    ### O/P: converts it to a action sequence list.
    '''
    if structure == "pddl":
        action_set = []
        for action in output.split("\n"):
            #if action.startswith('step'):
            try:
                action_set.append(''.join(action.split(": ")[1]))
            except IndexError:
                return False, False
        
        # convert the action set to the actions permissable in the domain
        game_action_set = copy.deepcopy(action_set)

        #for i in range(len(game_action_set)):
        #   game_action_set[i] = applicator[game_action_set[i].split(" ")[0]]
        #for i in range(len(game_action_set)):
        #    for j in range(len(game_action_set[i])):
        #        if game_action_set[i][j] in applicator.keys():
        #            game_action_set[i][j] = applicator[game_action_set[i]]
        return action_set, game_action_set