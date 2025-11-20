import subprocess, argparse, json

def dfa_to_lp(results):
    dfa_path = results + '/graphs/bisimulation.dfa'
    visualize_graph_from_dfa(results)
    lp_path = results + '/graphs/bisimulation.lp'
    command = f"python ./learner-strips/asp/scripts/make_lp_from_dfa.py {dfa_path} {lp_path}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        print(f"Error: {error}")
    else:
        print(f"Output: {output}")

def visualize_graph_from_dfa(file_path):
    dfa_path = file_path + '/graphs/bisimulation.dfa'
    output_path = file_path + '/graphs/bisimulation.dot'
    command = f"./learner-strips/sat/src/strips --dump-ts-dot --output {output_path} {dfa_path}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        print(f"Error: {error}")
    else:
        print(f"Output: {output}")
    command = f"dot -Tpdf -O {output_path}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        print(f"Error: {error}")
    else:
        print(f"Output: {output}")

def lp_to_domain(results, domain):
    domain_path = results + '/graphs/' + domain
    lp_path = results + '/graphs/bisimulation.lp'
    #visualize_graph_from_lp(lp_path)
    with open(domain_path, 'w') as file:
        benchmark = f'{lp_path} 6 0 -c opt_psym=2 -c opt_osym=2 -c opt_asym=2 -c max_precs=8 -c max_num_invariants=2 -c max_predicates=3 -c max_static=1 VERIFY 7 0 hanoi1op_3x2.lp hanoi1op_3x3.lp hanoi1op_3x4.lp hanoi1op_4x3.lp'
        file.write(benchmark)

def asp_solver(results, domain):
    domain_path = results + '/graphs/' + domain
    command = f"python ./learner-strips/asp/scripts/incremental_solver.py --remove_dir --results {results} {domain_path} 0"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output, error = process.communicate()
    if error:
        print(f"Error: {error}")
    else:
        print(f"Output: {output}")

def extract_info_after_last_answer(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the last occurrence of "Answer:"
    last_answer_index = max(i for i, line in enumerate(lines) if 'Answer:' in line)
    
    # Extract the information after the last "Answer:"
    info_after_last_answer = ''.join(lines[last_answer_index+1:]).strip()
    
    return info_after_last_answer

def transform_to_action_model(input_string):
    # Split the input string into lines
    lines = input_string.split('\n')

    # Initialize the action model and state predicates
    action_model = {
        'Fluent Predicates': [],
        'Static Predicates': [],
        'Invariant': [],
        'Actions': [],
        'Objects': []
    }
    state_predicates = {}

    # Parse the input string
    for line in lines:
        # If the line starts with 'action', 'object', 'p_static', 'labelname', 'some_binary_predicate', 'prec', 'eff', 'p_arity', 'inv_schema', 'a_arity', 'unequal', 'p_used', 'num_invariants', 'Optimization', or 'Progression', ignore it
        if any(line.startswith(prefix) for prefix in ['action', 'object', 'p_static', 'labelname', 'some_binary_predicate', 'prec', 'eff', 'p_arity', 'inv_schema', 'a_arity', 'unequal', 'p_used', 'num_invariants', 'Optimization', 'Progression']):
            continue

        # If the line starts with 'val', extract the state and associated predicate
        elif line.startswith('val'):
            # Extract the state and predicate from the line
            state = int(line.split(',')[-1].strip(')'))
            predicate = line.split('(')[2].strip(')')

            # Add the predicate to the state predicates
            if state not in state_predicates:
                state_predicates[state] = []
            state_predicates[state].append(predicate)

        # Otherwise, add the line to the action model
        else:
            # Determine the category of the line
            if 'Fluent Predicates' in line:
                category = 'Fluent Predicates'
            elif 'Static Predicates' in line:
                category = 'Static Predicates'
            elif 'Invariant' in line:
                category = 'Invariant'
            elif 'Actions' in line:
                category = 'Actions'
            elif 'Objects' in line:
                category = 'Objects'

            # Add the line to the action model
            action_model[category].append(line)

    return action_model, state_predicates

def save_outputs_to_file(input_string, file_path):
    # Call the transform_to_action_model function
    action_model, state_predicates = transform_to_action_model(input_string)

    # Convert the outputs to JSON format
    action_model_json = json.dumps(action_model, indent=4)
    state_predicates_json = json.dumps(state_predicates, indent=4)

    # Write the outputs to the file
    with open(file_path, 'w') as file:
        file.write("Action Model:\n")
        file.write(action_model_json)
        file.write("\n\nState Predicates:\n")
        file.write(state_predicates_json)


# Write an automated command executer that executes "python incremental_solver.py --remove_dir --results {arg.results} {domain} 0"
if __name__ == "__main__":
    # Define the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='benchmark.txt', help='Name of the benchmark domain, cf "learner_strips" folder.')
    parser.add_argument('--results_dir', type=str, default='./data/', help='Path to the data folder')
    args = parser.parse_args()
    dfa_to_lp(args.results_dir)
    lp_to_domain(args.results_dir, args.domain)
    asp_solver(args.results_dir, args.domain)
    output_string = extract_info_after_last_answer(args.results_dir + '/solver_stdout.txt')
    save_outputs_to_file(output_string, args.results_dir + 'pddl_format.txt')






