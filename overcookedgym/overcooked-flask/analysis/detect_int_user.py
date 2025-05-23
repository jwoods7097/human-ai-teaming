import pickle, argparse, os, csv, copy, json
from pathlib import Path
from rl_to_pddl import (
    print_occupancy, build_terrain_grid, get_initial_agent_positions,
    initialize_grid,  update_grid_for_interact_and_record,
    update_grid_for_move_and_record, construct_grid_from_state, process_actions
)



def extract_cells_with_object(grid_state):
    """
    Extracts cells from the grid state that contain the 'object' field.

    Args:
    - grid_state (dict): A dictionary where keys are cell positions (tuples), and
                         values are dictionaries representing the cell's properties.

    Returns:
    - dict: A filtered dictionary with only the cells that contain the 'object' field.
    """
    return {position: properties for position, properties in grid_state.items() if 'object' in properties}

def filter_effect_list_by_state(effect_list, state_snapshot):
    """
    Filters the effect list by:
    1. Removing entries where the object is not present or its state doesn't match in the state snapshot.
    2. Keeping only the most recent entry for duplicate objects based on their state.

    Args:
    - effect_list: A list containing effect entries for a single agent at a timestep.
                   Each effect entry is a tuple of (effect_object, timestamp).
    - state_snapshot: A 2D list representing the state of the grid at a timestep.

    Returns:
    - A filtered effect list with valid and deduplicated entries.
    """
    
    
    filtered_effect_list = []  # Store valid entries
    object_states_timestamps = {}  # Track the most recent entry for each object by its state

    for effect_entry in effect_list:
        # Unpack the effect entry
        try:
            effect_object, timestamp = effect_entry
        except ValueError:
            print(f"Unexpected effect_entry structure: {effect_entry}")
            continue  # Skip invalid entries

        is_valid = True  # Flag to determine if we should keep this entry
        # print("this is the object",effect_object)
        for position, object_properties in effect_object.items():
            # print(position)
            x, y = position  # Extract cell coordinates
            
            try:
                # Retrieve the actual state from the snapshot
                actual_state = state_snapshot[y][x]
                actual_object = actual_state.get('object', None)
                object_name = object_properties['name'] # comment this out when soup fixed
                if actual_object == None:
                    is_valid = False
                    break
                elif object_properties['id'] != actual_object['id']:
                    is_valid = False
                    break
                # print("its valid")

            except IndexError:
                # If the position is out of bounds in the state snapshot
                is_valid = False
                break
            # Check if this object's state already exists in the filtered list
            object_state_id = object_properties['id']
            # print(object_state_id)
            # print(object_states_timestamps)
            if object_state_id in object_states_timestamps:
                # Compare timestamps: keep the most recent entry
                if timestamp <= object_states_timestamps[object_state_id]:
                    is_valid = False
                    break

            # Only keep the effect entry if it is valid
            if is_valid:
                # print("it should be valid")
                for position, object_properties in effect_object.items():
                    object_states_timestamps[object_state_id] = timestamp  # Update the most recent timestamp
                filtered_effect_list.append(effect_entry)

    return filtered_effect_list

def check_precondition_in_effect_list(action, effect_list_other_agent):
    """
    Check if the preconditions of an action, including the state of the object, 
    are present in the effect list of the other agent.

    Args:
    - action: A dictionary containing the action details.
    - effect_list_other_agent: The effect list for the other agent at the same timestep.

    Returns:
    - True if the precondition (including the state) is present in the effect list, False otherwise.
    """
    preconditions = extract_cells_with_object(action.get('preconditions', {}))
    # print(preconditions)
    pre_position = None
    giver_timestep = 0
    for pre_position, pre_data in preconditions.items():
        pre_object = pre_data['object']
    for effect_entry in effect_list_other_agent:
        effect_object, giver_timestep = effect_entry  # Extract the effect details and timestamp
        # print(effect_entry)
        for position, effect_data in effect_object.items():
            if pre_position == position and pre_object is not None:
                # print(pre_position, pre_data)
                if pre_object['name'] == effect_data['name']:
                        # print(effect_data)
                        return True,effect_data, giver_timestep
    return False, None, giver_timestep  # Precondition with state not found

def check_if_int_goal(int_obj_id, goal_object_arr):
    for goal_object in goal_object_arr:
        for object in goal_object['state'][0]:
            if object['id'] == int_obj_id or int_obj_id == goal_object['id']:
                return True
    return False
def check_if_giver_loop(int_obj_id, giver_agent_id, snapshot_logs_array):
    # at each timestep, check the cell where the giver agent is at and if they are holding that object again
    giver_agent = 'Agent_0' if giver_agent_id == 0 else 'Agent_1'
    for t, curr_state in enumerate(snapshot_logs_array):
        # print(curr_state[0][0])
        flattened_state = [element for row in curr_state for element in row]
        for cell in flattened_state:
            # print("this is cell", cell)
            if cell[giver_agent]['is_present'] == True and cell[giver_agent]['held_objects'] is not None:
                giver_holding_obj = cell[giver_agent]['held_objects']
                # print(giver_holding_obj)
                if giver_holding_obj['id'] == int_obj_id:
                    print("giver got object back", int_obj_id)
                    return False
    return True

def check_if_receiver_loop(int_obj_id, rec_agent_id, snapshot_logs_array):
    # at each timestep, check the cell where the giver agent is at and if they are holding that object again
    rec_agent = 'Agent_0' if rec_agent_id == 0 else 'Agent_1'
    for t, curr_state in enumerate(snapshot_logs_array):
        # print(curr_state[0][0])
        flattened_state = [element for row in curr_state for element in row]
        for cell in flattened_state:
            # print("this is cell", cell)
            if cell[rec_agent]['is_present'] == True and cell[rec_agent]['held_objects'] is not None:
                rec_holding_obj = cell[rec_agent]['held_objects']
                # print(giver_holding_obj)
                if rec_holding_obj['id'] == int_obj_id:
                    print("receiver had object already", int_obj_id)
                    return False
    return True
def process_pkl(directory):
    for root, _, files in os.walk(directory):
        keys = ['task_reward', 
                'constructive_interdependencies', 
                'non-constructive interdependencies', 
                "looping interdependencies", 
                "non goal-reaching object interdependencies"]
        # int_dict_by_reward = {key: None for key in keys}
        for file in files:
            print("here", file)
            if file.endswith('.pkl'):  # Process only JSON files
                file_path = os.path.join(root, file)
                path = Path(file_path)
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                        traj_id = path.stem
                        human_id = int(traj_id.split("=")[1][0])
                        agent_id = 0 if human_id == 1 else 1
                        
                        cons_int, non_cons_int, loop_int, irr_int,acc_trig_ag, acc_trig_h, rec_ag, rec_h = detect_int(data, human_id, agent_id)
                        reward_hier = path.parent.name
                        
                        save_path = os.path.join(root, "obj_results.csv")
                        
                        with open(save_path, 'a', newline = '') as file_layout:
                            # print("here")
                            writer = csv.writer(file_layout)
                            row = [reward_hier, cons_int, non_cons_int, loop_int, irr_int]
                            print(row)
                            writer.writerow(row)      

                # append to dictionary here       
                except Exception as e:
                    print(f"Error processing {file_path}:,", {e})

def process_pkl_ac_dist(directory):    
    for root, _, files in os.walk(directory):
        keys = ['task_reward', 
                'ind_h',
                'trig_h', 
                'acc_trig_h', 
                'ind_ag',
                "trig_ag", 
                "acc_trig_ag"]
        # int_dict_by_reward = {key: None for key in keys}
        for file in files:
            
            if file.endswith('.pkl'):  # Process only JSON files
                # file is the trajectory name    
                # print("here", file)       
                file_path = os.path.join(root, file)
                path = Path(file_path)
                try:
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                        traj_id = path.stem
                        human_id = int(traj_id.split("=")[1][0])
                        agent_id = 0 if human_id == 1 else 1

                        # total actions = 405
                        # _, _, _, _, acc_trig_ag, acc_trig_h, rec_ag, rec_h = detect_int(data, human_id, agent_id)
                        action_count_h, ind_h, trig_h, action_count_ag, ind_ag, trig_ag = detect_ac_dist(data, human_id, agent_id)
                        print("here")
                        _, _, _, _, acc_trig_ag, acc_trig_h, rec_ag, rec_h = detect_int(data, human_id, agent_id)
                        if args.is_forced == True and human_id == 0 :
                            save_path = os.path.join(root, "traj_obj_results.csv")
                            # only save length
                            with open(save_path, 'a', newline = '') as file_layout:
                                writer = csv.writer(file_layout)
                                row = [action_count_h,
                                        len(ind_h), 
                                    len(trig_h), 
                                    len(acc_trig_h), 
                                    len(rec_h),
                                    action_count_ag,
                                    len(ind_ag), 
                                    len(trig_ag), 
                                    len(acc_trig_ag),
                                    len(rec_ag)]
                                writer.writerow(row)    
                                # also save the events
                            save_path = os.path.join(root, "traj_subj_results.csv")
                            with open(save_path, 'a', newline = '') as file_layout:
                                writer = csv.writer(file_layout)
                                row = [ind_h, 
                                    trig_h, 
                                    acc_trig_h, 
                                    rec_h,
                                    ind_ag, 
                                    trig_ag, 
                                    acc_trig_ag,
                                    rec_ag]
                                writer.writerow(row)  
                        elif args.is_forced != True:
                            save_path = os.path.join(root, "traj_obj_results.csv")
                            # only save length
                            with open(save_path, 'a', newline = '') as file_layout:
                                writer = csv.writer(file_layout)
                                row = [action_count_h,
                                        len(ind_h), 
                                    len(trig_h), 
                                    len(acc_trig_h), 
                                    len(rec_h),
                                    action_count_ag,
                                    len(ind_ag), 
                                    len(trig_ag), 
                                    len(acc_trig_ag),
                                    len(rec_ag)]
                                writer.writerow(row)    
                                # also save the events
                            save_path = os.path.join(root, "traj_subj_results.csv")
                            with open(save_path, 'a', newline = '') as file_layout:
                                writer = csv.writer(file_layout)
                                row = [ind_h, 
                                    trig_h, 
                                    acc_trig_h, 
                                    rec_h,
                                    ind_ag, 
                                    trig_ag, 
                                    acc_trig_ag,
                                    rec_ag]
                                writer.writerow(row)  

                except Exception as e:
                    print(f"Error processing {file_path}:", {e})

def detect_ac_dist(data, human_id, agent_id):
    ind_h = []
    trig_h = []
    ind_ag = []
    trig_ag = []
    action_count_h = 0
    action_count_ag = 0
    action_logs = data['action_logs']
    for joint_action in action_logs:
        for action in joint_action:
            if action['agent'] == human_id:
                action_count_h += 1
                if action['action'] != 'interact':
                    ind_h.append("movement")
                elif action['action'] == 'interact':
                    if action['action_type'] == 'put_onion_on_counter' or action['action_type'] == 'put_dish_on_counter' or action['action_type'] == 'put_soup_on_counter':
                        trig_h.append(action['action_type'])
            if action['agent'] == agent_id:
                action_count_ag += 1
                if action['action'] != 'interact':
                    ind_ag.append("movement")
                elif action['action'] == 'interact':
                    if action['action_type'] == 'put_onion_on_counter' or action['action_type'] == 'put_dish_on_counter' or action['action_type'] == 'put_soup_on_counter':
                        trig_ag.append(action['action_type'])
    return action_count_h, ind_h, trig_h, action_count_ag, ind_ag, trig_ag

def detect_int(data, human_id, algo_id):
    cons_int = 0
    loop_int = 0 
    irr_int = 0 
    non_cons_int = 0
    
    snapshots = data['snapshots']
    action_logs = data['action_logs']
    # print("check",snapshots[10][4][1])
    # Convert action logs to array with sequential indices
    action_logs_array = []
    snapshot_logs_array = []
    for i in range(len(snapshots)):
        snapshot_logs_array.append(snapshots[i])
    effect_list = [[],[]]
    effect_list_old = [[], []]
    goal_object_arr = []
    # these are the giver lists
    acc_trig_h = []
    acc_trig_ag = []
    # these are the receiver lists
    rec_h = []
    rec_ag = []
    for t,joint_action in enumerate(action_logs):
        if  t<500:
            for agent_id,action in enumerate(joint_action):
                if action['action'] == 'interact' and action['action_type'] == 'deliver_soup':
                        print("Deliver at", t)
                        x,y = action['from_pos']
                        if agent_id == 0:
                            goal_object_arr.append(snapshot_logs_array[t-1][y][x]['Agent_0']['held_objects'])
                        else:
                            goal_object_arr.append(snapshot_logs_array[t-1][y][x]['Agent_1']['held_objects'])
    print(goal_object_arr)
    for t,joint_action in enumerate(action_logs):
        if t<500:
            for agent_id, action in enumerate(joint_action):
                # action is of form dictionary with keys
                # dict_keys(['agent', 'action', 'from_pos', 'to_pos', 'timestep', 'move_successful', 'preconditions', 'effects'])
                other_agent_id = 0 if agent_id == 1 else 1
                # print("lets check for", agent_id, " at ", t)
                # print(effect_list_old[other_agent_id])
                is_int, int_obj, giver_timestep = check_precondition_in_effect_list(action, effect_list_old[other_agent_id])
                if is_int:
                    goal_reaching = False
                    not_giver_loop = False
                    not_rec_loop = False
                    print("precondition matched at", t)
                    int_obj_id = int_obj['id']
                    if check_if_int_goal(int_obj_id, goal_object_arr):
                        goal_reaching = True
                        print("object was in goal state")
                    print("object is ", int_obj_id)
                    print("receiver object id", agent_id)
                    print("giver object id", other_agent_id)
                    if check_if_giver_loop(int_obj_id, other_agent_id, snapshot_logs_array[t:len(snapshot_logs_array)]):
                        not_giver_loop = True
                        print('object did not get back to giver')
                    if check_if_receiver_loop(int_obj_id, agent_id, snapshot_logs_array[0:t]):
                        print('object was not at receiver')
                        not_rec_loop = True
                    if goal_reaching == True and not_giver_loop == True and not_rec_loop == True:
                        cons_int += 1
                        # this is the receiver
                        if agent_id == human_id:
                            acc_trig_ag.append(action_logs[giver_timestep][other_agent_id]['action_type'])
                            rec_h.append(action['action_type'])
                        elif agent_id == algo_id:
                            acc_trig_h.append(action_logs[giver_timestep][other_agent_id]['action_type'])
                            rec_ag.append(action['action_type'])                            
                    elif not_giver_loop == False or not_rec_loop == False:
                        loop_int += 1
                    elif goal_reaching == False:
                        irr_int += 1
                    else:
                        non_cons_int += 1
                # print(action)
                if action['action'] == 'interact':
                    if action['action_type'] == 'put_onion_in_pot_2_onion':
                        for cell, pre in action['preconditions'].items():
                            if cell != action['from_pos']:
                                x,y = cell
                                action['effects'] = {cell: snapshot_logs_array[t+20][y][x]} 
                        
                # print("For agent at", agent_id, t, action['effects'])
                for cell, effect_dict_item in action['effects'].items():
                    # for all the cells that the action affects, only consider the objects field in the dictionary
                    effect_dict_item = {cell : effect_dict_item}
                    for key,values in effect_dict_item.items():
                        if 'object' in values and values['object'] is not None:
                            effect_dict_item_cell = {key : values['object']}
                            effect_list[agent_id].append([effect_dict_item_cell, t])
                
                effect_list[agent_id] = filter_effect_list_by_state(effect_list[agent_id], snapshots[t])
                # print("curr effect list for", agent_id, "at time", t, "length is", len(effect_list[agent_id]))
                # check if precondition of this action is present in the effect list of the other agent
            # print("here")
            effect_list_old = copy.deepcopy(effect_list)
            # print("effect list saved at ", t)
            # print(effect_list_old[0])

    # print(acc_trig_ag, acc_trig_h, rec_ag, rec_h)
    return cons_int, non_cons_int, loop_int, irr_int, acc_trig_ag, acc_trig_h, rec_ag, rec_h

                



if __name__ == "__main__":
    # Parse the directory argument from the command line
    parser = argparse.ArgumentParser(description="Process all traj pkl in directory, of structure - user_study/algo/layout/")
    parser.add_argument("--directory", type=str, help="Path to the directory containing JSON files")
    parser.add_argument("--is_forced",type=bool)
    args = parser.parse_args()

    # Call the function with the provided directory
    # process_pkl(args.directory)
    process_pkl_ac_dist(args.directory)
    # with open('human-ai-final.pkl', 'rb') as f:
    #     data = pickle.load(f)
    # print(detect_int(data))

