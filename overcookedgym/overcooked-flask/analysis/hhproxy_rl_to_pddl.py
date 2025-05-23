# Global counters for object IDs
import copy
import argparse
import os
from pathlib import Path
soup_counter = 0
onion_counter = 0
dish_counter = 0

class Object:
    def __init__(self, name, position, state=None, id=None):
        self.name = name
        self.position = position
        self.state = state
        
        # Generate ID if not provided
        if id is None:
            global soup_counter, onion_counter, dish_counter
            if name == 'soup':
                soup_counter += 1
                self.id = f'soup_{soup_counter}'
            elif name == 'onion':
                onion_counter += 1 
                self.id = f'onion_{onion_counter}'
            elif name == 'dish':
                dish_counter += 1
                self.id = f'dish_{dish_counter}'
            else:
                raise ValueError(f"Unknown object name: {name}")
        else:
            self.id = id

def generate_object_id(name):
    """
    Generate a unique ID for an object based on its name.
    Args:
        name (str): Name of the object ('soup', 'onion', or 'dish')
    Returns:
        str: Unique ID in the format 'name_number'
    """
    global soup_counter, onion_counter, dish_counter
    if name == 'soup':
        soup_counter += 1
        return f'soup_{soup_counter}'
    elif name == 'onion':
        onion_counter += 1
        return f'onion_{onion_counter}'
    elif name == 'dish':
        dish_counter += 1
        return f'dish_{dish_counter}'
    else:
        raise ValueError(f"Unknown object name: {name}")



def build_terrain_grid(layout_list):
    terrain_grid = []
    for row in layout_list:
        terrain_row = []
        for cell in row:
            if cell == 'X':
                terrain_row.append('counter')
            elif cell == 'P':
                terrain_row.append('pot')
            elif cell == 'D':
                terrain_row.append('dish_dispenser')
            elif cell == 'O':
                terrain_row.append('onion_dispenser')
            elif cell == 'S':
                terrain_row.append('delivery')
            elif cell in [1, 2]:
                terrain_row.append('floor')  # agents start on floor
            elif cell is None:
                terrain_row.append('floor')
            else:
                raise ValueError(f"Unknown layout cell: {cell}")
        terrain_grid.append(terrain_row)
    return terrain_grid


def get_initial_agent_positions(layout_list):
    positions = {}
    for y, row in enumerate(layout_list):
        for x, cell in enumerate(row):
            if cell == 1 or cell == 2:
                positions[int(cell)-1] = (x, y)
    return [positions[i] for i in sorted(positions)]



def print_occupancy(grid):
    """
    Print a detailed view of the grid state showing terrain, agents, and objects.
    Symbols:
    - Agents: A0, A1 (with held objects in parentheses)
    - Objects: o=onion, d=dish, s=soup
    - Terrain: X=counter, P=pot, D=dish dispenser, O=onion dispenser, S=serving, .=floor
    """
    height = len(grid)
    width = len(grid[0])
    
    # Print column numbers for reference
    print('  ' + ' '.join([str(i) for i in range(width)]))
    
    agent_0_pos = None
    for y in range(height):
        row_str = [str(y)]  # Add row number
        for x in range(width):
            cell = grid[y][x]
            #print(cell)
            # First check for agents
            if cell['Agent_0']['is_present']:
                agent_0_pos = (x, y)
                if cell['Agent_0']['held_objects']:
                    row_str.append(f"A0({cell['Agent_0']['held_objects']['name'][:1]})")
                    # if cell['Agent_0']['held_objects']['name'] == 'soup':
                    #     print(f"soup: {cell['Agent_0']['held_objects']['state']}")
                    #     print("\n\n\n\n")
                else:
                    row_str.append("A0")
            elif cell['Agent_1']['is_present']:
                if cell['Agent_1']['held_objects']:
                    row_str.append(f"A1({cell['Agent_1']['held_objects']['name'][:1]})")
                else:
                    row_str.append("A1")
            # Then check for objects
            elif cell['object'] is not None:
                obj_name = cell['object']['name']
                if obj_name == 'onion':
                    row_str.append('o')
                elif obj_name == 'dish':
                    row_str.append('d')
                elif obj_name == 'soup':
                    # print("soup state")
                    # print(f"soup: {cell['object']['state']}")
                    # print("\n\n\n\n")
                    row_str.append('s')
                else:
                    row_str.append(obj_name[0])
            # Finally show terrain
            else:
                terrain = cell['terrain']
                if terrain == 'counter':
                    row_str.append('X')
                elif terrain == 'pot':
                    row_str.append('P')
                elif terrain == 'dish_dispenser':
                    row_str.append('D')
                elif terrain == 'onion_dispenser':
                    row_str.append('O')
                elif terrain == 'delivery':
                    row_str.append('S')
                elif terrain == 'floor':
                    row_str.append('.')
                else:
                    row_str.append(terrain[0].upper())
        print(' '.join(row_str))
    print(f"Agent 0 position: {agent_0_pos}")

def initialize_grid(terrain_grid, agent_positions):
    width = len(terrain_grid[0])  # Number of columns
    height = len(terrain_grid)    # Number of rows
    grid = []
    
    for y in range(height):
        row = []
        for x in range(width):
            cell = {
                'terrain': terrain_grid[y][x],
                'Agent_0': {
                    'is_present': False,
                    'hand_empty': True,
                    'held_objects': None,
                    'orientation': (0,-1)  # Default orientation
                },
                'Agent_1': {
                    'is_present': False,
                    'hand_empty': True,
                    'held_objects': None,
                    'orientation': (0,-1)  # Default orientation
                },
                'object': None,
                'is_empty': True
            }
            row.append(cell)
        grid.append(row)
    
    # Place agents in their initial positions
    for agent_idx, (x, y) in enumerate(agent_positions):
        grid[y][x][f'Agent_{agent_idx}']['is_present'] = True
        grid[y][x]['is_empty'] = False
    
    return grid



def get_preconditions_for_action(action_type, pos, target_pos, agent_key, current_held, target_cell):
    """
    Get preconditions for a given action type.
    Args:
        action_type: Type of action being performed
        pos: Current position of the agent
        target_pos: Target position for the action
        agent_key: Key for the agent in the grid
        current_held: Currently held object by the agent
        target_cell: Target cell information
    Returns:
        dict: Preconditions for the action
    """
    preconditions = {}
    
    if action_type == "pickup_onion_from_dispenser":
        preconditions = {
            pos: {
                agent_key: {
                    'hand_empty': True,
                    'held_objects': None
                }
            },
            target_pos: {
                'terrain': "onion_dispenser",
            }
        }
    elif action_type == "pickup_dish_from_dispenser":
        preconditions = {
            pos: {
                agent_key: {
                    'hand_empty': True,
                    'held_objects': None
                }
            },
            target_pos: {
                'terrain': "dish_dispenser",
            }
        }
    elif action_type == "pickup_onion_from_counter":
        preconditions = {
            pos: {
                agent_key: {
                    'hand_empty': True,
                    'held_objects': None
                }   
            },
            target_pos: {
                'terrain': 'counter',
                'object': {
                    'name': 'onion',
                    'position': target_pos,
                    'state': None
                }
            }
        }
    elif action_type == "pickup_dish_from_counter":
        preconditions = {
            pos: {
                agent_key: {
                    'hand_empty': True,
                    'held_objects': None
                }   
            },
            target_pos: {
                'terrain': 'counter',
                'object': {
                    'name': 'dish',
                    'position': target_pos,
                    'state': None
                }
            }
        }
    elif action_type == "put_onion_on_counter":
        preconditions = {
            pos: {
                agent_key: {
                    'hand_empty': False,
                    'held_objects': {
                        'name': 'onion',
                        'position': pos,
                        'state': None
                    }
                }
            },
            target_pos: {
                'terrain': 'counter',
                'object': None
            }
        }
    elif action_type == "put_dish_on_counter":
        preconditions = {
            pos: {
                agent_key: {
                    'hand_empty': False,
                    'held_objects': {
                        'name': 'dish',
                        'position': pos,
                        'state': None
                    }
                }
            },
            target_pos: {
                'terrain': 'counter',
                'object': None
            }
        }
    elif action_type == "put_onion_in_pot_0_onion":
        preconditions = {
            pos: {
                agent_key: {
                    'hand_empty': False,
                    'held_objects': {
                        'name': 'onion',
                        'position': pos,
                        'state': None
                    }
                }
            },
            target_pos: {
                'terrain': 'pot',
                'object': None  # Pot must be empty
            }
        }
    elif action_type == "put_onion_in_pot_1_onion":
        preconditions = {
            pos: {
                agent_key: {
                    'hand_empty': False,
                    'held_objects': {
                        'name': 'onion',
                        'position': pos,
                        'state': None
                    }
                }
            },
            target_pos: {
                'terrain': 'pot',
                'object': {
                    'name': 'soup',
                    'position': target_pos,
                    'state': ['onion', 1, 0]  # Pot must have 1 onion
                }
            }
        }
    elif action_type == "put_onion_in_pot_2_onion":
        preconditions = {
            pos: {
                agent_key: {
                    'hand_empty': False,
                    'held_objects': {
                        'name': 'onion',
                        'position': pos,
                        'state': None
                    }
                }
            },
            target_pos: {
                'terrain': 'pot',
                'object': {
                    'name': 'soup',
                    'position': target_pos,
                    'state': ['onion', 2, 0]  # Pot must have 2 onions
                }
            }
        }
    elif action_type == "pickup_soup_from_pot":
        preconditions = {
            pos: {
                agent_key: {
                    'hand_empty': False,
                    'held_objects': {
                        'name': 'dish',
                        'position': pos,
                        'state': None
                    }
                }
            },
            target_pos: {
                'terrain': 'pot',
                'object': {
                    'name': 'soup',
                    'position': target_pos,
                    'state': ('onion', 3, 20)  # Pot must have fully cooked soup
                }
            }
        }
    elif action_type == "put_soup_on_counter":
        preconditions = {
            pos: {
                agent_key: {
                    'hand_empty': False,
                    'held_objects': {
                        'name': 'soup',
                        'position': pos,
                        'state': ('onion', 3, 20)
                    }
                }
            },
            target_pos: {
                'terrain': 'counter',
                'object': None  # Counter must be empty
            }
        }
    elif action_type == "pickup_soup_from_counter":
        preconditions = {
            pos: {
                agent_key: {
                    'hand_empty': True,
                    'held_objects': None
                }
            },
            target_pos: {
                'terrain': 'counter',
                'object': {
                    'name': 'soup',
                    'position': target_pos,
                    'state': ('onion', 3, 20)
                }
            }
        }
    elif action_type == "deliver_soup":
        preconditions = {
            pos: {
                agent_key: {
                    'hand_empty': False,
                    'held_objects': {
                        'name': 'soup',
                        'position': pos,
                        'state': ('onion', 3, 20)
                    }
                }
            },
            target_pos: {
                'terrain': 'delivery'  # Target must be delivery station
            }
        }
    
    return preconditions

def can_perform_action(action_type, current_held, target_cell):
    """
    Determine if an action can be performed based on the current state.
    Args:
        action_type: Type of action being performed
        current_held: Currently held object by the agent
        target_cell: Target cell information
    Returns:
        bool: Whether the action can be performed
    """
    if action_type == 'pickup_onion_from_dispenser':
        return (
            current_held is None and
            target_cell['terrain'] == 'onion_dispenser'
        )
    elif action_type == 'pickup_dish_from_dispenser':
        return (
            current_held is None and
            target_cell['terrain'] == 'dish_dispenser'
        )
    elif action_type == 'pickup_onion_from_counter':
        return (
            current_held is None and
            target_cell['object'] is not None and
            target_cell['object']['name'] == 'onion'
        )
    elif action_type == 'pickup_dish_from_counter':
        return (
            current_held is None and
            target_cell['object'] is not None and
            target_cell['object']['name'] == 'dish'
        )
    elif action_type == 'put_onion_on_counter':
        return (
            current_held is not None and
            current_held['name'] == 'onion' and
            target_cell['terrain'] == 'counter' and
            target_cell['object'] is None
        )
    elif action_type == 'put_dish_on_counter':
        return (
            current_held is not None and
            current_held['name'] == 'dish' and
            target_cell['terrain'] == 'counter' and
            target_cell['object'] is None
        )
    elif action_type == 'put_soup_on_counter':
        return (
            current_held is not None and
            current_held['name'] == 'soup' and
            #current_held['state'] == ('onion', 3, 20) and
            target_cell['terrain'] == 'counter' and
            target_cell['object'] is None
        )
    elif action_type == 'pickup_soup_from_counter':
        return (
            current_held is None and
            target_cell['terrain'] == 'counter' and
            target_cell['object'] is not None and
            target_cell['object']['name'] == 'soup'
            #target_cell['object']['state'] == ('onion', 3, 20)
        )
    elif action_type == 'deliver_soup':
        return (
            current_held is not None and
            current_held['name'] == 'soup' and
            target_cell['terrain'] == 'delivery'
        )
    elif action_type == 'pickup_soup_from_pot':
        print(f"pickup soup from pot")
        print(f"target_cell: {target_cell}")
        return (
            current_held is not None and
            current_held['name'] == 'dish' and
            target_cell['terrain'] == 'pot' and
            target_cell['object'] is not None and
            target_cell['object']['name'] == 'soup' and
            target_cell['object']['state'][2] == 20
        )
    elif action_type in ['put_onion_in_pot_0_onion', 'put_onion_in_pot_1_onion', 'put_onion_in_pot_2_onion']:
        # Get current soup state
        current_soup = target_cell['object']
        onion_count = 0
        if current_soup is not None and current_soup['name'] == 'soup':
            onion_count = current_soup['state'][1]
        
        return (
            current_held is not None and
            current_held['name'] == 'onion' and
            target_cell['terrain'] == 'pot' and
            onion_count < 3  # Can't put more than 3 onions in pot
        )
    
    return False

def update_grid_for_move_and_record(agent_idx, other_agent_next_pos, from_pos, to_pos, timestep, grid, snapshots=None, action=None):
    """
    Update the grid for an agent's move and record the snapshot if snapshots dict is provided.
    Args:
        agent_idx: Index of the agent (0 or 1)
        from_pos: Current position of the agent (x, y)
        to_pos: Target position for the move (x, y)
        timestep: Current timestep
        grid: Current grid state
        snapshots: Dictionary to store snapshots
        action: The action being attempted (dx, dy) for movement or 'interact'
    Returns a dictionary containing the move log, preconditions, and effects.
    """
    if snapshots is None:
        snapshots = {}
    
    agent_key = f'Agent_{agent_idx}'
    x, y = from_pos
    orientation = grid[y][x][agent_key]['orientation']
    # print(f"Agent idx: {agent_idx}")
    # print(f"Orientation: {orientation}")
    # Record snapshot before move
    #record_grid_snapshot(grid, timestep, snapshots)
    
    # Get agent key
    agent_key = f'Agent_{agent_idx}'

    if agent_idx == 0:
        other_agent = 1
    else:
        other_agent = 0
    fx, fy = from_pos
    tx, ty = to_pos
    
    # Define preconditions as cell states before the move
    preconditions = {
        to_pos: {
            'terrain': 'floor',
            f'Agent_{other_agent}': {'is_present': False},
            'object': None,
        }
    }
    
    # Check if preconditions are met
    to_cell = grid[ty][tx]
    print(f"agent_idx: {agent_idx}")
    print(f"current_pos: {from_pos}")
    print(f"target_pos: {to_pos}")
    
    print(f"other_agent_next_pos: {other_agent_next_pos}")

    if agent_idx == 0:
        agent_1_pos = other_agent_next_pos[1]

        if grid[agent_1_pos[1]][agent_1_pos[0]]['terrain'] != 'floor' or grid[agent_1_pos[1]][agent_1_pos[0]]['object'] is not None or (agent_1_pos[0] == fx and agent_1_pos[1] == fy):
            agent_1_pos = other_agent_next_pos[0]

        can_move = (
            to_cell['terrain'] == 'floor' and
            not (agent_1_pos[0] == tx and agent_1_pos[1] == ty) and
            to_cell['object'] is None
        )
    else:
        can_move = (
            to_cell['terrain'] == 'floor' and
            to_cell['Agent_0']['is_present'] == False and
            to_cell['object'] is None
        )

    # if timestep == 5:
    #     print(f"Agent {agent_idx} is moving from {from_pos} to {to_pos}")
    #     print(f"to_cell: {to_cell}")
    #     print(f"can_move: {can_move}")
    
    # Only update orientation if there's actual movement (not 0,0)
    if action != 'interact' and action != (0, 0):
        grid[fy][fx][agent_key]['orientation'] = action

            
        # if agent_idx == 1:
        #     print(f"Agent {agent_idx} is moving from {from_pos} to {to_pos}")
        #     print(f"Orientation: {action}")
        #     print("Time step: ", timestep)
    
    # Only update positions if preconditions are met
    if can_move:
        # Update agent position and is_empty status
        grid[fy][fx][agent_key]['is_present'] = False
        grid[ty][tx][agent_key]['is_present'] = True
        if action != (0, 0):  # Only update orientation if there's actual movement
            grid[ty][tx][agent_key]['orientation'] = action
        
        # Update held object fields - copy the entire held object state
        held_obj = grid[fy][fx][agent_key]['held_objects']
        if held_obj is not None:
            held_obj = held_obj.copy()  # Create a copy to avoid reference issues
            held_obj['position'] = to_pos  # Update position to new location
        grid[fy][fx][agent_key]['held_objects'] = None
        grid[ty][tx][agent_key]['held_objects'] = held_obj
        
        
        # Update hand_empty status
        grid[fy][fx][agent_key]['hand_empty'] = True
        grid[ty][tx][agent_key]['hand_empty'] = held_obj is None
        
        
        # Update is_empty status for both cells
        grid[fy][fx]['is_empty'] = not (grid[fy][fx]['Agent_0']['is_present'] or grid[fy][fx]['Agent_1']['is_present'] or grid[fy][fx]['object'])
        grid[ty][tx]['is_empty'] = not (grid[ty][tx]['Agent_0']['is_present'] or grid[ty][tx]['Agent_1']['is_present'] or grid[ty][tx]['object'])
    else:
        # If preconditions not met, agent stays in current position
        to_pos = from_pos
        tx, ty = to_pos
    
    # Record snapshot after move
    #record_grid_snapshot(grid, timestep, snapshots)
    
    # Define effects as cell states after the move
    effects = {
        from_pos: {
            'terrain': grid[fy][fx]['terrain'],
            agent_key: grid[fy][fx][agent_key],
            'object': grid[fy][fx]['object'],
            'is_empty': grid[fy][fx]['is_empty']
        },
        to_pos: {
            'terrain': grid[ty][tx]['terrain'],
            agent_key: grid[ty][tx][agent_key],
            'object': grid[ty][tx]['object'],
            'is_empty': grid[ty][tx]['is_empty']
        }
    }
    
    # Create detailed action log
    action_log = {
        'agent': agent_idx,
        'action': action,  # Raw action (dx, dy) or 'interact'
        'from_pos': from_pos,
        'to_pos': to_pos,
        'timestep': timestep,
        'move_successful': can_move,
        'preconditions': preconditions,
        'effects': effects
    }
    
    return action_log, grid

def update_grid_for_interact_and_record(agent_idx, pos, timestep, grid, snapshots=None, next_state=None):
    """
    Update the grid for an agent's interact action and record the snapshot if snapshots dict is provided.
    Args:
        agent_idx: Index of the agent (0 or 1)
        pos: Current position of the agent (x, y)
        timestep: Current timestep
        grid: Current grid state
        snapshots: Dictionary to store snapshots
        next_state: State information from the next timestep to determine action type
    Returns a dictionary containing the action log, preconditions, and effects.
    """
    if snapshots is None:
        snapshots = {}
    
    # Record snapshot before interact
    #record_grid_snapshot(grid, timestep, snapshots)
    
    # Get agent key
    agent_key = f'Agent_{agent_idx}'
    x, y = pos
    
    # Get the cell in front of the agent based on orientation
    orientation = grid[y][x][agent_key]['orientation']
    #print(f"Agent idx: {agent_idx}")
    #print(f"Orientation: {orientation}")
    if orientation == (0, 1):  # Up
        target_cell = grid[y+1][x]
        target_pos = (x, y+1)
    elif orientation == (0, -1):  # Down
        target_cell = grid[y-1][x]
        target_pos = (x, y-1)
    elif orientation == (1, 0):  # Right
        target_cell = grid[y][x+1]
        target_pos = (x+1, y)
    else:  # Left
        target_cell = grid[y][x-1]
        target_pos = (x-1, y)
    

    
    # Determine the type of interact action based on current and next state
    action_type = None
    current_held = grid[y][x][agent_key]['held_objects']
    #print(current_held)
    effects = {}

    if next_state is not None:
        next_player = next_state['players'][agent_idx]
        next_held = next_player['held_object']

        print("Agent idx: ", agent_idx)
        print("Player position: ", pos)
        print("Player orientation: ", orientation)
        print("Player held: ", current_held)
        print("Next player held: ", next_held)
        print("Target cell: ", target_cell)
        print("Target pos: ", target_pos)
        print("Time step: ", timestep)
        #print("\n\n\n")
        if current_held is None and next_held is not None:
            print(target_cell)
            if target_cell['terrain'] == 'onion_dispenser':
                action_type = 'pickup_onion_from_dispenser'
            elif target_cell['terrain'] == 'dish_dispenser':
                action_type = 'pickup_dish_from_dispenser'
            if target_cell['object'] is not None:
                if target_cell['object']['name'] == 'onion':
                    action_type = 'pickup_onion_from_counter'
                elif target_cell['object']['name'] == 'dish':
                    action_type = 'pickup_dish_from_counter'
                elif target_cell['object']['name'] == 'soup':
                    action_type = 'pickup_soup_from_counter'
        
        # Check for put actions
        elif current_held is not None and next_held is None:
           # if timestep in [14,15,16,17]:
                #print(f"Agent {agent_idx} is holding {current_held['name']}")
            if target_cell['terrain'] == 'counter' and target_cell['object'] is None:
                if current_held['name'] == 'onion':
                    action_type = 'put_onion_on_counter'
                elif current_held['name'] == 'dish':
                    action_type = 'put_dish_on_counter'
                elif current_held['name'] == 'soup':
                    action_type = 'put_soup_on_counter'
            elif target_cell['terrain'] == 'pot':
                #print(f"target pos: {target_pos}")
                if current_held['name'] == 'onion':
                    if target_cell['object'] is None:
                        #print(f"time step: {timestep}")
                        action_type = 'put_onion_in_pot_0_onion'
                    elif target_cell['object']['name'] == 'soup' and target_cell['object']['state'][1] == 1:
                        action_type = 'put_onion_in_pot_1_onion'
                    elif target_cell['object']['name'] == 'soup' and target_cell['object']['state'][1] == 2:
                        action_type = 'put_onion_in_pot_2_onion'
            elif target_cell['terrain'] == 'delivery' and current_held['name'] == 'soup':
                action_type = 'deliver_soup'
                print(f"time step: {timestep}")
                print(f"SOUPPPP")
        
        elif current_held is not None and next_held is not None:
            if current_held['name'] == 'dish' and next_held['name'] == 'soup':
                action_type = 'pickup_soup_from_pot'
                #print(f"time step: {timestep}")
                #print(f"pickup soup from pot")

        #if action_type is None:
            #print(f"time step: {timestep}")

        # Get preconditions for the action
        preconditions = get_preconditions_for_action(
            action_type,
            pos,
            target_pos,
            agent_key,
            current_held,
            target_cell
        )

        # Check if action can be performed
        can_interact = can_perform_action(action_type, current_held, target_cell)

        if timestep == 155:
            print(f"can_interact: {can_interact} {action_type} {current_held} {target_pos}")
       # print(f"can_interact: {can_interact} {action_type} {current_held} {target_pos}")
        if can_interact:
            # Update grid based on action type
            if action_type in ['pickup_onion_from_dispenser', 'pickup_dish_from_dispenser']:
                # Update agent's held object with complete state
                
                held_obj = {
                    'name': 'onion' if action_type == 'pickup_onion_from_dispenser' else 'dish',
                    'position': pos,
                    'state': None,
                    'id': generate_object_id('onion' if action_type == 'pickup_onion_from_dispenser' else 'dish')  # Generate unique ID
                }
                grid[y][x][agent_key]['held_objects'] = held_obj
                grid[y][x][agent_key]['hand_empty'] = False

                effects = {
                    pos: {
                        agent_key: {
                            'hand_empty': False,
                            'held_objects': held_obj
                        }  
                    }
                }
                
            elif action_type in ['pickup_onion_from_counter', 'pickup_dish_from_counter']:
           
                picked_obj = target_cell['object']
                picked_obj['position'] = pos
                grid[y][x][agent_key]['held_objects'] = picked_obj
                grid[y][x][agent_key]['hand_empty'] = False
                
                # Remove object from target cell if it's not a dispenser
                
                target_cell['object'] = None
                target_cell['is_empty'] = True

                effects = {
                    pos: {
                        agent_key: {
                            'hand_empty': False,
                            'held_objects': picked_obj
                        }
                    },
                    target_pos: {
                        'object': None,
                        'is_empty': True
                    }
                }
                
            elif action_type in ['put_onion_on_counter', 'put_dish_on_counter']:
                # Clear agent's held object
                grid[y][x][agent_key]['held_objects'] = None
                grid[y][x][agent_key]['hand_empty'] = True
                
                # Place object on counter
                current_held['position'] = target_pos
                target_cell['object'] = current_held
                target_cell['is_empty'] = False
                print(f"Target Pos: {target_pos}")
                print(f"action_type: {action_type}")
                print("puttttt")

                effects = {
                    pos: {
                        agent_key: {
                            'hand_empty': True,
                            'held_objects': None
                        }
                    },
                    target_pos: {
                        'object': current_held,
                        'is_empty': False
                    }
                }
                
            elif action_type in ['put_onion_in_pot_0_onion', 'put_onion_in_pot_1_onion', 'put_onion_in_pot_2_onion']:
                # Get current soup state
                current_soup = target_cell['object']
                if current_soup is not None and current_soup['name'] == 'soup':
                    print(f"current_soup: {current_soup}")
                    print("\n\n\n\n")
                    onion_count = current_soup['state'][1]
                    onion_count += 1
                    # Create a completely new onion array with deep copies
                    new_onion_array = []
                    for onion in current_soup['state'][0]:
                        new_onion = {
                            'name': onion['name'],
                            'position': onion['position'],
                            'state': onion['state'],
                            'id': onion['id']
                        }
                        new_onion_array.append(new_onion)
                    # Add the new onion
                    new_onion = {
                        'name': current_held['name'],
                        'position': target_pos,
                        'state': current_held['state'],
                        'id': current_held['id']
                    }
                    new_onion_array.append(new_onion)
                    
                    # Create new soup object
                    new_soup = {
                        'name': 'soup',
                        'position': target_pos,
                        'state': (new_onion_array, onion_count, 0),
                        'id': current_soup['id']
                    }
                    
                    # Create new target cell
                    new_target_cell = {
                        'terrain': target_cell['terrain'],
                        'Agent_0': copy.deepcopy(target_cell['Agent_0']),
                        'Agent_1': copy.deepcopy(target_cell['Agent_1']),
                        'object': new_soup,
                        'is_empty': False
                    }
                    
                    # Update grid with new cell
                    grid[target_pos[1]][target_pos[0]] = new_target_cell
                    current_soup = new_soup

                    effects = {
                        pos: {
                            agent_key: {
                                'hand_empty': True,
                                'held_objects': None
                            }
                        },
                        target_pos: {
                            'object': copy.deepcopy(new_soup)
                        }
                    }
                
                else:
                    #print(f"current_held: {current_held}")
                    # Create new onion array with a single onion
                    new_onion = {
                        'name': current_held['name'],
                        'position': target_pos,
                        'state': current_held['state'],
                        'id': current_held['id']
                    }
                    soup_state = ([new_onion], 1, 0)
                    
                    # Create new soup object
                    new_soup = {
                        'name': 'soup',
                        'position': target_pos,
                        'state': soup_state,
                        "id": generate_object_id('soup')
                    }
                    
                    # Create new target cell
                    new_target_cell = {
                        'terrain': target_cell['terrain'],
                        'Agent_0': copy.deepcopy(target_cell['Agent_0']),
                        'Agent_1': copy.deepcopy(target_cell['Agent_1']),
                        'object': new_soup,
                        'is_empty': False
                    }
                    
                    # Update grid with new cell
                    grid[target_pos[1]][target_pos[0]] = new_target_cell
                    target_cell = new_target_cell

                    effects = {
                        pos: {
                            agent_key: {
                                'hand_empty': True,
                                'held_objects': None
                            }
                        },
                        target_pos: {
                            'object': copy.deepcopy(new_soup),
                            'is_empty': False
                        }
                    }
                grid[y][x][agent_key]['held_objects'] = None
                grid[y][x][agent_key]['hand_empty'] = True


            
            elif action_type == 'pickup_soup_from_pot':
                # Create a deep copy of the soup state array
                soup_state_obj = copy.deepcopy(target_cell['object']['state'][0])
                soup_state_obj.append(current_held)
                target_cell['object']['state'] = (soup_state_obj, target_cell['object']['state'][1], target_cell['object']['state'][2])
                held_obj = target_cell['object']
                held_obj['position'] = pos
                grid[y][x][agent_key]['held_objects'] = held_obj
                grid[y][x][agent_key]['hand_empty'] = False
                
                # print(f"PICKEED SOUPPPPP")
                # print(f"held_obj: {held_obj}")
                # Clear the pot
                target_cell['object'] = None
                target_cell['is_empty'] = True

                effects = {
                    pos: {
                        agent_key: {
                            'hand_empty': False,
                            'held_objects': held_obj
                        }
                    },
                    target_pos: {
                        'object': None,
                        'is_empty': True
                    }
                }
            
            elif action_type == 'put_soup_on_counter':
                
                held_obj = current_held
                held_obj['position'] = target_pos
                target_cell['object'] = held_obj
                
                target_cell['is_empty'] = False
                grid[y][x][agent_key]['held_objects'] = None
                grid[y][x][agent_key]['hand_empty'] = True
                
                # Place soup on counter
             

                effects = {
                    pos: {
                        agent_key: {
                            'hand_empty': True,
                            'held_objects': None
                        }
                    },
                    target_pos: {
                        'object': target_cell['object'],
                        'is_empty': False
                    }
                }
            
            elif action_type == 'pickup_soup_from_counter':
                
                    # Update agent's held object to soup
                held_obj = target_cell['object']
                held_obj['position'] = pos
                grid[y][x][agent_key]['held_objects'] = held_obj
                grid[y][x][agent_key]['hand_empty'] = False
                
                # Clear the counter
                target_cell['object'] = None
                target_cell['is_empty'] = True

                effects = {
                    pos: {
                        agent_key: {
                            'hand_empty': False,
                            'held_objects': held_obj
                        }
                    },
                    target_pos: {
                        'object': None,
                        'is_empty': True
                    }
                }
            
            elif action_type == 'deliver_soup':
             
                    # Clear agent's held object
                grid[y][x][agent_key]['held_objects'] = None
                grid[y][x][agent_key]['hand_empty'] = True

                effects = {
                    pos: {
                        agent_key: {
                            'hand_empty': True,
                            'held_objects': None
                        }
                    }
                }
        
    else:
        effects = {}
    
    # Record snapshot after interact
    #record_grid_snapshot(grid, timestep, snapshots)
    
    # Create detailed action log
    action_log = {
        'agent': agent_idx,
        'action': 'interact',
        'action_type': action_type,
        'from_pos': pos,
        'to_pos': pos,
        'success': can_interact,
        'preconditions': preconditions,
        'effects': effects
    }
    
    return action_log, grid

def construct_grid_from_state(state_info, terrain_grid):
    """
    Construct grid from state information.
    Args:
        state_info: Dictionary containing state information
        terrain_grid: The base terrain grid
    Returns:
        grid: Grid with agents and objects placed
    """
    # Initialize grid with terrain
    height = len(terrain_grid)
    width = len(terrain_grid[0])
    grid = []
    
    for y in range(height):
        row = []
        for x in range(width):
            cell = {
                'terrain': terrain_grid[y][x],
                'Agent_0': {
                    'is_present': False,
                    'hand_empty': True,
                    'held_objects': None,
                    'orientation': (0, -1)
                },
                'Agent_1': {
                    'is_present': False,
                    'hand_empty': True,
                    'held_objects': None,
                    'orientation': (0, -1)
                },
                'object': None,
                'is_empty': True
            }
            row.append(cell)
        grid.append(row)
    
    # Place agents
    for player_idx, player in enumerate(state_info['players']):
        x, y = player['position']
        grid[y][x][f'Agent_{player_idx}']['is_present'] = True
        grid[y][x][f'Agent_{player_idx}']['orientation'] = player['orientation']
        grid[y][x]['is_empty'] = False
        
        # Handle held objects with complete state
        if player['held_object'] is not None:
            held_obj = {
                'name': player['held_object']['name'],
                'position': (x, y),
                'state': player['held_object']['state']
            }
            grid[y][x][f'Agent_{player_idx}']['held_objects'] = held_obj
            grid[y][x][f'Agent_{player_idx}']['hand_empty'] = False
    
    # Place objects
    for obj in state_info['objects']:
        x, y = obj['position']
        grid[y][x]['object'] = obj
        grid[y][x]['is_empty'] = False
    
    return grid

def process_actions(step, grid, next_state=None):
    """
    Process actions and update grid accordingly.
    Args:
        step: Dictionary containing state info, agent idx, actions, and events
        grid: Current grid state
    Returns:
        grid: Updated grid after actions
        action_logs: List of action logs containing preconditions and effects
    """
    state_info = step['State_info']
    action_logs = []
    players = []
    
    for player_idx, player in enumerate(state_info['players']):
        info = {}
        info["position"] = player['position']
        info["orientation"] = player['orientation']
        players.append(info)
    
    #print(f"Orientation and position at timestep: {state_info['timestep']} - {players}")


    if next_state is not None:
        next_state_info = next_state['State_info']
    else:
        next_state_info = None
    
    if step['Player_actions'] is not None:
        
        # First collect all planned moves
        planned_moves = {}
        for player_idx, action in enumerate(step['Player_actions']):
            current_pos = state_info['players'][player_idx]['position']
            if action != 'interact':
                #print(f"action: {action}")
                dx, dy = action
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                planned_moves[player_idx] = (current_pos, next_pos)
            else:
                planned_moves[player_idx] = (current_pos, current_pos)

        # Check for conflicts
        next_positions = [move[1] for move in planned_moves.values()]
        has_conflict = len(next_positions) != len(set(next_positions))

        # Process actions, skipping conflicting moves
        for player_idx, action in enumerate(step['Player_actions']):
            current_pos = state_info['players'][player_idx]['position']

            if action == 'interact':
                # Handle interact action using new function
                if next_state_info is not None:
                    action_log, grid = update_grid_for_interact_and_record(
                        agent_idx=player_idx,
                        pos=current_pos,
                        timestep=state_info['timestep'],
                        grid=grid,
                        next_state=next_state_info  # Pass next state info if available
                    )
                    action_logs.append(action_log)
            else:
                # Handle movement action - action is just (dx, dy)
                # if state_info['timestep'] in [85,86,87,88,89,90,91,92,93,94]:
                #     print(f"Agent {player_idx} is moving from {current_pos} to {action}")
                dx, dy = action
                next_pos = (current_pos[0] + dx, current_pos[1] + dy)
                
                # Only execute move if there's no conflict
                if not has_conflict or next_pos not in [move[1] for idx, move in planned_moves.items() if idx != player_idx]:
                    
                    # print(f"Time step: {state_info['timestep']}")
                    # print(f"Agent {player_idx} is moving from {current_pos} to {next_pos}")
                    # Update grid and get preconditions/effects
                    action_log, grid = update_grid_for_move_and_record(
                        agent_idx=player_idx,
                        other_agent_next_pos = planned_moves[1] if player_idx == 0 else planned_moves[0],
                        from_pos=current_pos,
                        to_pos=next_pos,
                        timestep=state_info['timestep'],
                        grid=grid,
                        action=action
                    )
                    action_logs.append(action_log)
                else:
                    # Even if there's a conflict, update orientation to face the attempted movement direction
                        # Update orientation in current position
                        grid[current_pos[1]][current_pos[0]][f'Agent_{player_idx}']['orientation'] = action
                       

    # print("Grid at timestep: ", state_info['timestep'])
    # print_occupancy(grid)
    # print("--------------------------------")
    return grid, action_logs

def parse_trajectory_step(step, terrain_grid, next_state=None, grid=None):
    """
    Parse a single trajectory step into a grid snapshot.
    Args:
        step: Dictionary containing state info, agent idx, actions, and events
        terrain_grid: The base terrain grid
    Returns:
        grid: Grid snapshot for this step
        action_logs: List of action logs containing preconditions and effects
    """
    # First construct the grid from state
    if grid is None:
        grid = construct_grid_from_state(step['State_info'], terrain_grid)
    
    # Then process actions and update grid
    grid, action_logs = process_actions(step, grid, next_state)
    
    return grid, action_logs

def parse_trajectory(trajectory, terrain_grid):
    """
    Parse a complete trajectory and take grid snapshots at each step.
    Args:
        trajectory: List of state dictionaries
        terrain_grid: The base terrain grid
    Returns:
        snapshots: Dictionary of grid snapshots keyed by timestep
        action_logs: Dictionary of action logs keyed by timestep
    """
    snapshots = []
    action_logs = []

    if(trajectory[len(trajectory)-1]['State_info'] == []):
        #print("TRUE")
        #print(len(trajectory))
        trajectory = trajectory[:-1]
        # print(len(trajectory))
        # print("Modified last state")
        # print(trajectory[len(trajectory)-1])

    for i in range(len(trajectory)):                
        step = trajectory[i]
        next_state = trajectory[i+1] if i+1 < len(trajectory) else None
        timestep = step['State_info']['timestep']
        if i == 0:
            grid = None
        else:               # Update cooking state for all pots in the grid
            for row in grid:
                for cell in row:
                    if cell['terrain'] == 'pot' and cell['object'] is not None:
                        soup = cell['object']
                        if soup['name'] == 'soup' and soup['state'][1] == 3:  # If pot has 3 onions
                            cooking_time = soup['state'][2]
                            if cooking_time < 20:  # Only cook if not fully cooked
                                # Create new state tuple with updated cooking time
                                new_state = (soup['state'][0], soup['state'][1], cooking_time + 1)
                                soup['state'] = new_state

        grid, step_action_logs = parse_trajectory_step(step, terrain_grid, next_state, grid)
        #if grid is not None:
            #print(f"timestep: {timestep}")
            #if timestep < 10:
                # print(f"timestep: {timestep}")
                # print_occupancy(grid)

        for action_log in step_action_logs:
            if action_log['action'] == 'interact':
                if action_log['action_type'] == "deliver_soup":
                    # Check if the soup is delivered successfully
                    if action_log['success']:
                        print(f"Agent {action_log['agent']} successfully delivered soup at timestep {timestep}")
                    else:
                        print(f"Agent {action_log['agent']} failed to deliver soup at timestep {timestep}")
               
       
        #print(f"timestep: {timestep}")
        #print_occupancy(grid)
        #print("\n\n\n\n")
        
        # Create a deep copy of the grid before appending
        grid_copy = []
        for row in grid:
            row_copy = []
            for cell in row:
                cell_copy = {
                    'terrain': cell['terrain'],
                    'Agent_0': {
                        'is_present': cell['Agent_0']['is_present'],
                        'hand_empty': cell['Agent_0']['hand_empty'],
                        'orientation': cell['Agent_0']['orientation'],
                        'held_objects': copy.deepcopy(cell['Agent_0']['held_objects']) if cell['Agent_0']['held_objects'] is not None else None
                    },
                    'Agent_1': {
                        'is_present': cell['Agent_1']['is_present'],
                        'hand_empty': cell['Agent_1']['hand_empty'],
                        'orientation': cell['Agent_1']['orientation'],
                        'held_objects': copy.deepcopy(cell['Agent_1']['held_objects']) if cell['Agent_1']['held_objects'] is not None else None
                    },
                    'object': None,
                    'is_empty': cell['is_empty']
                }
                if cell['object'] is not None:
                    if cell['object']['name'] == 'soup':
                        # Deep copy soup state and its components
                        soup_state = cell['object']['state']
                        onion_array = copy.deepcopy(soup_state[0])
                        cell_copy['object'] = {
                            'name': 'soup',
                            'position': cell['object']['position'],
                            'state': (onion_array, soup_state[1], soup_state[2]),
                            'id': cell['object']['id']
                        }
                    else:
                        cell_copy['object'] = copy.deepcopy(cell['object'])
                row_copy.append(cell_copy)
            grid_copy.append(row_copy)
        snapshots.append(grid_copy)
        action_logs.append(step_action_logs)
    
    return snapshots, action_logs

def parse_pickle_file(pickle_file):
    """
    Parse a pickle trajectory file and store each step in an array.
    Args:
        pickle_file: Path to the pickle file
    Returns:
        trajectory: Array of trajectory steps, each containing state info, agent idx, actions, and events
    """
    import pickle
    
    with open(pickle_file, 'rb') as f:

        data = pickle.load(f)
    
    trajectory = []
    # Each step consists of 3 elements: state info, actions, events
    for n in range(0, len(data), 3):
        step = {
            'State_info': data[n] if n < len(data) else None,
            'Player_actions': data[n+1] if n+1 < len(data) else None,
            'Events': data[n+2] if n+2 < len(data) else None
        }
        trajectory.append(step)
    
    return trajectory


import pickle

def main():
    parser = argparse.ArgumentParser(description='Process Overcooked trajectory data')
    parser.add_argument('--layout_name', type=str, default='counter_circuit',
                      help='Name of the layout to use (default: cramped_room)')
    # parser.add_argument('--input_trajectory', type=str, default='human_trail.pkl',
    #                   help='Input trajectory file path (default: human_trail.pkl)')
    # parser.add_argument('--output_file', type=str, default='output_data.pkl',
    #                   help='Output file path (default: output_data.pkl)')
    parser.add_argument('--input_directory', type=str, default='/home/local/ASUAD/ubiswas2/Desktop/neurips25/ZSC-Eval/zsceval/scripts/overcooked/eval/user_study/fcp/test/',
                      help='Input trajectory directory')
    parser.add_argument('--output_directory', type=str, default='/home/local/ASURITE/vpalod/zsc-eval/ZSC-Eval/zsceval/scripts/overcooked/eval/logs',
                      help='Output directory (default: output_data.pkl)')
    
    args = parser.parse_args()
    
    layout_name = args.layout_name
    if layout_name == "counter_circuit":
        layout_list = [ 
            ['X','X','X','P','P','X','X','X'],
            ['X',None,None,2,None,None,None,'X'],
            ['D',None,'X','X','X','X',None,'S'],
            ['X',None,None,1,None,None,None,'X'],
            ['X','X','X','O','O','X','X','X']
        ]
    elif layout_name =="cramped_room":
        layout_list = [
            ['X', 'X', 'P', 'X', 'X'],
            ['O', None, None, 2, 'O'],
            ['X', 1, None, None, 'X'],
            ['X', 'D', 'X', 'S', 'X']
        ]
    elif layout_name == "forced_coordination":
        layout_list = [
            ['X', 'X', 'X', 'P', 'X'],
            ['O', None, 'X', 1, 'P'],
            ['O', 2, 'X', None, 'X'],
            ['D', None, 'X', None, 'X'],
            ['X', 'X', 'X', 'S', 'X']
        ]

    terrain_grid = build_terrain_grid(layout_list)
    agent_positions = get_initial_agent_positions(layout_list)
    grid = initialize_grid(terrain_grid, agent_positions)
    
    # file_path = "/home/local/ASURITE/vpalod/zsc-eval/ZSC-Eval/zsceval/scripts/overcooked/eval/traj_39_1.pkl"
    directory = Path(args.input_directory)
    print(os.walk(directory))
    for root, _, files in os.walk(directory):
        print("here")
        print(files)
        for file in files:
            print(file)
            if file.endswith('.pkl'):
                file_path = os.path.join(root, file)
                #Parse the pickle trajectory
                trajectory = parse_pickle_file(file_path)
                # print(trajectory[0])
                #trajectory = trajectory[:172]
                
                # Parse trajectory and get snapshots and action logs
                snapshots, action_logs = parse_trajectory(trajectory, terrain_grid)
                
                output_data = {
                    "snapshots": snapshots,
                    "action_logs": action_logs
                }
                input_file_name = os.path.basename(file_path)
                output_directory = os.path.join(args.output_directory, input_file_name)
                
                print(file_path)
                with open(output_directory, "wb") as pickle_file:
                    pickle.dump(output_data, pickle_file)
                
                # print(f"Snapshots and action logs have been saved to {args.output_file}")

                #Print grid and action logs for each timestep
                for timestep in range(len(snapshots)):
                    print(f"\nTimestep {timestep}:")
                    print("\nGrid State:")
                    print_occupancy(snapshots[timestep])
                    
                    if timestep < len(action_logs) and action_logs[timestep]:
                        print("\nAction Logs:")
                        for log in action_logs[timestep]:
                            try:
                                if log['action'] == 'interact':
                                    print(f"Agent {log['agent']} attempted to interact at position {log['from_pos']}")
                                else:
                                    print(f"Agent {log['agent']} attempted to move {log['action']} from {log['from_pos']} to {log['to_pos']}")
                                print("Preconditions:", log['preconditions'])
                                print("Effects:", log['effects'])
                                print(f"Move successful: {log['move_successful']}")
                            except:
                                print(log)
                    else:
                        print("\nNo actions in this timestep")
                            
                print("this is the file path", file_path, "end")


if __name__ == "__main__":
    
    main()
