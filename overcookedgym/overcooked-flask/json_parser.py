import os
import json
import shutil
from pathlib import Path
import numpy as np

def get_reward_category(reward):
    """Convert reward to the nearest category from the predefined values"""
    reward_values = [20, 40, 60, 80, 100, 120]
    # Find the closest reward value
    closest_reward = min(reward_values, key=lambda x: abs(x - reward))
    return f"reward_{closest_reward}"

def process_trajs():
    # Create trajs directory if it doesn't exist
    trajs_dir = Path("trajs")
    trajs_dir.mkdir(exist_ok=True)
    
    # Get all JSON files in the trajs directory
    json_files = list(trajs_dir.glob("*.json"))
    
    if not json_files:
        print("No JSON files found in the trajs directory.")
        return
    
    # Process each JSON file
    for json_file in json_files:
        try:
            print(f"\nProcessing {json_file.name}...")
            
            # Read and parse the JSON file
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                print(f"Skipping {json_file.name}: Not a valid trajectory file")
                continue
                
            # Get layout name and reward
            layout_name = data['mdp_params'][0]['layout_name']
            total_reward = sum(data['ep_rewards'][0])
            reward_category = get_reward_category(total_reward)
            
            # Create directory structure: trajs/layout_name/reward_category/
            target_dir = trajs_dir / layout_name / reward_category
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy the trajectory file to the appropriate directory
            target_file = target_dir / json_file.name
            shutil.copy2(json_file, target_file)
            print(f"Moved {json_file.name} to {target_dir}")
            
        except json.JSONDecodeError as e:
            print(f"Error reading {json_file.name}: Invalid JSON format - {str(e)}")
        except Exception as e:
            print(f"Error processing {json_file.name}: {str(e)}")

if __name__ == "__main__":
    process_trajs() 