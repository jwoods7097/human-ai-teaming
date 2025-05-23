import json
import numpy as np
from scipy import stats
from pathlib import Path
from collections import defaultdict

def get_layout_from_traj(traj_path):
    """Extract layout name from trajectory file"""
    try:
        # Convert relative path to absolute path
        base_dir = Path("/home/local/ASURITE/vpalod/coop-eval-user-study/overcookedgym/overcooked-flask")
        full_traj_path = base_dir / traj_path
        
        # Remove .npy extension if present
        if full_traj_path.suffix == '.npy':
            full_traj_path = full_traj_path.with_suffix('')
            
        with open(full_traj_path, 'r') as f:
            data = json.load(f)
        return data['mdp_params'][0]['layout_name']
    except Exception as e:
        print(f"Error reading {traj_path}: {str(e)}")
        return None

def analyze_demographics(questionnaire_data_list):
    """Analyze demographic information from questionnaire data"""
    ages = []
    gender_counts = defaultdict(int)
    played_counts = defaultdict(int)
    
    # Filter participants who have is_played field
    valid_participants = [data for data in questionnaire_data_list if 'is_played' in data]
    total_participants = len(valid_participants)
    
    for data in valid_participants:
        # Age
        try:
            age = int(data['age'])
            ages.append(age)
        except (ValueError, KeyError):
            pass
            
        # Gender
        if 'gender' in data:
            gender_counts[data['gender'].lower()] += 1
            
        # Played status
        played_counts[data['is_played'].lower()] += 1
    
    print("\nDemographic Information:")
    print(f"Total number of participants with is_played data: {total_participants}")
    
    if ages:
        print(f"\nAge Statistics:")
        print(f"Average age: {np.mean(ages):.2f}")
        print(f"Median age: {np.median(ages):.2f}")
        print(f"Age range: {min(ages)}-{max(ages)}")
    
    if gender_counts:
        print(f"\nGender Distribution:")
        for gender, count in gender_counts.items():
            percentage = (count / total_participants) * 100
            print(f"{gender.capitalize()}: {count} ({percentage:.1f}%)")
    
    if played_counts:
        print(f"\nPlayed Status:")
        for status, count in played_counts.items():
            percentage = (count / total_participants) * 100
            print(f"{status.upper()}: {count} ({percentage:.1f}%)")

def analyze_responses(questionnaire_data_list):
    # Initialize dictionaries to store responses by layout and participant
    q1_responses = {'counter_circuit': [], 'forced_coordination': []}
    q4_responses = {'counter_circuit': [], 'forced_coordination': []}
    q5_responses = {'counter_circuit': [], 'forced_coordination': []}
    
    # Store responses by participant for paired analysis
    participant_responses = defaultdict(lambda: {'counter_circuit': {'q1': [], 'q4': [], 'q5': []}, 
                                               'forced_coordination': {'q1': [], 'q4': [], 'q5': []}})
    
    # Process each questionnaire data
    for questionnaire_data in questionnaire_data_list:
        if "in_game" not in questionnaire_data.keys():
            continue
            
        participant_id = f"{questionnaire_data['name']}_{questionnaire_data['phone']}"
        
        for game in questionnaire_data['in_game']:
            traj_path = game['traj_path']
            layout = get_layout_from_traj(traj_path)
            
            if layout not in ['counter_circuit', 'forced_coordination']:
                continue
                
            # Extract Q1, Q4 and Q5 responses
            q1 = game['questionnaire']['Q1. My partner and I worked together to deliver the soups.']
            q4 = game['questionnaire']['Q4. My partner responded to my attempts to work with them.']
            q5 = game['questionnaire']['Q5. My partner attempted to work with me.']
            
            q1_responses[layout].append(q1)
            q4_responses[layout].append(q4)
            q5_responses[layout].append(q5)
            
            # Store for paired analysis
            participant_responses[participant_id][layout]['q1'].append(q1)
            participant_responses[participant_id][layout]['q4'].append(q4)
            participant_responses[participant_id][layout]['q5'].append(q5)
    
    # Perform statistical analysis for Q1
    print("\nAnalysis for Q1 (My partner and I worked together to deliver the soups):")
    print("Counter Circuit responses:", q1_responses['counter_circuit'])
    print("Forced Coordination responses:", q1_responses['forced_coordination'])
    
    # One-sample t-test against neutral midpoint (3) for counter_circuit
    if len(q1_responses['counter_circuit']) > 0:
        t_stat, p_val = stats.ttest_1samp(q1_responses['counter_circuit'], 3)
        print("\nOne-sample t-test for Counter Circuit (H0: mean = 3):")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_val:.4f}")
        print(f"Mean: {np.mean(q1_responses['counter_circuit']):.2f}")
    
    # One-sample t-test against neutral midpoint (3) for forced_coordination
    if len(q1_responses['forced_coordination']) > 0:
        t_stat, p_val = stats.ttest_1samp(q1_responses['forced_coordination'], 3)
        print("\nOne-sample t-test for Forced Coordination (H0: mean = 3):")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_val:.4f}")
        print(f"Mean: {np.mean(q1_responses['forced_coordination']):.2f}")
    
    # Perform statistical analysis
    print("\nAnalysis for Q4 (My partner responded to my attempts to work with them):")
    print("Counter Circuit responses:", q4_responses['counter_circuit'])
    print("Forced Coordination responses:", q4_responses['forced_coordination'])
    
    # One-sample t-test against neutral midpoint (3) for counter_circuit
    if len(q4_responses['counter_circuit']) > 0:
        t_stat, p_val = stats.ttest_1samp(q4_responses['counter_circuit'], 3)
        print("\nOne-sample t-test for Counter Circuit (H0: mean = 3):")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_val:.4f}")
        print(f"Mean: {np.mean(q4_responses['counter_circuit']):.2f}")
    
    # Paired t-test for participants who played both layouts
    paired_q4_cc = []
    paired_q4_fc = []
    for participant, responses in participant_responses.items():
        if (responses['counter_circuit']['q4'] and responses['forced_coordination']['q4']):
            paired_q4_cc.append(np.mean(responses['counter_circuit']['q4']))
            paired_q4_fc.append(np.mean(responses['forced_coordination']['q4']))
    
    if paired_q4_cc and paired_q4_fc:
        t_stat, p_val = stats.ttest_rel(paired_q4_cc, paired_q4_fc)
        print("\nPaired t-test between layouts (H0: mean difference = 0):")
        print(f"Number of participants with both layouts: {len(paired_q4_cc)}")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_val:.4f}")
    
    print("\nAnalysis for Q5 (My partner attempted to work with me):")
    print("Counter Circuit responses:", q5_responses['counter_circuit'])
    print("Forced Coordination responses:", q5_responses['forced_coordination'])
    
    # One-sample t-test against neutral midpoint (3) for counter_circuit
    if len(q5_responses['counter_circuit']) > 0:
        t_stat, p_val = stats.ttest_1samp(q5_responses['counter_circuit'], 3)
        print("\nOne-sample t-test for Counter Circuit (H0: mean = 3):")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_val:.4f}")
        print(f"Mean: {np.mean(q5_responses['counter_circuit']):.2f}")
    
    # Paired t-test for participants who played both layouts
    paired_q5_cc = []
    paired_q5_fc = []
    for participant, responses in participant_responses.items():
        if (responses['counter_circuit']['q5'] and responses['forced_coordination']['q5']):
            paired_q5_cc.append(np.mean(responses['counter_circuit']['q5']))
            paired_q5_fc.append(np.mean(responses['forced_coordination']['q5']))
    
    if paired_q5_cc and paired_q5_fc:
        t_stat, p_val = stats.ttest_rel(paired_q5_cc, paired_q5_fc)
        print("\nPaired t-test between layouts (H0: mean difference = 0):")
        print(f"Number of participants with both layouts: {len(paired_q5_cc)}")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_val:.4f}")

if __name__ == "__main__":
    # Load all questionnaire data from the questionnaire folder
    questionnaire_folder = Path("/home/local/ASURITE/vpalod/coop-eval-user-study/overcookedgym/overcooked-flask/questionnaire")
    questionnaire_data_list = []
    
    # Find all JSON files in the questionnaire folder
    json_files = list(questionnaire_folder.glob("*.json"))
    
    if not json_files:
        print("No JSON files found in the questionnaire folder.")
        exit(1)
    
    print(f"Found {len(json_files)} questionnaire files.")
    
    # Load each JSON file
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            questionnaire_data_list.append(data)
            print(f"Loaded {json_file.name}")
        except Exception as e:
            print(f"Error loading {json_file.name}: {str(e)}")
    
    if questionnaire_data_list:
        #analyze_demographics(questionnaire_data_list)
        analyze_responses(questionnaire_data_list)
    else:
        print("No valid questionnaire data found.") 