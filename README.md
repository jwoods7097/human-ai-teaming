# Interdependencies in Human-agent teams

This repository is the official implementation of Who is Helping Whom? Analyzing Inter-dependencies to Evaluate Cooperation in Human-AI Teaming. This is forked from https://github.com/liyang619/COLE-Platform.git.

## Requirements

Install [PantheonRL](https://github.com/Stanford-ILIAD/PantheonRL) in this repo

```shell
    conda create -n overcooked-vis python=3.7
    conda activate overcooked-vis
    python -m pip install "setuptools==65.5.0" "wheel==0.38.4"
    pip install gym==0.21.0
    pip install -r requirements.txt
    pip install -e .
```

Install mpi4py

```shell
conda install mpi4py
```

Install PyTorch (based on your CUDA version): https://pytorch.org/
(You don't actually need the GPU version to run the game)

Install human_aware_rl and its dependencies: overcooked_ai, baselines & stable_baselines

```shell
    cd overcookedgym/human_aware_rl
    pip install -e .
    cd overcooked_ai
    pip install -e .
    cd ..
    cd stable-baselines
    pip install -e .
    cd ..
    cd baselines
    pip install -e .
```

## How to load models for user study

You can get pretrained trained models [here](). For conducting the user study, folder named models has to be put at /overcookedgym/overcooked-flask/ and data has to be put at /overcookedgym/overcooked-flask/.

**Note:** The layout names in code and google drive are not aligned with the layout names in the paper. Here is the mapping:

```code
PYTHON_LAYOUT_NAME_TO_ENV_NAME = {
    "unident_s": "Asymmetric Advantages",
    "simple": "Cramped Room",
    "random1": "Coordination Ring",
    "random0": "Forced Coordination",
    "random3": "Counter Circuit"
}
```

In addition, you can load your own models if they are trained using the [Human-Aware-RL](https://github.com/HumanCompatibleAI/human_aware_rl/tree/neurips2019) framework.
Agents are loaded using the `get_agent_from_saved_model()` method, which loads tensorflow predictor models (`.pb` files), so you should save your agents in this style if you wish to load them into our framework. You can reference to the `save` method in `human_aware_rl/pbt/pbt.py` for saving agents that can be loaded.

To load your own models, you need to put them in the `./models` folder in a named folder (the folder names need to be the same for all layouts), and the models would be loaded upon starting the server. For example. If your algo is named `ABC`, then the folder structure should look like this:

```
-- models
  | --  simple
       | -- SP        <---- Baseline 1 
       | -- PBT       <---- Baseline 2
          ...
       | -- ABC       <---- Your Algorithm
  | --  unident_s
       | -- SP        <---- Baseline 1 
       | -- PBT       <---- Baseline 2
          ...
       | -- ABC       <---- Your Algorithm
  | --  random1
       | -- SP        <---- Baseline 1 
       | -- PBT       <---- Baseline 2
          ...
       | -- ABC       <---- Your Algorithm
  ...
```

## How to run user study

```shell
python overcookedgym/overcooked-flask/app.py --trajs_savepath ./trajs --ckpts ./models
```

- `--ckpts`: Folder containing all the AI models to be loaded. Default is `./models`.
- `--port`: The port where you run the server process.
- `--trajs_savepath`: Optional trajectory save path, default is `./trajs`.
- `--questionnaire_savepath`: Optional questionnaire save path, default is `./questionnaire`.
- `--ip`: Default is LOCALHOST, we **recommend you replace it with your public network IP**, because of a known bug of Flask that may cause extreme lag when playing the game. The same applies when debugging, you should visit your machine's IP in your browser instead of LOCALHOST.

---

## How to analyse trajectories

The trajectories from the user study are collected in `/overcookedgym/overcooked-flask/models` in JSON format.  To prepare these for analysis using the proposed metrics, we have to convert them to pickle (`.pkl`) format using the script:
`overcookedgym/overcooked-flask/analysis/trail_to_pkl.py`.

### **Purpose**

- **Input:** JSON trajectory files generated from human gameplay (found in `/overcookedgym/overcooked-flask/models`).
- **Output:** Corresponding pickle (`.pkl`) files, stored in the same location as their original JSON files.
- **Use case:** The `.pkl` format is required for subsequent analysis scripts that evaluate performance using custom metrics.

### **Command-Line Arguments**

- `directory` (required):
  **Type:** `str`
  The path to the directory containing the JSON trajectory files (all subdirectories are processed). (e.g, '/user_study/' )

#### Example usage:

```bash
python overcookedgym/overcooked-flask/analysis/trail_to_pkl.py overcookedgym/overcooked-flask/trajs
```

---

After converting trajectories to `.pkl` format, the next step is to use the mapping code to obtain a grounded, symbolic (PDDL-style) trajectory. This is done using:

`overcookedgym/overcooked-flask/analysis/rl_to_pddl.py`

### **Purpose:**

- Converts low-level trajectory pickle files into a sequence of symbolic grid states and action logs.
- Each action log contains symbolic actions, preconditions, and effects suitable for PDDL-based planning and analysis.
- Outputs a new `.pkl` file containing snapshots and logs for further symbolic or logic-based analysis.

### **Required arguments:**

- `--layout_name`: Name of the Overcooked layout (`counter_circuit`, `cramped_room`, or `forced_coordination`).
- `--input_directory`: Directory containing `.pkl` trajectory files (from previous conversion).
- `--output_directory`: Directory to save symbolic `.pkl` files.
- `--single`: (optional) If set, processes only a single hardcoded file for debugging.

#### **Example usage:**

```bash
python overcookedgym/overcooked-flask/analysis/rl_to_pddl.py --input_directory overcookedgym/overcooked-flask/trajs/forced_coordination/ --output_directory overcookedgym/overcooked-flask/trajs/forced_coordination/ --layout_name forced_coordination
```

This command processes all `.pkl` files in the input directory and saves the symbolic results in the output directory. Each output file contains:

- `"snapshots"`: symbolic grid states per timestep
- `"action_logs"`: symbolic actions with preconditions and effects per timestep

---

A domain-agnostic analysis module for detecting and categorizing interdependent interactions between agents in multi-agent environments.
Given a sequence of environment states ("snapshots") and corresponding action logs (as PDDL traces, generated by the mapping module), `overcookedgym/overcooked-flask/analysis/detect_int_user.py`) provides a framework for identifying and classifying interdependencies.

### **Purpose**

- **Detection:** Dynamically tracks the effects of each agent's actions across timesteps, maintaining effect lists for each agent.
- **Interdependence Identification:** At each timestep, the algorithm checks whether the preconditions for one agent’s action are satisfied by the effects of the other agent’s prior actions—thereby detecting interdependent events.
- **Categorization:** Each detected interdependence is classified as one of:
  - **Constructive:** The object or state transferred leads to progress toward the task goal.
  - **Looping:** The same object is repeatedly exchanged between agents without any change in state.
  - **Irrelevant:** The object or interaction does not contribute to the goal.
  - **Non-constructive:** The interdependence does not aid in achieving the goal or is redundant.
- **Domain-Agnostic:** By relying on a consistent PDDL-style symbolic schema, the module can be readily applied to different domains (as long as the environment logs are mapped accordingly).

### **Required arguments**

- `--directory`: Path to the directory containing processed symbolic `.pkl` trajectory files (output of the mapping module).
- `--is_forced` (optional, bool): If set, enables special handling for forced coordination layouts (affects output CSV structure).

#### **Example usage**

```bash
python overcookedgym/overcooked-flask/analysis/detect_int_user.py --directory overcookedgym/overcooked-flask/trajs/forced_coordination/ --is_forced true
```

#### **Output**

- For each trajectory file, outputs CSV files (`obj_results.csv`, `traj_obj_results.csv`, etc.) with counts and lists of each type of interdependence detected and the action distribution for each agent, enabling quantitative and qualitative analysis.

---

## Results

### ZSC Agent paired with Cooperative Partner

| Agent | Task Reward | Int `<sub>`cons `</sub>` | Int `<sub>`non-cons `</sub>` | %P `<sub>`tot-sub `</sub><sup>`trig `</sup>` | %P `<sub>`trig `</sub><sup>`not-trig-acc `</sup>` |
| ----- | ----------- | ---------------------------- | -------------------------------- | -------------------------------------------------- | ------------------------------------------------------- |
| COLE  | 36          | 0.6                          | 1.2                              | 38.5                                               | 88.88                                                   |
| MEP   | 43.33       | 1.834                        | 1.667                            | 41.28                                              | 75.55                                                   |
| HSP   | 0           | 0                            | 0.5                              | 28.57                                              | 100.0                                                   |
| FCP   | 0           | 0                            | 0.167                            | 21.79                                              | 100.0                                                   |

**Table:** ZSC Agents paired with a scripted coordination policy.

- %P `<sub>`tot-sub `</sub><sup>`trig `</sup>`: Interdependencies triggered by the scripted agents
- %P `<sub>`trig `</sub><sup>`not-trig-acc `</sup>`: Triggered interdependencies not accepted by the ZSC agents
- Task and teaming scores are averaged across 20 runs with the scripted agent.

---

### Task vs Teaming Score for ZSC paired with Human Participants

| Agent | Task Reward `<br>`Non-RC | Task Reward `<br>`RC | Int `<sub>`cons `</sub><br>`Non-RC | Int `<sub>`cons `</sub><br>`RC | Int `<sub>`non-cons `</sub><br>`Non-RC | Int `<sub>`non-cons `</sub><br>`RC |
| ----- | -------------------------- | ---------------------- | -------------------------------------- | ---------------------------------- | ------------------------------------------ | -------------------------------------- |
| COLE  | 76.21                      | 56.875                 | 1.89                                   | 11.375                             | 3.29                                       | 2.875                                  |
| MEP   | 50.00                      | 44.102                 | 0.928                                  | 8.692                              | 1.285                                      | 2.769                                  |
| HSP   | 41.11                      | 60.55                  | 1.388                                  | 12.055                             | 2.138                                      | 3.083                                  |
| FCP   | 22.55                      | 35.34                  | 0.97                                   | 7.06                               | 0.872                                      | 3.441                                  |

**Table:** ZSC Agents paired with human teammates. The task reward and teaming metrics are averaged across 36 runs with participants.

| Agent | %H-sub `<sub>`tot-sub `</sub><sup>`trig `</sup>` (Non-RC) | %H-sub `<sub>`tot-sub `</sub><sup>`trig `</sup>` (RC) | %H-sub `<sub>`trig `</sub><sup>`not trig-acc `</sup>` (Non-RC) | %H-sub `<sub>`trig `</sub><sup>`not trig-acc `</sup>` (RC) |
| ----- | --------------------------------------------------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------- | ---------------------------------------------------------------- |
| COLE  | 60.28                                                           | 45.28                                                       | 70.05                                                                | 38.34                                                            |
| MEP   | 66.82                                                           | 43.57                                                       | 82.39                                                                | 39.82                                                            |
| HSP   | 52.22                                                           | 42.92                                                       | 80.85                                                                | 40.58                                                            |
| FCP   | 48.30                                                           | 43.62                                                       | 98.41                                                                | 36.84                                                            |

**Table:** Analysis of triggered vs accepted interdependencies for the human player.

- %H-sub `<sub>`tot-sub `</sub><sup>`trig `</sup>`: Interdependencies triggered by the human partner
- %H-sub `<sub>`trig `</sub><sup>`not trig-acc `</sup>`: Triggered interdependencies not accepted by the ZSC agents

---

### Task vs Teaming Score for Best Performing Human-agent Team

| Agent | Task Reward RC | Task Reward Non-RC | Int `<sub>`cons `</sub>` RC | Int `<sub>`cons `</sub>` Non-RC | Int `<sub>`non-cons `</sub>` RC | Int `<sub>`non-cons `</sub>` Non-RC |
| ----- | -------------- | ------------------ | ------------------------------- | ----------------------------------- | ----------------------------------- | --------------------------------------- |
| COLE  | 120            | 100                | 8.0                             | 20.00                               | 3.0                                 | 0.33                                    |
| MEP   | 80.00          | 120                | 1.285                           | 24.0                                | 76.47                               | 0                                       |
| HSP   | 80             | 120                | 3.5                             | 25.00                               | 1.25                                | 1.667                                   |
| FCP   | 60             | 100                | 3.667                           | 20                                  | 1.334                               | 3.0                                     |

**Table:** Best performing team when ZSC Agents are paired with human teammates.

| Agent | %H-sub `<sub>`tot-sub `</sub><sup>`trig `</sup>` RC | %H-sub `<sub>`tot-sub `</sub><sup>`trig `</sup>` Non-RC | %H-sub `<sub>`trig `</sub><sup>`not trig-acc `</sup>` RC | %H-sub `<sub>`trig `</sub><sup>`not trig-acc `</sup>` Non-RC |
| ----- | --------------------------------------------------------- | ------------------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------ |
| COLE  | 40.58                                                     | 35.89                                                         | 11.76                                                          | 5.01                                                               |
| MEP   | 59.10                                                     | 46.57                                                         | 82.39                                                          | 0.20                                                               |
| HSP   | 28.67                                                     | 49.27                                                         | 33.34                                                          | 20.58                                                              |
| FCP   | 44.62                                                     | 48.68                                                         | 80.17                                                          | 18.92                                                              |

**Table:** Analysis of triggered vs accepted interdependencies for the best ZSC Agent–human team.

- %H-sub `<sub>`tot-sub `</sub><sup>`trig `</sup>`: Interdependencies triggered by the human partner
- %H-sub `<sub>`trig `</sub><sup>`not trig-acc `</sup>`: Triggered interdependencies not accepted by the ZSC agents
