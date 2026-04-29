for agent in cole
do
    for env in random0 random3
    do
        traj_dir="$HOME/ZSC/results/Overcooked/${env}/population/selfplay-${env}-${agent}-seed1/trajs/${env}"

        # convert trajectories to pickle
        python overcookedgym/overcooked-flask/analysis/trail_to_pkl.py "$traj_dir"

        if [ "$env" = "random3" ]
        then
            # convert pickle to pddl
            python overcookedgym/overcooked-flask/analysis/rl_to_pddl.py \
                --input_directory "$traj_dir" \
                --output_directory "$traj_dir" \
                --layout_name counter_circuit

            # detect interdependencies
            python overcookedgym/overcooked-flask/analysis/detect_int_proxy.py \
                --directory "$traj_dir"
        else
            # convert pickle to pddl
            python overcookedgym/overcooked-flask/analysis/rl_to_pddl.py \
                --input_directory "$traj_dir" \
                --output_directory "$traj_dir" \
                --layout_name forced_coordination

            # detect interdependencies
            python overcookedgym/overcooked-flask/analysis/detect_int_proxy.py \
                --directory "$traj_dir" \
                --is_forced true
        fi

        mv "$traj_dir/obj_results.csv" \
           "../ZSC-Eval/zsceval/scripts/overcooked/eval/selfplay_results/${env}_${agent}_seed1.csv"
    done
done