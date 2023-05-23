from search.run_one_replicate_ablation import run_one_replication
import argparse
import os
import utils.config_utils as config_utils
from itertools import product

### TODO Change path adapted for ablation studies
def create_path(save_path, label, problem_name, ablation_name, problem_kwargs, bo_kwargs, tr_kwargs):
    ### Desiderata:
    # - First name of kernel
    # - Second name of the task
    # - Third name of the graph
    # - Fourth different parameters of kernel, task, graph
    # => Results ordered more conveniently
    if problem_name == "diffusion":
        if problem_kwargs["random_graph_type"] == "ba":
            s = "_".join([problem_kwargs["random_graph_type"], f'm-{problem_kwargs["m"]}', f'beta-{problem_kwargs["beta"]}', f'gamma-{problem_kwargs["gamma"]}',
                f'n-{problem_kwargs["n"]}', f'epsilon-{problem_kwargs["epsilon"]}', f'fracinfect-{problem_kwargs["fraction_infected"]}', f'iter-{problem_kwargs["iteration_diffusion"]}'
                ])
        elif problem_kwargs["random_graph_type"] == "ws":
            s = "_".join([problem_kwargs["random_graph_type"], f'k-{problem_kwargs["k"]}', f'p-{problem_kwargs["p"]}', f'beta-{problem_kwargs["beta"]}', f'gamma-{problem_kwargs["gamma"]}',
                f'n-{problem_kwargs["n"]}', f'epsilon-{problem_kwargs["epsilon"]}', f'fracinfect-{problem_kwargs["fraction_infected"]}', f'iter-{problem_kwargs["iteration_diffusion"]}'
                ])
    elif problem_name == "diffusion_real":
        s = "_".join([f'beta-{problem_kwargs["beta"]}', f'gamma-{problem_kwargs["gamma"]}', f'fracinfect-{problem_kwargs["fraction_infected"]}'])
    elif problem_name == "centrality":
        if problem_kwargs["random_graph_type"] == "ba":
            s = "_".join([problem_kwargs["random_graph_type"], problem_kwargs["feature_name"], f'm-{problem_kwargs["m"]}', f'ninit-{problem_kwargs["n_init"]}', f'n-{problem_kwargs["n"]}'])
        if problem_kwargs["random_graph_type"] == "ws":
            s = "_".join([problem_kwargs["random_graph_type"], problem_kwargs["feature_name"], f'k-{problem_kwargs["k"]}', f'p-{problem_kwargs["p"]}', f'ninit-{problem_kwargs["n_init"]}', f'n-{problem_kwargs["n"]}'])
        if problem_kwargs["random_graph_type"] == "real_enron":
            s = "_".join([problem_kwargs["random_graph_type"], problem_kwargs["feature_name"], f'n_init-{problem_kwargs["n_init"]}'])
        if problem_kwargs["random_graph_type"] == "real_twitch":
            s = "_".join([problem_kwargs["random_graph_type"], problem_kwargs["feature_name"], f'n_init-{problem_kwargs["n_init"]}'])
        if problem_kwargs["random_graph_type"] == "real_fb":
            s = "_".join([problem_kwargs["random_graph_type"], problem_kwargs["feature_name"], f'n_init-{problem_kwargs["n_init"]}'])
    elif problem_name == "test_function":
        s = "_".join([problem_kwargs["test_function"], f'noise-{problem_kwargs["noise"]}', f'n-{problem_kwargs["n"]}'])
    elif problem_name == "team_opt":
        s = "_".join([problem_kwargs["random_graph_type"], f'n_skills-{problem_kwargs["n_skills"]}', f'alpha-{problem_kwargs["alpha"]}', f'n_individuals-{problem_kwargs["n_individuals"]}', f'threshold-{problem_kwargs["threshold"]}'])
    
    ##str_bo = [key + "-" + f"{value}" for key, value in bo_kwargs.items() if key not in ["tr_settings", ablation_name]]
    ##str_tr = [key + "-" + f"{value}" for key, value in tr_kwargs.items() if key != ablation_name]
    str_bo = [key + "-" + f"{value}" for key, value in bo_kwargs.items() if key not in ["tr_settings"]]
    str_tr = [key + "-" + f"{value}" for key, value in tr_kwargs.items()]
    s = "_".join([label, problem_name, s] + str_bo + str_tr)
    save_path = os.path.join(save_dir, s)
    return save_path


#d_label = {"ei_ego_network_1":"bo", "ei_ego_network_2":"bo_2", "random":"random", "local_search":"local_search", "dfs":"dfs", "bfs":"bfs"}

## TODO Manage option for gpu
if __name__ == "__main__":
    OVERWRITE = True
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='centrality')
    args = parser.parse_args()

    # load parameters
    config = config_utils.setup(f'config/{args.config}.yaml')
    
    ### Place where design save_path from config parameters
    save_dir = config["save_dir"]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ### Get different  configs
    labels = config["label"]                  # defines the method
    problem_name=config["problem_name"]
    ablation_name=config["problem_ablation"]
    problem_kwargs = config["problem_settings"]
    bo_kwargs = config["bo_settings"]
    tr_kwargs = bo_kwargs["tr_settings"]
    n_exp = getattr(config, "n_exp", 10)
    plot_result = getattr(config, "plot_result", True)
    animate = getattr(config, "animate", False)
    all_data_over_labels = {l: [] for l in labels}
    seed=0
    
    ## Get list of parameters for the ablation study
    if ablation_name in ["max_radius", "context_graph_nnode_init"]:
        list_ablation_parameter = bo_kwargs[ablation_name]
    elif ablation_name in ["n_nodes_min", "trust_region_multiplier", "succ_tol", "fail_tol"]:
        list_ablation_parameter = tr_kwargs[ablation_name]

    ### Make lists for multiple experiments ### Exchange label and ablation parameter here
    list_keys, list_values = [], []
    for key, value in problem_kwargs.items():
        if type(value) == list:
            list_keys.append(tuple(["pb", key]))
            list_values.append(value)
    for key, value in bo_kwargs.items():
        #if key != ablation_name:
        if type(value) == list:
            list_keys.append(tuple(["bo", key]))
            list_values.append(value)
    for key, value in tr_kwargs.items():
        #if key != ablation_name:
        if type(value) == list:
            list_keys.append(tuple(["tr", key]))
            list_values.append(value)
    if len(list_values) > 0:
        for label in labels:
            for t in product(*list_values):
                for i, el in enumerate(t):
                    type_param, key = list_keys[i]
                    if type_param == "pb":
                        problem_kwargs[key] = el
                    elif type_param == "bo":
                        bo_kwargs[key] = el
                    elif type_param == "tr":
                        tr_kwargs[key] = el

                bo_kwargs["tr_settings"] = tr_kwargs
                save_path = create_path(save_dir, label, problem_name, ablation_name, problem_kwargs, bo_kwargs, tr_kwargs)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                    print("Processing", save_path, "...")
                else:
                    "If folder already exists then perform optimization depending on OVERWRITE"
                    if OVERWRITE == False:
                        print(save_path + "found without overwriting, next config...")
                        continue
                bo_kwargs["tr_settings"] = tr_kwargs
                # for param in list_ablation_parameter:
                # if ablation_name in ["max_radius", "context_graph_nnode_init"]:
                #     bo_kwargs[ablation_name] = param
                # elif ablation_name in ["n_nodes_min", "trust_region_multiplier", "succ_tol", "fail_tol"]:
                #     tr_kwargs[ablation_name] = param
                #save_dir = os.path.join(save_path, ablation_name + f"-{param}")
                #if not os.path.exists(save_dir):
                #    os.makedirs(save_dir)
                print("State of the dict")
                print(bo_kwargs)
                for i in range(n_exp):
                    try:
                        run_one_replication(
                                label=label,
                                seed=seed + i,
                                problem_name=problem_name,
                                save_path=save_path,
                                batch_size=getattr(bo_kwargs, "batch_size", 1),
                                n_initial_points=getattr(bo_kwargs, "n_init", 10),
                                iterations=getattr(bo_kwargs, "max_iters", 50),
                                max_radius=getattr(bo_kwargs, "max_radius", 10),
                                context_graph_nnode_init=getattr(
                                    bo_kwargs, "context_graph_nnode_init", 100),
                                animation=animate,
                                trust_region_kwargs=getattr(
                                    bo_kwargs, "tr_settings", None),
                                problem_kwargs=problem_kwargs,
                            )
                    except Exception as e:
                        print("Configuration with label " + label + "failed, with error " + str(e) + "continue...")
                        continue

    else:
        raise NotImplementedError
        save_path = create_path(save_dir, problem_name, problem_kwargs, bo_kwargs)
        for label_idx, label in enumerate(labels):
            all_data = all_data_over_labels[label]
            for i in range(n_exp):
                try:
                    run_one_replication(
                        label=label,
                        seed=seed + i,
                        problem_name=problem_name,
                        save_path=save_path,
                        batch_size=getattr(bo_kwargs, "batch_size", 1),
                        n_initial_points=getattr(bo_kwargs, "n_init", 10),
                        iterations=getattr(bo_kwargs, "max_iters", 50),
                        max_radius=getattr(bo_kwargs, "max_radius", 10),
                        context_graph_nnode_init=getattr(
                            bo_kwargs, "context_graph_nnode_init", 100),
                        animation=animate,
                        trust_region_kwargs=getattr(
                            bo_kwargs, "tr_settings", None),
                        problem_kwargs=problem_kwargs,
                    )
                except Exception as e:
                    print("Configuration with label " + label + "failed, with error " + str(e) + "continue...")
                    continue