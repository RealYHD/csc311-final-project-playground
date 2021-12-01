import numpy as np

from neural_network import AutoEncoder, load_data, train, evaluate

def min_delta_tuning(init_hyperparameters, min_iteration_delta = 0.01, min_consistency = 3, init_dir = 1, max_iterations = 100, verbosity = 1):
    zero_train_matrix, train_matrix, valid_data, test_data = load_data("./data")

    curr_hyperparameters = {
        "k": init_hyperparameters["k"],
        "lr": init_hyperparameters["lr"],
        "lambda": init_hyperparameters["lambda"],
        "epochs": init_hyperparameters["epochs"]
    }

    results = {
        "hyperparameters": [],
        "training_log": [],
        "accuracy": []
    }

    curr_hyperparameter_vec = np.array(list(curr_hyperparameters.values()))
    default_growth_vec = np.full(4, 2)
    growth_vec = np.copy(default_growth_vec)
    accel_vec = np.full(4, 2)
    dir_vec = np.full(4, init_dir)

    consecutive_min_hit = 0
    last_iteration_acc = 0
    iteration_acc = 0
    last_acc = 0
    acc = 0
    iterations = 0
    while max_iterations == 0 or iterations < max_iterations:
        curr_hyperparameters = {
            "k": curr_hyperparameter_vec[0],
            "lr": curr_hyperparameter_vec[1],
            "lambda": curr_hyperparameter_vec[2],
            "epochs": curr_hyperparameter_vec[3]
        }
        results["hyperparameters"].append(curr_hyperparameters)
        
        # TODO: Deal with stdout
        u, q = train_matrix.shape

        for vec_i in range(len(curr_hyperparameter_vec)):
            param_mask = np.zeros(4)
            param_mask[vec_i] = 1

            acc_delta = acc - last_acc
            if iterations > 0:
                curr_hyperparameter_vec = curr_hyperparameter_vec * (growth_vec ** (dir_vec * param_mask))
                if vec_i == 0:
                    curr_hyperparameter_vec[vec_i] = max(curr_hyperparameter_vec[vec_i], 1)
            if verbosity > 0:
                print("#\tp\tk\tlr\tlambda\tepochs")
                print("{}\t{}\t{}\t{}\t{}\t{}".format(
                    iterations,
                    vec_i, 
                    curr_hyperparameter_vec[0],
                    curr_hyperparameter_vec[1],
                    curr_hyperparameter_vec[2],
                    curr_hyperparameter_vec[3]
                ))
            model = AutoEncoder(q, int(curr_hyperparameter_vec[0]))
            last_acc = acc
            train(model, curr_hyperparameter_vec[1], curr_hyperparameter_vec[2], train_matrix, zero_train_matrix, valid_data, int(curr_hyperparameter_vec[3]), verbosity > 1)
            acc = evaluate(model, zero_train_matrix, valid_data)
            if iterations > 0:
                acc_delta = acc - last_acc
                if acc_delta > 0:
                    growth_vec = growth_vec * (accel_vec ** (param_mask * dir_vec))
                elif acc_delta < 0:
                    dir_vec[vec_i] = -1 * dir_vec[vec_i]
                    # Essentially, undo the previous move.
                    curr_hyperparameter_vec = curr_hyperparameter_vec * (growth_vec ** (dir_vec * param_mask))
                    if vec_i == 0:
                        curr_hyperparameter_vec[vec_i] = max(curr_hyperparameter_vec[vec_i], 1)
                    
                    growth_vec = growth_vec * (accel_vec ** -1)


        # Calculate iteration gains
        last_iteration_acc = iteration_acc
        iteration_acc = acc
        iteration_acc_delta = np.abs(iteration_acc - last_iteration_acc)
        results["accuracy"].append(acc)
        if verbosity > 0:
            print("Accuracy: {}".format(iteration_acc))
        if iteration_acc_delta < min_iteration_delta:
            consecutive_min_hit += 1
            if verbosity > 0:
                print("{} of {} consecutive minimum changes for conclusion.".format(consecutive_min_hit, min_consistency))
            if consecutive_min_hit >= min_consistency:                    
                if verbosity > 0:
                    print("Accuracy delta {} was below set threshold of {}. Concluding tuning. Results as follows:".format(iteration_acc_delta, min_iteration_delta))
                    print("#\tk\tlr\tlambda\tepochs")
                    print("{}\t{}\t{}\t{}\t{}".format(
                        iterations, 
                        curr_hyperparameter_vec[0],
                        curr_hyperparameter_vec[1],
                        curr_hyperparameter_vec[2],
                        curr_hyperparameter_vec[3]
                    ))
                break
        else:
            consecutive_min_hit = 0
        iterations += 1
    return results


def main():
    # k: 5, learning rate: 0.05, number of epochs: 40, lambda: 0.00025 from rough permutation
    min_delta_tuning({
        "k": 1,
        "lr": 0.005,
        "lambda": 0.001,
        "epochs": 10
    })

if __name__ == "__main__":
    main()