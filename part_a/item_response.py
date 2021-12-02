import sys
from utils import *

import numpy as np


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(sparse_matrix, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    zero_sparse_matrix = sparse_matrix.toarray()
    zero_sparse_matrix[np.isnan(zero_sparse_matrix)] = 0
    num_users, num_questions = np.shape(zero_sparse_matrix)
    theta_mat = np.tile(np.expand_dims(theta, 1), (1, num_questions))
    beta_mat = np.tile(np.expand_dims(beta, 0), (num_users, 1))

    log_lklihood = zero_sparse_matrix * (theta_mat - beta_mat - np.logaddexp(0, theta_mat - beta_mat))\
        - (1-zero_sparse_matrix) * (np.logaddexp(0, theta_mat - beta_mat))
    log_lklihood = np.sum(log_lklihood) # Only include positives, and sum instead of multiply (log)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta_vec(sparse_matrix, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    num_users, num_questions = np.shape(sparse_matrix)

    c_mat = sparse_matrix.toarray()
    c_mat[np.isnan(c_mat)] = 0

    theta_mat = np.tile(np.expand_dims(theta, 1), (1, num_questions))
    beta_mat = np.tile(np.expand_dims(beta, 0), (num_users, 1))

    theta_grad = np.sum(c_mat - sigmoid(theta_mat - beta_mat), 1)
    theta = theta - (lr * theta_grad)

    beta_grad = np.sum(-c_mat + sigmoid(theta_mat - beta_mat), 0)
    beta = beta - (lr * beta_grad)

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta

def update_theta_beta(sparse_matrix, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    num_users, num_questions = np.shape(sparse_matrix)

    zero_sparse_matrix = sparse_matrix.toarray()
    zero_sparse_matrix[np.isnan(zero_sparse_matrix)] = 0

    for i in range(len(theta)):
        theta[i] = theta[i] - lr * np.sum(- (1 - sigmoid(theta[i] - beta)))
    
    for i in range(len(beta)):
        beta[i] = beta[i] - lr * np.sum(- (theta - beta[i]))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(sparse_matrix, val_data, lr, iterations, step_func = update_theta_beta_vec, verbosity = 1):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    num_users, num_questions = np.shape(sparse_matrix)
    rng = np.random.default_rng()
    theta = np.asarray(rng.uniform(0.001, 1, num_users)) # Size is of number of users based on ID.
    beta = np.asarray(rng.uniform(0.001, 1, num_questions)) # Size is of number of questions based on question ID.

    val_acc_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(sparse_matrix, theta=theta, beta=beta)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        if verbosity == 2:
            print("NLLK: {} \t Score: {}".format(neg_lld, score))
        elif verbosity == 1:
            sys.stdout.write("\rNLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = step_func(sparse_matrix, lr, theta, beta)
    if verbosity == 1:
        print()
    print("Final results: NLLK: {} \t Score: {}".format(neg_lld, score))

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])

def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    iteration_sets = [25, 50, 100, 200]
    lrs = [0.001, 0.01]

    for iterations in iteration_sets:
        for lr in lrs:
            print("Training with lr of {} and {} number of iterations.".format(lr, iterations))
            theta, beta, step_accs = irt(sparse_matrix, val_data, lr, iterations)
            acc = evaluate(test_data, theta, beta)
            print("Final accuracy: {}".format(acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    pass
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
