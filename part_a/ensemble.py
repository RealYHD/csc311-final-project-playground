import numpy as np
from scipy import sparse
from sklearn.impute import KNNImputer
import item_response as irt
from torch.autograd import Variable
import utils
import neural_network as nn
import torch

def predict_irt(u_id, q_id, theta, beta):
    """ Makes a prediction of whether or not u_id will answer
    q_id correctly given theta and beta.
    """
    return irt.sigmoid(theta[u_id] - beta[q_id])

def predict_knn_by_user(u_id, q_id, matrix, k):
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(matrix)
    return mat[u_id, q_id]

def predict_nn(u_id, q_id, zero_data_matrix, model):
    model.eval()
    inputs = Variable(zero_data_matrix[u_id]).unsqueeze(0)
    outputs = model(inputs)
    return outputs[0][q_id]

def prune_users(keep_users, data_set):
    user_set = set()
    user_set.update(keep_users) # hashset for O(n + m) lookup speed

    check_ind = len(data_set["user_id"])
    while check_ind > 0:
        check_ind = check_ind - 1
        u = data_set["user_id"][u]
        if not u in user_set:
            del data_set["user_id"][check_ind]
            del data_set["question_id"][check_ind]
            del data_set["is_correct"][check_ind]

def sample_sparse_matrix(n, sparse_matrix, gaurantee_users = True):
    coords = np.where(not np.isnan(sparse_matrix))
    np.random.choice(coords, n)
    selection_mask = np.zeros(np.shape(sparse_matrix))
    selection_mask[coords] = 1

    if gaurantee_users:
        gaurantee_mask = np.zeros(np.shape(sparse_matrix))
        for u_id in np.shape(sparse_matrix)[0]:
            user_row = sparse_matrix[u_id]
            q_id = np.random.choice(np.argwhere(~np.isnan(user_row)))
            gaurantee_mask[u_id, q_id] = 1
        selection_mask = gaurantee_mask + selection_mask
        selection_mask[selection_mask > 1] = 1
    selection_mask[selection_mask == 0] = np.nan

    return np.multiply(sparse_matrix, selection_mask)

def matrix_to_dict(sparse_matrix):
    sampled_data = {
        "user_id": [],
        "question_id": [],
        "is_correct": []
    }

    mat_pts = np.where(not np.isnan(sparse_matrix))
    for pt in mat_pts:
        u_id, q_id = pt
        is_correct = mat_pts[pt]
        sampled_data["user_id"].append(u_id)
        sampled_data["question_id"].append(q_id)
        sampled_data["is_correct"].append(is_correct)
    return sampled_data

def evaluate_ensemble(n = 200):
    train_data = utils.load_train_csv("../data")
    sparse_matrix = utils.load_train_sparse("../data")
    zero_sparse_matrix = sparse_matrix.copy()
    zero_sparse_matrix[np.isnan(sparse_matrix)] = 0
    val_data = utils.load_valid_csv("../data")
    test_data = utils.load_public_test_csv("../data")

    num_users, num_questions = np.shape(sparse_matrix)

    # Train IRT
    irt_model = irt.irt(matrix_to_dict(sample_sparse_matrix(n, sparse_matrix)), val_data, 0.01, 20)

    # Train NN
    sub_sparse_matrix = sample_sparse_matrix(n, sparse_matrix, False)
    sub_zero_matrix = sub_sparse_matrix.copy()
    sub_zero_matrix[np.isnan(sub_zero_matrix)] = 0
    nn_model = nn.AutoEncoder(num_questions, 5)
    nn.train(nn_model, 0.05, 0.00025, sub_sparse_matrix, sub_zero_matrix, 40, verbosity_mode=False)

    correct_pred = 0
    total = 0
    for i, u_id in enumerate(val_data["user_id"]):
        q_id = val_data["question_id"][i]
        answer_correct = val_data["is_correct"][i]
        
        # Eval IRT
        irt_prediction = predict_irt(u_id, q_id, irt_model[0], irt_model[1])

        # Eval NN
        nn_prediction = predict_nn(u_id, q_id, zero_sparse_matrix, nn_model)

        # Eval kNN
        knn_prediction = predict_knn_by_user(u_id, q_id, sparse_matrix, 11) # TODO: By item may have slightly higher accuracy.

        ensemble_prediction = irt_prediction + nn_prediction + knn_prediction
        ensemble_prediction = ensemble_prediction / 3.0
        ensemble_ind = ensemble_prediction >= 0.5
        if ensemble_ind == answer_correct:
            correct_pred = correct_pred + 1
        total = 0
    print("Final acc: {}".format(correct_pred / total))
