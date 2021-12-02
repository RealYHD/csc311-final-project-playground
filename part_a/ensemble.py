import sys
import numpy as np
from scipy import sparse
from sklearn.impute import KNNImputer
import torch
import item_response as irt
from torch.autograd import Variable
import utils
import neural_network as nn
import scipy.sparse

def predict_irt(u_id, q_id, theta, beta, binary = True):
    """ Makes a prediction of whether or not u_id will answer
    q_id correctly given theta and beta.
    """
    pred = irt.sigmoid(theta[u_id] - beta[q_id])
    if binary:
        return pred >= 0.5
    return pred

def impute_knn(sparse_matrix, k):
    nbrs = KNNImputer(n_neighbors=k)
    return nbrs.fit_transform(sparse_matrix.toarray())

def predict_knn_by_user(u_id, q_id, matrix, binary = True):
    pred = matrix[u_id, q_id]
    if binary:
        return pred >= 0.5
    return pred

def predict_nn(u_id, q_id, zero_data_matrix, model, binary = True):
    model.eval()
    inputs = Variable(zero_data_matrix[u_id]).unsqueeze(0)
    outputs = model(inputs)
    pred = outputs[0][q_id]
    if binary:
        return pred >= 0.5
    return pred

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

def sample_sparse_matrix(n, sparse_matrix, user_weights = [], guarantee_users = True):
    sparse_mat_shape = np.shape(sparse_matrix)
    num_users, num_questions = sparse_mat_shape
    composite = np.empty(sparse_mat_shape)
    composite[:] = np.nan
    sparse_matrix = sparse_matrix.toarray()

    if len(user_weights) == 0:
        user_weights = np.full(num_users, 1/num_users)

    if guarantee_users:
        composite = sparse_matrix
    
    selected = np.random.choice(num_users, n, replace=True, p=user_weights)
    primed = set()
    for user in selected:
        row = np.expand_dims(sparse_matrix[user], 0)
        if user in primed:
            composite = np.concatenate((composite, row))
        else:
            primed.add(user)
            composite[user] = row

    return sparse.csr_matrix(composite) # No reason other than to maintain standard.

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

def evaluate_ensemble(n = 300):
    sparse_matrix = utils.load_train_sparse("data")
    zero_sparse_matrix = sparse_matrix.toarray().copy()
    zero_sparse_matrix[np.isnan(zero_sparse_matrix)] = 0
    val_data = utils.load_valid_csv("data")
    test_data = utils.load_public_test_csv("data")

    num_users, num_questions = np.shape(sparse_matrix)

    # Train IRT
    irt_model = irt.irt(sample_sparse_matrix(300, sparse_matrix), val_data, 0.001, 180)

    # Train NN
    sub_sparse_matrix = sample_sparse_matrix(n, sparse_matrix, guarantee_users=False)
    sub_zero_matrix = sub_sparse_matrix.toarray().copy()
    sub_zero_matrix[np.isnan(sub_zero_matrix)] = 0
    nn_model = nn.AutoEncoder(num_questions, 5)
    nn.train(nn_model, 0.05, 0.00025, torch.FloatTensor(sub_sparse_matrix.toarray()), torch.FloatTensor(sub_zero_matrix), 40, val_data, verbosity=1)


    # kNN impute 
    knn_sub_sparse_mat = sample_sparse_matrix(n, sparse_matrix)
    knn_model = impute_knn(knn_sub_sparse_mat, k = 9)

    ind_correct = {
        "knn": 0,
        "nn": 0,
        "irt": 0
    }
    correct_pred = 0
    total = 0
    for i, u_id in enumerate(val_data["user_id"]):
        q_id = val_data["question_id"][i]
        answer_correct = val_data["is_correct"][i]
        
        # Eval IRT
        irt_prediction = predict_irt(u_id, q_id, irt_model[0], irt_model[1], binary=False)
        if (irt_prediction >= 0.5) == answer_correct:
            ind_correct["irt"] = 1 + ind_correct["irt"]

        # Eval NN
        nn_prediction = predict_nn(u_id, q_id, torch.FloatTensor(zero_sparse_matrix), nn_model, binary=False)
        if (nn_prediction >= 0.5) == answer_correct:
            ind_correct["nn"] = 1 + ind_correct["nn"]

        # Eval kNN
        knn_prediction = predict_knn_by_user(u_id, q_id, knn_model, binary=False) # TODO: By item may have slightly higher accuracy.
        if (knn_prediction >= 0.5) == answer_correct:
            ind_correct["knn"] = 1 + ind_correct["knn"]

        ensemble_prediction = irt_prediction + nn_prediction + knn_prediction
        ensemble_prediction = ensemble_prediction / 3.0
        ensemble_ind = ensemble_prediction >= 0.5
        if ensemble_ind == answer_correct:
            correct_pred = correct_pred + 1
        total = total + 1

        if i > 0:
            sys.stdout.write("\r Ind: {} knn cor.: {} irt cor.: {} nn cor.: {} ensemble acc.: {}".format(i, ind_correct["knn"], ind_correct["nn"], ind_correct["irt"], correct_pred / float(total)))
                        
    print("\nFinal acc: {}".format(correct_pred / total))

def main():
    evaluate_ensemble()

def test():
    sparse_matrix = utils.load_train_sparse("data")
    users = np.shape(sparse_matrix)[0]
    weight = np.full(users, 1/users)
    sample_sparse_matrix(200, sparse_matrix, user_weights=weight)

if __name__ == "__main__":
    main()

    # test()