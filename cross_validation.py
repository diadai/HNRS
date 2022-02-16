import numpy as np
np.random.seed(0)
def cross_validation(para_len, para_k):

    random_index = np.random.permutation(para_len).tolist()

    temp_num = para_len // para_k


    ret_train_index = []
    ret_test_index = []

    for i in range(para_k):
        temp_start_index = i * temp_num
        temp_end_index = (i + 1) * temp_num
        ret_test_index.append(random_index[temp_start_index: temp_end_index])

        temp_index = random_index[:temp_start_index] + random_index[temp_end_index:]

        ret_train_index.append(temp_index)

    return ret_train_index, ret_test_index


# if __name__ == '__main__':
#     temp_train_index , temp_test_index = cross_validation(21, 5)
#     print(temp_train_index[0], len(np.sort(temp_train_index[1])))
#     print(temp_test_index[0], len(np.sort(temp_test_index[0])))
