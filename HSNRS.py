import numpy as np
from fisvdd import fisvdd
import warnings
from collections import Counter
from scipy.spatial import distance

warnings.filterwarnings("ignore")

class HyperSphere:
    """class of the Hyper-sphere"""
    def __init__(self, data, label, para_s, class_num, class_list):
        self.data = data
        self.label = label
        self.s = para_s
        self.class_num = class_num
        self.class_list = class_list

    def get_matrix(self, para_a, para_b):
        """
        calculate the Gaussian kernel distance between two vectors
        """
        dist_sq = distance.cdist(para_a, para_b)
        cur_sim_vec = np.exp(-np.square(dist_sq) / (2.0 * self.s * self.s))
        return cur_sim_vec

    def get_radius_and_centers(self, fd_k):
        """
        calculate the radius and centers
        """
        # radius
        used = 0
        term_1 = 1
        cur_sim_vec = self.get_matrix(fd_k.sv, fd_k.sv)
        term_2 = -2 * np.dot(cur_sim_vec[:, 0], fd_k.alpha)
        term_3 = np.dot(np.dot(fd_k.alpha.T, cur_sim_vec), fd_k.alpha)
        radius = np.sqrt(term_1 + term_2 + term_3)

        #center
        center = np.dot(fd_k.alpha.T, fd_k.sv)
        return radius, center

    def init_hyper_sphere(self):
        """
        initial hyper-spheres
        return: list of radius and center of hypersphere
        """
        radius_list = []
        center_list = []
        sv_list = []
        sv_index_list = []
        alpha_list = []
        for i in range(self.class_num):
            class_k_index = np.where(self.label == self.class_list[i])[0][:]
            data_k = self.data[class_k_index, :]
            #s_k = data_k.shape[1]*np.std(data_k)
            # s_k = np.sqrt(data_k.shape[1]/2)
            fd_k = fisvdd(data_k, self.s)
            fd_k.find_sv()
            each_radius, each_center = self.get_radius_and_centers(fd_k)
            radius_list.append(each_radius)
            center_list.append(each_center.reshape(1, len(each_center)))
            sv_list.append(fd_k.sv)
            sv_index_list.append(fd_k.index)
            alpha_list.append(fd_k.alpha)
        parameters = {
            "radius": radius_list,
            "center": center_list,
            "sv" : sv_list,
            "sv_index": sv_index_list,
            "alpha": alpha_list
        }
        self.parameter = parameters
        return parameters

    def get_purity(self, remove_i, parameter):
        attributes = list(range(self.data.shape[1]))
        attributes.remove(remove_i)
        dis_list = self.get_distance_to_center(attributes, parameter)
        compare_distance_list = []
        for i in range(self.class_num):
            compare_distance = np.where(dis_list[i] > parameter["radius"][i])[0]
            compare_distance_list.append(compare_distance)
        return compare_distance_list


    def get_distance_to_center(self, att, parameter):
        distance_list = []
        for i in range(self.class_num):
            class_k_index = np.where(self.label == self.class_list[i])[0][:]
            data_k = self.data[class_k_index, :]
            data_index = list(range(data_k.shape[0]))
            non_sv_index = list(set(data_index) - set(self.parameter["sv_index"][i]))
            temp_data = data_k[non_sv_index, :]

            # term_2 = -2*self.get_matrix(temp_data[:, att], parameter["center"][i][:, att])
            a = self.get_matrix(temp_data[:, att], parameter["sv"][i][:, att])
            term_2 = -2*np.dot(self.get_matrix(temp_data[:, att], parameter["sv"][i][:, att]), parameter["alpha"][i])
            cur_sim_vec = self.get_matrix(parameter["sv"][i][:, att], parameter["sv"][i][:, att])
            term_3 = np.dot(np.dot(parameter["alpha"][i].T, cur_sim_vec), parameter["alpha"][i])
            a = 1+term_2+term_3
            temp_distance = np.sqrt(1+term_2+term_3)
            distance_list.append(temp_distance)
        return distance_list

def get_label_list(label):
    """
    get the number and list of label types of data
    """
    class_list = []
    class_dict = Counter(label[:, 0])
    for item in class_dict:
        class_list.append(item)
    class_num = len(class_list)
    return class_num, class_list

def get_attribute_reduction(data, label, para_s):
    """

    """
    # init
    class_num, class_list = get_label_list(label)
    hyper_sphere = HyperSphere(data, label, para_s, class_num, class_list)
    parameters = hyper_sphere.init_hyper_sphere()

    attributes_reduction = list(range(data.shape[1]))
    for i in range(data.shape[1]):
        if len(attributes_reduction) <= 1:
            break

        the_remove_i = attributes_reduction.index(i)
        attributes_reduction.remove(i)  # remove the ith attribute
        distance_list = hyper_sphere.get_purity(the_remove_i, parameters)
        num_mis = 0
        for j in range(class_num):
            num_mis += len(distance_list[j])
        print(num_mis/data.shape[0])  #or num_mis/data.shape[0] < 0.01
        if num_mis == 0:
            hyper_sphere = HyperSphere(data[:, attributes_reduction], label, para_s, class_num, class_list)
            parameters = hyper_sphere.init_hyper_sphere()
        else:
            attributes_reduction.append(i)
            attributes_reduction.sort()

    attributes_reduction = attributes_reduction
    return attributes_reduction






