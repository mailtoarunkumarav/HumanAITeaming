import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import datetime
import sys
from Functions import FunctionHelper
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

import os, re
sys.path.append("../..")
from HelperUtility.PrintHelper import PrintHelper as PH

import math
np.random.seed(500)

class GaussianProcessRegressor:

    # Constructor
    def __init__(self, output_gen_time, id, kernel_type, number_of_test_datapoints, noise, linspacexmin, linspacexmax,
                 linspaceymin, linspaceymax, signal_variance, number_of_dimensions, number_of_observed_samples, X, y, y_orig,
                 number_of_restarts_likelihood, bounds, lengthscale_bounds, signal_variance_bounds, Xmin, Xmax,
                 ymin, ymax, Xs, ys, ys_orig, char_len_scale, len_weights, len_weights_bounds, weight_params_estimation, fun_helper_obj):

        self.id = id
        self.output_gen_time = output_gen_time
        self.kernel_type = kernel_type
        self.number_of_test_datapoints = number_of_test_datapoints
        self.noise = noise
        self.linspacexmin = linspacexmin
        self.linspacexmax = linspacexmax
        self.linspaceymin = linspaceymin
        self.linspaceymax = linspaceymax
        self.signal_variance = signal_variance
        self.number_of_dimensions = number_of_dimensions
        self.number_of_observed_samples = number_of_observed_samples
        self.X = X
        self.y = y
        self.y_orig = y_orig
        self.number_of_restarts_likelihood = number_of_restarts_likelihood
        self.bounds = bounds
        self.lengthscale_bounds = lengthscale_bounds
        self.signal_variance_bounds = signal_variance_bounds
        self.Xmin = Xmin
        self.Xmax = Xmax
        self.ymin = ymin
        self.ymax = ymax
        self.Xs = Xs
        self.ys = ys
        self.ys_orig = ys_orig
        self.char_len_scale = char_len_scale
        self.len_weights = len_weights
        self.len_weights_bounds = len_weights_bounds
        self.weight_params_estimation = weight_params_estimation
        self.fun_helper_obj = fun_helper_obj
        self.L_X_X = None
        self.K_Xs_Xs = None
        self.lengthscale_list = []

    def gp_fit(self, X, y):
        self.X = X
        self.y = y
        self.L_X_X = None
        self.K_Xs_Xs = None

    def refit_utils_std_ys(self, y_orig):
        self.y_orig = y_orig
        self.ys = (self.ys_orig - np.mean(y_orig)) / np.std(y_orig)

    def refit_std_y(self, ynew_original):
        self.y_orig = np.append(self.y_orig, [ynew_original], axis=0)
        y_mean = np.mean(self.y_orig)
        y_std = np.std(self.y_orig)
        self.y = (self.y_orig - y_mean) /y_std
        self.ys = (self.ys_orig - y_mean)/y_std
        return self.y

    # Define Plot Prior
    def plot_graph(self, plot_params):
        plt.figure(plot_params['plotnum'])
        plt.clf()
        for eachplot in plot_params['plotvalues']:
            if (len(eachplot) == 2):
                plt.plot(eachplot[0], eachplot[1])
            elif (len(eachplot) == 3):
                plt.plot(eachplot[0], eachplot[1], eachplot[2])
            elif (len(eachplot) == 4):
                if(eachplot[3].startswith("label=")):
                    plt.plot(eachplot[0], eachplot[1], eachplot[2], label=eachplot[3][6:])
                    plt.legend(loc='upper right',prop={'size': 14})
                else:
                    flag = eachplot[3]
                    if flag.startswith('lw'):
                        plt.plot(eachplot[0], eachplot[1], eachplot[2], lw=eachplot[3][2:])
                    elif flag.startswith('ms'):
                        plt.plot(eachplot[0], eachplot[1], eachplot[2], ms=eachplot[3][2:])

            elif (len(eachplot) == 5):

                if(eachplot[3].startswith("label=")):
                    flag = eachplot[4]
                    if flag.startswith('lw'):
                        plt.plot(eachplot[0], eachplot[1], eachplot[2], label=eachplot[3][6:], lw=eachplot[4][2:])
                    elif flag.startswith('ms'):
                        plt.plot(eachplot[0], eachplot[1], eachplot[2], label=eachplot[3][6:], ms=eachplot[4][2:])
                    plt.legend(loc='upper right',prop={'size': 14})

                else:
                    flag = eachplot[3]
                    if flag.startswith('lw'):
                        plt.plot(eachplot[0], eachplot[1], eachplot[2], lw=eachplot[3][2:])
                    elif flag.startswith('ms'):
                        plt.plot(eachplot[0], eachplot[1], eachplot[2], ms=eachplot[3][2:])

        if 'gca_fill' in plot_params.keys():
            if len(plot_params['gca_fill']) == 3:
                plt.gca().fill_between(plot_params['gca_fill'][0], plot_params['gca_fill'][1],
                                       plot_params['gca_fill'][2],
                                       color="#66cc66", alpha=0.5)
            else:
                if plot_params['gca_fill'][3].startswith('color'):
                    color = plot_params['gca_fill'][3][6:]
                    PH.printme(PH.p1, len(plot_params['gca_fill']), color)
                    plt.gca().fill_between(plot_params['gca_fill'][0], plot_params['gca_fill'][1],
                                           plot_params['gca_fill'][2], color=color)

        # plt.axis(plot_params['axis'])
        plt.xlim(plot_params['axis'][0], plot_params['axis'][1])
        plt.title(plot_params['title'])
        plt.xlabel(plot_params['xlabel'])
        plt.ylabel(plot_params['ylabel'])
        plt.savefig(plot_params['file']+".pdf", bbox_inches='tight')


    # Define the kernel function
    def computekernel(self, data_point1, data_point2):

        if self.kernel_type == 'SE':
             result = self.sq_exp_kernel(data_point1, data_point2, self.char_len_scale, self.signal_variance)

        elif self.kernel_type == 'MATERN5':
            result = self.matern5_kernel(data_point1, data_point2, self.char_len_scale, self.signal_variance)

        elif self.kernel_type == 'MATERN3':
            result = self.matern3_kernel(data_point1, data_point2, self.char_len_scale, self.signal_variance)

        elif self.kernel_type == 'MKL':
            result = self.multi_kernel(data_point1, data_point2, self.char_len_scale, self.signal_variance)

        elif self.kernel_type == 'POLY':
            result = self.poly_kernel(data_point1, data_point2, self.degree)

        elif self.kernel_type == 'LIN':
            result = self.linear_kernel(data_point1, data_point2, self.char_len_scale, self.signal_variance)

        elif self.kernel_type == 'PER':
            result = self.periodic_kernel(data_point1, data_point2, self.char_len_scale, self.signal_variance)

        return result

    def matern3_kernel(self, data_point1, data_point2, char_len_scale, signal_variance):

        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                difference = (data_point1[i, :] - data_point2[j, :])
                l2_difference = np.sqrt(np.dot(difference, difference.T))
                each_kernel_val = (signal_variance ** 2) * (1 + (np.sqrt(3)*l2_difference/char_len_scale)) * \
                                  (np.exp((-1 * np.sqrt(3) / char_len_scale) * l2_difference))
                kernel_mat[i, j] = each_kernel_val
        return kernel_mat

    def matern5_kernel(self, data_point1, data_point2, char_len_scale, signal_variance):

        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                difference = (data_point1[i, :] - data_point2[j, :])
                l2_difference = np.sqrt(np.dot(difference, difference.T))
                each_kernel_val = (signal_variance**2)* (1 + (np.sqrt(5)*l2_difference/char_len_scale) + (5*(l2_difference**2)/(
                        3*(char_len_scale**2)))) * (np.exp((-1 * np.sqrt(5) * l2_difference / char_len_scale)))
                kernel_mat[i, j] = each_kernel_val
        return kernel_mat


    def periodic_kernel(self, data_point1, data_point2, char_len_scale, signal_variance):

        p = 2
        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                difference = (data_point1[i, :] - data_point2[j, :])
                each_kernel_val = (signal_variance ** 2) * (np.exp((-2 / (char_len_scale**2)) * ((np.sin(difference*(np.pi/p)))**2)))
                kernel_mat[i, j] = each_kernel_val
        return kernel_mat

    def multi_kernel_arxiv(self, data_point1, data_point2, char_len_scale, signal_variance):

        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                difference = (data_point1[i, :] - data_point2[j, :])
                l2_difference = np.sqrt(np.dot(difference, difference.T))
                l2_difference_sq = np.dot(difference, difference.T)
                sek = (signal_variance ** 2) * (np.exp((-1 / (2*char_len_scale**2)) * l2_difference_sq))
                mat3 = (signal_variance ** 2) * (1 + (np.sqrt(3)*l2_difference/char_len_scale)) * \
                                  (np.exp((-1 * np.sqrt(3) / char_len_scale) * l2_difference))
                lin = signal_variance + np.dot(data_point1[i, :], data_point2[j, :].T) * (char_len_scale**2)

                p = 2
                # periodic = (signal_variance ** 2) * (np.exp((-2 / (char_len_scale**2)) * ((np.pi/p)*((np.sin(difference)))**2)))

                degree_val1 = 4
                poly1 = signal_variance+np.power(np.dot(data_point1[i, :], data_point2[j, :].T), degree_val1)

                degree_val2 = 6
                poly2 = signal_variance+np.power(np.dot(data_point1[i, :], data_point2[j, :].T), degree_val2)
                each_kernel_val = self.len_weights[0] * sek + self.len_weights[1] * mat3 + self.len_weights[2] * lin \
                                  + self.len_weights[3] * poly1 + self.len_weights[4] * poly2

                kernel_mat[i, j] = each_kernel_val
        return kernel_mat

    def multi_kernel(self, data_point1, data_point2, char_len_scale, signal_variance):

        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                difference = (data_point1[i, :] - data_point2[j, :])
                l2_difference = np.sqrt(np.dot(difference, difference.T))
                l2_difference_sq = np.dot(difference, difference.T)
                sek = (signal_variance ** 2) * (np.exp((-1 / (2*(self.len_weights[1])**2)) * l2_difference_sq))
                lin = self.len_weights[3] + np.dot(data_point1[i, :], data_point2[j, :].T)
                degree_val1 = 3
                poly1 = signal_variance + np.power(np.dot(np.dot(data_point1[i, :], data_point2[j, :].T), self.len_weights[5]), degree_val1)
                each_kernel_val = self.len_weights[0] * sek + self.len_weights[2] * lin + self.len_weights[4] * poly1
                kernel_mat[i, j] = each_kernel_val
        return kernel_mat

    def sq_exp_kernel_matversion(self, data_point1, data_point2, char_length_scale, signal_variance):

        # Define the SE kernel function
        total_squared_distances = np.sum(data_point1 ** 2, 1).reshape(-1, 1) + np.sum(data_point2 ** 2, 1) - 2 * np.dot(
            data_point1, data_point2.T)
        kernel_val = (signal_variance **2) * np.exp(-(total_squared_distances * (1 / ((char_length_scale**2) * 2.0))))
        # print (kernel_val)
        return kernel_val

    def sq_exp_kernel_vanilla(self, data_point1, data_point2, char_len_scale, signal_variance):

        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                difference = (data_point1[i, :] - data_point2[j, :])
                l2_difference_sq = np.dot(difference, difference.T)
                each_kernel_val = (signal_variance ** 2) * (np.exp((-1 / (2 * char_len_scale ** 2)) * l2_difference_sq))
                kernel_mat[i, j] = each_kernel_val
        return kernel_mat

    def sq_exp_kernel(self, data_point1, data_point2, char_len_scale, signal_variance):

        if self.number_of_dimensions == 1:
            return self.sq_exp_kernel_vanilla(data_point1, data_point2, char_len_scale, signal_variance)
        else:
            return self.ard_sq_exp_kernel(data_point1, data_point2, char_len_scale, signal_variance)

    def ard_sq_exp_kernel(self, data_point1, data_point2, char_len_scale, signal_variance):

        # Element wise squaring the vector of given length scales
        char_len_scale = np.array(char_len_scale) ** 2
        # Computing inverse of a diagonal matrix by reciprocating each item in the diagonal
        inv_char_len = 1 / char_len_scale
        # Creating a Diagonal matrix with squared l values
        inv_sq_dia_len = np.diag(inv_char_len)

        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                difference = ((data_point1[i, :] - data_point2[j, :]))
                product1 = np.dot(difference, inv_sq_dia_len)
                final_product = np.dot(product1, difference.T)
                each_kernel_val = (signal_variance**2) * (np.exp((-1 / 2.0) * final_product))
                kernel_mat[i, j] = each_kernel_val
        return kernel_mat

    def poly_kernel(self, data_point1, data_point2, degree):

        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                each_kernel_val = 1+np.power(np.dot(data_point1[i, :], data_point2[j, :].T), degree)
                # each_kernel_val = each_kernel_val/number_of_observed_samples
                kernel_mat[i, j] = each_kernel_val
        return kernel_mat

    def linear_kernel(self, data_point1, data_point2, char_len_scale, signal_variance):

        kernel_mat = np.zeros(shape=(len(data_point1), len(data_point2)))
        for i in np.arange(len(data_point1)):
            for j in np.arange(len(data_point2)):
                each_kernel_val = signal_variance + np.dot(data_point1[i, :], data_point2[j, :].T) * (char_len_scale**2)
                # each_kernel_val = each_kernel_val/number_of_observed_samples
                kernel_mat[i, j] = each_kernel_val
        return kernel_mat

    def optimize_log_marginal_likelihood_l(self, input):

        # 0 to n-1 elements represent the nth eleme
        init_charac_length_scale = np.array(input[: self.number_of_dimensions])
        signal_variance = input[len(input)-1]

        if self.kernel_type == 'SE':
            K_x_x = self.sq_exp_kernel(self.X, self.X, init_charac_length_scale, signal_variance)
        elif self.kernel_type == 'MATERN3':
            K_x_x = self.matern3_kernel(self.X, self.X, init_charac_length_scale, signal_variance)
        elif self.kernel_type == 'MATERN5':
            K_x_x = self.matern5_kernel(self.X, self.X, init_charac_length_scale, signal_variance)
        elif self.kernel_type == 'LIN':
            K_x_x = self.linear_kernel(self.X, self.X, init_charac_length_scale, signal_variance)
        elif self.kernel_type == 'PER':
            K_x_x = self.periodic_kernel(self.X, self.X, init_charac_length_scale, signal_variance)

        # K_x_x = self.sq_exp_kernel_vanilla(self.X, self.X, init_charac_length_scale, signal_variance)
        eye = 1e-6 * np.eye(len(self.X))
        Knoise = K_x_x + eye
        # Find L from K = L *L.T instead of inversing the covariance function
        L_x_x = np.linalg.cholesky(Knoise)
        factor = np.linalg.solve(L_x_x, self.y)
        log_marginal_likelihood = -0.5 * (np.dot(factor.T, factor) +
                                          self.number_of_observed_samples * np.log(2 * np.pi) +
                                          np.log(np.linalg.det(Knoise)))
        return log_marginal_likelihood[0]

    def optimize_log_marginal_likelihood_weight_params(self, input):

        self.len_weights = input[0:(len(input) - 1)]
        self.signal_variance = input[len(input) - 1]

        K_x_x = self.multi_kernel(self.X, self.X, self.char_len_scale, self.signal_variance)
        eye = 1e-3 * np.eye(len(self.X))
        Knoise = K_x_x + eye
        # Find L from K = L *L.T instead of inversing the covariance function
        # L_x_x = np.linalg.cholesky(Knoise)

        try:
            L_x_x = np.linalg.cholesky(Knoise)
            factor = np.linalg.solve(L_x_x, self.y)

        except np.linalg.LinAlgError:
            PH.printme(PH.p1, "!!!!!!!!!!!Matrix is not positive definite, params: ", input, "Eigen Vals:", np.linalg.eigvals(Knoise), "\n",
                           K_x_x)
            L_x_x = np.linalg.cholesky(Knoise)

        factor = np.linalg.solve(L_x_x, self.y)
        log_marginal_likelihood = -0.5 * (np.dot(factor.T, factor) + len(self.X) * np.log(2 * np.pi) +
                                          np.log(np.linalg.det(Knoise)))
        # PH.printme(PH.p1, "Trying.... ", input, "\t:Logl: ", log_marginal_likelihood)
        return log_marginal_likelihood[0]

    def get_kernel_matrix_condition_number(self, input):

        if self.kernel_type == "MKL":
            self.len_weights = input
            K_x_x = self.multi_kernel(self.X, self.X, self.char_len_scale, self.signal_variance)

        elif self.kernel_type == "SE":
            self.char_len_scale = input
            K_x_x = self.sq_exp_kernel(self.X, self.X, self.char_len_scale, self.signal_variance)

        K_x_x = self.multi_kernel(self.X, self.X, self.char_len_scale, self.signal_variance)
        eye = 1e-3 * np.eye(len(self.X))
        Knoise = K_x_x + eye
        condition_num = np.linalg.cond(Knoise)
        return condition_num


    def optimize_log_marginal_likelihood_weight_params_const(self, input):

        self.len_weights = input

        K_x_x = self.multi_kernel(self.X, self.X, self.char_len_scale, self.signal_variance)
        eye = 1e-3 * np.eye(len(self.X))
        Knoise = K_x_x + eye
        # Find L from K = L *L.T instead of inversing the covariance function
        # L_x_x = np.linalg.cholesky(Knoise)

        try:
            L_x_x = np.linalg.cholesky(Knoise)
            factor = np.linalg.solve(L_x_x, self.y)

        except np.linalg.LinAlgError:
            PH.printme(PH.p1, "!!!!!!!!!!!Matrix is not positive definite, params: ", input, "Eigen Vals:", np.linalg.eigvals(Knoise), "\n",
                           K_x_x)
            L_x_x = np.linalg.cholesky(Knoise)

        factor = np.linalg.solve(L_x_x, self.y)
        log_marginal_likelihood = -0.5 * (np.dot(factor.T, factor) + len(self.X) * np.log(2 * np.pi) +
                                          np.log(np.linalg.det(Knoise)))
        # PH.printme(PH.p1, "Trying.... ", input, "\t:Logl: ", log_marginal_likelihood)
        return log_marginal_likelihood[0]


    def gaussian_predict(self, Xs):

        # Commenting to speed up the comuptations and avoid prior and posterior samples
        # # compute the covariances between the unseen data points i.e K**
        # K_xs_xs = self.computekernel(Xs, Xs)
        #
        # # Cholesky decomposition to find L from covariance matrix K i.e K = L*L.T
        # L_xs_xs = np.linalg.cholesky(K_xs_xs + 1e-6 * np.eye(self.number_of_test_datapoints))
        #
        # # Sample 3 standard normals for each of the unseen data points
        # standard_normals = np.random.normal(size=(self.number_of_test_datapoints, 3))
        #
        # # multiply them by the square root of the covariance matrix L
        # f_prior = np.dot(L_xs_xs, standard_normals)
        f_prior = None

        # Compute mean and variance
        mean, variance, factor1 = self.compute_mean_var(Xs, self.X, self.y)
        diag_variance = np.diag(variance)

        f_post = None
        return mean, diag_variance, f_prior, f_post

    def gaussian_predict_on_conditioned_X(self, Xs, X, y):

        f_prior = None

        # Compute mean and variance
        mean, variance, factor1 = self.compute_mean_var(Xs, X, y)
        diag_variance = np.diag(variance)
        f_post = None
        return mean, diag_variance, f_prior, f_post

    def compute_mean_var(self, Xs, X, y):

        # Apply the kernel function to our training points

        K_x_x = self.computekernel(X, X)
        eye = 1e-10 * np.eye(len(X))

        if self.id == "ai" or self.id == "baseline":
            eye = 0.01 * np.eye(len(X))

        L_X_X = np.linalg.cholesky(K_x_x + eye)

        K_x_xs = self.computekernel(X, Xs)
        factor1 = np.linalg.solve(L_X_X, K_x_xs)
        factor2 = np.linalg.solve(L_X_X, y)
        mean = np.dot(factor1.T, factor2).flatten()

        K_Xs_Xs = self.computekernel(Xs, Xs)
        variance = K_Xs_Xs - np.dot(factor1.T, factor1)

        return mean, variance, factor1

    def runGaussian(self, pwd_qualifier, role, plot_posterior):

        PH.printme(PH.p1, "!!!!!!!!!!Gaussian Process Fitting Started!!!!!!!!!" )

        # if self.id == "ai" or self.id == "baseline":
        if self.kernel_type == 'MKL':
            if self.weight_params_estimation:

                PH.printme(PH.p1, "Maximising the weights of the multi kernel")
                x_max_value = None
                log_like_max = - 1 * float("inf")

                random_points_a = []
                random_points_b = []
                random_points_c = []
                random_points_d = []
                random_points_e = []
                random_points_f = []

                # Data structure to create the starting points for the scipy.minimize method
                random_data_point_each_dim = np.random.uniform(self.len_weights_bounds[0][0],
                                                               self.len_weights_bounds[0][1],
                                                               self.number_of_restarts_likelihood).reshape(1,
                                                                                                           self.number_of_restarts_likelihood)
                random_points_a.append(random_data_point_each_dim)

                random_data_point_each_dim = np.random.uniform(self.len_weights_bounds[1][0],
                                                               self.len_weights_bounds[1][1],
                                                               self.number_of_restarts_likelihood).reshape(1,
                                                                                                           self.number_of_restarts_likelihood)
                random_points_b.append(random_data_point_each_dim)

                random_data_point_each_dim = np.random.uniform(self.len_weights_bounds[2][0],
                                                               self.len_weights_bounds[2][1],
                                                               self.number_of_restarts_likelihood).reshape(1,
                                                                                                           self.number_of_restarts_likelihood)
                random_points_c.append(random_data_point_each_dim)

                random_data_point_each_dim = np.random.uniform(self.len_weights_bounds[3][0],
                                                               self.len_weights_bounds[3][1],
                                                               self.number_of_restarts_likelihood).reshape(1,
                                                                                                           self.number_of_restarts_likelihood)
                random_points_d.append(random_data_point_each_dim)

                random_data_point_each_dim = np.random.uniform(self.len_weights_bounds[4][0],
                                                               self.len_weights_bounds[4][1],
                                                               self.number_of_restarts_likelihood).reshape(1,
                                                                                                           self.number_of_restarts_likelihood)
                random_points_e.append(random_data_point_each_dim)

                random_data_point_each_dim = np.random.uniform(self.len_weights_bounds[5][0],
                                                               self.len_weights_bounds[5][1],
                                                               self.number_of_restarts_likelihood).reshape(1,
                                                                                                           self.number_of_restarts_likelihood)
                random_points_f.append(random_data_point_each_dim)

                # Vertically stack the arrays of randomly generated starting points as a matrix
                random_points_a = np.vstack(random_points_a)
                random_points_b = np.vstack(random_points_b)
                random_points_c = np.vstack(random_points_c)
                random_points_d = np.vstack(random_points_d)
                random_points_e = np.vstack(random_points_e)
                random_points_f = np.vstack(random_points_f)
                variance_start_points = np.random.uniform(self.signal_variance_bounds[0],
                                                          self.signal_variance_bounds[1],
                                                          self.number_of_restarts_likelihood)

                for ind in np.arange(self.number_of_restarts_likelihood):

                    tot_init_points = []

                    param_a = random_points_a[0][ind]
                    tot_init_points.append(param_a)
                    param_b = random_points_b[0][ind]
                    tot_init_points.append(param_b)
                    param_c = random_points_c[0][ind]
                    tot_init_points.append(param_c)
                    param_d = random_points_d[0][ind]
                    tot_init_points.append(param_d)
                    param_e = random_points_e[0][ind]
                    tot_init_points.append(param_e)
                    param_f = random_points_f[0][ind]
                    tot_init_points.append(param_f)
                    tot_init_points.append(variance_start_points[ind])
                    total_bounds = self.len_weights_bounds.copy()
                    total_bounds.append(self.signal_variance_bounds)

                    maxima = opt.minimize(lambda x: -self.optimize_log_marginal_likelihood_weight_params(x),
                                          tot_init_points,
                                          method='L-BFGS-B',
                                          tol=0.01,
                                          options={'maxfun': 200, 'maxiter': 40},
                                          bounds=total_bounds)

                    params = maxima['x']
                    log_likelihood = self.optimize_log_marginal_likelihood_weight_params(params)
                    if log_likelihood > log_like_max:
                        PH.printme(PH.p1, "New maximum log likelihood ", log_likelihood, " found for params ", params)
                        x_max_value = maxima['x']
                        log_like_max = log_likelihood

                self.len_weights = x_max_value[0:(len(maxima['x']) - 1)]
                self.signal_variance = x_max_value[len(maxima['x']) - 1]

                PH.printme(PH.p1, "Opt weights: ", self.len_weights, "   variance:", self.signal_variance)

        # if self.id == "GroundTruth" or self.id == "HumanExpert":
        if self.kernel_type == "LIN" or self.kernel_type == "SE":

            log_like_max = - 1 * float("inf")
            if self.weight_params_estimation:
                PH.printme(PH.p1, "Hyper Params estimating.. for ", self.kernel_type)
                # Estimating Length scale itself

                x_max_value = None

                # Data structure to create the starting points for the scipy.minimize method
                random_points = []
                starting_points = []

                # Depending on the number of dimensions and bounds, generate random multi-starting points to find maxima
                for dim in np.arange(self.number_of_dimensions):
                    random_data_point_each_dim = np.random.uniform(self.lengthscale_bounds[dim][0],
                                                                   self.lengthscale_bounds[dim][1],
                                                                   self.number_of_restarts_likelihood). \
                        reshape(1, self.number_of_restarts_likelihood)
                    random_points.append(random_data_point_each_dim)

                # Vertically stack the arrays of randomly generated starting points as a matrix
                random_points = np.vstack(random_points)

                # Reformat the generated random starting points in the form [x1 x2].T for the specified number of restarts
                for sample_num in np.arange(self.number_of_restarts_likelihood):
                    array = []
                    for dim_count in np.arange(self.number_of_dimensions):
                        array.append(random_points[dim_count, sample_num])
                    starting_points.append(array)
                starting_points = np.vstack(starting_points)

                variance_start_points = np.random.uniform(self.signal_variance_bounds[0],
                                                          self.signal_variance_bounds[1],
                                                          self.number_of_restarts_likelihood)

                total_bounds = self.lengthscale_bounds.copy()
                total_bounds.append(self.signal_variance_bounds)

                for ind in np.arange(self.number_of_restarts_likelihood):

                    init_len_scale = starting_points[ind]
                    init_var = variance_start_points[ind]

                    init_points = np.append(init_len_scale, init_var)
                    maxima = opt.minimize(lambda x: -self.optimize_log_marginal_likelihood_l(x),
                                          init_points,
                                          method='L-BFGS-B',
                                          tol=0.01,
                                          options={'maxfun': 100, 'maxiter': 100},
                                          bounds=total_bounds)

                    len_scale_temp = maxima['x'][:self.number_of_dimensions]
                    variance_temp = maxima['x'][len(maxima['x']) - 1]
                    params = np.append(len_scale_temp, variance_temp)
                    log_likelihood = self.optimize_log_marginal_likelihood_l(params)

                    if (log_likelihood > log_like_max):
                        PH.printme(PH.p1, "New maximum log likelihood ", log_likelihood, " found for l= ",
                                   maxima['x'][: self.number_of_dimensions], " var:", maxima['x'][len(maxima['x']) - 1])

                        x_max_value = maxima
                        log_like_max = log_likelihood

                self.char_len_scale = x_max_value['x'][:self.number_of_dimensions]
                self.signal_variance = x_max_value['x'][len(maxima['x']) - 1]
                PH.printme(PH.p1, "Opt Length scale: ", self.char_len_scale, "\nOpt variance: ", self.signal_variance)
                self.lengthscale_list.append(self.char_len_scale)



        if self.number_of_dimensions == 1 and plot_posterior:
            mean, variance, factor1 = self.compute_mean_var(self.Xs, self.X, self.y)
            diag_variance = np.diag(variance)
            standard_deviation = np.sqrt(diag_variance)

            # Computing kernel
            # self.plot_kernel("Kernel Samples", None)

            if self.kernel_type == 'SE':
                title = "SE Kernel"
            elif self.kernel_type == 'MATERN3':
                title = "Matern 3/2 Kernel"
            elif self.kernel_type == 'MKL':
                title = "Multiple Kernel Learning"
            elif self.kernel_type == 'LIN':
                title = "Linear Kernel"

            plot_posterior_distr_params = {'plotnum': 'GP_Posterior_Distr_' + "_" + role,
                                           # 'axis': [self.linspacexmin, self.linspacexmax, linspaceymin, linspaceymax],
                                           # 'axis': [0, 1, self.linspaceymin, self.linspaceymax],
                                           'axis': [-0.1, 1.1, -3, 3],
                                           'plotvalues': [[self.X, self.y, 'r+', 'ms20'], [self.Xs, self.ys, 'b-', 'label=True Fn'],
                                                          [self.Xs, mean, 'g--', 'label=Mean Fn', 'lw2']],
                                           'title': title,
                                           'file': pwd_qualifier + '_GP_' + role,
                                           'gca_fill': [self.Xs.flat, mean - 2 * standard_deviation,
                                                        mean + 2 * standard_deviation],
                                           'xlabel': 'x',
                                           'ylabel': 'output, f(x)'
                                           }
            self.plot_graph(plot_posterior_distr_params)

        PH.printme(PH.p1, "!!!!!!!!!!Finished!!!!!!!!!" )

    def plot_posterior_predictions(self, pwd_qualifier, Xs, ys, mean, standard_deviation):

        pattern = re.compile("R[0-9]+_+")
        match = pattern.search(pwd_qualifier)
        span = match.span()
        file_name = pwd_qualifier[span[1] - (span[1] - span[0]):]

        plot_posterior_distr_params = {'plotnum': 'GP_Posterior_Distr_'+file_name,
                                       # 'axis': [0, 1, 0, 1],
                                       'axis': [-0.1, 1.1, -3, 3],
                                       'plotvalues': [[self.X, self.y, 'r+', 'ms20'], [Xs, ys, 'b-', 'label=True Fn'],
                                                      [self.Xs, mean, 'g--','label=Mean Fn','lw2']],
                                       'file': pwd_qualifier,
                                       'gca_fill': [self.Xs.flat, mean - 2 * standard_deviation,
                                                    mean + 2 * standard_deviation],
                                       'title': "MKL",
                                       'xlabel': 'x',
                                       'ylabel': 'output, f(x)'
                                       }

        self.plot_graph(plot_posterior_distr_params)


if __name__ == "__main__":

    PH(os.getcwd())
    timenow = datetime.datetime.now()
    PH.printme(PH.p1, "\nStart time: ", timenow.strftime("%H%M%S_%d%m%Y"))
    stamp = timenow.strftime("%H%M%S_%d%m%Y")

    kernel_type = 'SE'
    char_len_scale = 0.2
    number_of_test_datapoints = 500
    noise = 0.0
    random_seed = 500
    signal_variance = 1
    degree = 2

    # Linear Sin Function
    linspacexmin = 0
    linspacexmax = 10
    linspaceymin = 0
    linspaceymax = 10

    number_of_dimensions = 1
    number_of_observed_samples = 30
    hyper_params_estimation = False
    weight_params_estimation = False
    degree_estimation = False
    number_of_restarts_likelihood = 100
    oned_bounds = [[linspacexmin, linspacexmax]]
    sphere_bounds = [[linspacexmin, linspacexmax], [linspacexmin, linspacexmax]]
    michalewicz2d_bounds = [[0, np.pi], [0, np.pi]]
    random_bounds = [[0, 1], [1, 2]]
    # bounds = sphere_bounds
    bounds = oned_bounds
    # bounds = random_bounds

    Xmin = linspacexmin
    Xmax = linspacexmax
    ymax = linspaceymax
    ymin = linspaceymin

    a = 0.14
    b = 0.1
    lengthscale_bounds = [[0.1, 1]]
    signal_variance_bounds = [0.1, 1]
    true_func_type = "custom"
    fun_helper_obj = FunctionHelper(true_func_type)
    len_weights = [0.1, 0.3, 0.2]
    len_weights_bounds = [[0.1, 5] for i in range(4)]

    # Commenting for regression data - Forestfire
    random_points = []
    X = []

    # Generate specified (number of observed samples) random numbers for each dimension
    for dim in np.arange(number_of_dimensions):
        random_data_point_each_dim = np.random.uniform(bounds[dim][0], bounds[dim][1],
                                                       number_of_observed_samples).reshape(1, number_of_observed_samples)
        random_points.append(random_data_point_each_dim)

    # Vertically stack the arrays obtained for each dimension in the form a matrix, so that it can be reshaped
    random_points = np.vstack(random_points)

    # Generated values are to be reshaped into input points in the form of x1=[x11, x12, x13].T
    for sample_num in np.arange(number_of_observed_samples):
        array = []
        for dim_count in np.arange(number_of_dimensions):
            array.append(random_points[dim_count, sample_num])
        X.append(array)
    X = np.vstack(X)

    # Linear Sin Function
    x_obs = np.linspace(linspacexmin, 3, 15)
    x_obs = np.append(x_obs, np.linspace(7, linspacexmax, 15))
    X = x_obs.reshape(-1, 1)


    y_arr = []
    for each_x in X:
        val = fun_helper_obj.get_true_func_value(each_x)
        y_arr.append(val)

    y = np.vstack(y_arr)
    # y =  sinc_function(X)

    X = np.divide((X - Xmin), (Xmax - Xmin))
    y = (y - ymin) / (ymax - ymin)

    random_points = []
    Xs = []

    # Generate specified (number of unseen data points) random numbers for each dimension
    for dim in np.arange(number_of_dimensions):
        random_data_point_each_dim = np.linspace(bounds[dim][0], bounds[dim][1],
                                                 number_of_test_datapoints).reshape(1,number_of_test_datapoints)
        random_points.append(random_data_point_each_dim)
    random_points = np.vstack(random_points)

    # Generated values are to be reshaped into input points in the form of x1=[x11, x12, x13].T
    for sample_num in np.arange(number_of_test_datapoints):
        array = []
        for dim_count in np.arange(number_of_dimensions):
            array.append(random_points[dim_count, sample_num])
        Xs.append(array)
    Xs = np.vstack(Xs)

    ys_arr = []
    for each_xs in Xs:
        val_xs = fun_helper_obj.get_true_func_value(each_xs)
        ys_arr.append(val_xs)

    ys = np.vstack(ys_arr)
    # ys = sinc_function(Xs)

    Xs = np.divide((Xs - Xmin), (Xmax - Xmin))
    ys = (ys - ymin) / (ymax - ymin)

    gaussianObject = GaussianProcessRegressor(str(stamp), kernel_type, number_of_test_datapoints, noise, random_seed, linspacexmin,
                                     linspacexmax, linspaceymin, linspaceymax, signal_variance,
                                     number_of_dimensions, number_of_observed_samples, X, y, hyper_params_estimation,
                                     number_of_restarts_likelihood, lengthscale_bounds, signal_variance_bounds, fun_helper_obj, Xmin,
                                     Xmax, ymin, ymax, Xs, ys, char_len_scale, len_weights, len_weights_bounds, weight_params_estimation,
                                     degree_estimation, degree)


    count = 1

    PH.printme(PH.p1, "kernel_type: ", kernel_type, "\tnumber_of_test_datapoints: ",  number_of_test_datapoints, "\tnoise:", noise,  "\trandom_seed:",
          random_seed,  "\nlinspacexmin:", linspacexmin,"\tlinspacexmax:", linspacexmax, "\tlinspaceymin:",  linspaceymin,
          "\tlinspaceymax:",    linspaceymax, "\nsignal_variance:",  signal_variance, "\tnumber_of_dimensions:", number_of_dimensions,
          "\tnumber_of_observed_samples:", number_of_observed_samples,  "\nhyper_params_estimation:",  hyper_params_estimation,
          "\tnumber_of_restarts_likelihood:", number_of_restarts_likelihood, "\tlengthscale_bounds:",
          lengthscale_bounds, "\tsignal_variance_bounds:",    signal_variance_bounds,   "\nXmin:", Xmin, "\tXmax:", Xmax, "\tymin:", ymin,
          "\tymax:", ymax, "\tchar_len_scale:", char_len_scale, "\tlen_weights:", len_weights, "\tlen_weights_bounds:",
          len_weights_bounds, "\tweight_params_estimation:", weight_params_estimation, "\nX:", X, "\ty:", y)

    kernel_types = ['SE', 'MATERN3', 'MKL']

    runs = 1

    for run in range(runs):
        for kernel in kernel_types:
            PH.printme(PH.p1, "\n\nKernel: ", kernel)
            if kernel == 'MKL':
                gaussianObject.weight_params_estimation = True
                gaussianObject.hyper_params_estimation = False
                gaussianObject.degree_estimation = False
            elif kernel == "POLY":
                gaussianObject.weight_params_estimation = False
                gaussianObject.hyper_params_estimation = False
                gaussianObject.degree_estimation = True
            else:
                gaussianObject.hyper_params_estimation = True
            gaussianObject.runGaussian(count, kernel, None, None)
        gaussianObject.weight_params_estimation = False
        gaussianObject.hyper_params_estimation = False
        gaussianObject.degree_estimation = False
        gaussianObject.hyper_params_estimation = False

    timenow = datetime.datetime.now()
    PH.printme(PH.p1, "\nEnd time: ", timenow.strftime("%H%M%S_%d%m%Y"))
    plt.show()

