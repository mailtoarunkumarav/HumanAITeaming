import numpy as np
import scipy.optimize as opt
from GP_Regressor_Wrapper import GPRegressorWrapper
from HelperUtility.PrintHelper import PrintHelper as PH
from Acquisition_Function import AcquisitionFunction
from scipy.optimize import NonlinearConstraint, Bounds
from scipy.misc import derivative


import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
import nlopt

class AIModel:

    def __init__(self, epsilon_distance, minimiser_restarts, lambda_reg, lambda_mul, llk_threshold, global_opt_method,
                 last_suggestion_opt):
        self.epsilon_distance = epsilon_distance
        self.number_minimiser_restarts = minimiser_restarts
        self.lambda_reg = lambda_reg
        self.lambda_mul = lambda_mul
        self.min_acq_difference = None
        self.max_acq_difference = None
        self.min_llk = None
        self.max_llk = None
        self.llk_threshold = llk_threshold
        self.global_opt_method = global_opt_method
        self.last_suggestion_opt = last_suggestion_opt

    def obtain_ai_model_suggestion(self, plot_files_identifier, gp_aimodel, acq_func_obj, noisy_suggestions, ai_suggestion_count,
                                   plot_iterations):

        if gp_aimodel.he_suggestions is None:
            self.kernel_hyperparameter_tuning(gp_aimodel)

        elif gp_aimodel.he_suggestions is not None:
            PH.printme(PH.p1, "\nOptimising kernel with the suggestions provided by human expert...")
            self.kernel_optimisation_with_ai_suggestions(plot_files_identifier, gp_aimodel, acq_func_obj, noisy_suggestions,
                                                     ai_suggestion_count)
            PH.printme(PH.p1, "******* Kernel Optimisation Complete *******")

            debugging_bool = True
            if debugging_bool:
                derivative_list = []
                for i in range(len(gp_aimodel.he_suggestions["x_suggestions_best"])):
                    current_index = (gp_aimodel.number_of_observed_samples + gp_aimodel.HE_input_iterations[i] - 1)

                    data_conditioned_on_current_he_suggestion_X = gp_aimodel.X[0:current_index]
                    data_conditioned_on_current_he_suggestion_y = gp_aimodel.y[0:current_index]
                    exp_suggestion = gp_aimodel.he_suggestions["x_suggestions_best"][i]

                    der = derivative(lambda xx: acq_func_obj.data_conditioned_upper_confidence_bound_util("ai", noisy_suggestions,
                                xx, data_conditioned_on_current_he_suggestion_X, data_conditioned_on_current_he_suggestion_y,
                                gp_aimodel, ai_suggestion_count), exp_suggestion)
                    derivative_list.append(der)
                PH.printme(PH.p1, "\n\nFinal list of derivatives: ", derivative_list[-1])
                temp = np.append(gp_aimodel.char_len_scale, gp_aimodel.signal_variance)
                PH.printme(PH.p1, "Log likeli:", gp_aimodel.optimize_log_marginal_likelihood_l(temp))


        if gp_aimodel.kernel_type == 'MKL':
            PH.printme(PH.p1, "Final Optimisation : Opt weights: ", gp_aimodel.len_weights, "  Signal variance: ",
                       gp_aimodel.signal_variance)
        elif gp_aimodel.kernel_type == 'SE':
            PH.printme(PH.p1, "Final Optimisation : Char len scale: ", gp_aimodel.char_len_scale, "  Signal variance: ",
                       gp_aimodel.signal_variance)
            gp_aimodel.lengthscale_list.append(gp_aimodel.char_len_scale)
            PH.printme(PH.p1, gp_aimodel.lengthscale_list )

        gp_aimodel.runGaussian(plot_files_identifier + "Test_AI_Post" + str(ai_suggestion_count), "AIModel", True)
        xnew, acq_func_values = acq_func_obj.max_acq_func("ai", noisy_suggestions, gp_aimodel, ai_suggestion_count)

        # uncomment to  plot Acq functions
        if gp_aimodel.number_of_dimensions == 1 and plot_iterations != 0 and ai_suggestion_count % plot_iterations == 0:
            plot_axes = [0, 1, acq_func_values.min() * 0.7, acq_func_values.max() * 2]
            # print(acq_func_values)
            acq_func_obj.plot_acquisition_function(plot_files_identifier + "acq_" + str(ai_suggestion_count), gp_aimodel.Xs,
                                                   acq_func_values, plot_axes)

        PH.printme(PH.p1, "Best value for acq function is found at ", xnew)
        return xnew


    def kernel_hyperparameter_tuning(self, gp_aimodel):

        x_max_value = None
        log_like_max = -1 * float("inf")

        if gp_aimodel.kernel_type == "SE":

            # Data structure to create the starting points for the scipy.minimize method
            random_points = []
            starting_points = []

            # Depending on the number of dimensions and bounds, generate random multi-starting points to find maxima
            for dim in np.arange(gp_aimodel.number_of_dimensions):
                random_data_point_each_dim = np.random.uniform(gp_aimodel.lengthscale_bounds[dim][0],
                                                               gp_aimodel.lengthscale_bounds[dim][1],
                                                               gp_aimodel.number_of_restarts_likelihood). \
                    reshape(1, self.number_minimiser_restarts)
                random_points.append(random_data_point_each_dim)

            # Vertically stack the arrays of randomly generated starting points as a matrix
            random_points = np.vstack(random_points)

            # Reformat the generated random starting points in the form [x1 x2].T for the specified number of restarts
            for sample_num in np.arange(self.number_minimiser_restarts):
                array = []
                for dim_count in np.arange(gp_aimodel.number_of_dimensions):
                    array.append(random_points[dim_count, sample_num])
                starting_points.append(array)
            starting_points = np.vstack(starting_points)

            variance_start_points = np.random.uniform(gp_aimodel.signal_variance_bounds[0],
                                                      gp_aimodel.signal_variance_bounds[1],
                                                      gp_aimodel.number_of_restarts_likelihood)

            total_bounds = gp_aimodel.lengthscale_bounds.copy()
            total_bounds.append(gp_aimodel.signal_variance_bounds)

            for ind in np.arange(self.number_minimiser_restarts):

                init_len_scale = starting_points[ind]
                init_var = variance_start_points[ind]

                init_points = np.append(init_len_scale, init_var)
                maxima = opt.minimize(lambda x: -gp_aimodel.optimize_log_marginal_likelihood_l(x),
                                      init_points,
                                      method='L-BFGS-B',
                                      tol=0.01,
                                      options={'maxfun': 100, 'maxiter': 100},
                                      bounds=total_bounds)

                len_scale_temp = maxima['x'][:gp_aimodel.number_of_dimensions]
                variance_temp = maxima['x'][len(maxima['x']) - 1]
                params = np.append(len_scale_temp, variance_temp)
                log_likelihood = gp_aimodel.optimize_log_marginal_likelihood_l(params)

                if (log_likelihood > log_like_max):
                    PH.printme(PH.p1, "New maximum log likelihood ", log_likelihood, " found for l= ",
                               maxima['x'][: gp_aimodel.number_of_dimensions], " var:", maxima['x'][len(maxima['x']) - 1])

                    x_max_value = maxima
                    log_like_max = log_likelihood

            gp_aimodel.char_len_scale = x_max_value['x'][:gp_aimodel.number_of_dimensions]
            gp_aimodel.signal_variance = x_max_value['x'][len(maxima['x']) - 1]

            stage_one_best_kernel = x_max_value['x'][:gp_aimodel.number_of_dimensions]

            PH.printme(PH.p1, "Opt Len Scale: ", gp_aimodel.char_len_scale, "  Signal variance: ", gp_aimodel.signal_variance,
                       "   Maximum Liklihood:", log_like_max)

        elif gp_aimodel.kernel_type == "MKL":

            best_solutions = np.array([])
            start_points_list = []

            random_points_a = []
            random_points_b = []
            random_points_c = []
            random_points_d = []
            random_points_e = []
            random_points_f = []

            # Data structure to create the starting points for the scipy.minimize method
            random_data_point_each_dim = np.random.uniform(gp_aimodel.len_weights_bounds[0][0], gp_aimodel.len_weights_bounds[0][1],
                                                           self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
            random_points_a.append(random_data_point_each_dim)

            random_data_point_each_dim = np.random.uniform(gp_aimodel.len_weights_bounds[1][0], gp_aimodel.len_weights_bounds[1][1],
                                                           self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
            random_points_b.append(random_data_point_each_dim)

            random_data_point_each_dim = np.random.uniform(gp_aimodel.len_weights_bounds[2][0], gp_aimodel.len_weights_bounds[2][1],
                                                           self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
            random_points_c.append(random_data_point_each_dim)

            random_data_point_each_dim = np.random.uniform(gp_aimodel.len_weights_bounds[3][0], gp_aimodel.len_weights_bounds[3][1],
                                                           self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
            random_points_d.append(random_data_point_each_dim)

            random_data_point_each_dim = np.random.uniform(gp_aimodel.len_weights_bounds[4][0], gp_aimodel.len_weights_bounds[4][1],
                                                           self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
            random_points_e.append(random_data_point_each_dim)

            random_data_point_each_dim = np.random.uniform(gp_aimodel.len_weights_bounds[4][0], gp_aimodel.len_weights_bounds[4][1],
                                                           self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
            random_points_f.append(random_data_point_each_dim)

            variance_start_points = np.random.uniform(gp_aimodel.signal_variance_bounds[0], gp_aimodel.signal_variance_bounds[1],
                                                      self.number_minimiser_restarts)

            # Vertically stack the arrays of randomly generated starting points as a matrix
            random_points_a = np.vstack(random_points_a)
            random_points_b = np.vstack(random_points_b)
            random_points_c = np.vstack(random_points_c)
            random_points_d = np.vstack(random_points_d)
            random_points_e = np.vstack(random_points_e)
            random_points_f = np.vstack(random_points_f)

            for ind in np.arange(self.number_minimiser_restarts):

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
                total_bounds = gp_aimodel.len_weights_bounds.copy()
                # total_bounds.append(gp_aimodel.bounds)
                total_bounds.append(gp_aimodel.signal_variance_bounds)

                maxima = opt.minimize(lambda x: -gp_aimodel.optimize_log_marginal_likelihood_weight_params(x),
                                      tot_init_points,
                                      method='L-BFGS-B',
                                      tol=0.01,
                                      options={'maxfun': 200, 'maxiter': 40},
                                      bounds=total_bounds)

                params = maxima['x']
                log_likelihood = gp_aimodel.optimize_log_marginal_likelihood_weight_params(params)
                if log_likelihood > log_like_max:
                    PH.printme(PH.p1, "New maximum log likelihood ", log_likelihood, " found for params ", params)
                    x_max_value = maxima['x']
                    log_like_max = log_likelihood

                start_points_list.append(tot_init_points)
                best_solutions = np.append(best_solutions, np.array(log_likelihood))

            stage_one_best_kernel = x_max_value[0:(len(maxima['x']) - 1)]

            gp_aimodel.len_weights = x_max_value[0:(len(maxima['x']) - 1)]
            gp_aimodel.signal_variance = x_max_value[len(maxima['x']) - 1]

            PH.printme(PH.p1, "Opt weights: ", gp_aimodel.len_weights, "  Signal variance: ", gp_aimodel.signal_variance)
            PH.printme(PH.p1, "\n Maximum Liklihood:", log_like_max)



    def kernel_optimisation_with_ai_suggestions(self, plot_files_identifier, gp_aimodel, acq_func_obj, noisy_suggestions,
                                               ai_suggestion_count):

        x_max_value = None
        log_like_max = -1 * float("inf")

        if self.global_opt_method == "SLSQP":

            all_constraint_list = []

            for i in range(len(gp_aimodel.he_suggestions["x_suggestions_best"])):

                current_index = (gp_aimodel.number_of_observed_samples + gp_aimodel.HE_input_iterations[i] - 1)

                data_conditioned_on_current_he_suggestion_X = gp_aimodel.X[0:current_index]
                data_conditioned_on_current_he_suggestion_y = gp_aimodel.y[0:current_index]
                exp_suggestion = gp_aimodel.he_suggestions["x_suggestions_best"][i]
                ai_suggestion = gp_aimodel.ai_suggestions[current_index]

                nlc1 = NonlinearConstraint(lambda x: self.kernel_correction_with_acq(x, gp_aimodel, acq_func_obj, noisy_suggestions,
                                                                                     ai_suggestion_count, exp_suggestion, ai_suggestion,
                                                                                     data_conditioned_on_current_he_suggestion_X,
                                                                                     data_conditioned_on_current_he_suggestion_y), 0,
                                           np.inf)
                all_constraint_list.append(nlc1)

                # Derivative constraint
                nlc2 = NonlinearConstraint(lambda x: self.derivative_constraint(x, gp_aimodel, acq_func_obj, noisy_suggestions,
                                                                                     ai_suggestion_count, exp_suggestion, ai_suggestion,
                                                                                     data_conditioned_on_current_he_suggestion_X,
                                                                                     data_conditioned_on_current_he_suggestion_y), 0, 0)
                all_constraint_list.append(nlc2)

        if self.last_suggestion_opt:
            # # If only last suggestion used in the optimisation
            # constraint_list = all_constraint_list[-1]

            # Derivative constraints
            constraint_list = all_constraint_list[-2:]

        else:
            constraint_list = all_constraint_list


        # Data structure to create the starting points for the scipy.minimize method
        random_points = []
        starting_points = []

        # Depending on the number of dimensions and bounds, generate random multi-starting points to find maxima
        for dim in np.arange(gp_aimodel.number_of_dimensions):
            random_data_point_each_dim = np.random.uniform(gp_aimodel.lengthscale_bounds[dim][0],
                                                           gp_aimodel.lengthscale_bounds[dim][1],
                                                           gp_aimodel.number_of_restarts_likelihood). \
                reshape(1, self.number_minimiser_restarts)
            random_points.append(random_data_point_each_dim)

        # Vertically stack the arrays of randomly generated starting points as a matrix
        random_points = np.vstack(random_points)

        # Reformat the generated random starting points in the form [x1 x2].T for the specified number of restarts
        for sample_num in np.arange(self.number_minimiser_restarts):
            array = []
            for dim_count in np.arange(gp_aimodel.number_of_dimensions):
                array.append(random_points[dim_count, sample_num])
            starting_points.append(array)
        starting_points = np.vstack(starting_points)

        variance_start_points = np.random.uniform(gp_aimodel.signal_variance_bounds[0],
                                                  gp_aimodel.signal_variance_bounds[1],
                                                  gp_aimodel.number_of_restarts_likelihood)

        total_bounds = gp_aimodel.lengthscale_bounds.copy()
        total_bounds.append(gp_aimodel.signal_variance_bounds)

        for ind in np.arange(self.number_minimiser_restarts):

            init_len_scale = starting_points[ind]
            init_var = variance_start_points[ind]

            init_points = np.append(init_len_scale, init_var)

            if self.global_opt_method == "SLSQP":
                maxima = opt.minimize(lambda x: -gp_aimodel.optimize_log_marginal_likelihood_l(x), init_points, method='SLSQP',
                                      constraints=constraint_list, bounds=total_bounds)

            len_scale_temp = maxima['x'][:gp_aimodel.number_of_dimensions]
            variance_temp = maxima['x'][len(maxima['x']) - 1]
            params = np.append(len_scale_temp, variance_temp)
            log_likelihood = gp_aimodel.optimize_log_marginal_likelihood_l(params)

            if (log_likelihood > log_like_max):
                PH.printme(PH.p1, "New maximum log likelihood ", log_likelihood, " found for l= ",
                           maxima['x'][: gp_aimodel.number_of_dimensions], " var:", maxima['x'][len(maxima['x']) - 1])

                x_max_value = maxima
                log_like_max = log_likelihood

        gp_aimodel.char_len_scale = x_max_value['x'][:gp_aimodel.number_of_dimensions]
        gp_aimodel.signal_variance = x_max_value['x'][len(maxima['x']) - 1]

        PH.printme(PH.p1, "Opt Len Scale: ", gp_aimodel.char_len_scale, "  Signal variance: ", gp_aimodel.signal_variance,
                   "   Maximum Liklihood:", log_like_max)


    def kernel_correction_with_acq(self, x, gp_aimodel, acq_func_obj, noisy_suggestions, ai_suggestion_count, exp_suggestion, ai_suggestion,
                                   data_conditioned_on_current_he_suggestion_X, data_conditioned_on_current_he_suggestion_y):

        gp_aimodel.char_len_scale = x[:gp_aimodel.number_of_dimensions]
        gp_aimodel.signal_variance = x[len(x) - 1]

        if acq_func_obj.acq_type == "ucb":
            acq_value_he = acq_func_obj.data_conditioned_upper_confidence_bound_util("ai", noisy_suggestions, exp_suggestion,
                                                                                 data_conditioned_on_current_he_suggestion_X,
                                                                                 data_conditioned_on_current_he_suggestion_y,
                                                                                 gp_aimodel, ai_suggestion_count)

            acq_value_ai = acq_func_obj.data_conditioned_upper_confidence_bound_util("ai", noisy_suggestions,
                                                                                     ai_suggestion,
                                                                                     data_conditioned_on_current_he_suggestion_X,
                                                                                     data_conditioned_on_current_he_suggestion_y,
                                                                                     gp_aimodel, ai_suggestion_count)

            acq_difference = acq_value_he-acq_value_ai
        return acq_difference

    def derivative_constraint(self, x, gp_aimodel, acq_func_obj, noisy_suggestions, ai_suggestion_count, exp_suggestion, ai_suggestion,
                                   data_conditioned_on_current_he_suggestion_X, data_conditioned_on_current_he_suggestion_y):

        gp_aimodel.char_len_scale = x[:gp_aimodel.number_of_dimensions]
        gp_aimodel.signal_variance = x[len(x) - 1]

        if acq_func_obj.acq_type == "ucb":
            derivative_value = derivative(lambda xx: acq_func_obj.data_conditioned_upper_confidence_bound_util("ai", noisy_suggestions,
                                                                                                     xx,
                                                                                 data_conditioned_on_current_he_suggestion_X,
                                                                                 data_conditioned_on_current_he_suggestion_y,
                                                                                 gp_aimodel, ai_suggestion_count), exp_suggestion)

        return derivative_value

    def obtain_twostg_aimodel_suggestions(self, plot_files_identifier, gp_aimodel, acq_func_obj, noisy_suggestions, ai_suggestion_count,
                                   plot_iterations):

        x_max_value = None
        log_like_max = -1 * float("inf")

        if gp_aimodel.kernel_type == "SE":

            # Data structure to create the starting points for the scipy.minimize method
            random_points = []
            starting_points = []

            # Depending on the number of dimensions and bounds, generate random multi-starting points to find maxima
            for dim in np.arange(gp_aimodel.number_of_dimensions):
                random_data_point_each_dim = np.random.uniform(gp_aimodel.lengthscale_bounds[dim][0],
                                                               gp_aimodel.lengthscale_bounds[dim][1],
                                                               gp_aimodel.number_of_restarts_likelihood). \
                    reshape(1, self.number_minimiser_restarts)
                random_points.append(random_data_point_each_dim)

            # Vertically stack the arrays of randomly generated starting points as a matrix
            random_points = np.vstack(random_points)

            # Reformat the generated random starting points in the form [x1 x2].T for the specified number of restarts
            for sample_num in np.arange(self.number_minimiser_restarts):
                array = []
                for dim_count in np.arange(gp_aimodel.number_of_dimensions):
                    array.append(random_points[dim_count, sample_num])
                starting_points.append(array)
            starting_points = np.vstack(starting_points)

            variance_start_points = np.random.uniform(gp_aimodel.signal_variance_bounds[0],
                                                      gp_aimodel.signal_variance_bounds[1],
                                                      gp_aimodel.number_of_restarts_likelihood)

            total_bounds = gp_aimodel.lengthscale_bounds.copy()
            total_bounds.append(gp_aimodel.signal_variance_bounds)

            for ind in np.arange(self.number_minimiser_restarts):

                init_len_scale = starting_points[ind]
                init_var = variance_start_points[ind]

                init_points = np.append(init_len_scale, init_var)
                maxima = opt.minimize(lambda x: -gp_aimodel.optimize_log_marginal_likelihood_l(x),
                                      init_points,
                                      method='L-BFGS-B',
                                      tol=0.01,
                                      options={'maxfun': 100, 'maxiter': 100},
                                      bounds=total_bounds)

                len_scale_temp = maxima['x'][:gp_aimodel.number_of_dimensions]
                variance_temp = maxima['x'][len(maxima['x']) - 1]
                params = np.append(len_scale_temp, variance_temp)
                log_likelihood = gp_aimodel.optimize_log_marginal_likelihood_l(params)

                if (log_likelihood > log_like_max):
                    PH.printme(PH.p1, "New maximum log likelihood ", log_likelihood, " found for l= ",
                               maxima['x'][: gp_aimodel.number_of_dimensions], " var:", maxima['x'][len(maxima['x']) - 1])

                    x_max_value = maxima
                    log_like_max = log_likelihood

            gp_aimodel.char_len_scale = x_max_value['x'][:gp_aimodel.number_of_dimensions]
            gp_aimodel.signal_variance = x_max_value['x'][len(maxima['x']) - 1]

            stage_one_best_kernel = x_max_value['x'][:gp_aimodel.number_of_dimensions]

            PH.printme(PH.p1, "******* Stage 1 Optimisation complete *******\nOpt Len Scale: ", gp_aimodel.char_len_scale,
                       "  Signal variance: ", gp_aimodel.signal_variance, "   Maximum Liklihood:", log_like_max,  "   Stage One "
                                                                                                    "Best kernel: ",stage_one_best_kernel)

        elif gp_aimodel.kernel_type == "MKL":

            best_solutions = np.array([])
            start_points_list = []

            random_points_a = []
            random_points_b = []
            random_points_c = []
            random_points_d = []
            random_points_e = []
            random_points_f = []

            # Data structure to create the starting points for the scipy.minimize method
            random_data_point_each_dim = np.random.uniform(gp_aimodel.len_weights_bounds[0][0], gp_aimodel.len_weights_bounds[0][1],
                                                           self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
            random_points_a.append(random_data_point_each_dim)

            random_data_point_each_dim = np.random.uniform(gp_aimodel.len_weights_bounds[1][0], gp_aimodel.len_weights_bounds[1][1],
                                                           self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
            random_points_b.append(random_data_point_each_dim)

            random_data_point_each_dim = np.random.uniform(gp_aimodel.len_weights_bounds[2][0], gp_aimodel.len_weights_bounds[2][1],
                                                           self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
            random_points_c.append(random_data_point_each_dim)

            random_data_point_each_dim = np.random.uniform(gp_aimodel.len_weights_bounds[3][0], gp_aimodel.len_weights_bounds[3][1],
                                                           self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
            random_points_d.append(random_data_point_each_dim)

            random_data_point_each_dim = np.random.uniform(gp_aimodel.len_weights_bounds[4][0], gp_aimodel.len_weights_bounds[4][1],
                                                           self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
            random_points_e.append(random_data_point_each_dim)

            random_data_point_each_dim = np.random.uniform(gp_aimodel.len_weights_bounds[4][0], gp_aimodel.len_weights_bounds[4][1],
                                                           self.number_minimiser_restarts).reshape(1, self.number_minimiser_restarts)
            random_points_f.append(random_data_point_each_dim)

            variance_start_points = np.random.uniform(gp_aimodel.signal_variance_bounds[0], gp_aimodel.signal_variance_bounds[1],
                                                      self.number_minimiser_restarts)

            # Vertically stack the arrays of randomly generated starting points as a matrix
            random_points_a = np.vstack(random_points_a)
            random_points_b = np.vstack(random_points_b)
            random_points_c = np.vstack(random_points_c)
            random_points_d = np.vstack(random_points_d)
            random_points_e = np.vstack(random_points_e)
            random_points_f = np.vstack(random_points_f)

            for ind in np.arange(self.number_minimiser_restarts):

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
                total_bounds = gp_aimodel.len_weights_bounds.copy()
                # total_bounds.append(gp_aimodel.bounds)
                total_bounds.append(gp_aimodel.signal_variance_bounds)

                maxima = opt.minimize(lambda x: -gp_aimodel.optimize_log_marginal_likelihood_weight_params(x),
                                      tot_init_points,
                                      method='L-BFGS-B',
                                      tol=0.01,
                                      options={'maxfun': 200, 'maxiter': 40},
                                      bounds=total_bounds)

                params = maxima['x']
                log_likelihood = gp_aimodel.optimize_log_marginal_likelihood_weight_params(params)
                if log_likelihood > log_like_max:
                    PH.printme(PH.p1, "New maximum log likelihood ", log_likelihood, " found for params ", params)
                    x_max_value = maxima['x']
                    log_like_max = log_likelihood

                start_points_list.append(tot_init_points)
                best_solutions = np.append(best_solutions, np.array(log_likelihood))

            stage_one_best_kernel = x_max_value[0:(len(maxima['x']) - 1)]

            gp_aimodel.len_weights = x_max_value[0:(len(maxima['x']) - 1)]
            gp_aimodel.signal_variance = x_max_value[len(maxima['x']) - 1]

            PH.printme(PH.p1, "******* Stage 1 Optimisation complete *******\nOpt weights: ", gp_aimodel.len_weights,
                       "  Signal variance: ", gp_aimodel.signal_variance, "\n Maximum Liklihood:", log_like_max, "   Stage One Best "
                                                                                                                 "kernel: ",
                       stage_one_best_kernel)

        if gp_aimodel.he_suggestions is not None:

            PH.printme(PH.p1, "\nStarting stage 2")
            self.distance_maximiser_for_likelihood(gp_aimodel, acq_func_obj, noisy_suggestions, ai_suggestion_count, stage_one_best_kernel,
                                                   None, log_like_max)

            PH.printme(PH.p1, "******* Stage 2 Optimisation complete *******")

        if gp_aimodel.kernel_type == 'MKL':
            PH.printme(PH.p1, "Final Optimisation : Opt weights: ", gp_aimodel.len_weights, "  Signal variance: ", gp_aimodel.signal_variance)
        elif gp_aimodel.kernel_type == 'SE':
            PH.printme(PH.p1, "Final Optimisation : Char len scale: ", gp_aimodel.char_len_scale, "  Signal variance: ",
                       gp_aimodel.signal_variance)
        PH.printme(PH.p1, "\n")
        gp_aimodel.runGaussian(plot_files_identifier + "Test_AI_Post" + str(ai_suggestion_count), "AIModel", True)
        xnew, acq_func_values = acq_func_obj.max_acq_func("ai", noisy_suggestions, gp_aimodel, ai_suggestion_count)

        # uncomment to  plot Acq functions
        if gp_aimodel.number_of_dimensions == 1 and plot_iterations != 0 and ai_suggestion_count % plot_iterations == 0:
            plot_axes = [0, 1, acq_func_values.min() * 0.7, acq_func_values.max() * 2]
            # print(acq_func_values)
            acq_func_obj.plot_acquisition_function(plot_files_identifier + "acq_" + str(ai_suggestion_count), gp_aimodel.Xs,
                                                   acq_func_values, plot_axes)

        PH.printme(PH.p1, "Best value for acq function is found at ", xnew)
        return xnew

    def distance_maximiser_for_likelihood(self, gp_aimodel, acq_func_obj, noisy_suggestions, ai_suggestion_count, stage_one_best_kernel,
                                       start_points_list, best_solutions):

        empty_gradient_info = None
        condition_num_max = 25000

        compromised_likelihood = self.llk_threshold * best_solutions
        PH.printme(PH.p1, "Start point: ", stage_one_best_kernel, "Best:", best_solutions[0], "comp: ",compromised_likelihood)


        # # SLSQP
        if self.global_opt_method == "SLSQP":
            if gp_aimodel.kernel_type == 'SE':
                nlc1 = NonlinearConstraint(lambda x: gp_aimodel.optimize_log_marginal_likelihood_l(x), compromised_likelihood, np.inf)
                total_bounds = [[0.05, 1] for i in range(gp_aimodel.number_of_dimensions)]

            elif gp_aimodel.kernel_type == 'MKL':
                nlc1 = NonlinearConstraint(lambda x: gp_aimodel.optimize_log_marginal_likelihood_weight_params_const(x),
                                           compromised_likelihood, np.inf)
                total_bounds = [[0.05, 1] for i in range(6)]
            # # Constraint on condition number
            nlc2 = NonlinearConstraint(lambda x: gp_aimodel.get_kernel_matrix_condition_number(x), 1, 25000)

            constraint_list = []
            constraint_list.append(nlc1)
            constraint_list.append(nlc2)

            maxima = opt.minimize(lambda x: -self.constrained_distance_maximiser(x, empty_gradient_info, gp_aimodel, acq_func_obj,
                                                                                 noisy_suggestions, ai_suggestion_count),
                                  stage_one_best_kernel, method='SLSQP',
                                  constraints=constraint_list,
                                  bounds=total_bounds
                                  )
            params = maxima['x']

        # # # # # Differential Evolution
        elif self.global_opt_method == "DE":
            PH.printme(PH.p1, "Differential Evolution Implementation")
            self.global_opt_method = "DE"

            # # NL Constraint on Log likelihood - MKL based
            if gp_aimodel.kernel_type == 'MKL':
                nlc1 = NonlinearConstraint(lambda x: gp_aimodel.optimize_log_marginal_likelihood_weight_params_const(x)[0],
                                       compromised_likelihood,
                                       np.inf, keep_feasible=True)
                bounds = Bounds([0, 0, 0, 0, 0], [1, 1, 1, 1, 1])

            # # # NL Constraint on log likelihood - SE based
            elif gp_aimodel.kernel_type == 'SE':
                nlc1 = NonlinearConstraint(lambda x: gp_aimodel.optimize_log_marginal_likelihood_l(x),
                                       compromised_likelihood,
                                       np.inf,  keep_feasible=False)

            # # Constraint on condition number
            nlc2 = NonlinearConstraint(lambda x: gp_aimodel.get_kernel_matrix_condition_number(x), 1,
                                       25000, keep_feasible=True)

            # # Constraint on positivity
            nlc3 = NonlinearConstraint(lambda x: x, 0,
                                       np.inf, keep_feasible=True)
            bounds = Bounds([0.01], [1])

            maxima = opt.differential_evolution(
                lambda x: -self.constrained_distance_maximiser(x, empty_gradient_info, gp_aimodel, acq_func_obj,
                                        noisy_suggestions, ai_suggestion_count), bounds, constraints=(nlc1, nlc2,nlc3),
                seed=1, disp=False,
                maxiter=100,
                x0=stage_one_best_kernel)
            params = maxima['x']

        # # NLOPT Implementation
        elif self.global_opt_method == "NLOPT":

            PH.printme(PH.p1, "NLOPT Implementation")
            self.global_opt_method = "NLOPT"
            condition_num_max = 25000
            if gp_aimodel.kernel_type == 'MKL':
                nlopt_obj = nlopt.opt(nlopt.GN_ORIG_DIRECT, 5)
                nlopt_obj.set_lower_bounds([0, 0, 0, 0, 0])
                nlopt_obj.set_upper_bounds([1, 1, 1, 1, 1])

            if gp_aimodel.kernel_type == 'SE':
                nlopt_obj = nlopt.opt(nlopt.GN_ORIG_DIRECT, 1)
                nlopt_obj.set_lower_bounds([0.05])
                nlopt_obj.set_upper_bounds([1])

            nlopt_obj.set_max_objective(lambda x, grad: self.constrained_distance_maximiser(x, grad, gp_aimodel, acq_func_obj,
                                                                                            noisy_suggestions, ai_suggestion_count))
            nlopt_obj.add_inequality_constraint(lambda x, grad: self.ineq_llk_constraint1(x, grad, gp_aimodel, compromised_likelihood),
                                                1e-8)
            nlopt_obj.add_inequality_constraint(lambda x, grad: self.ineq_llk_constraint2(x, grad, gp_aimodel, condition_num_max), 1e-8)
            nlopt_obj.set_xtol_rel(1e-8)
            maxima = nlopt_obj.optimize(stage_one_best_kernel)
            max_constrained_likelihood = nlopt_obj.last_optimum_value()
            PH.printme(PH.p1, "result code = ", nlopt_obj.last_optimize_result())
            params = maxima
            PH.printme(PH.p1, "Best Params: ", maxima, "Maximum Likelihood: ", max_constrained_likelihood)

        if gp_aimodel.kernel_type == 'MKL':
            gp_aimodel.len_weights = params
            PH.printme(PH.p1, "Condition Num: ", gp_aimodel.get_kernel_matrix_condition_number(params), "l",
                    gp_aimodel.optimize_log_marginal_likelihood_weight_params_const(params))

        elif gp_aimodel.kernel_type == 'SE':
            gp_aimodel.char_len_scale = params
            PH.printme(PH.p1, "Condition Num: ", gp_aimodel.get_kernel_matrix_condition_number(params), "l",
                       gp_aimodel.optimize_log_marginal_likelihood_l(params))

        distance = self.constrained_distance_maximiser(params, empty_gradient_info, gp_aimodel, acq_func_obj, noisy_suggestions,
                                                       ai_suggestion_count)
        PH.printme(PH.p1, "New constrained distance maximum for stage two ", distance, " found for params ", params)

    def ineq_llk_constraint1(self, x, grad, gp_aimodel, compromised_likelihood):

        if gp_aimodel.kernel_type == 'MKL':
            new_likelihood = gp_aimodel.optimize_log_marginal_likelihood_weight_params_const(x)[0]
        elif gp_aimodel.kernel_type == 'SE':
            new_likelihood = gp_aimodel.optimize_log_marginal_likelihood_l(x)
        constraint = compromised_likelihood - new_likelihood
        return constraint[0]

    def ineq_llk_constraint2(self, x, grad, gp_aimodel, condition_num_max):

        condition_num = gp_aimodel.get_kernel_matrix_condition_number(x)
        constraint = condition_num - condition_num_max
        return constraint


    def constrained_distance_maximiser(self, inputs, grad, gp_aimodel, acq_func_obj, noisy_suggestions, ai_suggestion_count):

        acq_difference_sum = 0

        for i in range(len(gp_aimodel.he_suggestions["x_suggestions_best"])):

            data_conditioned_on_current_he_suggestion_X = gp_aimodel.X[0:(gp_aimodel.number_of_observed_samples +
                                                                          gp_aimodel.HE_input_iterations[i] - 1)]
            data_conditioned_on_current_he_suggestion_y = gp_aimodel.y[0:(gp_aimodel.number_of_observed_samples +
                                                                          gp_aimodel.HE_input_iterations[i] - 1)]

            # Xs_random = np.random.uniform(0, 1, 10).reshape(10, 1)
            # random_acq_values = []

            if acq_func_obj.acq_type == "ucb":

                best_acq_value = acq_func_obj.data_conditioned_upper_confidence_bound_util("ai", noisy_suggestions,
                                                                                           gp_aimodel.he_suggestions[
                                                                                               "x_suggestions_best"][
                                                                                               i],
                                                                                           data_conditioned_on_current_he_suggestion_X,
                                                                                           data_conditioned_on_current_he_suggestion_y,
                                                                                           gp_aimodel,
                                                                                           ai_suggestion_count)


                worst_acq_value = acq_func_obj.data_conditioned_upper_confidence_bound_util("ai", noisy_suggestions,
                                                                                            gp_aimodel.he_suggestions[
                                                                                                "x_suggestions_worst"][
                                                                                                i],
                                                                                            data_conditioned_on_current_he_suggestion_X,
                                                                                            data_conditioned_on_current_he_suggestion_y,
                                                                                            gp_aimodel,
                                                                                            ai_suggestion_count)

                # for each_Xs in Xs_random:
                #     value = acq_func_obj.data_conditioned_upper_confidence_bound_util("ai", noisy_suggestions, each_Xs,
                #                                                                       data_conditioned_on_current_he_suggestion_X,
                #                                                                       data_conditioned_on_current_he_suggestion_y,
                #                                                                       gp_aimodel, ai_suggestion_count)
                #     random_acq_values.append(value)

            if acq_func_obj.acq_type == "ei":
                y_max = gp_aimodel.y.max()
                best_acq_value = acq_func_obj.data_conditioned_expected_improvement_util("ai", noisy_suggestions,
                                                                                         gp_aimodel.he_suggestions[
                                                                                             "x_suggestions_best"][i],
                                                                                         data_conditioned_on_current_he_suggestion_X,
                                                                                         data_conditioned_on_current_he_suggestion_y,
                                                                                         y_max, gp_aimodel)
                worst_acq_value = acq_func_obj.data_conditioned_expected_improvement_util("ai", noisy_suggestions,
                                                                                          gp_aimodel.he_suggestions[
                                                                                              "x_suggestions_worst"][i],
                                                                                          data_conditioned_on_current_he_suggestion_X,
                                                                                          data_conditioned_on_current_he_suggestion_y,
                                                                                          y_max, gp_aimodel)

                # for each_Xs in Xs_random:
                #     value = acq_func_obj.data_conditioned_expected_improvement_util("ai", noisy_suggestions, each_Xs,
                #                                                                     data_conditioned_on_current_he_suggestion_X,
                #                                                                     data_conditioned_on_current_he_suggestion_y,
                #                                                                     y_max, gp_aimodel)
                #     random_acq_values.append(value)


            acq_diff_value = best_acq_value - worst_acq_value
            # PH.printme(PH.p1, "Likeli:",constrained_likelihood,str(i)+" acq_diff:", acq_diff_value, "Best:", best_acq_value,
            #            " Worst:", worst_acq_value, "  params: ", params)
            acq_difference_sum += acq_diff_value

            if acq_diff_value > 100:
                PH.printme(PH.p1, "Acq difference value is greater than 100")

        if acq_difference_sum < self.min_acq_difference:
            self.min_acq_difference = acq_difference_sum
        if acq_difference_sum > self.max_acq_difference:
            self.max_acq_difference = acq_difference_sum

        # PH.printme(PH.p1, "Const. Llk:", constrained_likelihood, "    Dist:", acq_difference_sum, "   Weights:", inputs, "   Variance:",
        #            gp_aimodel.signal_variance)

        if self.global_opt_method == "NLOPT":
            acq_difference_sum = np.float64(acq_difference_sum[0])

        return acq_difference_sum

