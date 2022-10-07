import numpy as np

from GP_Regressor_Wrapper import GPRegressorWrapper
from HelperUtility.PrintHelper import PrintHelper as PH
from Acquisition_Function import AcquisitionFunction

class HumanExpertModel:

    def __init__(self, human_expert_input_method):
        self.human_expert_input_method = human_expert_input_method

    def construct_human_expert_model(self, run_count, pwd_qualifier, number_of_observations_groundtruth, function_type, gp_wrapper_obj,
                                     number_of_random_observations_humanexpert, noisy_suggestions, estimated_kernel, controlled_obs):

        PH.printme(PH.p1, "Constructing Kernel for ground truth....")
        gp_groundtruth = gp_wrapper_obj.construct_gp_object(pwd_qualifier, "GroundTruth", number_of_observations_groundtruth,
                                                            function_type, None, estimated_kernel, controlled_obs)
        gp_groundtruth.runGaussian(pwd_qualifier + "R" + str(run_count + 1), "GT", True)
        PH.printme(PH.p1, "Ground truth kernel construction complete")

        PH.printme(PH.p1, "Construct GP object for Expert")
        gp_humanexpert = gp_wrapper_obj.construct_gp_object(pwd_qualifier, "HumanExpert",
                                                            number_of_random_observations_humanexpert, function_type, None,
                                                            estimated_kernel, controlled_obs)

        gp_humanexpert.initial_random_observations = {"observations_X": gp_humanexpert.X, "observations_y": gp_humanexpert.y,
                                                      "observations_y_orig": gp_humanexpert.y_orig}

        # Linear kernel for Human expert
        PH.printme(PH.p1, "Human Expert with info from Ground truth kernel: Var:", gp_groundtruth.signal_variance, "   Lengthscale:",
                   gp_groundtruth.char_len_scale)

        if noisy_suggestions:
            PH.printme(PH.p1, "Adding noise to the Human expert model....")
            gp_humanexpert.len_weights = np.array([])
            for i in range(len(gp_groundtruth.len_weights)):
                if gp_groundtruth.len_weights[i] == 0:
                    value = gp_groundtruth.len_weights[i] + 0.1
                elif gp_groundtruth.len_weights[i] == 1:
                    value = gp_groundtruth.len_weights[i] - 0.1
                else:
                    value = np.random.normal(gp_groundtruth.len_weights[i], 0.05)
                gp_humanexpert.len_weights = np.append(gp_humanexpert.len_weights, value)
            PH.printme(PH.p1, "After adding noise to the Human Expert model", gp_humanexpert.len_weights)

        else:

            if gp_humanexpert.kernel_type == "MKL":
                gp_humanexpert.len_weights = gp_groundtruth.len_weights
            else:
                gp_humanexpert.char_len_scale = gp_groundtruth.char_len_scale

        gp_humanexpert.signal_variance = gp_groundtruth.signal_variance

        gp_humanexpert.runGaussian(pwd_qualifier + "R" + str(run_count + 1), "HE_Initial", True)
        return gp_humanexpert

    def obtain_human_expert_suggestions(self, suggestion_count, file_identifier, gp_humanexpert, acq_func_obj, noisy_suggestions,
                                        plot_iterations):

        PH.printme(PH.p1, "Compute Human Expert Suggestion: ", suggestion_count)
        # PH.printme(PH.p1, "Weights from Ground truth: ", gp_humanexpert.len_weights)

        if self.human_expert_input_method != "suggestion_correction":
            PH.printme(PH.p1, "Calculating the worst suggestion .... ")
            xnew_worst, acq_func_values_worst = acq_func_obj.min_acq_func("HumanExpert", noisy_suggestions, gp_humanexpert,
                                                                          suggestion_count)
            xnew_orig_worst = np.multiply(xnew_worst.T, (gp_humanexpert.Xmax - gp_humanexpert.Xmin)) + gp_humanexpert.Xmin

            PH.printme(PH.p1, "This is the worst value found after minimising Acq. ", "\tXworst: ", xnew_worst, "\tOriginal:",
                       xnew_orig_worst)
        else:
            xnew_worst = None

        PH.printme(PH.p1, "Calculating the best suggestion .... ")
        xnew_best, acq_func_values_best = acq_func_obj.max_acq_func("HumanExpert", noisy_suggestions, gp_humanexpert, suggestion_count)
        xnew_orig_best = np.multiply(xnew_best.T, (gp_humanexpert.Xmax - gp_humanexpert.Xmin)) + gp_humanexpert.Xmin

        if gp_humanexpert.number_of_dimensions == 1 and plot_iterations != 0 and suggestion_count % plot_iterations == 0:
            plot_axes = [0, 1, acq_func_values_best.min() * 0.7, acq_func_values_best.max() * 2]
            acq_func_obj.plot_acquisition_function(file_identifier + "_Test_Best_acq_" + str(suggestion_count), gp_humanexpert.Xs,
                                               acq_func_values_best, plot_axes)

        PH.printme(PH.p1, "This is the best value found after maximising Acq. ", "\tXBest: ", xnew_best, "\tOriginal:", xnew_orig_best)

        ynew_orig_best = gp_humanexpert.fun_helper_obj.get_true_func_value(xnew_orig_best)
        y_temp = gp_humanexpert.refit_std_y(ynew_orig_best)
        ynew_best = y_temp[-1]

        PH.printme(PH.p1, "(", xnew_best, ynew_best, ") is the new best value suggested..    Original: ",
                   (xnew_orig_best, ynew_orig_best))

        # Add the new observation point to the existing set of observed samples along with its true value
        X = gp_humanexpert.X
        X = np.append(X, [xnew_best], axis=0)
        gp_humanexpert.X = X

        PH.printme(PH.p1, "\n\n@@@@@@@@@@ Best: ", xnew_best, "   Worst:", xnew_worst, "\n\n")


        # Plotting the posteriors
        if gp_humanexpert.number_of_dimensions == 1 and plot_iterations != 0 and suggestion_count % plot_iterations == 0:
            with np.errstate(invalid='ignore'):
                mean, diag_variance, f_prior, f_post = gp_humanexpert.gaussian_predict(gp_humanexpert.Xs)
                standard_deviation = np.sqrt(diag_variance)
            gp_humanexpert.plot_posterior_predictions(file_identifier + "_Suggestion" + "_" + str(
                suggestion_count), gp_humanexpert.Xs, gp_humanexpert.ys, mean, standard_deviation)

        return xnew_best, xnew_worst
