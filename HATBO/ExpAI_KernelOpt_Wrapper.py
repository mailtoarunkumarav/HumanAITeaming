import matplotlib

matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
import numpy as np
import datetime
import sys
import os
import random
from matplotlib.ticker import MaxNLocator

from GP_Regressor_Wrapper import GPRegressorWrapper
from HelperUtility.PrintHelper import PrintHelper as PH
from Acquisition_Function import AcquisitionFunction
from HumanExpert_Model import HumanExpertModel
from Baseline_Model import BaselineModel
from AI_Model import AIModel
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import sys, getopt


sys.path.append("..")

# setting up the global parameters for plotting graphs i.e, graph size and suppress warning from multiple graphs
# being plotted
plt.rcParams["figure.figsize"] = (6, 6)
# plt.rcParams["font.size"] = 12
plt.rcParams['figure.max_open_warning'] = 0
# np.seterr(divide='ignore', invalid='ignore')

# To fix the random number genration, currently not able, so as to retain the random selection of points
random_seed = 200
np.random.seed(random_seed)


# Class for starting Bayesian Optimization with the specified parameters
class ExpAIKerOptWrapper:

    def kernel_opt_wrapper(self, pwd_qualifier, full_time_stamp, function_type, external_input, cmd_inputs):

        number_of_runs = 10
        number_of_restarts_acq = 100
        number_of_minimiser_restarts = 100

        # Epsilons is the value used during the maximization of PI and EI ACQ functions
        # Greater value like Epsilon = 10 is more of exploration and Epsilon = 0.0001 (exploitation)
        # Epsilon1 used in PI : 3
        epsilon1 = 3
        # epsilon2 used in EI : 4
        epsilon2 = 0.01
        # Kappa value to be used during the maximization of UCB ACQ function, but this is overriden in the case if
        # Kappa is calculated at each iteration as a function of the iteration and other parameters
        # kappa=10 is more of exploration and kappa = 0.1 is more of exploitation
        nu = 0.1

        # Number of observations for human expert and ground truth models
        # number_of_observations_groundtruth = 50
        number_of_observations_groundtruth = 50
        number_of_random_observations_humanexpert = 3

        # Total number of iterations
        # number_total_suggestions = 12
        # number_total_suggestions = 25
        number_total_suggestions = 5

        epsilon_distance = 0.6

        noisy_suggestions = False

        plot_iterations = 1

        # acquisition type
        # acq_fun = 'ei'
        # acq_fun = 'ucb'

        # acq_fun_list = ['ei', 'ucb']
        acq_fun_list = ['ucb']
        # acq_fun_list = ['ei']

        total_regret_ai = {}
        total_regret_baseline_mlbo = {}
        total_regret_baseline_somlbo = {}
        total_regret_baseline_bo_eo = {}
        lambda_reg = 0.7
        lambda_mul = 10

        llk_threshold = 1.0

        # sec_stg_opt = "DE"
        sec_stg_opt = "SLSQP"
        # sec_stg_opt = "NLOPT"

        estimated_kernel = "SE"
        # estimated_kernel = "MKL"

        human_expert_input_method = "suggestion_correction"
        # human_expert_input_method = "distance_maximisation"

        # simulated_human = False
        simulated_human = True

        controlled_obs = True
        # controlled_obs = False

        # last_suggestion = False
        last_suggestion = True

        PH.printme(PH.p1, "\n###################################################################",
                   "\nAcq. Functions:", acq_fun_list, "   Minimiser Restarts:",
                   number_of_minimiser_restarts, "\nRestarts for Acq:", number_of_restarts_acq, "  Eps1:",
                   epsilon1, "   eps2:", epsilon2, "   No_obs_GT:", number_of_observations_groundtruth, "   Random Obs:",
                   number_of_random_observations_humanexpert, "\n   Total Suggestions: ", number_total_suggestions, "    Eps Dist.:",
                   epsilon_distance, "Func Type:", function_type, "\nNoisy:", noisy_suggestions,
                   "   plot iterations:", plot_iterations, "   Lambda:", lambda_reg, "\n lambda Multiplier:", lambda_mul,
                   "    Threshold Value: ", llk_threshold, "    2nd Stage Optimiser:", sec_stg_opt, "    Estimatd Kernel:",
                   estimated_kernel,
                   "\nOptimisation method: SLSQP + Constraint on Condition Number ",
                   "\nSpecial Inputs: Simulated Human:", simulated_human,
                   "    Controlled Observations: ", controlled_obs,
                   "    Last suggestion in optimisation: ", last_suggestion,
                   "\n\n"
                   )
        timenow = datetime.datetime.now()
        PH.printme(PH.p1, "Generating results Start time: ", timenow.strftime("%H%M%S_%d%m%Y"))

        # Run Optimization for the specified number of runs
        for run_count in range(number_of_runs):

            gp_wrapper_obj = GPRegressorWrapper()
            human_expert_model_obj = HumanExpertModel(human_expert_input_method)

            gp_humanexpert = human_expert_model_obj.construct_human_expert_model(run_count, pwd_qualifier,
                                                                                 number_of_observations_groundtruth,
                                                                                 function_type, gp_wrapper_obj,
                                                                                 number_of_random_observations_humanexpert,
                                                                                 noisy_suggestions, estimated_kernel, controlled_obs)

            # # Generating kernel for human expert
            humanexpert_kernel = self.kernel_sampling("Human Expert Kernel", gp_humanexpert, False)

            initial_random_observations_X = gp_humanexpert.X
            initial_random_observations_y = gp_humanexpert.y
            initial_random_observations_y_orig = gp_humanexpert.y_orig

            HE_input_iterations = [4, 7, 10]

            number_of_humanexpert_suggestions = len(HE_input_iterations)

            PH.printme(PH.p1, number_of_humanexpert_suggestions, " Human Expert Input Iterations: ", HE_input_iterations)

            acq_func_obj = AcquisitionFunction(None, number_of_restarts_acq, nu, epsilon1, epsilon2)

            for acq_fun in acq_fun_list:

                observations_pool_X = initial_random_observations_X
                observations_pool_y = initial_random_observations_y
                observations_pool_y_orig = initial_random_observations_y_orig

                PH.printme(PH.p1, "\n\n########Generating results for Acquisition Function: ", acq_fun.upper(), "#############")
                plot_files_identifier = pwd_qualifier + "R" + str(run_count + 1) + "_" + acq_fun.upper()
                acq_func_obj.set_acq_func_type(acq_fun)

                he_suggestions_best = []
                he_suggestions_worst = []

                PH.printme(PH.p1, "Construct GP object for AI Model")
                gp_aimodel = gp_wrapper_obj.construct_gp_object(pwd_qualifier, "ai", number_of_random_observations_humanexpert,
                                                                function_type, gp_humanexpert.initial_random_observations,
                                                                estimated_kernel, controlled_obs)

                gp_aimodel.HE_input_iterations = HE_input_iterations
                gp_aimodel.he_suggestions = None
                gp_aimodel.ai_suggestions = initial_random_observations_X

                aimodel_obj = AIModel(epsilon_distance, number_of_minimiser_restarts, lambda_reg, lambda_mul, llk_threshold, sec_stg_opt,
                                      last_suggestion)

                PH.printme(PH.p1, "Construct GP object for baseline - Standard BO")
                gp_baseline_mlbo = gp_wrapper_obj.construct_gp_object(pwd_qualifier, "baseline", number_of_random_observations_humanexpert,
                                                                 function_type, gp_humanexpert.initial_random_observations,
                                                                 estimated_kernel, controlled_obs)

                gp_baseline_somlbo = gp_wrapper_obj.construct_gp_object(pwd_qualifier, "baseline",
                                                                        number_of_random_observations_humanexpert,
                                                                      function_type, gp_humanexpert.initial_random_observations,
                                                                      estimated_kernel, controlled_obs)

                gp_baseline_mlbo.runGaussian(plot_files_identifier, "Initial_Posterior_MLBO", True)
                gp_baseline_somlbo.runGaussian(plot_files_identifier, "Initial_Posterior_SOMLBO", True)
                PH.printme(PH.p1, "*************GP Constructions complete************")
                baseline_model_obj = BaselineModel()

                for suggestion_count in range(1, number_total_suggestions + 1):

                    aimodel_obj.max_acq_difference = -1 * float("inf")
                    aimodel_obj.max_llk = -1 * float("inf")
                    aimodel_obj.min_acq_difference = 1 * float("inf")
                    aimodel_obj.min_llk = 1 * float("inf")

                    if human_expert_model_obj.human_expert_input_method == "distance_maximisation":
                        if suggestion_count in HE_input_iterations:
                            PH.printme(PH.p1, "\n\n*******************\nStarting suggestion:", suggestion_count,
                                       " with Human Expert inputs...")
                            PH.printme(PH.p1, "Generating Human Expert Suggestions")
                            gp_humanexpert.gp_fit(observations_pool_X, observations_pool_y)
                            gp_humanexpert.refit_utils_std_ys(observations_pool_y_orig)

                            if simulated_human:
                                xnew_best, xnew_worst = human_expert_model_obj.obtain_human_expert_suggestions(suggestion_count,
                                        plot_files_identifier + "_HE", gp_humanexpert, acq_func_obj, noisy_suggestions, plot_iterations)
                            else:
                                PH.printme(PH.p1, "Taking inputs from human expert")
                                xnew_best = None
                                xnew_worst = None
                                expert_input = {}

                                def onclick_best(xbest_event):
                                    PH.printme(PH.p1, "Xbest:", xbest_event.xdata)
                                    expert_input['xbest'] = xbest_event.xdata

                                def onclick_worst(xworst_event):
                                    PH.printme(PH.p1, "Xworst:", xworst_event.xdata)
                                    expert_input['xworst'] = xworst_event.xdata

                                xbest_fig, xbest_ax = plt.subplots()
                                xbest_ax.title.set_text('Input Xbest point for suggestion ')
                                xbest_ax.set_xlim([0, 1])
                                xbest_ax.set_ylim([np.min(gp_aimodel.y)-2, np.max(gp_aimodel.y)+2])
                                xbest_ax.plot(gp_aimodel.X, gp_aimodel.y, "ro")
                                cid_xbest = xbest_fig.canvas.mpl_connect('button_press_event', onclick_best)

                                xworst_fig, xworst_ax = plt.subplots()
                                xworst_ax.title.set_text('Input Xworst point for suggestion ')
                                xworst_ax.set_xlim([0, 1])
                                xworst_ax.set_ylim([np.min(gp_aimodel.y)-2, np.max(gp_aimodel.y)+2])
                                xworst_ax.plot(gp_aimodel.X, gp_aimodel.y, "ro")
                                cid_xworst = xworst_fig.canvas.mpl_connect('button_press_event', onclick_worst)
                                plt.show()
                                xnew_best = expert_input['xbest']
                                xnew_worst = expert_input['xworst']

                                PH.printme(PH.p1, "Final inputs - Best:", xnew_best, " Worst:",xnew_worst)

                            # xnew_worst will return None if human_expert_input_method = suggestion_correction
                            he_suggestions_best.append(xnew_best)
                            he_suggestions_worst.append(xnew_worst)

                            gp_aimodel.he_suggestions = {"x_suggestions_best": he_suggestions_best,
                                                         "x_suggestions_worst": he_suggestions_worst}
                            PH.printme(PH.p1, "Aggregated Expert Suggestions: ", gp_aimodel.he_suggestions)

                        else:
                            PH.printme(PH.p1, "\n\nPredicting suggestion:" + str(suggestion_count) + " without Human Expert inputs")

                        PH.printme(PH.p1, "Human expert inputs via distance maximisation")
                        xnew_ai = aimodel_obj.obtain_twostg_aimodel_suggestions(plot_files_identifier + "_AI_", gp_aimodel, acq_func_obj,
                                                                                noisy_suggestions, suggestion_count, plot_iterations)
                        PH.printme(PH.p1, "\n\nDistance optimisation details:\nMax Diff:", aimodel_obj.max_acq_difference, "\tMin Diff:",
                                   aimodel_obj.min_acq_difference)
                        xnew = xnew_ai

                    elif human_expert_model_obj.human_expert_input_method == "suggestion_correction":
                        PH.printme(PH.p1, " \nPredicting suggestion:" + str(suggestion_count) + " for AI model")
                        xnew_ai = aimodel_obj.obtain_ai_model_suggestion(plot_files_identifier + "_AI_", gp_aimodel, acq_func_obj,
                                                                         noisy_suggestions, suggestion_count, plot_iterations)

                        gp_aimodel.ai_suggestions = np.append(gp_aimodel.ai_suggestions, [xnew_ai], axis=0)

                        if suggestion_count in HE_input_iterations:
                            PH.printme(PH.p1, "\nCorrecting suggestion:", suggestion_count," with Human Expert inputs...")
                            PH.printme(PH.p1, "Generating Human Expert Suggestions")
                            gp_humanexpert.gp_fit(observations_pool_X, observations_pool_y)
                            gp_humanexpert.refit_utils_std_ys(observations_pool_y_orig)

                            if simulated_human:
                                xnew_best, xnew_worst = human_expert_model_obj.obtain_human_expert_suggestions(suggestion_count,
                                        plot_files_identifier + "_HE", gp_humanexpert, acq_func_obj, noisy_suggestions, plot_iterations)
                            else:
                                plt.close('all')
                                PH.printme(PH.p1, "Taking inputs from human expert")
                                xnew_best = None
                                xnew_worst = None


                                # Figure Input
                                expert_input = {}

                                def onclick_best(xbest_event):
                                    PH.printme(PH.p1, "Xbest:", xbest_event.xdata)
                                    expert_input['xbest'] = xbest_event.xdata

                                def onclick_worst(xworst_event):
                                    PH.printme(PH.p1, "Xworst:", xworst_event.xdata)
                                    expert_input['xworst'] = xworst_event.xdata

                                xbest_fig, xbest_ax = plt.subplots()
                                xbest_ax.title.set_text('Input Xbest point for suggestion ')
                                xbest_ax.set_xlim([0, 1])
                                xbest_ax.set_ylim([np.min(gp_aimodel.y)-2, np.max(gp_aimodel.y)+2])
                                xbest_ax.plot(gp_aimodel.X, gp_aimodel.y, "ro")
                                cid_xbest = xbest_fig.canvas.mpl_connect('button_press_event', onclick_best)

                                xworst_fig, xworst_ax = plt.subplots()
                                xworst_ax.title.set_text('Input Xworst point for suggestion ')
                                xworst_ax.set_xlim([0, 1])
                                xworst_ax.set_ylim([np.min(gp_aimodel.y)-2, np.max(gp_aimodel.y)+2])
                                xworst_ax.plot(gp_aimodel.X, gp_aimodel.y, "ro")
                                cid_xworst = xworst_fig.canvas.mpl_connect('button_press_event', onclick_worst)
                                plt.show()
                                xnew_best = expert_input['xbest']
                                xnew_worst = expert_input['xworst']

                                PH.printme(PH.p1, "Final inputs - Best:", xnew_best, " Worst:",xnew_worst)
                                xnew_best = np.array([xnew_best])

                            # xnew_worst will return None if human_expert_input_method = suggestion_correction
                            he_suggestions_best.append(xnew_best)
                            he_suggestions_worst.append(xnew_worst)

                            gp_aimodel.he_suggestions = {"x_suggestions_best": he_suggestions_best,
                                                         "x_suggestions_worst": he_suggestions_worst}
                            PH.printme(PH.p1, "Aggregated Expert Suggestions: ", gp_aimodel.he_suggestions)

                            xnew = xnew_best
                            PH.printme(PH.p1, "Human expert suggestion added to the pool of observations")

                        else:
                            xnew = xnew_ai
                            PH.printme(PH.p1, "AI model suggestion added to the pool of observations")

                    xnew_orig = np.multiply(xnew.T, (gp_aimodel.Xmax - gp_aimodel.Xmin)) + gp_aimodel.Xmin

                    # Add the new observation point to the existing set of observed samples along with its true value
                    observations_pool_X = np.append(observations_pool_X, [xnew], axis=0)
                    ynew_orig = gp_aimodel.fun_helper_obj.get_true_func_value(xnew_orig)

                    # objective function noisy
                    if gp_aimodel.fun_helper_obj.true_func_type == "LIN1D" or gp_aimodel.fun_helper_obj.true_func_type == "LINSIN1D":
                        # or gp_aimodel.fun_helper_obj.true_func_type == "BEN1D":
                        ynew_orig = ynew_orig + np.random.normal(0, 0.1)

                    # Standardising Y
                    # ynew_ai = (ynew_ai_orig - gp_aimodel.ymin) / (gp_aimodel.ymax - gp_aimodel.ymin)
                    # observations_pool_y = np.append(observations_pool_y, [ynew_ai], axis=0)
                    observations_pool_y_orig = np.append(observations_pool_y_orig, [ynew_orig], axis=0)
                    observations_pool_y = (observations_pool_y_orig - np.mean(observations_pool_y_orig)) / np.std(observations_pool_y_orig)
                    gp_aimodel.gp_fit(observations_pool_X, observations_pool_y)
                    gp_aimodel.refit_utils_std_ys(observations_pool_y_orig)
                    ynew = observations_pool_y[-1]

                    PH.printme(PH.p1, "AI model: (", xnew, ynew, ") is the new best value added..    Original: ", (xnew_orig, ynew_orig))

                    if gp_aimodel.number_of_dimensions == 1 and plot_iterations != 0 and suggestion_count % plot_iterations == 0:
                        with np.errstate(invalid='ignore'):
                            mean_ai, diag_variance_ai, f_prior_ai, f_post_ai = gp_aimodel.gaussian_predict(gp_aimodel.Xs)
                            standard_deviation_ai = np.sqrt(diag_variance_ai)

                        gp_aimodel.plot_posterior_predictions(plot_files_identifier + "_AI_suggestion" + "_" + str(suggestion_count),
                                                              gp_aimodel.Xs, gp_aimodel.ys, mean_ai, standard_deviation_ai)

                    # # # Baseline model MLBO
                    PH.printme(PH.p1, "\nSuggestion for MLBO Baseline at iteration", suggestion_count)
                    xnew_baseline_mlbo, ynew_baseline_mlbo = baseline_model_obj.obtain_baseline_suggestion(suggestion_count,
                                                                                                           plot_files_identifier,
                                                                                                           gp_baseline_mlbo, acq_func_obj,
                                                                                                           noisy_suggestions,
                                                                                                           plot_iterations, "MLBO")
                    PH.printme(PH.p1, "MLBO Baseline: (", xnew_baseline_mlbo, ynew_baseline_mlbo, ") is the new value added for llK "
                                                                                                 "baseline")

                    # # # # Baseline SOMLBO
                    PH.printme(PH.p1, "\nSuggestion for SOMLBO Baseline at iteration", suggestion_count)
                    xnew_baseline_somlbo, ynew_baseline_somlbo = baseline_model_obj.obtain_baseline_suggestion(suggestion_count,
                                                                                                               plot_files_identifier,
                                                                                                               gp_baseline_somlbo,
                                                                                                               acq_func_obj,
                                                                                                               noisy_suggestions,
                                                                                                               plot_iterations, "SOMLBO")
                    PH.printme(PH.p1, "SOMLBO Baseline: (", xnew_baseline_somlbo, ynew_baseline_somlbo, ") is the new value added for llK "
                                                                                                        "baseline")
                    gp_baseline_somlbo.gp_fit(observations_pool_X, observations_pool_y)
                    gp_baseline_somlbo.refit_utils_std_ys(observations_pool_y_orig)

                    # # #L-inf Norm calculations
                    if gp_aimodel.kernel_type == 'MKL':
                        PH.printme(PH.p1, "\n\nCalculating L-infinity norm for the kernels obtained")
                        sampled_ai_kernel = self.kernel_sampling("AI Kernel", gp_aimodel, False)
                        sampled_baseline_kernel = self.kernel_sampling("Baseline Kernel - MLBO", gp_baseline_mlbo, False)

                        ai_kernel_diff = humanexpert_kernel.reshape(-1, 1) - sampled_ai_kernel.reshape(-1, 1)
                        ai_inf_norm = np.linalg.norm(ai_kernel_diff, ord=np.inf, axis=0)
                        ai_l2_norm = np.linalg.norm(ai_kernel_diff, ord=2, axis=0)
                        PH.printme(PH.p1, "AI Model: kernel difference: L-Inf: ", ai_inf_norm, "\tL2: ", ai_l2_norm)

                        base_kernel_diff = humanexpert_kernel.reshape(-1, 1) - sampled_baseline_kernel.reshape(-1, 1)
                        base_inf_norm = np.linalg.norm(base_kernel_diff, ord=np.inf, axis=0)
                        base_l2_norm = np.linalg.norm(base_kernel_diff, ord=2, axis=0)
                        PH.printme(PH.p1, "Base Model: kernel difference: L-Inf: ", base_inf_norm, "\tL2: ", base_l2_norm)

                        PH.printme(PH.p1, "\nPlotting kernel weights of GT, AI, Baseline Kernels")
                        PH.printme(PH.p1,
                                   # "HE:", gp_humanexpert.len_weights,
                                   "\n", "AI:", gp_aimodel.len_weights, "\nBL-MLBO:", gp_baseline_mlbo.len_weights)
                        self.plot_kernel_weights(pwd_qualifier, full_time_stamp, run_count, suggestion_count, gp_humanexpert.len_weights,
                                                 gp_aimodel.len_weights, gp_baseline_mlbo.len_weights)
                    plt.close('all')
                    # plt.show()

                gp_aimodel.runGaussian(pwd_qualifier + "R" + str(run_count + 1) + "_" + acq_fun.upper(), "AI_final", True)
                gp_baseline_mlbo.runGaussian(pwd_qualifier + "R" + str(run_count + 1) + "_" + acq_fun.upper(), "Base_final_MLBO", True)
                gp_baseline_somlbo.runGaussian(pwd_qualifier + "R" + str(run_count + 1) + "_" + acq_fun.upper(), "Base_final_SOMLBO", True)

                true_max = gp_humanexpert.fun_helper_obj.get_true_max()

                ai_regret = {}
                baseline_regret_mlbo = {}
                baseline_regret_somlbo = {}

                for i in range(number_of_random_observations_humanexpert + number_total_suggestions):

                    if acq_fun not in ai_regret or acq_fun not in baseline_regret_mlbo or acq_fun not in baseline_regret_somlbo:
                        ai_regret[acq_fun] = []
                        baseline_regret_mlbo[acq_fun] = []
                        baseline_regret_somlbo[acq_fun] = []

                    if i <= number_of_random_observations_humanexpert - 1:
                        ai_regret[acq_fun].append(true_max - np.max(gp_aimodel.y_orig[0:number_of_random_observations_humanexpert]))
                        baseline_regret_mlbo[acq_fun].append(true_max - np.max(gp_baseline_mlbo.y_orig[
                                                                          0:number_of_random_observations_humanexpert]))
                        baseline_regret_somlbo[acq_fun].append(true_max - np.max(gp_baseline_somlbo.y_orig[
                                                                          0:number_of_random_observations_humanexpert]))
                    else:
                        ai_regret[acq_fun].append(true_max - np.max(gp_aimodel.y_orig[0:i + 1]))
                        baseline_regret_mlbo[acq_fun].append(true_max - np.max(gp_baseline_mlbo.y_orig[0:i + 1]))
                        baseline_regret_somlbo[acq_fun].append(true_max - np.max(gp_baseline_somlbo.y_orig[0:i + 1]))

                if acq_fun not in total_regret_ai or acq_fun not in total_regret_baseline_mlbo or acq_fun not in \
                        total_regret_baseline_somlbo:
                    total_regret_ai[acq_fun] = []
                    total_regret_baseline_mlbo[acq_fun] = []
                    total_regret_baseline_somlbo[acq_fun] = []

                total_regret_ai[acq_fun].append(ai_regret[acq_fun])
                total_regret_baseline_mlbo[acq_fun].append(baseline_regret_mlbo[acq_fun])
                total_regret_baseline_somlbo[acq_fun].append(baseline_regret_somlbo[acq_fun])

            PH.printme(PH.p1, "\n###########\nTotal AI Regret:\n", total_regret_ai, "\nTotal Baseline Regret MLBO:\n",
                       total_regret_baseline_mlbo, "\nTotal Baseline Regret SOMLBO:\n", total_regret_baseline_somlbo,
                       "\n###################\n")
            PH.printme(PH.p1, "Iterations and Lenghtscales\nAI model:")
            PH.printme(PH.p1, repr(np.vstack(gp_aimodel.lengthscale_list)))
            PH.printme(PH.p1, "\nBaseline STD-BO: ")
            PH.printme(PH.p1, repr(np.vstack(gp_baseline_mlbo.lengthscale_list)))
            PH.printme(PH.p1, "\nBaseline STD-BO(MOD): ")
            PH.printme(PH.p1, repr(np.vstack(gp_baseline_somlbo.lengthscale_list)))
            PH.printme(PH.p1, "\n\n@@@@@@@@@@@@@@ Round ", str(run_count + 1) + " complete @@@@@@@@@@@@@@@@\n\n")
            plt.close('all')

        PH.printme(PH.p1, "Tot_AI:\n", total_regret_ai, "\n\nTot Base-MLBO:\n", total_regret_baseline_mlbo, "\n\nTot Base-SOMLBO:\n",
                   total_regret_baseline_somlbo)
        # # # Plotting regret
        plt.close('all')
        self.plot_regret(pwd_qualifier, full_time_stamp, acq_fun_list, total_regret_ai, total_regret_baseline_mlbo,
                         total_regret_baseline_somlbo, len(gp_aimodel.y))

        endtimenow = datetime.datetime.now()
        PH.printme(PH.p1, "\nEnd time: ", endtimenow.strftime("%H%M%S_%d%m%Y"))

    def plot_kernel_weights(self, pwd_qualifier, full_time_stamp, run_count, suggestion_count, groundtruth_kernel_weights,
                            ai_kernel_weights, baseline_kernel_weights):
        xaxis_kernel_type = np.arange(len(groundtruth_kernel_weights))

        fig_name = "R" + str(run_count + 1) + "_S" + str(suggestion_count) + "_Kernel_weights_" + full_time_stamp
        plt.figure(str(fig_name))
        plt.ylim(0, 2)

        for k_num in range(len(groundtruth_kernel_weights)):
            plt.bar(xaxis_kernel_type[k_num], groundtruth_kernel_weights[k_num], color='blue', width=0.1)
            plt.bar(xaxis_kernel_type[k_num] + 0.1, ai_kernel_weights[k_num], color='green', width=0.1, )
            plt.bar(xaxis_kernel_type[k_num] + 0.2, baseline_kernel_weights[k_num], color='red', width=0.1)

        plt.title("Kernel Weights of MKL")

        plt.xlabel('Kernel Type')
        plt.ylabel('Weights')

        plt.xticks(xaxis_kernel_type + 0.1, ['w_se', 'l_se', 'w_Lin', 'c_lin', 'w_poly', 'l_poly'])

        plt.legend(('GroundTruth', 'AI-Model', 'Baseline'))
        plt.savefig(pwd_qualifier + fig_name + '.pdf')

    def kernel_sampling(self, msg, gp_plotting, plot_bool):

        if gp_plotting.number_of_dimensions != 1:
            return None

        PH.printme(PH.p1, "plotting kernel for ", msg,
                   # " \twith kernel weights: ", gp_plotting.len_weights
                   )

        kernel_mat = np.zeros(shape=(200, 200))
        xbound = np.linspace(0, 5, 200).reshape(-1, 1)
        X1, X2 = np.meshgrid(xbound, xbound)
        for x_i in range(len(xbound)):
            for x_j in range(len(xbound)):
                num = gp_plotting.computekernel(np.array([xbound[x_i]]), np.array([xbound[x_j]]))
                kernel_mat[x_i][x_j] = num

        if function_type == "LIN1D":
            minimum = np.min(kernel_mat)
            maximum = np.max(kernel_mat)
            kernel_mat = np.divide(kernel_mat - minimum, maximum - minimum)

        if not plot_bool:
            return kernel_mat

        fig = plt.figure(msg)
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X1, X2, kernel_mat, rstride=1, cstride=1,
                               cmap='viridis', linewidth=1, antialiased=False)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%0.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=20)

    def plot_regret(self, pwd_name, full_time_stamp, acq_fun_list, total_regret_ai, total_regret_base_mlbo, total_regret_base_somlbo,
                    total_number_of_obs):

        iterations_axes_values = [i + 1 for i in np.arange(total_number_of_obs)]
        fig_name = 'Regret_' + full_time_stamp
        plt.figure(str(fig_name))
        plt.clf()
        ax = plt.subplot(111)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # # # AI model
        colors = ["#713326", "#22642E", "#0D28C3", "#EB0F0F"]
        count = 0
        for acq in acq_fun_list:
            regret_ai = np.vstack(total_regret_ai[acq])
            regret_mean_ai = np.mean(regret_ai, axis=0)
            regret_std_dev_ai = np.std(regret_ai, axis=0)
            regret_std_dev_ai = regret_std_dev_ai / np.sqrt(total_number_of_obs)
            PH.printme(PH.p1, "\nAI Regret Details\nTotal Regret:", acq.upper(), "\n", regret_ai, "\n\n", acq.upper(), " Regret Mean",
                       regret_mean_ai, "\n\n", acq.upper(), " Regret Deviation\n", regret_std_dev_ai)

            ax.plot(iterations_axes_values, regret_mean_ai, colors[count])
            # plt.errorbar(iterations_axes_values, ei_regret_mean, yerr= ei_regret_std_dev )
            plt.gca().fill_between(iterations_axes_values, regret_mean_ai + regret_std_dev_ai,
                                   regret_mean_ai - regret_std_dev_ai, color=colors[count], alpha=0.25, label="AI-" + acq.upper())

            count += 1

            # Baseline model-MLBO
            regret_base_mlbo = np.vstack(total_regret_base_mlbo[acq])
            regret_mean_base_mlbo = np.mean(regret_base_mlbo, axis=0)
            regret_std_dev_base_mlbo = np.std(regret_base_mlbo, axis=0)
            regret_std_dev_base_mlbo = regret_std_dev_base_mlbo / np.sqrt(total_number_of_obs)
            PH.printme(PH.p1, "\nBaseline Regret Details\nTotal Regret \n", regret_base_mlbo, "\n\nRegret Mean", regret_mean_base_mlbo,
                       "\n\nRegret Deviation\n", regret_std_dev_base_mlbo)

            ax.plot(iterations_axes_values, regret_mean_base_mlbo, colors[count])
            # plt.errorbar(iterations_axes_values, ei_regret_mean, yerr= ei_regret_std_dev )
            plt.gca().fill_between(iterations_axes_values, regret_mean_base_mlbo + regret_std_dev_base_mlbo,
                                   regret_mean_base_mlbo - regret_std_dev_base_mlbo, color=colors[count], alpha=0.25,
                                   label="STD-BO")

            count += 1

            # # Baseline model-SOMLBO
            regret_base_somlbo = np.vstack(total_regret_base_somlbo[acq])
            regret_mean_base_somlbo = np.mean(regret_base_somlbo, axis=0)
            regret_std_dev_base_somlbo = np.std(regret_base_somlbo, axis=0)
            regret_std_dev_base_somlbo = regret_std_dev_base_somlbo / np.sqrt(total_number_of_obs)
            PH.printme(PH.p1, "\nBaseline Regret Details\nTotal Regret \n", regret_base_somlbo, "\n\nRegret Mean", regret_mean_base_somlbo,
                       "\n\nRegret Deviation\n", regret_std_dev_base_somlbo)

            ax.plot(iterations_axes_values, regret_mean_base_somlbo, colors[count])
            # plt.errorbar(iterations_axes_values, ei_regret_mean, yerr= ei_regret_std_dev )
            plt.gca().fill_between(iterations_axes_values, regret_mean_base_somlbo + regret_std_dev_base_somlbo,
                                   regret_mean_base_somlbo - regret_std_dev_base_somlbo, color=colors[count], alpha=0.25,
                                   label="STD-BO(MOD)")

            count += 1

        plt.axis([1, len(iterations_axes_values), 0, 3 * np.maximum(np.max(regret_mean_ai), np.maximum(np.max(regret_mean_base_mlbo),
                                                                    np.max(regret_mean_base_somlbo)))])
        plt.title('Regret')
        plt.xlabel('Evaluations')
        plt.ylabel('Simple Regret')
        legend = ax.legend(loc=1, fontsize='x-small')
        plt.savefig(pwd_name + fig_name + '.pdf')


if __name__ == "__main__":
    timenow = datetime.datetime.now()
    timestamp = timenow.strftime("%H%M%S_%d%m%Y")
    input = None
    ker_opt_wrapper_obj = ExpAIKerOptWrapper()

    function_type = "ACK1D"
    input = None

    argv = sys.argv[1:]
    cmd_inputs = {}

    try:
        opts, args = getopt.getopt(argv, "d:s:t:r:", ["dataset=", "subspaces=", "iterations=", "runs="])
    except getopt.GetoptError:
        print('python ExpAI_KernelOpt_Wrapper.py -d <dataset/function> -t <number_of_iterations> -r <runs>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-d", "--dataset"):
            cmd_inputs["dataset"] = arg
        elif opt in ("-t", "--iterations"):
            cmd_inputs["iterations"] = int(arg)
        elif opt in ("-r", "--runs"):
            cmd_inputs["runs"] = int(arg)
        else:
            print('python ExpAI_KernelOpt_Wrapper.py -d <dataset/function> -t <number_of_iterations> -r <runs>')
            sys.exit()

    full_time_stamp = function_type + "_" + timestamp
    directory_full_qualifier_name = os.getcwd() + "/HAT_Results/" + full_time_stamp + "/"
    PH(directory_full_qualifier_name)
    ker_opt_wrapper_obj.kernel_opt_wrapper(directory_full_qualifier_name, full_time_stamp, function_type, input, cmd_inputs)
