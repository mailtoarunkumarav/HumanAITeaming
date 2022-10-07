from scipy.stats import norm
import numpy as np
import scipy.optimize as opt
import matplotlib
matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
from HelperUtility.PrintHelper import PrintHelper as PH
import re

# Class to handle the Acquisition Functions related tasks required for the Bayesian Optimization
class AcquisitionFunction():

    # Initializing the parameters required for the ACQ functions
    def __init__(self, acq_type, number_of_restarts, nu=1, epsilon1=3, epsilon2=4 ):
        self.acq_type = acq_type
        self.number_of_restarts = number_of_restarts
        self.nu = nu
        self.epsilon1 = epsilon1
        self.epsilon2 = epsilon2

    # Method to set the type of ACQ function to be used for the Optimization process
    def set_acq_func_type(self, type):
        self.acq_type = type

    # Expected improvement ACQ function
    def expected_improvement(self, role, noisy_suggestions, mean, std_dev, y_max):
        with np.errstate(divide='ignore'):
            z_value = (mean - y_max - self.epsilon2) / std_dev
            zpdf = norm.pdf(z_value)
            zcdf = norm.cdf(z_value)
            # commented to fit the code for plotting acq functions
            # ei_acq_func = np.dot(zcdf, (mean - y_max - self.epsilon2)) + np.dot(std_dev, zpdf)
            ei_acq_func = np.multiply(zcdf, (mean - y_max - self.epsilon2)) + np.multiply(std_dev, zpdf)
            ei_acq_func[std_dev == 0] = 0

        return ei_acq_func

    # Probability improvement ACQ function
    def probability_improvement(self, role, mean, std_dev, y_max):
        z_value = (mean - y_max - self.epsilon1) / std_dev
        zcdf = norm.cdf(z_value)
        return zcdf

    # UCB ACQ function
    def ucb_acq_func(self, role, noisy_suggestions, mean, std_dev, iteration_count, number_of_dimensions):


        with np.errstate(divide='ignore') or np.errstate(invalid='ignore'):

            # Constant parameters to be used while maximizing the ACQ function
            delta = 0.1;d = number_of_dimensions;b = 1;a = 1;r = 1;v=1;

            # Different forms for the beta value
            # formula beta1 = 2log(t22π2/ (3δ)) + 2dlogt2dbr * sqrt(log(4da / δ))
            # formula beta2 = 2log(dt2π2/6δ)
            # formula beta3 = 2 log(td/2+2π2/3δ)

            # beta1 = 2 * np.log((iteration_count ** 2) * (2 * (np.pi ** 2)) * (1 / (3 * delta))) +\
            #         (2 * d) * np.log((iteration_count ** 2) * d * b * r * (np.sqrt(np.log(4 * d * a * (1 / delta)))))
            #
            # beta2 = 2 * np.log(d*(iteration_count**2)*(np.pi**2)*(1/(6*delta)))

            # beta3 = 2 * np.log((iteration_count**((d/2)+2))* (np.pi**2) * (1/(3*delta)))

            beta = 2 * np.log((iteration_count ** 2) * (2 * (np.pi ** 2)) * (1 / (3 * delta))) + (2 * d) * np.log((iteration_count ** 2)
                                                                    * d * b * r * (np.sqrt(np.log(4 * d * a * (1 / delta)))))

            # Uncomment to add noisy suggestion
            if role == "HumanExpert" and noisy_suggestions:

                # simple Gamma distribution
                mu = beta
                var = 0.01
                shape = (mu * mu/var)
                scale = var/mu
                beta = np.random.gamma(shape, scale)

                # # # Gamma distribution - RGP-UCB
                # theta = 1
                # k_t = (np.log((1/(2*np.sqrt(2)))*(iteration_count**2+1)))/np.log(1+(theta/2))
                # beta = np.random.gamma(k_t, theta)

                # # Beta distribution
                # mu = beta3
                # var = 0.01
                # shape1_alpha = mu * ((((1 - mu) * mu) / var ** 2) - 1)
                # shape2_beta = shape1_alpha * ((1 / mu) - 1)
                # print(beta3, shape1_alpha, shape2_beta)
                # beta3 = np.random.beta(shape1_alpha, shape2_beta)
                #
                # # Normal distribution
                # # mu = beta3
                # # var = 0.5
                # # beta3 = np.random.normal(mu, var)

            self.nu = np.sqrt(v * beta)
            # self.nu = 0.1 * beta3
            result = mean + self.nu * std_dev
            return result

    # Helper method to invoke the EI acquisition function
    def expected_improvement_util(self, role, noisy_suggestions, x, y_max, gp_obj):

        with np.errstate(divide='ignore') or np.errstate(invalid='ignore'):

            # Use Gaussian Process to predict the values for mean and variance at given x
            mean, variance, fprior, f_post = gp_obj.gaussian_predict(np.array([x]))
            std_dev = np.sqrt(variance)
            result = self.expected_improvement(role, noisy_suggestions, mean, std_dev, y_max)
            # Since scipy.minimize function is used to find the minima and so converting it to maxima by * with -1
            return result

    def probability_improvement_util(self, role, x, y_max, gp_obj):
        with np.errstate(divide='ignore') or np.errstate(invalid='ignore'):
            # Use Gaussian Process to predict the values for mean and variance at given x
            mean, variance, fprior, f_post = gp_obj.gaussian_predict(np.matrix(x))
            # mean, variance, fprior, f_post = gp_obj.gaussian_predict(np.array(x))
            std_dev = np.sqrt(variance)
            result = self.probability_improvement(role, mean, std_dev, y_max)
            # Since scipy.minimize function is used to find the minima and so converting it to maxima by * with -1
            return result

    def upper_confidence_bound_util(self, role, noisy_suggestions, x, gp_obj, iteration_count):
        with np.errstate(divide='ignore') or np.errstate(invalid='ignore'):
            # Use Gaussian Process to predict the values for mean and variance at given x
            mean, variance, fprior, f_post = gp_obj.gaussian_predict(np.matrix(x))
            std_dev = np.sqrt(variance)
            result = self.ucb_acq_func(role, noisy_suggestions, mean, std_dev, iteration_count, gp_obj.number_of_dimensions)
            # Since scipy.minimize function is used to find the minima and so converting it to maxima by * with -1
            return result

    def data_conditioned_upper_confidence_bound_util(self, role, noisy_suggestions, Xs, X, y, gp_obj, iteration_count):
        with np.errstate(divide='ignore') or np.errstate(invalid='ignore'):
            # Use Gaussian Process to predict the values for mean and variance at given x
            mean, variance, fprior, f_post = gp_obj.gaussian_predict_on_conditioned_X(np.matrix(Xs), X, y)
            std_dev = np.sqrt(variance)
            result = self.ucb_acq_func(role, noisy_suggestions, mean, std_dev, iteration_count, gp_obj.number_of_dimensions)
            # Since scipy.minimize function is used to find the minima and so converting it to maxima by * with -1
            return result

    def data_conditioned_expected_improvement_util(self, role, noisy_suggestions, Xs, X, y, y_max, gp_obj):

        with np.errstate(divide='ignore') or np.errstate(invalid='ignore'):

            # Use Gaussian Process to predict the values for mean and variance at given x
            mean, variance, fprior, f_post = gp_obj.gaussian_predict_on_conditioned_X(np.array([Xs]), X, y)
            std_dev = np.sqrt(variance)
            result = self.expected_improvement(role, noisy_suggestions, mean, std_dev, y_max)
            # Since scipy.minimize function is used to find the minima and so converting it to maxima by * with -1
            return result

    # Method to maximize the ACQ function as specified the user
    def max_acq_func(self, role, noisy_suggestions, gp_obj, iteration_count, print_bool="TT"):

        # Initialize the xmax value and the function values to zeroes
        x_max_value = np.zeros(gp_obj.number_of_dimensions)
        fmax = - 1 * float("inf")

        # Temporary data structures to store xmax's, function values of each run of finding maxima using scipy.minimize
        tempmax_x=[]
        tempfvals=[]

        # Data structure to create the starting points for the scipy.minimize method
        random_points = []
        starting_points = []
        # Depending on the number of dimensions and bounds, generate random multiple starting points to find maxima
        for dim in np.arange(gp_obj.number_of_dimensions):
            random_data_point_each_dim = np.random.uniform(gp_obj.bounds[dim][0], gp_obj.bounds[dim][1],
                                                           self.number_of_restarts).reshape(1, self.number_of_restarts)
            random_points.append(random_data_point_each_dim)

        # Vertically stack the arrays of randomly generated starting points as a matrix
        random_points = np.vstack(random_points)

        # Reformat the generated random starting points in the form [x1 x2].T for the specified number of restarts
        for sample_num in np.arange(self.number_of_restarts):
            array = []
            for dim_count in np.arange(gp_obj.number_of_dimensions):
                array.append(random_points[dim_count, sample_num])
            starting_points.append(array)
        starting_points = np.vstack(starting_points)

        # Normalizing code
        starting_points = np.divide((starting_points - gp_obj.Xmin), (gp_obj.Xmax - gp_obj.Xmin))

        # Find maxima of the ACQ function using PI
        if (self.acq_type == 'pi'):

            # Obtain the maximum value of the unknown function from the samples observed already
            y_max = gp_obj.y.max()
            PH.printme(print_bool,"ACQ Function : PI ")

            # Obtain the maxima of the ACQ function by starting the optimization at different start points
            for starting_point in starting_points:

                # Find the maxima in the bounds specified for the PI ACQ function
                max_x = opt.minimize(lambda x: -self.probability_improvement_util(role, x, y_max, gp_obj), starting_point,
                                     method='L-BFGS-B',
                                     tol=0.001,
                                     options={'maxfun': 200, 'maxiter': 20},
                                     # bounds=gp_obj.bounds)
                                     bounds=[[0, 1] for bnds in range(gp_obj.number_of_dimensions)])

                # Use gaussian process to predict mean and variances for the maximum point identified
                mean, variance, f_prior, f_post = gp_obj.gaussian_predict(np.matrix(max_x['x']))
                std_dev = np.sqrt(variance)
                fvalue = self.probability_improvement(role, mean, std_dev, y_max)

                # Store the maxima of ACQ function and the corresponding value at the maxima, required for debugging
                tempmax_x.append(max_x['x'])
                tempfvals.append(fvalue)
                # Compare the values obtained in the current run to find the best value overall and store accordingly
                if (fvalue > fmax):
                    PH.printme(print_bool,"New best Fval: ",fvalue," found at: ", max_x['x'])
                    x_max_value = max_x['x']
                    fmax = fvalue

            PH.printme(print_bool,"PI Best is ", fmax, "at ", x_max_value)

            # Calculate the ACQ function values at each of the unseen data points to plot the ACQ function
            with np.errstate(invalid='ignore'):
                mean, variance, f_prior, f_post = gp_obj.gaussian_predict(gp_obj.Xs)
                std_dev = np.sqrt(variance)
                acq_func_values = self.probability_improvement(role, mean, std_dev, y_max)

            # used to verify if the maxima value is really found at 0
            # if(x_max_value==[0]):
            #     pH.printme(print_bool,'\n\ntemp fvalues and xmax',tempfvals, tempmax_x)

        # Find maxima of the ACQ function using UCB
        elif (self.acq_type == "ucb"):
            # PH.printme(print_bool, "ACQ Function : UCB ")

            # Obtain the maxima of the ACQ function by starting the optimization at different start points
            for starting_point in starting_points:

                # Find the maxima in the bounds specified for the UCB ACQ function
                max_x = opt.minimize(lambda x: -self.upper_confidence_bound_util(role, noisy_suggestions, x, gp_obj, iteration_count),
                                     starting_point,
                                     method='L-BFGS-B',
                                     tol=0.001,
                                     options={'maxfun': 200, 'maxiter': 40},
                                     # bounds=gp_obj.bounds)
                                     bounds=[[0, 1] for bnds in range(gp_obj.number_of_dimensions)])

                # Use gaussian process to predict mean and variances for the maximum point identified
                mean, variance, f_prior, f_post = gp_obj.gaussian_predict(np.matrix(max_x['x']))
                std_dev = np.sqrt(variance)
                fvalue = self.ucb_acq_func(role, noisy_suggestions, mean, std_dev, iteration_count, gp_obj.number_of_dimensions)

                # Store the maxima of ACQ function and the corresponding value at the maxima, required for debugging
                tempmax_x.append(max_x['x'])
                tempfvals.append(fvalue)

                # Compare the values obtained in the current run to find the best value overall and store accordingly
                if fvalue > fmax:
                    PH.printme(print_bool, "New UCB Maximum is: ", fvalue, " found at: ", max_x['x'])
                    x_max_value = max_x['x']
                    fmax = fvalue

            PH.printme(print_bool, "Final UCB maximum is ", fmax, "at ", x_max_value)

            # Calculate the ACQ function values at each of the unseen data points to plot the ACQ function
            with np.errstate(invalid='ignore'):
                mean, variance, f_prior, f_post = gp_obj.gaussian_predict(gp_obj.Xs)
                std_dev = np.sqrt(variance)
                acq_func_values = self.ucb_acq_func(role, noisy_suggestions, mean, std_dev, iteration_count, gp_obj.number_of_dimensions)

            # used to verify if the maxima value is really found at 0
            # if(x_max_value==[0]):
            #     pH.printme(print_bool,tempfvals, tempmax_x)

        # Find maxima of the ACQ function using EI

        elif (self.acq_type == 'ei'):

            # Obtain the maximum value of the unknown function from the samples observed already
            y_max = gp_obj.y.max()
            PH.printme(print_bool,"ACQ Function : EI ")

            # Obtain the maxima of the ACQ function by starting the optimization at different start points
            for starting_point in starting_points:

                # Find the maxima in the bounds specified for the PI ACQ function
                max_x = opt.minimize(lambda x: -self.expected_improvement_util(role, noisy_suggestions, x, y_max, gp_obj), starting_point,
                                     method='L-BFGS-B',
                                     tol=0.01,
                                     options={'maxfun': 200, 'maxiter': 40},
                                     # bounds=gp_obj.bounds)
                                     # bounds = [[0, 1],[0,1]])
                                     bounds=[[0, 1] for bnds in range(gp_obj.number_of_dimensions)])

                # Use gaussian process to predict mean and variances for the maximum point identified
                mean, variance, f_prior, f_post = gp_obj.gaussian_predict(np.array([max_x['x']]))
                std_dev = np.sqrt(variance)
                fvalue = self.expected_improvement(role, noisy_suggestions, mean, std_dev, y_max)

                # Store the maxima of ACQ function and the corresponding value at the maxima, required for debugging
                tempmax_x.append(max_x['x'])
                tempfvals.append(fvalue)

                # Compare the values obtained in the current run to find the best value overall and store accordingly
                if (fvalue > fmax):
                    PH.printme(print_bool,"New best Fval: ", fvalue, " found at: ", max_x['x'])
                    x_max_value = max_x['x']
                    fmax = fvalue

            PH.printme(print_bool,"EI Best is ", fmax, "at ", x_max_value)

            # # Uncomment to add noisy suggestion
            if role == "HumanExpert" and noisy_suggestions:
                PH.printme(PH.p1, "Adding noise to the Human expert suggested point using EI Acq")
                perturbation = np.random.uniform(-0.2, 0.2)
                x_max_value = x_max_value + perturbation

            # Calculate the ACQ function values at each of the unseen data points to plot the ACQ function
            with np.errstate(invalid='ignore'):
                mean, variance, f_prior, f_post = gp_obj.gaussian_predict(gp_obj.Xs)
                std_dev = np.sqrt(variance)
                acq_func_values = self.expected_improvement(role, noisy_suggestions, mean, std_dev, y_max)

            # used to verify if the maxima value is really found at 0
            # if(x_max_value == [0]):
            #     pH.printme(print_bool,tempfvals, tempmax_x)

        return x_max_value, acq_func_values

    # Method to maximize the ACQ function as specified the user
    def min_acq_func(self, role, noisy_suggestions, gp_obj, iteration_count, print_bool="TT"):

        # # Initialize the xmax value and the function values to zeroes
        x_min_value = np.zeros(gp_obj.number_of_dimensions)
        fmin = 1 * float("inf")

        # Temporary data structures to store xmax's, function values of each run of finding maxima using scipy.minimize
        tempmax_x = []
        tempfvals = []

        # Data structure to create the starting points for the scipy.minimize method
        random_points = []
        starting_points = []
        # Depending on the number of dimensions and bounds, generate random multiple starting points to find maxima
        for dim in np.arange(gp_obj.number_of_dimensions):
            random_data_point_each_dim = np.random.uniform(gp_obj.bounds[dim][0], gp_obj.bounds[dim][1],
                                                           self.number_of_restarts).reshape(1, self.number_of_restarts)
            random_points.append(random_data_point_each_dim)

        # Vertically stack the arrays of randomly generated starting points as a matrix
        random_points = np.vstack(random_points)

        # Reformat the generated random starting points in the form [x1 x2].T for the specified number of restarts
        for sample_num in np.arange(self.number_of_restarts):
            array = []
            for dim_count in np.arange(gp_obj.number_of_dimensions):
                array.append(random_points[dim_count, sample_num])
            starting_points.append(array)
        starting_points = np.vstack(starting_points)

        # Normalizing code
        starting_points = np.divide((starting_points - gp_obj.Xmin), (gp_obj.Xmax - gp_obj.Xmin))

        # Find maxima of the ACQ function using PI
        if (self.acq_type == 'pi'):

            # Obtain the maximum value of the unknown function from the samples observed already
            y_max = gp_obj.y.max()
            PH.printme(print_bool, "ACQ Function : PI ")

            # Obtain the maxima of the ACQ function by starting the optimization at different start points
            for starting_point in starting_points:

                # Find the maxima in the bounds specified for the PI ACQ function
                min_x = opt.minimize(lambda x: self.probability_improvement_util(role, x, y_max, gp_obj), starting_point,
                                     method='L-BFGS-B',
                                     tol=0.001,
                                     options={'maxfun': 200, 'maxiter': 40},
                                     # bounds=gp_obj.bounds)
                                     bounds=[[0, 1] for bnds in range(gp_obj.number_of_dimensions)])

                # Use gaussian process to predict mean and variances for the maximum point identified
                mean, variance, f_prior, f_post = gp_obj.gaussian_predict(np.matrix(min_x['x']))
                std_dev = np.sqrt(variance)
                fvalue = self.probability_improvement(role, mean, std_dev, y_max)

                # Store the maxima of ACQ function and the corresponding value at the maxima, required for debugging
                tempmax_x.append(min_x['x'])
                tempfvals.append(fvalue)
                # Compare the values obtained in the current run to find the best value overall and store accordingly
                if (fvalue < fmin):
                    PH.printme(print_bool, "New best Fval: ", fvalue, " found at: ", min_x['x'])
                    x_min_value = min_x['x']
                    fmin = fvalue

            PH.printme(print_bool, "PI Best is ", fmin, "at ", x_min_value)

            # Calculate the ACQ function values at each of the unseen data points to plot the ACQ function
            with np.errstate(invalid='ignore'):
                mean, variance, f_prior, f_post = gp_obj.gaussian_predict(gp_obj.Xs)
                std_dev = np.sqrt(variance)
                acq_func_values = self.probability_improvement(role, mean, std_dev, y_max)

        # Find maxima of the ACQ function using UCB
        elif (self.acq_type == "ucb"):
            # PH.printme(print_bool, "ACQ Function : UCB ")

            # Obtain the maxima of the ACQ function by starting the optimization at different start points
            for starting_point in starting_points:

                # Find the maxima in the bounds specified for the UCB ACQ function
                min_x = opt.minimize(lambda x: self.upper_confidence_bound_util(role, noisy_suggestions, x, gp_obj, iteration_count),
                                     starting_point,
                                     method='L-BFGS-B',
                                     tol=0.001,
                                     options={'maxfun': 200, 'maxiter': 40},
                                     # bounds=gp_obj.bounds)
                                     bounds=[[0, 1] for bnds in range(gp_obj.number_of_dimensions)])

                # Use gaussian process to predict mean and variances for the maximum point identified
                mean, variance, f_prior, f_post = gp_obj.gaussian_predict(np.matrix(min_x['x']))
                std_dev = np.sqrt(variance)
                fvalue = self.ucb_acq_func(role, noisy_suggestions, mean, std_dev, iteration_count, gp_obj.number_of_dimensions)

                # Store the maxima of ACQ function and the corresponding value at the maxima, required for debugging
                tempmax_x.append(min_x['x'])
                tempfvals.append(fvalue)

                # Compare the values obtained in the current run to find the best value overall and store accordingly
                if (fvalue < fmin):
                    PH.printme(print_bool, "New UCB Minimum is: ", fvalue, " found at: ", min_x['x'])
                    x_min_value = min_x['x']
                    fmin = fvalue

            PH.printme(print_bool, "Final UCB minimum is ", fmin, "at ", x_min_value)

            # Calculate the ACQ function values at each of the unseen data points to plot the ACQ function
            with np.errstate(invalid='ignore'):
                mean, variance, f_prior, f_post = gp_obj.gaussian_predict(gp_obj.Xs)
                std_dev = np.sqrt(variance)
                acq_func_values = self.ucb_acq_func(role, noisy_suggestions, mean, std_dev, iteration_count, gp_obj.number_of_dimensions)

            # used to verify if the maxima value is really found at 0
            # if(x_max_value==[0]):
            #     pH.printme(print_bool,tempfvals, tempmax_x)

        # Find maxima of the ACQ function using EI

        elif (self.acq_type == 'ei'):

            # Obtain the maximum value of the unknown function from the samples observed already
            y_max = gp_obj.y.max()
            PH.printme(print_bool, "ACQ Function : EI ")

            # Obtain the maxima of the ACQ function by starting the optimization at different start points
            for starting_point in starting_points:

                # Find the maxima in the bounds specified for the PI ACQ function
                min_x = opt.minimize(lambda x: self.expected_improvement_util(role, noisy_suggestions, x, y_max, gp_obj), starting_point,
                                     method='L-BFGS-B',
                                     tol=0.01,
                                     options={'maxfun': 20, 'maxiter': 40},
                                     # bounds=gp_obj.bounds)
                                     # bounds = [[0, 1],[0,1]])
                                     bounds=[[0, 1] for bnds in range(gp_obj.number_of_dimensions)])

                # Use gaussian process to predict mean and variances for the maximum point identified
                mean, variance, f_prior, f_post = gp_obj.gaussian_predict(np.array([min_x['x']]))
                std_dev = np.sqrt(variance)
                fvalue = self.expected_improvement(role, noisy_suggestions, mean, std_dev, y_max)

                # Store the maxima of ACQ function and the corresponding value at the maxima, required for debugging
                tempmax_x.append(min_x['x'])
                tempfvals.append(fvalue)

                # Compare the values obtained in the current run to find the best value overall and store accordingly
                if (fvalue < fmin):
                    PH.printme(print_bool, "New best minimum Fval: ", fvalue, " found at: ", min_x['x'])
                    x_min_value = min_x['x']
                    fmin = fvalue

            PH.printme(print_bool, "EI Best minimum is ", fmin, "at ", x_min_value)

            # Uncomment to add noisy suggestion
            if role == "HumanExpert" and noisy_suggestions:
                PH.printme(PH.p1, "Adding noise to the Human expert suggested point using EI Acq")
                perturbation = np.random.uniform(-0.2, 0.2)
                x_min_value = x_min_value + perturbation

            # Calculate the ACQ function values at each of the unseen data points to plot the ACQ function
            with np.errstate(invalid='ignore'):
                mean, variance, f_prior, f_post = gp_obj.gaussian_predict(gp_obj.Xs)
                std_dev = np.sqrt(variance)
                acq_func_values = self.expected_improvement(role, noisy_suggestions, mean, std_dev, y_max)

        return x_min_value, acq_func_values


    # Helper method to plot the values found for the specified ACQ function at unseen data points
    def plot_acquisition_function(self, pwd_qualifier, Xs, acq_func_values, plot_axis):

        pattern = re.compile("R[0-9]+_+")
        match = pattern.search(pwd_qualifier)
        span = match.span()
        file_name = pwd_qualifier[span[1] - (span[1] - span[0]):]
        # Set the parameters of the ACQ functions plot
        plt.figure(file_name)
        plt.clf()
        plt.plot(Xs, acq_func_values)
        plt.axis(plot_axis)
        plt.title('Acquisition Function')
        plt.savefig(pwd_qualifier+".pdf", bbox_inches='tight')


    def plot_graph(self, count, Xs, len_values, plot_axis):

        # Set the parameters of the ACQ functions plot
        plt.figure('lengthscale - ' + str(count))
        plt.clf()
        plt.plot(Xs, len_values)
        # plt.axis(plot_axis)
        plt.title('lengthscale')
        plt.savefig('len'+str(count), bbox_inches='tight')
