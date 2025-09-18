# ------------------------------------------
# Imports
# ------------------------------------------
import os
import json
import numpy as np
from datetime import datetime as dt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


class Scoring:
    """
    This class is used to compute the scores for the competition.

    Atributes:
        * start_time (datetime): The start time of the scoring process.
        * end_time (datetime): The end time of the scoring process.
        * training_labels (np.array): The training labels used for standardization.
        * reference_data (dict): The reference data.
        * ingestion_result (dict): The ingestion result.
        * ingestion_duration (float): The ingestion duration.
        * scores_dict (dict): The scores dictionary.
    """

    def __init__(self, name=""):
        # Initialize class variables
        self.start_time = None
        self.end_time = None
        self.training_labels = None
        self.reference_data = None
        self.ingestion_result = None
        self.ingestion_duration = None
        self.scores_dict = {}

    def start_timer(self):
        self.start_time = dt.now()

    def stop_timer(self):
        self.end_time = dt.now()

    def get_duration(self):
        if self.start_time is None:
            print("[-] Timer was never started. Returning None")
            return None

        if self.end_time is None:
            print("[-] Timer was never stoped. Returning None")
            return None

        return self.end_time - self.start_time

    def load_training_labels(self, reference_dir):
        """
        Load the training_labels (for standardization).

        Args:
            reference_dir (str): The reference data directory name.
        """
        print("[*] Reading training labels")
        training_labels_file = os.path.join(reference_dir, "label.npy")
        training_labels = np.load(training_labels_file)                     # float64 (101, 256, 5)
        self.training_labels = training_labels.reshape(101*256, -1)[:, :2]  # float64 (101*256, 2)

    def load_reference_data(self, reference_dir):
        """
        Load the reference data.

        Args:
            reference_dir (str): The reference data directory name.
        """
        print("[*] Reading reference data")
        reference_data_file = os.path.join(reference_dir, "label_test.npy")
        test_labels = np.load(reference_data_file)
        self.reference_data = test_labels[:, :2]

    def load_ingestion_result(self, predictions_dir):
        """
        Load the ingestion result.

        Args:
            predictions_dir (str): The predictions directory name.
        """
        print("[*] Reading ingestion result")

        ingestion_result_file = os.path.join(predictions_dir, "result.json")
        with open(ingestion_result_file) as f:
            self.ingestion_result = json.load(f)

    def check_result(self):
        if "means" not in self.ingestion_result:
            raise KeyError("[-] Means not found in the result!")

        if "errorbars" not in self.ingestion_result:
            raise KeyError("[-] Errorbars not found in the result!")

        test_set_size = self.reference_data.shape[0]

        if test_set_size != len(self.ingestion_result["means"]):
            raise ValueError(f"[-] Number of samples in means should be {test_set_size}")

        if test_set_size != len(self.ingestion_result["errorbars"]):
            raise ValueError(f"[-] Number of samples in errorbars should be {test_set_size}")

    def compute_scores(self):
        """
        Compute the scores for the competition.

        """

        self.label_scaler = StandardScaler()
        self.label_scaler.fit(self.training_labels)

        # Compute the score for Phase 1.
        # The rank on the leaderboard is sorted by this.
        def _score_phase1(true_cosmo, infer_cosmo, errorbar):
            sq_error = (true_cosmo - infer_cosmo)**2
            scale_factor = 1000  # This is a constant that scales the error term. 
            score = - np.sum(sq_error/errorbar**2 + np.log(errorbar**2) + scale_factor * sq_error, 1)
            score = np.mean(score)
            if score >= -10**6:  # Set a minimum of the score (to properly display on Codabench)
                return score
            else:
                return -10**6

        # Compute the mean of the MSEs of 2 POIs (both are standardized).
        # This is only for reference on the leaderboard.
        def _mse_phase1(true_cosmo, infer_cosmo):
            true_cosmo_scaled = self.label_scaler.transform(true_cosmo)
            infer_cosmo_scaled = self.label_scaler.transform(infer_cosmo)
            return mean_squared_error(true_cosmo_scaled, infer_cosmo_scaled)

        # Compute the mean of the R^2 coefficients of 2 POIs (standardization would not change the R^2).
        # This is only for reference on the leaderboard.
        def _r2_phase1(true_cosmo, infer_cosmo):
            return r2_score(true_cosmo, infer_cosmo)

        # Compute the fraction of times scalar mu is within intervals defined by vectors p16 and p84
        def _coverage_phase1(mu, p16, p84):
            """
            Args:
                * mu (array): The true \mu value.
                * p16 (array): The 16th percentile.
                * p84 (array): The 84th percentile.
            """
            coverage = np.mean((mu >= p16) & (mu <= p84), 0)
            return coverage

        print("[*] Computing scores")
        means = np.array(self.ingestion_result["means"])
        errorbars = np.array(self.ingestion_result["errorbars"])
        avg_score = _score_phase1(self.reference_data, means, errorbars)
        avg_errorbar = np.mean(errorbars, 0)

        mse = _mse_phase1(self.reference_data, means)
        r2_coeff = _r2_phase1(self.reference_data, means)

        p16 = means - errorbars
        p84 = means + errorbars
        avg_coverage = np.mean(_coverage_phase1(self.reference_data, p16, p84))

        print("------------------")
        print(f"Avg Score: {avg_score}")
        print(f"Avg Errorbar: {avg_errorbar}")
        print(f"Mean squared error: {mse}")
        print(f"R-squared: {r2_coeff}")
        print(f"Avg Coverage: {avg_coverage}")
        print("------------------")

        self.scores_dict = {
            "avg_score": avg_score,
            "MSE": mse,
            "R_squared": r2_coeff,
            "avg_coverage": avg_coverage
        }

    def write_scores(self, output_dir):

        print("[*] Writing scores")
        score_file = os.path.join(output_dir, "scores.json")
        with open(score_file, "w") as f_score:
            f_score.write(json.dumps(self.scores_dict, indent=4))
