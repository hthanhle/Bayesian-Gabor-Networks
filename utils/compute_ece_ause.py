"""
Created on Wed Nov 25 10:38:43 2020
Compute ECE and AUSE metrics
@author: Hieu Phan
"""
import matplotlib.pyplot as plt
from scipy.special import softmax
import numpy as np
import pickle


class Bin:
    def __init__(self, lower_bound, upper_bound):
        self.correct = 0.0
        self.count = 0.0
        self.total_confidence = 0.0
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def reset(self):
        self.correct = 0.0
        self.count = 0.0
        self.total_confidence = 0.0

    def get_acc(self):
        if self.count == 0:
            return 0
        acc = self.correct / self.count
        return acc

    def get_conf(self):
        if self.count == 0:
            return 0
        conf = self.total_confidence / self.count
        return conf

    def assign_scores(self, scores, corrects):
        in_bin = np.all([scores > self.lower_bound, scores <= self.upper_bound], axis=0)
        self.count = self.count + in_bin.sum()
        self.correct = self.correct + corrects[in_bin].sum()
        self.total_confidence = self.total_confidence + scores[in_bin].sum()


class ECEHelper:
    def __init__(self, n_bins=10):
        bin_boundaries = np.linspace(0, 1, n_bins)
        self.n_bins = n_bins
        self.bins = [Bin(lower_bound=bin_boundaries[i], upper_bound=bin_boundaries[i + 1]) for i in range(n_bins - 1)]

    def reset(self):
        for bin in self.bins:
            bin.reset()

    def distribute_to_bins(self, logits, labels):
        n_classes = logits.shape[1]
        logits_ = np.transpose(logits, [0, 2, 3, 1]).reshape(
            (-1, n_classes))  # flatten. Expected shape: [height*width*1, num_classes]
        labels_ = labels.reshape(-1)  # flatten. Expected shape: [height*width*1]

        probs = softmax(logits_, axis=-1)
        confidence = np.max(probs, axis=-1)
        predictions = np.argmax(probs, axis=-1)

        corrects = predictions == labels_
        for i, bin in enumerate(self.bins):
            bin.assign_scores(confidence, corrects)

    def get_total_count(self):
        count = 0
        for bin in self.bins:
            count += bin.count
        return count

    def get_ece(self):
        ece = 0
        total_count = self.get_total_count()

        for bin in self.bins:
            if bin.count > 0:
                error = abs(bin.get_acc() - bin.get_conf()) * (bin.count / total_count)
                ece += error
        return ece

    def get_max_ece(self):
        max_ece = 0
        for bin in self.bins:
            error = abs(bin.get_acc() - bin.get_conf())
            if error > max_ece:
                max_ece = error
        return max_ece

    def save(self, filename):
        data = {}
        data['conf'] = np.array([bin.get_conf() for bin in self.bins])
        data['acc'] = np.array([bin.get_acc() for bin in self.bins])
        data['count'] = np.array([bin.count for bin in self.bins])
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        with open(filename, 'rb') as handle:
            b = pickle.load(handle)
        return b

    def visualize(self, filename=None, save_results=False):
        plt.clf()
        # Visualize the bins
        # total_count = self.get_total_count()
        # hist_positions = [(bin.lower_bound + bin.upper_bound) / 2 for bin in self.bins]
        # hist_height = [bin.count / total_count for bin in self.bins]
        # plt.bar(hist_positions, hist_height,width=1/self.n_bins, color="black", alpha=0.15)

        # Visualize the matched points
        conf = [bin.get_conf() for bin in self.bins]
        acc = [bin.get_acc() for bin in self.bins]
        plt.tight_layout()
        plt.plot(conf, acc, "ro")
        plt.ylabel("Accuracy")
        plt.xlabel("Confidence")
        plt.xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        plt.ylim((-0.05, 1.05))
        plt.xlim((-0.05, 1.05))
        plt.plot([-0.05, 1.05], [-0.05, 1.05], '--', c='darkgray')
        if filename != None:
            plt.savefig(filename, dpi=1200)
            plt.clf()
        else:
            plt.show()

        if save_results == True:
            np.save('./conf.npy', conf)
            np.save('./acc.npy', acc)


class AUSEHelper():
    def __init__(self, step=0.01):
        self.fractions = list(np.arange(start=0.0, stop=1.0, step=step))  # ([0.0, 0.01, ..., 0.99], 100 elements)
        self.squared_error_values = np.array([])
        self.uncertainty = np.array([])
        self.scores_by_error = np.array([])
        self.scores_by_uncertainty = np.array([])

    def store_values(self, logits, labels, uncertainty):
        n_classes = logits.shape[1]
        logits_ = np.transpose(logits, [0, 2, 3, 1]).reshape(
            (-1, n_classes))  # flatten. Expected shape: [height*width*1, num_classes]
        labels_ = labels.reshape(-1)  # flatten. Expected shape: [height*width*1]
        uncertainty = uncertainty.reshape(-1)
        probs = softmax(logits_, axis=-1)

        one_hot = np.zeros((labels_.size, n_classes))  # one-hot coding. Expected shape: [height*width*1, num_classes]
        one_hot[np.arange(labels_.size), labels_] = 1

        # Here we use MSE to measure the error between the expected predictions (i.e. the one-hot of the labels) and the actual predictions (i.e. predicted scores)        
        squared_error = np.sum(np.power(one_hot - probs, 2), axis=-1)
        self.squared_error_values = np.concatenate([self.squared_error_values, squared_error])
        self.uncertainty = np.concatenate([self.uncertainty, uncertainty])

    def sparsification_error_by_removing_fractions(self):
        sorted_id_by_uncertainty = np.argsort(
            self.uncertainty)  # get the indices that would sort an array in ASCENDING order
        sorted_id_by_squared_error = np.argsort(self.squared_error_values)
        n = self.uncertainty.shape[0]

        for fraction in self.fractions:
            n_retains = int((1.0 - fraction) * n)

            brier_scores_by_uncertainty = np.mean(self.squared_error_values[sorted_id_by_uncertainty[
                                                                            :n_retains]])  # '[:n_retains]' is to take the pixels (from the beginning) upto 'n_retains'. That means we are REMOVING the last pixels (i.e. the pixels with high UNCERTAINTY)
            self.scores_by_uncertainty = np.append(self.scores_by_uncertainty, brier_scores_by_uncertainty)

            brier_scores_by_error = np.mean(self.squared_error_values[sorted_id_by_squared_error[:n_retains]])
            self.scores_by_error = np.append(self.scores_by_error, brier_scores_by_error)

        self.scores_by_uncertainty = self.scores_by_uncertainty / self.scores_by_uncertainty[
            0]  # sparsification error. Note that dividing by the first element (i.e. largest one) is for normalization
        self.scores_by_error = self.scores_by_error / self.scores_by_error[0]  # oracle

    def save(self, filename):
        data = {}
        data['scores_by_error'] = self.scores_by_error
        data['scores_by_uncertainty'] = self.scores_by_uncertainty
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        with open(filename, 'rb') as handle:
            b = pickle.load(handle)
            self.scores_by_error = b['scores_by_error']
            self.scores_by_uncertainty = b['scores_by_uncertainty']

    def get_ause(self):
        if len(self.scores_by_error) == 0 or len(self.scores_by_uncertainty) == 0:
            self.sparsification_error_by_removing_fractions()
        sparsification_errors = self.scores_by_uncertainty - self.scores_by_error

        ause = np.trapz(y=sparsification_errors, x=self.fractions)

        return ause

    def visualize(self, filename=None, save_results=False):
        if len(self.scores_by_error) == 0 or len(self.scores_by_uncertainty) == 0:
            self.sparsification_error_by_removing_fractions()
        plt.tight_layout()
        plt.plot(self.fractions, self.scores_by_error, 'b--', label='Oracle sparsification')
        plt.plot(self.fractions, self.scores_by_uncertainty, 'r-', label='BGN sparsification')
        plt.legend()
        plt.ylabel("MSE")
        plt.xlabel("Percentage of removed pixels")
        if filename != None:
            plt.savefig(filename, dpi=1200)
            plt.clf()
        else:
            plt.show()

        if save_results == True:
            np.save('./fraction_removed_pixels.npy', self.fractions)
            np.save('./scores_by_error.npy', self.scores_by_error)
            np.save('./scores_by_uncertainty.npy', self.scores_by_uncertainty)
