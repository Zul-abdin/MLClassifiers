from math import log
import random
import numpy
import operator
import copy
import time


class NaiveBayes:
    def __init__(self, img_size_y, filenames):
        # user-defined self variables

        # Number of chars in each image (y dimension)
        # Number of lines to read per image until starting a new image
        self.img_size_y = img_size_y
        # List of all files (ordered in terms of execution order. FIRST WILL BE USED AS TRAINING)
        # FORMAT:
        # [["training_img", "training_labels"], ["validate_img", "validate_labels"], ["test_img", "test_labels"]]
        self.filenames = filenames

        # internally used self variables
        self.num_images_full = 0  #
        # Number of training images
        self.num_images = 0  #
        # labels mapped to all of their respective images (represented as strings) (FULL SET)
        self.label_images_full = dict()
        # labels mapped to all of their respective images (represented as strings) (FRACTIONAL SET)
        self.label_images = dict()
        # Probability of each label in training label
        self.train_label_probability = dict()  #
        # Probabilities of each feature for every label
        # (Example: Consider digits and label 0. Value will be a list of all 3 probabilities relating to every index in the image)
        self.feature_probability = dict()
        # List of all features in data
        self.valid_features = set()
        # Laplace smoothing constant
        self.k = 1
        # Percentage of samples to be taken
        self.sample_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # Number of trials for each percentage of samples
        self.trials = 5

    # Calculates and prints data based on 2 dicts with filenames as keys and lists of accuracies/time as values
    def print_data(self, acc, t_time, percent_data_used):
        print("Sampling " + str(percent_data_used) + "% of Training File: " + "(Training Time =  " + str(numpy.sum(t_time)) + ")")
        for filename in acc:
            print("\tAccuracy for predicting file " + filename + ": ")
            print("\t\tMean: " + str(numpy.mean(acc[filename])))
            print("\t\tStd Dev: " + str(numpy.std(acc[filename])))
        print()

    def fractional_training(self):
        for perc in self.sample_sizes:
            acc = dict()
            train_time = 0
            for i in range(self.trials):
                # Initializing fractional sets from full sets and re-used self variables
                self.num_images = int(self.num_images_full * (perc / 100))
                self.label_images = copy.deepcopy(self.label_images_full)
                self.train_label_probability = dict()
                self.feature_probability = dict()

                # Remove proper amount of randomly chosen images from fractional set
                for i in range(self.num_images_full - self.num_images):
                    available_labels = []
                    for label in self.label_images:
                        if len(self.label_images[label]) > 0:
                            available_labels.append(label)
                    random_label = random.choice(available_labels)
                    self.label_images[random_label].remove(random.choice(self.label_images[random_label]))

                # Actual Training happens here
                start_time = time.time()
                self.analyze_label_probability()
                for label in self.label_images:
                    self.analyze_feature_probability(label)
                end_time = time.time()
                train_time = end_time - start_time
                acc_dict = self.make_predictions()
                for filename in acc_dict:
                    if filename not in acc:
                        acc[filename] = []
                    acc[filename].append(acc_dict[filename])
            self.print_data(acc, train_time, perc)

    def training(self):
        with open(self.filenames[0][0]) as imageReader, open(self.filenames[0][1]) as labelReader:
            for labelLine in labelReader:
                label = labelLine[:-1]
                image = ""
                for line_num in range(self.img_size_y):
                    imageLine = next(imageReader).strip('\n')
                    [self.valid_features.add(c) for c in imageLine]
                    image += imageLine
                self.num_images_full += 1
                if label not in self.label_images_full:
                    self.label_images_full[label] = [image]
                else:
                    self.label_images_full[label].append(image)
        self.fractional_training()

    # Requires self.num_images and self.label_images to be accurate
    def analyze_label_probability(self):
        for key in self.label_images.keys():
            self.train_label_probability[key] = ((len(self.label_images[key]) + self.k) / (float(self.num_images) + (len(self.valid_features) * self.k)))

    def analyze_feature_probability(self, label):
        valid_label = ""
        for label_x in self.label_images:
            if len(self.label_images[label_x]) > 0:
                valid_label = label_x
                break
        probability_list = [dict() for i in range(len(self.label_images[valid_label][0]))]
        for ind in range(len(self.label_images[valid_label][0])):
            counts = dict()
            for feature in self.valid_features:
                counts[feature] = 0
            for image in self.label_images[label]:
                counts[image[ind]] += 1

            # Counts[feature] has now become probability of feature
            for feature in counts:
                counts[feature] = (((counts[feature]) + self.k) / (float(len(self.label_images[label])) + (len(self.valid_features) * self.k)))
            probability_list[ind] = counts

        self.feature_probability[label] = probability_list

    def make_predictions(self):
        acc = dict()
        valid_label = ""
        for label_x in self.label_images:
            if len(self.label_images[label_x]) > 0:
                valid_label = label_x
                break
        for filenames in self.filenames[1:]:
            with open(filenames[0]) as imageReader, open(filenames[1]) as labelReader:
                total_tries = 0
                correct = 0
                for labelLine in labelReader:
                    total_tries += 1
                    label = labelLine[:-1]
                    image = ""
                    for line_num in range(self.img_size_y):
                        imageLine = next(imageReader).strip('\n')
                        image += imageLine
                    ans = dict()
                    for label_v in self.train_label_probability:
                        ans[label_v] = 0.0
                    for feat_label in self.feature_probability:
                        prob_dict_list = self.feature_probability[feat_label]
                        sum_log = 0.0
                        for ind in range(len(self.label_images[valid_label][0])):
                            sum_log += log(prob_dict_list[ind][image[ind]])
                        sum_log += log(self.train_label_probability[feat_label])
                        ans[feat_label] = sum_log
                    best_prediction = max(ans.items(), key=operator.itemgetter(1))[0]
                    if label == best_prediction:
                        correct += 1
            # Returns [filename, Accuracy of Predictions]
            acc[filenames[0]] = (float(correct)/total_tries)

        return acc


digit_predictor = NaiveBayes(28,
                             [['digitdata/trainingimages', 'digitdata/traininglabels'],  # Training Files Here
                              ['digitdata/validationimages', 'digitdata/validationlabels'],  # Validation Files Here
                              ['digitdata/testimages', 'digitdata/testlabels']])  # Test Files Here
digit_predictor.training()

face_predictor = NaiveBayes(70,
                             [['facedata/facedatatrain', 'facedata/facedatatrainlabels'],  # Training Files Here
                              ['facedata/facedatavalidation', 'facedata/facedatavalidationlabels'],  # Validation Files Here
                              ['facedata/facedatatest', 'facedata/facedatatestlabels']])  # Test Files Here

face_predictor.training()