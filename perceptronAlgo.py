import random
import time
import copy
import numpy


class Perceptron:
    def __init__(self, img_size_y, filenames, enumerate_features):

        # user-defined self variables

        # Number of chars in each image (y dimension)
        # Number of lines to read per image until starting a new image
        self.img_size_y = img_size_y
        # List of all files (ordered in terms of execution order. FIRST WILL BE USED AS TRAINING)
        # FORMAT:
        # [["training_img", "training_labels"], ["validate_img", "validate_labels"], ["test_img", "test_labels"]]
        self.filenames = filenames
        # Enumeration function
        self.enumerate_features = enumerate_features

        # internally used self variables
        self.num_images_full = 0
        # Number of training images
        self.num_images = 0
        # labels mapped to all of their respective images (represented as strings) (FULL SET)
        self.label_images_full = dict()
        # labels mapped to all of their respective images (represented as strings) (FRACTIONAL SET)
        self.label_images = dict()
        # List of all features in data
        self.valid_features = set()
        # Percentage of samples to be taken
        self.sample_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # Number of trials for each percentage of samples
        self.max_iter = 3
        # Dict with label as keys and the weight vector (as a list) as values
        self.weights = dict()
        # Number of trials for each percentage of samples
        self.trials = 5
        # Threshold for changing true weights
        self.threshold = -1

    # Calculates and prints data based on 2 dicts with filenames as keys and lists of accuracies/time as values
    def print_data(self, acc, t_time, percent_data_used):
        print("Sampling " + str(percent_data_used) + "% of Training File: " + "(Training Time =  " + str(
            numpy.sum(t_time)) + ")")
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
                self.weights = dict()

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
                self.init_weights()
                self.update_weights()
                end_time = time.time()
                train_time = end_time - start_time
                acc_dict = self.make_predictions()
                for filename in acc_dict:
                    if filename not in acc:
                        acc[filename] = []
                    acc[filename].append(acc_dict[filename])
            self.print_data(acc, train_time, perc)

    def make_predictions(self):
        acc = dict()
        for filenames in self.filenames[1:]:
            with open(filenames[0]) as imageReader, open(filenames[1]) as labelReader:
                total_tries = 0
                correct = 0
                for labelLine in labelReader:
                    total_tries += 1
                    label = labelLine[:-1]
                    img = []
                    for line_num in range(self.img_size_y):
                        imageLine = next(imageReader).strip('\n')
                        [img.append(self.enumerate_features[c]) for c in imageLine]
                    scores = dict()
                    for label_in in self.label_images:
                        scores[label_in] = self.calc_score(label_in, img)
                    prediction = max(scores, key=scores.get)
                    # print(prediction)
                    if label == prediction:
                        correct += 1
            # Returns [filename, Accuracy of Predictions]
            acc[filenames[0]] = (float(correct) / total_tries)

        return acc

    def calc_score(self, label, img):
        sum = 0
        for ind in range(len(img)):
            sum += (img[ind] * self.weights[label][ind])
        return sum

    def update_weights(self):
        did_update = False
        for iter_num in range(self.max_iter):

            # Version 2
            rand_order = []
            for label in self.label_images:
                for img in self.label_images[label]:
                    rand_order.append([label, img])
            random.shuffle(rand_order)

            scores_ex = dict()
            for item in rand_order:
                for label_ex in self.label_images:
                    scores_ex[label_ex] = self.calc_score(label_ex, item[1])
                prediction_ex = max(scores_ex, key=scores_ex.get)

                if prediction_ex != item[0]:
                    did_update = True
                    for ind in range(len(item[1])):
                        if abs(self.weights[prediction_ex][ind] - self.weights[item[0]][ind]) >= self.threshold:
                            self.weights[item[0]][ind] += item[1][ind]
                        self.weights[prediction_ex][ind] -= item[1][ind]

            # Version 1 [Not random order when iterating to train. This makes the predictions/weights be overwhelmingly biased]
            # for label in self.label_images:
            # The dict to hold all scores for each label (max is the prediction)
            #    scores = dict()
            #    # Begin calculating scores
            #    for img in self.label_images[label]:
            #        for label_in in self.label_images:
            #            scores[label_in] = self.calc_score(label_in, img)
            #        prediction = max(scores, key=scores.get)

            # If prediction is not correct...
            #        if prediction != label:
            #            did_update = True
            #            for ind in range(len(img)):
            #                if abs(self.weights[prediction][ind] - self.weights[label][ind]) >= self.threshold:
            #                    self.weights[label][ind] += img[ind]
            #                self.weights[prediction][ind] -= img[ind]
            if not did_update:
                break

    def training(self):
        with open(self.filenames[0][0]) as imageReader, open(self.filenames[0][1]) as labelReader:
            for labelLine in labelReader:
                label = labelLine[:-1]
                image = []
                for line_num in range(self.img_size_y):
                    imageLine = next(imageReader).strip('\n')
                    [self.valid_features.add(c) for c in imageLine]
                    [image.append(self.enumerate_features[c]) for c in imageLine]
                self.num_images_full += 1
                if label not in self.label_images_full:
                    self.label_images_full[label] = [image]
                else:
                    self.label_images_full[label].append(image)
        self.fractional_training()

    def init_weights(self):
        for label in self.label_images:
            self.weights[label] = []

        valid_label = ""
        for label_x in self.label_images:
            if len(self.label_images[label_x]) > 0:
                valid_label = label_x
                break

        for label in self.label_images:
            for char in self.label_images[valid_label][0]:
                # self.weights[label].append(random.uniform(-1, 1))
                self.weights[label].append(0)


enumeration_func = {' ': 0, '+': 1, '#': 2}

digit_predictor = Perceptron(28,
                             [['digitdata/trainingimages', 'digitdata/traininglabels'],  # Training Files Here
                              ['digitdata/validationimages', 'digitdata/validationlabels'],  # Validation Files Here
                              ['digitdata/testimages', 'digitdata/testlabels']],
                             enumeration_func)  # Test Files Here
digit_predictor.training()

face_predictor = Perceptron(70,
                            [['facedata/facedatatrain', 'facedata/facedatatrainlabels'],  # Training Files Here
                             ['facedata/facedatavalidation', 'facedata/facedatavalidationlabels'],
                             # Validation Files Here
                             ['facedata/facedatatest', 'facedata/facedatatestlabels']],
                            enumeration_func)  # Test Files Here

face_predictor.training()
