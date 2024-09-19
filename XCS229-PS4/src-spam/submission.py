import collections

import numpy as np

import util
import svm


def get_words(message):
    """Get the normalized list of words from a message string.

    This function should split a message into words, normalize them, and return
    the resulting list. For splitting, you should split on spaces. Please use
    split(' ') as your choice of splitting maneuver. 
    For normalization, you should convert everything to lowercase.

    Note for enterprising students:  There are myriad ways to split sentences for
    this algorithm.  For instance, you might want to exclude punctuation (unless
    it's organized in an email address format) or exclude numbers (unless they're
    organized in a zip code or phone number format).  Clearly this can become quite
    complex.  For our purposes, please split using the space character ONLY (ie split(' ')). 
    This is intended to balance your understanding with our ability to autograde the
    assignment.  Thanks and have fun with the rest of the assignment!

    Args:
        message: A string containing an SMS message

    Returns:
       The list of normalized words from the message.

    REMINDER: Please use split(' ') as your choice of splitting maneuver
    """

    # *** START CODE HERE ***
    
    # words = message.split(' ')
    # lower_case = [word.lower() for word in words]

    lower_message = message.lower()
    # punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    # for char in punctuation:
    #     lower_message = lower_message.replace(char, ' ')
    
    words = lower_message.split(' ')
    return words
    # *** END CODE HERE ***


def create_dictionary(messages):
    """Create a dictionary mapping words to integer indices.

    This function should create a dictionary of word to indices using the provided
    training messages. Use get_words to process each message.

    Rare words are often not useful for modeling. Please only add words to the dictionary
    if they occur in at least *five messages*.

    Args:
        messages: A list of strings containing SMS messages

    Returns:
        A python dict mapping words to integers.
    """

    # *** START CODE HERE ***
    my_dict = {}
    for message in messages:
        words = get_words(message)
        array_words = np.array(words)
        unique_words = np.unique(array_words)
        for i in unique_words:
            if i in my_dict:
                my_dict[i] += 1
            else:
                my_dict[i] = 1
    
    common_words = [word for word, count in my_dict.items() if count >= 5]
    common_dict = dict(zip(common_words, range(len(common_words))))
    return common_dict
    # *** END CODE HERE ***


def transform_text(messages, word_dictionary):
    """Transform a list of text messages into a numpy array for further processing.

    This function should create a numpy array that contains the number of times each word
    of the vocabulary appears in each message.
    Each row in the resulting array should correspond to each message
    and each column should correspond to *a word of the vocabulary*.

    Use the provided word dictionary to map words to column indices. Ignore words that
    are not present in the dictionary. Use get_words to get the words for a message.

    Args:
        messages: A list of strings where each string is an SMS message.
        word_dictionary: A python dict mapping words to integers.

    Returns:
        A numpy array marking the words present in each message.
        Where the component (i,j) is the number of occurrences of the
        j-th vocabulary word in the i-th message.
    """
    # *** START CODE HERE ***
    # def check_for_int(messages):
    #     for message in messages:
    #         if isinstance(message, int):
    #             return True  # Found an integer
    #     return False
    
    # contains_int = check_for_int(messages)
    # print(f"Does messages contain an int? {contains_int}")
    # print(type(messages))
    # print(messages)

    count_array = np.zeros((len(messages), len(word_dictionary)))
    for count, message in enumerate(messages):
        words = get_words(message)
        unique_words = np.unique(np.array(words))
        for word in unique_words:
            if word in word_dictionary:
                count_array[count, word_dictionary[word]] += 1
    
    return count_array

    # *** END CODE HERE ***


def fit_naive_bayes_model(matrix, labels):
    """Fit a naive bayes model.

    This function should fit a Naive Bayes multinomial event model with Laplace smoothing given a training matrix and labels.

    The function should return the state of that model as a dictionary with the following keys:

        phi_{y=1} - the model parameter that matches p(y=1)
        phi_{y=0} - the model parameter that matches p(y=0)
        phi_{k|y=1} - the model parameter that matches p(x_j = k|y = 1) (for any j)
        phi_{k|y=0} - the model parameter that matches p(x_j = k|y = 0) (for any j)

    Refer to the remark from the assignment's pdf, about how to represent the parameters to avoid underflow. 

    Args:
        matrix: A numpy array containing word counts for the training data
        labels: The binary (0 or 1) labels for that training data

    Returns: The trained model
    """

    model = dict.fromkeys(['phi_{y=0}', 'phi_{y=1}', 'phi_{k|y=0}', 'phi_{k|y=1}'])

    # *** START CODE HERE ***
    labels = np.array(labels)
    n = len(labels)

    phi_y1 = np.sum(labels)/n

    model['phi_{y=1}'] = np.log(phi_y1)
    model['phi_{y=0}'] = np.log(1 - phi_y1)

    spam_matrix = matrix[labels == 1]
    ham_matrix = matrix[labels == 0]

    spam_sum = np.sum(spam_matrix, axis=0)
    ham_sum = np.sum(ham_matrix, axis=0)

    no_spam_words = np.sum(spam_sum)
    no_ham_words = np.sum(ham_sum)

    V = matrix.shape[1]

    model['phi_{k|y=1}'] = np.log((1 + spam_sum) / (V + no_spam_words))
    model['phi_{k|y=0}'] = np.log((1 + ham_sum) / (V + no_ham_words))

    return model
    # *** END CODE HERE ***


def predict_from_naive_bayes_model(model, matrix):
    """Use a Naive Bayes model to compute predictions for a target matrix.

    This function should be able to predict on the models that fit_naive_bayes_model
    outputs.

    Args:
        model: A trained model from fit_naive_bayes_model
        matrix: A numpy array containing word counts

    Returns: A numpy array containing the predictions from the model
    """
    # *** START CODE HERE ***
    log_p_x_given_y1 = matrix @ model['phi_{k|y=1}']
    log_p_x_given_y0 = matrix @ model['phi_{k|y=0}']

    y1_sum_log = log_p_x_given_y1 + model['phi_{y=1}']
    y0_sum_log = log_p_x_given_y0 + model['phi_{y=0}']

    prediction = 1 / (1 + np.exp(y0_sum_log - y1_sum_log))
    prediction = np.where(prediction >= 0.5, 1, 0)

    return prediction
    # *** END CODE HERE ***


def get_top_five_naive_bayes_words(model, dictionary):
    """Compute the top five words that are most indicative of the spam (i.e positive) class.

    Ues the metric given in part-c as a measure of how indicative a word is.
    Return the words in sorted form, with the most indicative word first.

    Args:
        model: The Naive Bayes model returned from fit_naive_bayes_model
        dictionary: A mapping of word to integer ids

    Returns: A list of the top five most indicative words in sorted order with the most indicative first
    """
    # *** START CODE HERE ***
    indicativeness = model['phi_{k|y=1}'] - model['phi_{k|y=0}']

    words = np.array(list(dictionary.keys()))

    largest5_indices = np.argpartition(indicativeness, -5)[-5:]
    
    print(largest5_indices)

    largest5_words = words[largest5_indices]

    return list(largest5_words)
    # *** END CODE HERE ***


def compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, radius_to_consider):
    """Compute the optimal SVM radius using the provided training and evaluation datasets.

    You should only consider radius values within the radius_to_consider list.
    You should use accuracy as a metric for comparing the different radius values.

    Args:
        train_matrix: The word counts for the training data
        train_labels: The spma or not spam labels for the training data
        val_matrix: The word counts for the validation data
        val_labels: The spam or not spam labels for the validation data
        radius_to_consider: The radius values to consider

    Returns:
        The best radius which maximizes SVM accuracy.
    """
    # *** START CODE HERE ***
    accuracies = []
    for radius in radius_to_consider:
        prediction = svm.train_and_predict_svm(train_matrix, train_labels, val_matrix, radius)
        accuracies.append(np.sum(prediction == val_labels)/len(prediction))

    print(accuracies)
    max_index = accuracies.index(max(accuracies))
    return radius_to_consider[max_index]

    # *** END CODE HERE ***


def main():
    train_messages, train_labels = util.load_spam_dataset('spam_train.tsv')
    val_messages, val_labels = util.load_spam_dataset('spam_val.tsv')
    test_messages, test_labels = util.load_spam_dataset('spam_test.tsv')

    dictionary = create_dictionary(train_messages)

    print('Size of dictionary: ', len(dictionary))

    util.write_json('spam_dictionary_(soln)', dictionary)

    train_matrix = transform_text(train_messages, dictionary)

    np.savetxt('spam_sample_train_matrix_(soln)', train_matrix[:100,:])

    val_matrix = transform_text(val_messages, dictionary)
    test_matrix = transform_text(test_messages, dictionary)

    naive_bayes_model = fit_naive_bayes_model(train_matrix, train_labels)

    naive_bayes_predictions = predict_from_naive_bayes_model(naive_bayes_model, test_matrix)

    np.savetxt('spam_naive_bayes_predictions_(soln)', naive_bayes_predictions)

    naive_bayes_accuracy = np.mean(naive_bayes_predictions == test_labels)

    print('Naive Bayes had an accuracy of {} on the testing set'.format(naive_bayes_accuracy))

    top_5_words = get_top_five_naive_bayes_words(naive_bayes_model, dictionary)

    print('The top 5 indicative words for Naive Bayes are: ', top_5_words)

    util.write_json('spam_top_indicative_words_(soln)', top_5_words)

    optimal_radius = compute_best_svm_radius(train_matrix, train_labels, val_matrix, val_labels, [0.01, 0.1, 1, 10])

    util.write_json('spam_optimal_radius_(soln)', optimal_radius)

    print('The optimal SVM radius was {}'.format(optimal_radius))

    svm_predictions = svm.train_and_predict_svm(train_matrix, train_labels, test_matrix, optimal_radius)

    svm_accuracy = np.mean(svm_predictions == test_labels)

    print('The SVM model had an accuracy of {} on the testing set'.format(svm_accuracy, optimal_radius))


if __name__ == "__main__":
    main()
