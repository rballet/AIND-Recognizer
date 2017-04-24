import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Likelihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []

    # TODO implement the recognizer
    for idx, test_word in enumerate(test_set.wordlist):
        X, lengths = test_set.get_item_Xlengths(idx)
        prob_dict = {}
        best_logL = -float("Inf")
        best_guess = ''
        for model_word in models.keys():
            try:
                logL = models[model_word].score(X,lengths)
                prob_dict[model_word] = logL
                if logL > best_logL:
                    best_guess, best_logL = model_word, logL
            except:
                prob_dict[model_word] = best_logL
        probabilities.append(prob_dict)
        guesses.append(best_guess)
    return probabilities, guesses
