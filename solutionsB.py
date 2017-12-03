#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import nltk
import math
import time


from itertools import chain
from collections import Counter
from nltk.tag import CRFTagger

START_SYMBOL = '*'
STOP_SYMBOL = 'STOP'
RARE_SYMBOL = '_RARE_'
RARE_WORD_MAX_FREQ = 5
LOG_PROB_OF_ZERO = -1000


# TODO: IMPLEMENT THIS FUNCTION - DONE
# Receives a list of tagged sentences and processes each sentence to generate a list of words and a list of tags.
# Each sentence is a string of space separated "WORD/TAG" tokens, with a newline character in the end.
# Remember to include start and stop symbols in yout returned lists, as defined by the constants START_SYMBOL and STOP_SYMBOL.
# brown_words (the list of words) should be a list where every element is a list of the tags of a particular sentence.
# brown_tags (the list of tags) should be a list where every element is a list of the tags of a particular sentence.
def split_wordtags(brown_train):
    brown_words = []
    brown_tags = []

    start = START_SYMBOL + " " + START_SYMBOL + " "
    for line in brown_train:
        splited_line = (start + line + " " + STOP_SYMBOL).split()
        words, tags = zip(*(s.split("/") if "/" in s else [s, s] for s in splited_line))
        brown_words.append(list(words))
        brown_tags.append(list(tags))

    return brown_words, brown_tags


# TODO: IMPLEMENT THIS FUNCTION - DONE
# This function takes tags from the training data and calculates tag trigram probabilities.
# It returns a python dictionary where the keys are tuples that represent the tag trigram, and the values are the log probability of that trigram
def calc_trigrams(brown_tags):
    q_values = {}

    bigrams = list(chain(*[list(nltk.bigrams(i)) for i in brown_tags]))
    trigrams = list(chain(*[list(nltk.trigrams(i)) for i in brown_tags]))
    biprop = Counter(bigrams)
    triprop = Counter(trigrams)
    for i in triprop:
        q_values[i] = math.log2(triprop[i] / (1.0 * biprop[i[0:2]]))

    return q_values


# This function takes output from calc_trigrams() and outputs it in the proper format
def q2_output(q_values, filename):
    outfile = open(filename, "w")
    trigrams = q_values.keys()

    # trigrams.sort()
    trigrams = sorted(trigrams)
    for trigram in trigrams:
        output = " ".join(['TRIGRAM', trigram[0], trigram[1], trigram[2], str(q_values[trigram])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION - DONE
# Takes the words from the training data and returns a set of all of the words that occur more than 5 times (use RARE_WORD_MAX_FREQ)
# brown_words is a python list where every element is a python list of the words of a particular sentence.
# Note: words that appear exactly 5 times should be considered rare!
def calc_known(brown_words):
    all_words = list(chain(*brown_words))
    counter = Counter(all_words)
    known_words = set([elem for elem in counter if elem != START_SYMBOL and elem != STOP_SYMBOL and
                       counter[elem] > RARE_WORD_MAX_FREQ])
    return known_words


# TODO: IMPLEMENT THIS FUNCTION - DONE
# Takes the words from the training data and a set of words that should not be replaced for '_RARE_'
# Returns the equivalent to brown_words but replacing the unknown words by '_RARE_' (use RARE_SYMBOL constant)
def replace_rare(brown_words, known_words):
    brown_words_rare = brown_words
    for i in brown_words_rare:
        for j in range(len(i)):
            if i[j] not in known_words:
                i[j] = RARE_SYMBOL

    return brown_words_rare


# This function takes the ouput from replace_rare and outputs it to a file
def q3_output(rare, filename):
    outfile = open(filename, 'w')
    for sentence in rare:
        outfile.write(' '.join(sentence[2:-1]) + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION - DONE
# Calculates emission probabilities and creates a set of all possible tags
# The first return value is a python dictionary where each key is a tuple in which the first element is a word
# and the second is a tag, and the value is the log probability of the emission of the word given the tag
# The second return value is a set of all possible tags for this data set
def calc_emission(brown_words_rare, brown_tags):
    e_values = {}

    zipped = list(chain(*[list(zip(brown_words_rare[i], brown_tags[i])) for i in range(0, len(brown_words_rare))]))
    words_tags_counter = Counter(zipped)
    tags = chain(*brown_tags)
    tags_counter = Counter(tags)
    taglist = set(tags_counter.keys())

    for i in words_tags_counter:
        e_values[i] = math.log2(words_tags_counter[i] / (1.0 * tags_counter[i[1]]))

    return e_values, taglist


# This function takes the output from calc_emissions() and outputs it
def q4_output(e_values, filename):
    outfile = open(filename, "w")
    emissions = e_values.keys()
    # emissions.sort()  for python 2
    emissions = sorted(emissions)
    for item in emissions:
        output = " ".join([item[0], item[1], str(e_values[item])])
        outfile.write(output + '\n')
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function takes data to tag (brown_dev_words), a set of all possible tags (taglist), a set of all known words (known_words),
# trigram probabilities (q_values) and emission probabilities (e_values) and outputs a list where every element is a tagged sentence 
# (in the WORD/TAG format, separated by spaces and with a newline in the end, just like our input tagged data)
# brown_dev_words is a python list where every element is a python list of the words of a particular sentence.
# taglist is a set of all possible tags
# known_words is a set of all known words
# q_values is from the return of calc_trigrams()
# e_values is from the return of calc_emissions()
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. Remember also that the output should not contain the "_RARE_" symbol, but rather the
# original words of the sentence!
def viterbi(brown_dev_words, taglist, known_words, q_values, e_values):
    tagged = []

    ################################
    temp_brown_dev_words = brown_dev_words[:]

    for sentence in temp_brown_dev_words:

        temp = []
        t = [START_SYMBOL, START_SYMBOL, 0]  # make it tuple, like tuple(t) and use it as key in q_values

        for words in sentence:
            d_word_tag_values = {}
            if (words in known_words):
                for key in e_values:
                    if (key[0] == words):
                        t[-1] = key[-1]
                        if (q_values.get(tuple(t), LOG_PROB_OF_ZERO) != LOG_PROB_OF_ZERO):
                            d_word_tag_values[(key, tuple(t))] = e_values[key] + q_values[tuple(t)]
                        else:
                            d_word_tag_values[(key, None)] = e_values[key] + LOG_PROB_OF_ZERO
            else:
                for key in e_values:
                    if (key[0] == RARE_SYMBOL):
                        t[-1] = key[-1]
                        if (q_values.get(tuple(t), LOG_PROB_OF_ZERO) != LOG_PROB_OF_ZERO):
                            d_word_tag_values[(key, tuple(t))] = e_values[key] + q_values[tuple(t)]
                        else:
                            d_word_tag_values[(key, None)] = e_values[key] + LOG_PROB_OF_ZERO

            key_tuple_state = max(d_word_tag_values, key=d_word_tag_values.get)
            t[-1] = key_tuple_state[0][-1]

            temp.append((words, t[-1]))

            t.append(0)
            t = t[1:]

        tagged.append(temp)
    ################################

    return tagged


# This function takes the output of viterbi() and outputs it to file
def q5_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        ################################################################### courtesy Pushpendra pratap
        temp = []
        for i in sentence:
            temp.append('/'.join(i))
        outfile.write(' '.join(temp[:]) + '\n')
        ###################################################################
#        outfile.write(sentence)
    outfile.close()


# TODO: IMPLEMENT THIS FUNCTION
# This function uses nltk to create the taggers described in question 6
# brown_words and brown_tags is the data to be used in training
# brown_dev_words is the data that should be tagged
# The return value is a list of tagged sentences in the format "WORD/TAG", separated by spaces. Each sentence is a string with a 
# terminal newline, not a list of tokens. 
def nltk_tagger(brown_words, brown_tags, brown_dev_words):
    # Hint: use the following line to format data to what NLTK expects for training
    training = [zip(brown_words[i], brown_tags[i]) for i in range(0, len(brown_words))]

    # IMPLEMENT THE REST OF THE FUNCTION HERE
    tagged = []

    ######################################################################
    default_tagger = nltk.DefaultTagger('NOUN')
    bigram_tagger = nltk.BigramTagger(training, backoff=default_tagger)
    trigram_tagger = nltk.TrigramTagger(training, backoff=bigram_tagger)
    for i in brown_dev_words:
        tagged.append(trigram_tagger.tag(i))
    ######################################################################

    return tagged


# This function takes the output of nltk_tagger() and outputs it to file
def q6_output(tagged, filename):
    outfile = open(filename, 'w')
    for sentence in tagged:
        # outfile.write(sentence)

        ###################################################################
        temp = []
        for i in sentence:
            temp.append('/'.join(i))
        outfile.write(' '.join(temp[:]) + '\n')
        ###################################################################

    outfile.close()

DATA_PATH = 'data/'
OUTPUT_PATH = 'output/'


def main():
    # start timer
    for item in ["UD_Ukrainian", "Brown", ]:

        print("in process " + item)
        # open Brown training data
        infile = open(DATA_PATH + item + "_tagged_train.txt", "r", encoding="utf-8")
        brown_train = infile.readlines()
        infile.close()

        # split words and tags, and add start and stop symbols (question 1)
        brown_words, brown_tags = split_wordtags(brown_train)

        # calculate tag trigram probabilities (question 2)
        q_values = calc_trigrams(brown_tags)

        # question 2 output
        q2_output(q_values, OUTPUT_PATH + item + '_B2.txt')

        # calculate list of words with count > 5 (question 3)
        known_words = calc_known(brown_words)

        # get a version of brown_words with rare words replace with '_RARE_' (question 3)
        brown_words_rare = replace_rare(brown_words, known_words)

        # question 3 output
        q3_output(brown_words_rare, OUTPUT_PATH + item + "_B3.txt")

        # calculate emission probabilities (question 4)
        e_values, taglist = calc_emission(brown_words_rare, brown_tags)

        # question 4 output
        q4_output(e_values, OUTPUT_PATH + item + "_B4.txt")

        # delete unneceessary data
        del brown_train
        del brown_words_rare

        # open Brown development data (question 5)
        infile = open(DATA_PATH + item + "_test.txt", "r")
        brown_dev = infile.readlines()
        infile.close()

        # format Brown development data here
        brown_dev_words = []
        for sentence in brown_dev:
            brown_dev_words.append(sentence.split(" ")[:-1])

        # do viterbi on brown_dev_words (question 5)
        viterbi_tagged = viterbi(brown_dev_words, taglist, known_words, q_values, e_values)

        # question 5 output
        q5_output(viterbi_tagged, OUTPUT_PATH + item + "_B5.txt")

        # # do nltk tagging here
        # nltk_tagged = nltk_tagger(brown_words, brown_tags, brown_dev_words)
        #
        # # question 6 output
        # q6_output(nltk_tagged, OUTPUT_PATH + item + "_B6.txt")

    for item in ["Brown", "UD_Ukrainian"]:
        print("in crf process " + item)
        # open Brown training data
        infile = open(DATA_PATH + item + "_tagged_train.txt", "rb", encoding="utf-8")
        brown_train = infile.readlines()
        infile.close()

        brown_words, brown_tags = split_wordtags(brown_train)
        train_words_tags = []
        ct = CRFTagger()
        for i in range(len(brown_words)):
            tmp = []
            for j in range(len(brown_words[i])):
                tmp.append((brown_words[i][j].decode('utf-8'), brown_tags[i][j].decode('utf-8')))
            train_words_tags.append(tmp)

        ct.train(train_words_tags, u'model.crf.tagger')

        # open Brown development data (question 5)
        infile = open(DATA_PATH + item + "_test.txt", "r")
        brown_dev = infile.readlines()
        infile.close()

        # format Brown development data here
        tests_words = []
        for sentence in brown_dev:
            tests_words.append([i for i in sentence.split(" ")[:-1]])

        result_cfg = ct.tag_sents(tests_words)
        with open(OUTPUT_PATH + item + "_CFG.txt", "w") as file:
            for line in result_cfg:
                for word in line:
                    file.write(word[0] + "/" + word[1] + " ")
                file.write("\n")

        # print total time to run Part B
        print("Part B time: ", str(time.clock()), ' sec')


if __name__ == "__main__": main()
