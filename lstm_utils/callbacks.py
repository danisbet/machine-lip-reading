from keras import backend as K
import numpy as np
import keras
import editdistance
import csv
import os
from spell import Spell
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
def labels_to_text(labels):
    # 26 is space, 27 is CTC blank char
    text = ''
    for c in labels:
        if c >= 0 and c < 26:
            text += chr(c + ord('a'))
        elif c == 26:
            text += ' '
    return text

def decode(y_pred, input_length, greedy=False, beam_width=10, top_paths=1):
    """Decodes the output of a softmax.
    Can use either greedy search (also known as best path)
    or a constrained dictionary search.
    # Arguments
        y_pred: tensor `(samples, time_steps, num_categories)`
            containing the prediction, or output of the softmax.
        input_length: tensor `(samples, )` containing the sequence length for
            each batch item in `y_pred`.
        greedy: perform much faster best-path search if `true`.
            This does not use a dictionary.
        beam_width: if `greedy` is `false`: a beam search decoder will be used
            with a beam of this width.
        top_paths: if `greedy` is `false`,
            how many of the most probable paths will be returned.
    # Returns
        Tuple:
            List: if `greedy` is `true`, returns a list of one element that
                contains the decoded sequence.
                If `false`, returns the `top_paths` most probable
                decoded sequences.
                Important: blank labels are returned as `-1`.
            Tensor `(top_paths, )` that contains
                the log probability of each decoded sequence.
    """
    #str_list = []
    #print("y_pred in decode", y_pred.shape)
    #print("y_pred in decode", np.squeeze(y_pred).shape)

    # for i, seq in enumerate(np.squeeze(y_pred)):
    #     #print(seq[:,27])
    #     max_ind = np.argmax(seq, axis = 1)
    #     #print(max_ind)
    #     max_ind = labels_to_text(max_ind)
    #     str_list.append(max_ind)
    #print("str_list",str_list)
    #print("input_length",input_length)

    #
    decoded = K.ctc_decode(y_pred=y_pred, input_length=input_length,
                           greedy=greedy, beam_width=beam_width, top_paths=top_paths)
    paths = [path.eval(session=K.get_session()) for path in decoded[0]]
    # print ("I am paths\n", paths)
    # #logprobs  = decoded[1].eval(session=K.get_session())
    spell = Spell(path=CURRENT_PATH+"/dictionary.txt")
    preprocessed = []
    postprocessors=[labels_to_text, spell.correction]
    #for output in str_list:
    for output in paths[0]:
        # out_temp = list(set(output))
        # out = ''
        # for i in reversed(range(len(out_temp))):
        #     out += out_temp[i]
        out = output
        for postprocessor in postprocessors:
            out = postprocessor(out)
        preprocessed.append(out)
    #print("preprocessed",preprocessed)
    return preprocessed


class Statistics(keras.callbacks.Callback):
    def __init__(self, model, x_train, y_train, input_len_train, label_len_train, num_samples_stats=256, output_dir=None):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.input_len_train = input_len_train
        self.label_len_train = label_len_train
        self.output_dir = output_dir
        self.num_sample_stats = num_samples_stats
        if output_dir is not None and not os.path.exists(output_dir):
            os.makedirs(self.output_dir)


    def get_statistics(self, num):
        import tensorflow as tf
        num_left = num
        data = []
        source_str = []

        while num_left > 0:
            num_proc = min(self.x_train.shape[0], num_left)
            input_data = {'the_input': self.x_train[0:num_proc], 'the_labels': self.y_train[0:num_proc],
             'label_length': self.label_len_train[0:num_proc], 'input_length': self.input_len_train[0:num_proc]}
            output_layer = self.model.get_layer('ctc').input[0]
            input_layer = self.model.get_layer('padding1').input
            fn = K.function([input_layer,K.learning_phase()],[output_layer,K.learning_phase()])
            y_pred = fn([input_data['the_input'],0])[0]
            print ("I am y_pred", y_pred.shape)

            decoded_res = decode(y_pred, input_data['input_length'])

            for i in range(0, num_proc):
                source_str.append(labels_to_text(self.y_train[i].astype(int)))
            #for k in reversed(range(len(decoded_res))):
            #   data = []
            for j in range(0, num_proc):
                data.append((decoded_res[j], source_str[j]))
            if num_left == num:
                print("predicted word, source word:", data)


            num_left -= num_proc

        mean_cer, mean_cer_norm    = self.get_mean_character_error_rate(data)
        #mean_wer, mean_wer_norm    = self.get_mean_word_error_rate(data)

        return {
            'samples': num,
            'cer': (mean_cer, mean_cer_norm),
        }

    def get_mean_tuples(self, data, individual_length, func):
        total = 0.0
        total_norm = 0.0
        length = len(data)
        for i in range(0, length):
            val = float(func(data[i][0], data[i][1]))
            total += val
            total_norm += val / individual_length
        return (total / length, total_norm / length)
    def get_mean_character_error_rate(self, data):
        mean_individual_length = np.mean([len(pair[1]) for pair in data])
        return self.get_mean_tuples(data, mean_individual_length, editdistance.eval)

    def get_mean_word_error_rate(self, data):
        mean_individual_length = np.mean([len(pair[1].split()) for pair in data])
        return self.get_mean_tuples(data, mean_individual_length, wer_sentence)


    def on_train_begin(self, logs={}):
        with open(os.path.join(self.output_dir, 'stats.csv'), 'wb') as csvfile:
            csvw = csv.writer(csvfile)
            csvw.writerow(
                ["Epoch", "Samples", "Mean CER", "Mean CER (Norm)"])

    def on_epoch_end(self, epoch, logs={}):
        stats = self.get_statistics(self.num_sample_stats)
        print('\n\n[Epoch %d] Out of %d samples: [CER: %.3f - %.3f] \n'
              % (epoch, stats['samples'], stats['cer'][0], stats['cer'][1]))