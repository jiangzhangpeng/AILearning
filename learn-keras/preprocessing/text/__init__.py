#encoding:utf-8
import keras
str = 'Transform a list of num_samples sequences (lists of scalars) into a 2D Numpy array of shape  ' \
      '(num_samples, num_timesteps). num_timesteps is either the maxlen argument if provided, or the ' \
      'length of the longest sequence otherwise. Sequences that are shorter than num_timesteps are padded ' \
      'with value at the end. Sequences longer than num_timesteps are truncated so that it fits the ' \
      'desired length. Position where padding or truncation happens is determined by padding or truncating, ' \
      'respectively.Return: 2D Numpy array of shape (num_samples, num_timesteps).'
#text_to_word_sequence
t2w = keras.preprocessing.text.text_to_word_sequence(str)
print(t2w)

#one_hot
oh = keras.preprocessing.text.one_hot(str,1000)
print(oh)

