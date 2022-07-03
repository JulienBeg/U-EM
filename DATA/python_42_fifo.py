#! /usr/bin/python3

"""
%
% Framework for developping attacks in Matlab under Unix
% for the DPA contest V4, AES256 RSM implementation
%
% Version 1, 24/07/2013
%
% Guillaume Duc <guillaume.duc@telecom-paristech.fr>
%
"""

import numpy as np
import scipy
import scipy.io

# FIFO filenames (TODO: adapt them)
#fifo_in_filename = 'fifo_from_wrapper'
#fifo_out_filename = 'fifo_to_wrapper'
fifo_in_filename = 'python_42_fifo.py_from_wrapper'
fifo_out_filename = 'python_42_fifo.py_to_wrapper'

# Number of the attacked subkey
# TODO: adapt it
attacked_subkey = np.uint8(0)

# Open the two communication FIFO
fifo_in = open(fifo_in_filename, 'rb')
fifo_out = open(fifo_out_filename,'wb')

# Retrieve the number of traces
#num_traces = np.fromfile(fifo_in, count=1, dtype=np.uint32)
num_traces = fifo_in.read(4)
num_traces = np.frombuffer(num_traces, count= 1 , dtype = np.uint32)[0]

#std_special = open("/home/julien/std_special.txt","w+")
#std_special.write("OK Ca fonctionnne jusqu'ici =) " + str(num_traces))
#std_special.close()

# Send start of attack string
start = np.array([10, 46, 10],dtype=np.uint8)
#start.tofile(fifo_out)
fifo_out.write(start.tobytes())

matrixTra = np.zeros((num_traces,1704402),dtype=np.float32)
matrixOffset = np.zeros((num_traces,16),dtype=np.uint8)
matrixPlaintext = np.zeros((num_traces,16),dtype=np.uint8)

# Main iteration
for iteration in range(num_traces):

    """
    fifo_in.close()
    fifo_in = open(fifo_in_filename, 'rb')

    fifo_out.close()
    fifo_out = open(fifo_out_filename,'ab')
    """
    fifo_in.flush()
    fifo_out.flush()


    # Read trace
    """
    plaintext = np.fromfile(fifo_in,count =  16, dtype=np.uint8) #% 16x1 uint8
    ciphertext = np.fromfile(fifo_in,count =  16, dtype=np.uint8) #% 16x1 uint8
    offset = np.fromfile(fifo_in,count =  1, dtype=np.uint8) #% 1x1 uint8
    samples = np.fromfile(fifo_in,count = 435002, dtype=np.int8) #% 435002x1 int8
    """

    plaintext = np.frombuffer(fifo_in.read(16),count =  16, dtype=np.uint8)
    ciphertext = np.frombuffer(fifo_in.read(16),count =  16, dtype=np.uint8)
    shuffle0 =  np.frombuffer(fifo_in.read(16),count =  16, dtype=np.uint8)
    shuffle10 =  np.frombuffer(fifo_in.read(16),count =  16, dtype=np.uint8)
    offset = np.frombuffer(fifo_in.read(16),count =  16, dtype=np.uint8)
    samples = np.frombuffer(fifo_in.read(1704402 * 4),count = 1704402, dtype=np.float32)

    matrixTra[iteration] = samples
    matrixOffset[iteration] = offset
    matrixPlaintext[iteration] = plaintext


    """
    % TODO: put your attack code here
    %
    %  Must produce bytes which is a 256 lines x 16 columns array
    % (matrix) where for each byte of the attacked subkey (the columns of
    % the array), all the 256 possible values of this byte are sorted
    % according to their probability (first position: most probable, last:
    % least probable), i.e. if your attack is successful, the value of the
    % key is the first line of the array.
    """

    resultat = np.array([np.arange(256,dtype=np.uint8) for _ in range(16)])

    # Send result

    """
    attacked_subkey.tofile(file_out)
    resultat.tofile(fifo_out)
    """
    fifo_out.write(attacked_subkey.tobytes())
    fifo_out.write(resultat.tobytes())

scipy.io.savemat('matrixTra.mat',{'sample':matrixTra, 'offset':matrixOffset, 'plain':matrixPlaintext})

# Close the two FIFOs
fifo_in.close()
fifo_out.close()
