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

T0 = [
        270581,270581,270581,270581,
        270580,270581,270580,270580,
        270581,270580,270581,270581,
        270581,270580,270580,270580
]

T1 = [
        307824,307822,307825,307821,
        307821,307823,307824,307823,
        307824,307823,307824,307823,
        307823,307824,307823,307823
]

import numpy as np

folder_index = int(input("Folder index ?"))
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
num_traces = fifo_in.read(4)
num_traces = np.frombuffer(num_traces, count= 1 , dtype = np.uint32)[0]

std_special = open("/home/julien/PoI_k"+str(folder_index),"a+")
std_special.write("Plain Y1 Y2 \n")

# Send start of attack string
start = np.array([10, 46, 10],dtype=np.uint8)
fifo_out.write(start.tobytes())

# Main iteration
for iteration in range(num_traces):

    fifo_in.flush()
    fifo_out.flush()

    # Read trace
    plaintext = np.frombuffer(fifo_in.read(16),count =  16, dtype=np.uint8)
    ciphertext = np.frombuffer(fifo_in.read(16),count =  16, dtype=np.uint8)
    shuffle0 =  np.frombuffer(fifo_in.read(16),count =  16, dtype=np.uint8)
    shuffle10 =  np.frombuffer(fifo_in.read(16),count =  16, dtype=np.uint8)
    offset = np.frombuffer(fifo_in.read(16),count =  16, dtype=np.uint8)
    samples = np.frombuffer(fifo_in.read(1704402 * 4),count = 1704402, dtype=np.float32)

    Plain = plaintext[0]
    Y2 = samples[T1[folder_index]]
    Y1 = samples[T0[folder_index]]

    std_special.write(str(Plain) +" "+ str(Y1) + " " + str(Y2) + "\n")

    resultat = np.array([np.arange(256,dtype=np.uint8) for _ in range(16)])

    # Send result
    fifo_out.write(attacked_subkey.tobytes())
    fifo_out.write(resultat.tobytes())

# Close the two FIFOs
fifo_in.close()
fifo_out.close()
