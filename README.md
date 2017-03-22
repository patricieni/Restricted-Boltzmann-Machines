# Restricted-Boltzmann-Machines
Branch to test different things on RBM


input_data_test = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    input_data_1_test = np.array([1, 1, 1, 1,
                                  1, 1, 1, 1,
                                  1, 1, 1, 1,
                                  1, 1, 1, 1])
    input_data_2_test = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])


    # Reconstruct partial input
    i = 0
    c1 = 0
    for i in range(0, 10000):
        input_data = np.array([1, 1, 1, 1,
                               0, 0, 0, 0,
                               1, 1, 1, 1,
                               0, 0, 0, 0])

        reconstructed_data = rbm.gibbs(4, input_data)

        if i > 1000:
            if np.sum((reconstructed_data - input_data_test) ** 2) == 0:
                c1 += 1
        i += 1
    print("Ex 1: Reconstructed Data from last step: ", reconstructed_data.astype(int))
    print("Ex 1: Probability for correct state. Should be close to 1 ", c1/9000)

    j = 0
    c2 = 0
    for j in range(0, 10000):
        input_data_1 = np.array([0.5, 0.5, 0.5, 0.5,
                                 0.5, 1, 1, 0.5,
                                 0.5, 1, 1, 0.5,
                                 0.5, 0.5, 0.5, 0.5
                                 ])

        rec_data = rbm.gibbs(4, input_data_1)

        if j > 1000:
            if np.sum((rec_data - input_data_1_test) ** 2) == 0:
                c2 += 1
        j += 1
    print("Ex 2: Reconstructed Data from last step: ", rec_data.astype(int))
    print("Ex 2: Probability to get matrix of ones. There are other possibilities as well. 8 to be more precise. 12 % ", c2/9000)

    m = 0
    c3 = 0
    for m in range(0, 10000):
        input_data_2 = np.array([0.5, 0.5, 0, 0.5,
                                 0.5, 0.5, 1, 0.5,
                                 0.5, 0.5, 1, 0.5,
                                 0.5, 0.5, 0, 0.5
                                 ])

        rec_data2 = rbm.gibbs(4, input_data_2)

        if m > 1000:
            if np.sum((rec_data2 - input_data_2_test) ** 2) == 0:
                c3 += 1
        m += 1
    print("Ex 3: Reconstructed Data from last step: ", rec_data2.astype(int))
    print("Ex 3: Probability for correct state. Should be  close to 1 ", c3/9000)


    plt.show()