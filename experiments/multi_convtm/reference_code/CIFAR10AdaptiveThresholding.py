import argparse
import os  # PATCH (multi_convtm)
from tmu.models.classification.vanilla_classifier import TMClassifier
import numpy as np
from keras.datasets import cifar10
import cv2
from tmu.preprocessing.standard_binarizer.binarizer import StandardBinarizer
from time import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_clauses", default=2000, type=int)
    parser.add_argument("--T", default=500, type=int)
    parser.add_argument("--s", default=10.0, type=float)
    parser.add_argument("--max_included_literals", default=32, type=int)
    parser.add_argument("--device", default="GPU", type=str)
    parser.add_argument("--weighted_clauses", default=True, type=bool)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--patch_size", default=10, type=int)

    args = parser.parse_args()

    (X_train_org, Y_train), (X_test_org, Y_test) = cifar10.load_data()
    
    X_train = np.copy(X_train_org)
    X_test = np.copy(X_test_org)
    
    Y_train=Y_train.reshape(Y_train.shape[0])
    Y_test=Y_test.reshape(Y_test.shape[0])

    for i in range(X_train.shape[0]):
            for j in range(X_train.shape[3]):
                    X_train[i,:,:,j] = cv2.adaptiveThreshold(X_train_org[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    for i in range(X_test.shape[0]):
            for j in range(X_test.shape[3]):
                X_test[i,:,:,j] = cv2.adaptiveThreshold(X_test_org[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # PATCH (multi_convtm): tmu 0.8.3 asserts uint32 inputs; values are unchanged (0/1 bits, labels 0-9).
    X_train, X_test = X_train.astype(np.uint32), X_test.astype(np.uint32)
    Y_train, Y_test = Y_train.astype(np.uint32), Y_test.astype(np.uint32)
    if os.environ.get('MC_EPOCHS'): args.epochs = int(os.environ['MC_EPOCHS'])  # PATCH (multi_convtm): calibration runs only

    tm = TMClassifier(
        number_of_clauses=args.num_clauses,
        T=args.T,
        s=args.s,
        max_included_literals=args.max_included_literals,
        platform=args.device,
        weighted_clauses=args.weighted_clauses,
        # PATCH (multi_convtm): removed type_i_ii_ratio=args.type_i_ii_ratio (never defined -> AttributeError); tmu default 1.0
        patch_dim=(args.patch_size, args.patch_size)
    )

    for epoch in range(args.epochs):
        start_training = time()
        tm.fit(X_train, Y_train)
        stop_training = time()

        start_testing = time()
        Y_test_predicted, Y_test_scores = tm.predict(X_test, return_class_sums=True)
        stop_testing = time()

        result_test = 100*(Y_test_scores.argmax(axis=1) == Y_test).mean()

        print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (epoch+1, result_test, stop_training-start_training, stop_testing-start_testing))

        np.savetxt("CIFAR10AdaptiveThresholding_%d_%d_%d_%.1f_%d_%d_%d.txt" % (epoch+1, args.num_clauses, args.T, args.s, args.patch_size, args.max_included_literals, args.weighted_clauses), Y_test_scores, delimiter=',')