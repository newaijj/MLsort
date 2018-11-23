[X_CV X_CV_sorted] = generateData(200, 10, 0, 99);
save X_CV.mat X_CV;
save X_CV_sorted.mat X_CV_sorted;

[X_test X_test_sorted] = generateData(200, 10, 0, 99);
save X_test.mat X_test;
save X_test_sorted.mat X_test_sorted;
