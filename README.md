# rusty-titanic
Attempt at solving the Titanic ML challenge using Rust & rusty-machine.

https://www.kaggle.com/c/titanic

$ ./target/release/gaussian_accuracy
2019-11-20 09:50:18,844 INFO  [gaussian_accuracy] accuracy: 0.7798507462686567
2019-11-20 09:50:18,844 INFO  [gaussian_accuracy] Duration: 308ms 503us 444ns

$ ./target/release/svm_accuracy
2019-11-20 09:50:25,573 INFO  [svm_accuracy] accuracy: 0.8283582089552238
2019-11-20 09:50:25,573 INFO  [svm_accuracy] Duration: 717ms 275us 977ns

$ ./target/release/logistic_regression_accuracy
2019-11-20 09:50:33,570 INFO  [logistic_regression_accuracy] accuracy: 0.8171641791044776
2019-11-20 09:50:33,570 INFO  [logistic_regression_accuracy] Duration: 265ms 315us 228ns

$ ./target/release/nnet_accuracy
2019-11-20 09:50:37,848 INFO  [nnet_accuracy] row_accuracy: 0.8097014925373134
2019-11-20 09:50:37,848 INFO  [nnet_accuracy] Duration: 247ms 122us 276ns
