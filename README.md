# rusty-titanic
Attempt at solving the Titanic ML challenge using Rust & rusty-machine.

 - [Kaggle Titanic](https://www.kaggle.com/c/titanic)
 - [rusty-machine](https://github.com/AtheMathmo/rusty-machine)

```
$ ./target/release/gaussian_accuracy
2019-11-20 09:50:18,844 INFO  [gaussian_accuracy] accuracy: 0.7611940298507462
2019-11-20 09:50:18,844 INFO  [gaussian_accuracy] Duration: 353ms 807us 671ns
$ ./target/release/svm_accuracy
2019-11-20 09:50:25,573 INFO  [svm_accuracy] accuracy: 0.8432835820895522
2019-11-20 09:50:25,573 INFO  [svm_accuracy] Duration: 806ms 776us 437ns
$ ./target/release/logistic_regression_accuracy
2019-11-20 09:50:33,570 INFO  [logistic_regression_accuracy] accuracy: 0.8544776119402985
2019-11-20 09:50:33,570 INFO  [logistic_regression_accuracy] Duration: 354ms 805us 183ns
$ ./target/release/nnet_accuracy
2019-11-20 09:50:37,848 INFO  [nnet_accuracy] row_accuracy: 0.8283582089552238
2019-11-20 09:50:37,848 INFO  [nnet_accuracy] Duration: 678ms 763us 652ns
```
