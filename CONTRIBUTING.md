# Contributing

Run `python -m unittest discover -s tests -v` and the quickstart before proposing a change. Keep raw recordings, local absolute paths, credentials and model weights out of commits.

For model changes, preserve input cases, targets, split availability and evaluation weights. Compare against recent linear and persistence. Report all declared folds, failures, unsupported cases and censored RUL medians. Do not select a configuration on test RMSE and describe the same test as independent validation.

Good next contributions include an input adapter with independently validated circuit conversion, tests on additional devices, and a controlled small CNN/LSTM comparison. A CNN is planned work, not an existing project result.

Use synthetic/minimal examples in issues. For measurement concerns, provide units, the circuit/measurement method and acquisition timing rather than private raw files.
