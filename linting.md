src/classifier/__init__.py:1:1: F401 '.mlp_classifier.MLPClassifier' imported but unused
src/classifier/mlp_classifier.py:5:1: I100 Import statements are in the wrong order. 'from encoders import one_hot_encode' should be before 'import numpy' and in a different group.
src/classifier/mlp_classifier.py:6:1: I201 Missing newline between import groups. 'from nn_modules import Module, Softmax' is identified as Third Party and 'from encoders import one_hot_encode' is identified as Third Party.
src/classifier/mlp_classifier.py:10:18: ANN101 Missing type annotation for self in method
src/classifier/mlp_classifier.py:10:80: E501 line too long (85 > 79 characters)
src/classifier/mlp_classifier.py:10:85: ANN204 Missing return type annotation for special method
src/classifier/mlp_classifier.py:27:13: ANN101 Missing type annotation for self in method
src/classifier/mlp_classifier.py:27:20: N803 argument name 'X' should be lowercase
src/classifier/mlp_classifier.py:27:49: ANN001 Missing type annotation for function argument 'batch_size'
src/classifier/mlp_classifier.py:31:80: E501 line too long (81 > 79 characters)
src/classifier/mlp_classifier.py:44:10: N806 variable 'X_batches' in function should be lowercase
src/classifier/mlp_classifier.py:47:18: N806 variable 'X' in function should be lowercase
src/classifier/mlp_classifier.py:62:23: ANN101 Missing type annotation for self in method
src/classifier/mlp_classifier.py:62:30: N803 argument name 'X' should be lowercase
src/classifier/mlp_classifier.py:83:17: ANN101 Missing type annotation for self in method
src/classifier/mlp_classifier.py:83:23: ANN001 Missing type annotation for function argument 'X'
src/classifier/mlp_classifier.py:83:24: N803 argument name 'X' should be lowercase
src/encoders/__init__.py:1:1: F401 '.one_hot.one_hot_encode' imported but unused
src/encoders/one_hot.py:4:20: ANN001 Missing type annotation for function argument 'y'
src/encoders/one_hot.py:4:23: ANN001 Missing type annotation for function argument 'classes'
src/encoders/one_hot.py:4:31: ANN201 Missing return type annotation for public function
src/nn_modules/__init__.py:1:1: F401 '.linear.Linear' imported but unused
src/nn_modules/__init__.py:2:1: F401 '.module.Module' imported but unused
src/nn_modules/__init__.py:3:1: F401 '.relu.ReLU' imported but unused
src/nn_modules/__init__.py:4:1: F401 '.softmax.Softmax' imported but unused
src/nn_modules/linear.py:13:18: ANN101 Missing type annotation for self in method
src/nn_modules/linear.py:13:60: ANN204 Missing return type annotation for special method
src/nn_modules/linear.py:31:74: C812 missing trailing comma
src/nn_modules/linear.py:37:17: ANN101 Missing type annotation for self in method
src/nn_modules/linear.py:59:18: ANN101 Missing type annotation for self in method
src/nn_modules/linear.py:75:16: ANN101 Missing type annotation for self in method
src/nn_modules/module.py:2:17: ANN101 Missing type annotation for self in method
src/nn_modules/module.py:2:23: ANN001 Missing type annotation for function argument 'x'
src/nn_modules/module.py:2:25: ANN201 Missing return type annotation for public function
src/nn_modules/module.py:5:18: ANN101 Missing type annotation for self in method
src/nn_modules/module.py:5:24: ANN001 Missing type annotation for function argument 'd'
src/nn_modules/module.py:5:26: ANN201 Missing return type annotation for public function
src/nn_modules/module.py:8:16: ANN101 Missing type annotation for self in method
src/nn_modules/module.py:8:22: ANN001 Missing type annotation for function argument 'alpha'
src/nn_modules/module.py:8:28: ANN201 Missing return type annotation for public function
src/nn_modules/relu.py:13:18: ANN101 Missing type annotation for self in method
src/nn_modules/relu.py:13:23: ANN204 Missing return type annotation for special method
src/nn_modules/relu.py:16:17: ANN101 Missing type annotation for self in method
src/nn_modules/relu.py:34:18: ANN101 Missing type annotation for self in method
src/nn_modules/relu.py:34:24: ANN001 Missing type annotation for function argument 'd'
src/nn_modules/softmax.py:11:18: ANN101 Missing type annotation for self in method
src/nn_modules/softmax.py:11:23: ANN204 Missing return type annotation for special method
src/nn_modules/softmax.py:14:17: ANN101 Missing type annotation for self in method
src/nn_modules/softmax.py:31:77: C812 missing trailing comma
src/nn_modules/softmax.py:34:18: ANN101 Missing type annotation for self in method
src/nn_modules/softmax.py:34:24: ANN001 Missing type annotation for function argument 'd'
src/nn_modules/softmax.py:51:14: C812 missing trailing comma
src/sanity_test.py:2:1: I201 Missing newline between import groups. 'from sklearn.datasets import make_blobs, make_moons' is identified as Third Party and 'import numpy' is identified as Third Party.
src/sanity_test.py:4:1: I100 Import statements are in the wrong order. 'from classifier import MLPClassifier' should be before 'from sklearn.datasets import make_blobs, make_moons' and in a different group.
src/sanity_test.py:5:1: I201 Missing newline between import groups. 'from nn_modules import Linear, ReLU' is identified as Third Party and 'from classifier import MLPClassifier' is identified as Third Party.
src/sanity_test.py:8:23: ANN201 Missing return type annotation for public function
src/sanity_test.py:9:6: N806 variable 'X' in function should be lowercase
src/sanity_test.py:11:6: N806 variable 'X_test' in function should be lowercase
src/sanity_test.py:16:80: C812 missing trailing comma
src/sanity_test.py:26:24: ANN201 Missing return type annotation for public function
src/sanity_test.py:27:6: N806 variable 'X' in function should be lowercase
src/sanity_test.py:28:6: N806 variable 'X_test' in function should be lowercase
src/sanity_test.py:28:80: E501 line too long (80 > 79 characters)
