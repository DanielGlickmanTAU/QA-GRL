from experiments.test_race import Test

#used because pychamrm is problematic with running the same tests twice in the same time.
if __name__ == '__main__':
    print('starting test')
    Test().test_race_classification_params()