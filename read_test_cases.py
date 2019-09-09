import yaml


def read_test_case(name):
    with open('test_cases.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    parameters = data[name]

    return parameters['deadhead_mean'], parameters['deadhead_variance']
