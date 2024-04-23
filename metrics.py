

def character_accuracy(predicted, target):
    '''
    Calculate the characteer-level accuracy of the input string compared to target string

    :param: target: The target string
    :param: predicted: The predicted string
    :return: The accuracy as a percentage
    '''

    if len(target) == 0:
        return 0.0 # or should we print error message

    correct = sum(1 for i, j in zip(target, predicted) if i == j)
    return (correct / len(target))