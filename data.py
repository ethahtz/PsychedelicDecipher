from datasets import load_dataset
from utils import CHAR_SPACE
import pickle


def preprocess_str(text):
    result = text.lower()

    # Filter out all non-number/alphabetical characters except space
    result = list([val for val in result if val in CHAR_SPACE])
    result = ''.join(result)

    # Replace multiple spaces with one space
    result = result.split()
    result = ' '.join(result)

    return result


def get_datasets(preprocessed_train="data/train_data.pkl"):
    
    dataset = load_dataset("wikitext", 'wikitext-103-raw-v1')

    if preprocessed_train is not None:
        print("Loading preprocessed train data...")
        with open(preprocessed_train, 'rb') as f:
            accumulated_train_data = pickle.load(f)
    else:
        accumulated_train_data = []
            
        for i in range(len(dataset['train'])):
            curr_text = dataset['train'][i]['text']
            processed = preprocess_str(curr_text)
            accumulated_train_data.append(processed)
        
        accumulated_train_data = ' '.join(accumulated_train_data)

    accumulated_test_data = ''

    for i in range(len(dataset['test'])):
        curr_text = dataset['test'][i]['text']
        processed = preprocess_str(curr_text)
        accumulated_test_data += processed

    return accumulated_train_data, accumulated_test_data