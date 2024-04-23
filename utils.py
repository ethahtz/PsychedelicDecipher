import random

CHAR_SPACE = [chr(i) for i in range(ord('0'), ord('9')+1)] + [chr(i) for i in range(ord('a'), ord('z')+1)] + [' ']
ENGLISH_PRIOR_ORDER = [' ', 'e', 't', 'a', 'i', 'n', 'o', 'r', 's', 'h', 'l', 'd', 'c', 'u', 'm', 'f', 'g', 'p', 'w', 'b', 'y', 'v', 'k', '1', '0', '2', '9', 'j', 'x', '3', '5', '8', '4', 'z', '6', '7', 'q']

class Subst_cipher():
    def __init__(self, key=None, domain=[chr(i) for i in range(ord('0'), ord('9')+1)] + [chr(i) for i in range(ord('a'), ord('z')+1)]):
        if key == None:
            key = [ch for ch in domain]
            random.shuffle(key)

        self.p2c = {plain:cipher for plain, cipher in zip(domain, key)}
        self.c2p = {cipher:plain for plain, cipher in zip(domain, key)}

    def encrypt(self, plaintext):
        return [self.p2c[ch] if ch in self.p2c else ch for ch in plaintext]

    def decrypt(self, ciphertext):
        return [self.c2p[ch] if ch in self.p2c else ch for ch in ciphertext]


def compute_frequecy(text, space=CHAR_SPACE):
    """
    text: list of strings or string
    """
    count = {c: 0 for c in space}
    total = 0

    if isinstance(text, list):
        for line in text:
            for c in line:
                count[c] += 1
                total += 1
    else:
        for c in text:
            count[c] += 1
            total += 1

    freq_sorted_vocab = sorted(count, key=lambda x: count[x], reverse=True)

    freq = {c: count[c] / total for c in freq_sorted_vocab}

    return freq, freq_sorted_vocab


def unigram_freq_decipher(cipher, english_prior_order=ENGLISH_PRIOR_ORDER):

    _, cipher_freq_order_list = compute_frequecy(cipher)

    english_prior_order = list(range(0, 37))

    lis = []
    for i in range(len(cipher)):
        idx = cipher_freq_order_list.index(cipher[i])
        lis.append(english_prior_order[idx])

    return ''.join(lis)


def most_freq_decipher(cipher, english_prior_order=ENGLISH_PRIOR_ORDER):

    return english_prior_order[0] * len(cipher)

