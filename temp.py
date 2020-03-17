def load_token_list():
    with open('./token_list', 'r') as f:
        return [token.strip() for token in f.readlines()]


token_list = load_token_list()
print(token_list)
print('Super@dmin' in token_list)
