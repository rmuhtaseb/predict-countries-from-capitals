import argparse
import pickle

from util import predict_country

def main():
    my_parser = argparse.ArgumentParser(description='Predict Countries from Capital')

    my_parser.add_argument('--city1',
                       default='Athens',
                       help='City 1')

    my_parser.add_argument('--country1',
                       default='Greece',
                       help='Country 1')

    my_parser.add_argument('--city2',
                       default='Cairo',
                       help='City 2')

    my_parser.add_argument('--data_path',
                       default='./data/word_embeddings_subset.p',
                       help='Word Embeddings Data Path')

    args = my_parser.parse_args()

    word_embeddings = pickle.load(open(args.data_path, "rb"))

    prediction = predict_country(args.city1, args.country1, args.city2, word_embeddings)

    print(f'Predicted Country for City={args.city2}:\n{prediction[0]}\nSimilarity Score: {prediction[1]}')


if __name__ == "__main__":
    main()
