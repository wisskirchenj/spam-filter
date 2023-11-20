from spam.preprocess import load_and_preprocess_data


def main():
    data = load_and_preprocess_data()
    print(data[:200])


if __name__ == '__main__':
    main()
