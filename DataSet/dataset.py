from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer

VALID_ENCODINGS = ['Binarty', 'MultiLabel']

ENCODERS = {'MultiLabel': LabelBinarizer}


class Dataset:
    def __init__(self, df, label_encoder: str, label: str):
        if label_encoder not in VALID_ENCODINGS:
            raise ValueError(f'encode: {label_encoder} is not a valid encoding type')

        if label not in df.columns:
            raise ValueError(f'Unable to find label: {label} in dataframe columns')

        self.data, self.target, self.label_mapping = self.encode_labels(df, label_encoder, label)

    @staticmethod
    def encode_labels(df, label_encoder: str, label: str):
        encoder = ENCODERS.get(label_encoder)

        x = df.loc[:, list(set(df.columns)-{label})]
        y = df.loc[:, label]
        mapping = dict(zip(y.unique(), range(len(y.unique()))))
        y = y.replace(mapping)

        return x, encoder().fit_transform(y), mapping


