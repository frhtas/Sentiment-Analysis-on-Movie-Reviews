# Load Huggingface transformers
from transformers import TFBertModel, BertConfig, BertTokenizerFast, TFAutoModel

# Then what you need from tensorflow.keras
from tensorflow.keras.layers import Input, Dropout, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.optimizers import LAMB, AdamW
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical

# And pandas for data import + sklearn because you allways need sklearn
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

import re


class BertModel():
    def __init__(self):
        self.stars = [u"\u2605\u2606\u2606\u2606\u2606", 
                      u"\u2605\u2605\u2606\u2606\u2606", 
                      u"\u2605\u2605\u2605\u2606\u2606", 
                      u"\u2605\u2605\u2605\u2605\u2606",
                      u"\u2605\u2605\u2605\u2605\u2605",
                      u"\u2606\u2606\u2606\u2606\u2606"]
        self.get_model()


    def get_model(self):
        # Name of the BERT model to use
        model_name = 'bert-base-cased'

        # Max length of tokens
        self.max_length = 51

        # Load transformers config and set output_hidden_states to False
        config = BertConfig.from_pretrained(model_name)
        config.output_hidden_states = False

        # Load BERT tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained_model_name_or_path = model_name, config = config)

        # Build model input
        input_ids = Input(shape=(self.max_length,), name='input_ids', dtype='int32')
        attention_mask = Input(shape=(self.max_length,), name='attention_mask', dtype='int32') 
        inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

        bert = TFAutoModel.from_pretrained(model_name)
        embeddings = bert.bert(inputs)[1]  # access pooled activations with [1]

        # convert bert embeddings into 5 output classes
        output = Flatten()(embeddings)
        output = Dense(256, activation='relu')(output)
        output = Dense(128, activation='relu')(output)
        output = Dense(5, activation='softmax', name='outputs')(output)

        self.model = Model(inputs=inputs, outputs=output)

        # Take a look at the model
        self.model.summary()

        optimizer = AdamW(learning_rate=1e-5, weight_decay=1e-6)
        loss = CategoricalCrossentropy()
        acc = CategoricalAccuracy('accuracy')

        self.model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

        self.model.load_weights('D:/DERSLER/Bilgisayar Mühendisliği/BLM 2021-2022 Güz/BLM_2122_1/Doğal Dil İşlemeye Giriş/Proje/bert_weights.h5')


    def predict_review(self, review):
        review = self.simple_clean(review)
        data = {'PhraseId': [0, 1], 'SentenceId': [0, 1], 'Phrase': [review, '* '*51]}
        test = pd.DataFrame(data=data)

        x = self.tokenizer(text=test.Phrase.to_list(),
                            add_special_tokens=True,
                            max_length=self.max_length,
                            truncation=True,
                            padding=True, 
                            return_tensors='tf',
                            return_token_type_ids = False,
                            return_attention_mask = True,
                            verbose = True)

        items = tf.data.Dataset.from_tensor_slices((x['input_ids'], x['attention_mask']))

        items = items.map(self.map_func)
        items = items.batch(32)

        predictions = self.model.predict(items).argmax(axis=-1)
        print(self.stars[predictions[0]])

        return self.stars[predictions[0]]


    def map_func(self, input_ids, masks):
        return {'input_ids': input_ids, 'attention_mask': masks}

    
    def simple_clean(self, text):
        text = re.sub(r'@|#', r'', text.lower())     # Returns a string with @-symbols and hashtags removed.
        text = re.sub(r'http.*', r'', text.lower())  # Returns a string with any websites starting with 'http.' removed.
        return ' '.join(re.findall(r'\w+', text.lower())) # Returns a string with only English unicode word characters ([a-zA-Z0-9_]).




if __name__ == "__main__":
    model = BertModel()
    model.predict_review("I liked this movie, it was good.")