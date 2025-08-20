import pickle
from tf_keras.preprocessing.text import Tokenizer
from tf_keras.models import load_model
from kiwipiepy import Kiwi
from pandas import pandas as pd
from cleanutils import clean_message
from tf_keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import sentencepiece as spm
from tokenizers import Tokenizer

class KiwiModel:
    def __init__(self, model_name, tokenizer_name):
        self.name = model_name
        self.kiwi = Kiwi()
        self.tokenizer = pickle.load(open(tokenizer_name, 'rb'))
        self.model = load_model(model_name)
    
    def get_tokens(self, text):
        if pd.isna(text) or not text:
            return []
        text = clean_message(text, False)
        tokens = self.kiwi.tokenize(str(text))
        filtered_tokens = []
        for token in tokens:
            if token.tag[0] in ['N', 'V', 'M', 'S', 'U']:
                filtered_tokens.append(token.form)
        return filtered_tokens

    def tokenize_test(self, X_test):
        X_test = self.tokenizer.texts_to_sequences(X_test)
        max_len = 200
        X_test = pad_sequences(X_test, maxlen=max_len)
        return X_test
    
    def clean_message(self, msg):
        msg = clean_message(msg, False)
        tokens = self.get_tokens(msg)
        return tokens

    def tokenize_message(self, msg):
        return msg

class JamoModel:
    def __init__(self, model_name, tokenizer_name):
        self.name = model_name
        self.tokenizer = pickle.load(open(tokenizer_name, 'rb'))
        self.model = load_model(model_name)
    
    def tokenize_test(self, X_test):
        X_test = self.tokenizer.texts_to_sequences(X_test)
        max_len = 800
        X_test = pad_sequences(X_test, maxlen=max_len)
        return X_test
    
    def clean_message(self, msg):
        msg = clean_message(msg, True)
        return msg
    
    def tokenize_message(self, msg):
        return msg

class SpmModel:
    def __init__(self, model_name, tokenizer_name):
        self.name = model_name
        self.tokenizer = Tokenizer.from_pretrained(tokenizer_name)
        self.model = load_model(model_name)
    
    def tokenize_test(self, X_test):
        max_len = 200
        X_test = X_test.apply(lambda x: self.tokenizer.encode(x).ids)
        X_test = pad_sequences(X_test, maxlen=max_len)
        return X_test
    
    def clean_message(self, msg):
        msg = clean_message(msg, False)
        return msg
    
    def tokenize_message(self, msg):
        tokens = map(lambda x: self.tokenizer.decode([x]), self.tokenizer.encode(msg).ids)
        return list(tokens)

def eval_model(name, model, X_test, Y_test):
    print(f"\n{name}")

    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print(f"테스트 Loss: {loss:.4f}")
    print(f"테스트 정확도: {acc:.4f}")

    # 1. 모델 예측 (확률값)
    Y_pred_probs = model.predict(X_test)

    # 2. 확률값을 이진 클래스로 변환 (임계값 0.5 기준)
    Y_pred_classes = (Y_pred_probs > 0.5).astype(int).flatten()

    # 정밀도, 재현율, F1 점수 계산
    precision = precision_score(Y_test, Y_pred_classes)
    recall = recall_score(Y_test, Y_pred_classes)
    f1 = f1_score(Y_test, Y_pred_classes)

    print(f"테스트 정밀도 (Precision): {precision:.4f}")
    print(f"테스트 재현율 (Recall): {recall:.4f}")
    print(f"테스트 F1 점수 (F1 Score): {f1:.4f}")

    return Y_pred_classes

def predict_model(model, message):
    X_test = pd.Series([message])
    X_test = model.tokenize_test(X_test)
    prob = model.model.predict(X_test, verbose=0)
    prob = prob[0][0] * 100
    return prob

interact = True
models = {
    "rnn": KiwiModel("rnn.keras", "tokenizer.pickle"),
    "cnn": KiwiModel("cnn.keras", "tokenizer.pickle"),
    "rnn-spm": SpmModel("rnn-spm.keras", "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"),
    "cnn-spm": SpmModel("cnn-spm.keras", "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"),
    #"rnn-jamo": JamoModel("rnn-jamo.keras", "tokenizer-jamo.pickle"),
    #"cnn-jamo": JamoModel("cnn-jamo.keras", "tokenizer-jamo.pickle"),
    "cnn-jamo-deep": JamoModel("cnn-jamo-deep.keras", "tokenizer-jamo.pickle"),
}

if interact:
    while True:
        msg = input()
        for name, model in models.items():
            cleaned = model.clean_message(msg)
            prob = predict_model(model, cleaned)
            cleaned = model.tokenize_message(cleaned)
            print(f"{prob:5.2f}%: {name}, {cleaned}")

else:
    #data_test = pd.read_json("test-kiwi.json")
    #data_jamo_test = pd.read_json("test-jamo.json")
    data_test = pd.read_json("dataset-test.json")
    data_jamo_test = pd.read_json("dataset-jamo-test.json")
    data_spm_test = pd.read_json("dataset-test-spm.json")
    tests = [
        {
            "X_test": data_test["tokens"],
            "Y_test": data_test["type"],
            "model": models["rnn"],
        },
        {
            "X_test": data_test["tokens"],
            "Y_test": data_test["type"],
            "model": models["cnn"],
        },
        {
            "X_test": data_spm_test["message"],
            "Y_test": data_spm_test["type"],
            "model": models["rnn-spm"],
        },
        {
            "X_test": data_spm_test["message"],
            "Y_test": data_spm_test["type"],
            "model": models["cnn-spm"],
        },
        # {
        #     "X_test": data_jamo_test["message"],
        #     "Y_test": data_jamo_test["type"],
        #     "model": models["cnn-jamo"],
        # },
        {
            "X_test": data_jamo_test["message"],
            "Y_test": data_jamo_test["type"],
            "model": models["cnn-jamo-deep"],
        },
    ]

    for test in tests:
        X_test = test["X_test"]
        X_test = test["model"].tokenize_test(X_test)
        Y_pred = eval_model(test["model"].name, test["model"].model, X_test, test["Y_test"])
        Y_pred = pd.Series(Y_pred)
        df = pd.DataFrame({
            'X': test["X_test"],
            'Y': test["Y_test"],
            'predict': Y_pred
        })
        df = df[df["Y"] != df["predict"]]
        df.to_csv(test["model"].name + ".csv", index=False, encoding="utf-8-sig")
