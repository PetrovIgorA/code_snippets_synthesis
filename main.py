import os
import time
import numpy as np
import gensim as gs
from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.models import Sequential

pst = time.time()

directory = "./dataset"
maxLenProgram = 2000
countGenerate = 5

print("Download files...")
st = time.time()

programs = []
for filename in os.listdir(directory):
    if filename[0].isdigit():
        with open(directory + "/" + filename, 'r', encoding='utf-8') as file:
            program = file.read()
            j = program.rfind("import")
            if j >= 0:
                program = program[program.rfind("import"):]
                program = program[program.find(';') + 1:].split()
            else:
                program = program.split()
            if len(program) <= maxLenProgram:
                programs.append(program)

print("time: ", time.time() - st)

print("Word2Vec...")
st = time.time()

wordModel = gs.models.Word2Vec(programs, size=70, min_count=1, window=10, iter=100)
pretrainWeights = wordModel.wv.syn0
wordsCount, emdeddingSize = pretrainWeights.shape

print(pretrainWeights.shape)
print("time: ", time.time() - st)

print("data preparation...")
st = time.time()

def wordToIndex(word):
    return wordModel.wv.vocab[word].index

def indexToWord(index):
    return wordModel.wv.index2word[index]

train_X = np.zeros([len(programs), maxLenProgram], dtype=np.int32)
train_Y = np.zeros([len(programs)], dtype=np.int32)
for i, program in enumerate(programs):
    for j, word in enumerate(program[:-1]):
        train_X[i, j] = wordToIndex(word)
    train_Y[i] = wordToIndex(program[-1])

print("time: ", time.time() - st)

print("craete model...")
st = time.time()

model = Sequential()
model.add(Embedding(input_dim=wordsCount, output_dim=emdeddingSize, weights=[pretrainWeights]))
model.add(LSTM(units=emdeddingSize))
model.add(Dropout(0.2))
model.add(Dense(units=wordsCount))
model.add(Activation('softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

print("time: ", time.time() - st)

print("Training:")
st = time.time()

def sample(preds, temperature=1.0):
    if temperature <= 0:
        return np.argmax(preds)
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate(program):
    indexes = [wordToIndex(word) for word in program.split()]
    for i in range(countGenerate):
        prediction = model.predict(x=np.array(indexes))
        index = sample(prediction[-1], temperature=0.7)
        indexes.append(index)
    return ' '.join(indexToWord(index) for index in indexes)

def epochs(epoch, _):
    i = 0
    programs = [
    """public abstract class AbstractDiscoveredOperation implements Operation {

	private final OperationMethod operationMethod ; 

	private final OperationInvoker invoker ; 

	public AbstractDiscoveredOperation ( DiscoveredOperationMethod operationMethod , OperationInvoker invoker )  {
		this.operationMethod = operationMethod ; 
		this.invoker = """]
    for program in programs:
        with open("generated/gen" + str(i) + ".java", 'w', encoding='utf-8') as file:
            file.write(generate(program))
            i += 1

model.fit(train_X, train_Y, batch_size=2, epochs=2, callbacks=[LambdaCallback(on_epoch_end=epochs)])

print("time: ", time.time() - st)

print("Program time: ", time.time() - pst)