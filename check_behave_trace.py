import pickle

behave_trace = {}
with open('data/client_behave_trace', 'rb') as fin:
    behave_trace = pickle.load(fin)

print(behave_trace[1])