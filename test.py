import argparse

Ap = argparse.ArgumentParser()
Ap.add_argument('-q', '--q', action='store_true')
qwe = vars(Ap.parse_args())['q']

print(qwe)