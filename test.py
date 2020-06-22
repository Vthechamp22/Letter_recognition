import argparse

Ap = argparse.ArgumentParser()
Ap.add_argument('-q', '--q')
qwe = vars(Ap.parse_args())['q']

print(qwe)