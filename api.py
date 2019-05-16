import sys

import datasets
import functions

df = datasets.mergeData()

method = sys.argv[1]
term = sys.argv[2]

if(method == 'search'):
    print(df.loc[df['title'] == term].to_json(orient='index'))
    sys.stdout.flush()

if(method == 'match'):
    id = df.loc[df['title']==term].index[0]
    functions.getSuggestions(df, id)
    sys.stdout.flush()

if(method == 'match2'):
    print('{"query":')
    print('"Heat",')
    print('"movies":[')
    print('"Leon: The Professional",')
    print('"Running Out of Time"]}')
    sys.stdout.flush()
