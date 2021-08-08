import os
from ran import fs


def load_by_filename(folder):
    '''
    '''

    result = []
    for p in fs.list_files(folder):
        n = os.path.basename(p)
        c = os.path.splitext(n)[0]
        result.append((c, p))
    return result

