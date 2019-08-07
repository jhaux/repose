import numpy as np


def average_keypoints(kpts):
    """Average multiple keypoint detections weighted by confidence.
    [n_detections, n_keypoints, 3] -> [1, n_keypoints, 3]"""
    if len(kpts.shape) == 2:
        assert kpts.shape[-1] == 3
        kpts = np.expand_dims(kpts, 0)
    if kpts.shape[0] == 1:
        return kpts
    elif kpts.shape[0] == 0:
        # if no detections just predict all at center of image with zero
        # confidence
        dummy = 0.5 * np.ones((1, kpts.shape[1], 2), dtype=kpts.dtype)
        dummy_conf = np.zeros((1, kpts.shape[1], 1), dtype=kpts.dtype)
        return np.concatenate([dummy, dummy_conf], -1)
    total_weights = np.sum(kpts[:, :, [2]], axis=0, keepdims=True)
    total_weights = np.maximum(1e-6, total_weights)
    avg = np.sum(kpts[:, :, :2] * kpts[:, :, [2]],
                 axis=0,
                 keepdims=True) / total_weights
    new_conf = total_weights / kpts.shape[0]
    return np.concatenate([avg, new_conf], axis=2)


class SIstring(object):
    def __init__(self, val, std=None, unit=None):
        self.value = val
        self.std = std
        self.unit = unit

    def __str__(self):
        kind = 'num' if self.unit is None else 'SI'
        pm = '' if self.std is None else ' +- {}'.format(self.std)
        unit = '' if self.unit is None else '{{{}}}'.format(self.unit)
        return '\\{}{{{}{}}}{}'.format(kind, self.value, pm, unit)


def LatexString(python_string):
    r'''Turns a python string into a latex-compatible string, with properly
    escaped symbols.
    (I.e. ``[# $ % & ~ _ ^ \ { }] -> [\# \$ \% \& \~ \_ \^ \\ \{ \}]``
    '''

    latex_specials = [r'#', r'$', r'%', r'&', r'~',
                      r'_', r'^', r'\\', r'{', r'}']

    for symbol in latex_specials:
        python_string = python_string.replace(symbol, r'\{}'.format(symbol))

    return python_string


def make_latex_table(content, cols=None, rows=None, breake_every=10):
    '''Generate a table with entries given as kwargs.

    Args:
        content (dict): Each key, list pair corresponds to one row/column
            of entries.
        cols (list): Strings to enter as column headings.
        rows (list): see above. If rows is not None, cols must be None.
        breake_every (int): Number of columns, after which the rest of the
            columns are appended as new lines.

    Returns:
        str: The table.
    '''

    assert cols is None or rows is None, 'use only one of rows or cols.'

    top_headings = cols if cols is not None else rows

    lines = np.empty([1+len(content), len(top_headings)], dtype=object)

    lines[0, :] = [str(t) for t in top_headings]

    for i, (k, v) in enumerate(content.items(), 1):
        if i == 1:
            col_types = ['S' if isinstance(stuff, SIstring) else 'c'
                         for stuff in v]
        lines[i, 0] = str(k)
        lines[i, 1:] = [str(stuff) for stuff in v]

    if cols is None:
        lines = lines.T

    n_rows, n_cols = lines.shape
    print(breake_every, n_cols-1)
    n_splits = int(np.ceil((n_cols - 1) / breake_every))
    print(n_splits)
    all_lines = np.array_split(lines[:, 1:],
                               n_splits,
                               axis=1)

    broken_lines = []
    names = np.expand_dims(lines[:, 0], 1)
    for subset in all_lines:
        if subset.size > 0:
            broken_lines += [np.concatenate([names, subset], axis=1)]

    lengths = np.array([[len(i) for i in j] for j in lines])
    col_widths = [str(val) for val in np.max(lengths, 0)]

    tab_str = r'% This table was automatically compiled. '
    tab_str += 'If there are numbers in the original table make sure to add '
    tab_str += '\\usepackage{siuntix} in your preamble.\n'

    tab_str += '\\begin{tabular}{c' + ''.join(col_types) + '}\n'
    tab_str += '\\toprule\n'

    for chunk, lines in enumerate(broken_lines):
        for i, line in enumerate(lines):
            l_str = ''
            for j, val in enumerate(line):
                val_str = '{: <' + col_widths[j] + '}'
                val_str = val_str.format(val)
                if j < len(line) - 1:
                    val_str += ' & '

                l_str += val_str
            l_str += ' \\\\\n'

            if i == 0:
                l_str += '\\midrule\n'

            tab_str += l_str
        if len(broken_lines) > 1 and chunk < len(broken_lines) - 1:
            tab_str += '\\midrule\\midrule\n'

    tab_str += '\\bottomrule\n'
    tab_str += '\\end{tabular}'

    return tab_str


def compile_table(table_string, table_name):
    '''Can produce a pdf containing only the table IF there is nothing strange
    in there.
    '''

    import os

    document_str = r'''
    \documentclass{{standalone}}
    \usepackage{{booktabs}}
    \usepackage{{siunitx}}
    \begin{{document}}
    {}
    \end{{document}}
    '''.format(table_string)

    with open(table_name, 'w+') as table_file:
        table_file.write(document_str)

    os.system('pdflatex {}'.format(table_name))
    # os.system('rm {}'.format(table_name))


def set_preamble(preamble_file='preamble/mpl_preamble.tex'):
    import matplotlib as mpl

    latex_custom_preamble = {
        "font.family": "serif",       # use serif/main font for text elements
        "text.usetex": True,          # use inline math for ticks
        "pgf.rcfonts": False,
        "pgf.texsystem": "pdflatex",
        'hatch.linewidth': 10.,
        'hatch.color': 'w'
        }

    if preamble_file is not None:
        preamble = load_preamble(preamble_file)
        latex_custom_preamble["text.latex.preamble"] = preamble
        latex_custom_preamble["pgf.preamble"] = preamble,

    mpl.rcParams.update(latex_custom_preamble)


def load_preamble(preamble_file='preamble/mpl_preamble.tex',
                  commands_file='src/shared_commands.tex'):
    with open(preamble_file, 'r') as f:
        file_data = f.readlines()

    for l in file_data:
        l.replace('utf8', 'utf8x')  # Some bug Johann told me about

    with open(commands_file, 'r') as f:
        file_data += f.readlines()

    return file_data


def get_document_lengths():
    # Fill with default values
    lengths_backup = {'columnwidth': 3.1756,
                      'linewidth': 3.1756,
                      'textwidth': 6.875,
                      'textheight': 8.875}

    log_path = 'eccv2018submission.log'

    try:
        with open(log_path, 'r') as log:
            lines = log.readlines()

            lengths = {}
            for l in lines:
                if '___' in l:
                    name, length = l.split('=')
                    name = name.strip(' ')
                    name = name.strip('_').lower()
                    length = length[:-3]
                    lengths[name] = float(length)
    except Exception:
        print('Could not load thesis.log file')
        lengths = lengths_backup

    if lengths == {}:
        print('lengths was empty')
        lengths = lengths_backup

    return lengths
