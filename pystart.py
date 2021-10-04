# -*- coding: utf-8 -*-

print("Make today a great day!")

GPU_STR = "0,1,2,3"

TRAINING_DATA = '/array1/data/front_row_data/training'
TESTING_DATA = '/array1/data/front_row_data/testing'

import os
from sys import platform
import random
import code
import shutil

from functools import reduce

getwd = os.getcwd

def setwd(path):
    owd = os.getcwd()
    os.chdir(path)
    return owd

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Could not import matplotlib.pyplot as plt.")

try:
    import numpy as np
    from numpy import all, any, array, asarray, allclose, alltrue
    from numpy import abs, sqrt, sin, cos, log, log10, log2, exp, power
    from numpy import mod, mean, median, var, std, max, min, argmax, argmin, sum
    from numpy import divmod, rint, conj, exp2, reciprocal, gcd, lcm
except ImportError:
    print("Could not import numpy.")

try:
    from sklearn.utils import shuffle
except:
    pass

try:
    import pandas as pd
except:
    print("Could not import pandas.")

import time
has_len = lambda x: hasattr(x, '__len__')
timestamp = lambda: time.strftime("%m_%d_%y_%H-%M-%S", time.strptime(time.asctime()))
is_function = lambda x: x.__class__.__name__ == 'function'
if_func = is_function

def is_null(x, return_arr=True):
    if not has_len(x):
        return x is None
    y = list(map(lambda e: e is None, x))
    y = y if return_arr else all(y)
    return True if y == [] or len(y) == 1 else y

empty = is_null

isinstance_vec = lambda x, t: asarray(list(map(lambda e: isinstance(e, t), x)))
types = lambda x: list(map(lambda e: type(e), x))

def is_char_array(x):
    return all(isinstance_vec(x, str))

def is_int_array(x):
    return all(isinstance_vec(x, int))

def is_complex_array(x):
    return all(isinstance_vec(x, complex))

def nchar(x):
    if not is_char_array(x):
        raise ValueError("`x` must be a character vector")
    return array(list(map(lambda s: len(s), x))).astype(bool)

def nzchar(x):
    if not is_char_array(x):
        raise ValueError("`x` must be a character vector")
    return ~nchar(x) #np.logical_not(empty(x))

def as_complex(x):
    x = np.asarray(x)
    if len(x.shape) < 2:
        raise ValueError("`x` must be at least a 2D array")
    return 1j * x[:, 1] + x[:, 0]

def factors(n):    
    return set(reduce(list.__add__, 
                ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

from itertools import zip_longest
def groupr(iterable, n, padvalue=None):
  "groupr(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
  return list(zip_longest(*[iter(iterable)]*n, fillvalue=padvalue))


##############################################################################
# Environment Management                                                     #
##############################################################################


import inspect
caller = get_caller_name = lambda: inspect.stack()[1][3]

def tryb():
    l = locals()
    g = globals()
    for k, v in l.items():
        g[k] = v
    code.interact(local=g)

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def __iter__(self):
        return iter(self.__dict__.items())
    def add_to_namespace(self, **kwargs):
        self.__dict__.update(kwargs)

def env(**kwargs):
    return Namespace(**kwargs)
environment = env


import pynvml as N

N.nvmlInit()

DEV_COUNT = N.nvmlDeviceGetCount()
MB = 1024 * 1024

gpus = nvidia = nvidia_smi = lambda: os.system('nvidia-smi')

def set_cuda_devices(i=""):
    """Set one or more GPUs to use for training by index or all by default
        Args:
            `i` may be a list of indices or a scalar integer index
                default='' # <- Uses all GPUs if you pass nothing
    """
    def list2csv(l):
        s = ''
        ll = len(l) - 1
        for i, x in enumerate(l):
            s += str(x)
            s += ',' if i < ll else ''
        return s 
    if i.__eq__(''): # Defaults to ALL
        i = list(range(DEV_COUNT))
    if isinstance(i, list):
        i = list2csv(i)

    # ensure other gpus not initialized by tf
    os.environ['CUDA_VISIBLE_DEVICES'] = str(i)
    print("CUDA_VISIBLE_DEVICES set to {}".format(i))
    
def set_gpu_tf(gpu="", gpu_max_memory=None):
    """Set gpu for tensorflow upon initialization.  Call this BEFORE importing tensorflow"""
    set_cuda_devices(str(gpu))
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('\nUsed gpus:', gpus)
    if gpus:
        try:
            for gpu in gpus:
                print("Setting memory_growth=True for gpu {}".format(gpu))
                tf.config.experimental.set_memory_growth(gpu, True)
                if gpu_max_memory is not None:
                    print("Setting GPU max memory to: {} mB".format(gpu_max_memory))
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu, 
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=gpu_max_memory)]
                        )
        except RuntimeError as e:
            print(e)

def get_gpu_available_memory():
    return list(
        map(
            lambda x: N.nvmlDeviceGetMemoryInfo(
                N.nvmlDeviceGetHandleByIndex(x)).free // MB, range(DEV_COUNT)
            )
        )

def get_based_gpu_idx():
    mem_free = get_gpu_available_memory()
    idx = np.argmax(mem_free)
    print("GPU:{} has {} available MB".format(idx, mem_free[idx]))
    return idx

def set_based_gpu():
    idx = get_based_gpu_idx()
    set_gpu_tf(str(idx))


# inp = input("Do you wish to select GPUs and import tensorflow? (y/n)\n>> ")
# if inp.lower() in ['y', 'yes', 'ye', 'yee']:
# set_min_gpu()
try:
    pass
    # set_based_gpu()
    # gpu_str = GPU_STR #input("Enter GPU slots for tensorflow to see:\n>> ")
    # max_mem = "" #input("Enter GPU max memory (mB). Leave blank if none desired:\n>> ")
    # set_gpu_tf(gpu_str, int(max_mem) if max_mem != "" else None)
except:
    print("Could not configure GPU devices for tensorflow...")

def add_to_namespace(x, **kwargs):
    if not hasattr(x, '__dict__'):
        raise ValueError(
            "Cannot update nonexistant `__dict__` for object of type {}".format(type(x)))
    x.__dict__.update(kwargs)
    return x

def add_to_namespace_dict(x, _dict):
    x.__dict__.update(_dict)
    return x

def exists_here(object_str):
    if str(object_str) != object_str:
        print("Warning: Object passed in was not a string, and may have unexpected behvaior")
    return object_str in list(globals())

def stopifnot(predicate):
    predicate_str = predicate
    if is_strlike(predicate):
        predicate = eval(predicate)
    if is_bool(predicate) and predicate not in [True, 1]:
        import sys
        sys.exit("\nPredicate:\n\n  {}\n\n is not True... exiting.".format(
            predicate_str))

def add_to_globals(x):
    if type(x) == dict:
        if not all(list(map(lambda k: is_strlike(k), list(x)))):
            raise KeyError("dict `x` must only contain keys of type `str`")
    elif type(x) == list:
        if type(x[0]) == tuple:
            if not all(list(map(lambda t: is_strlike(t[0]), x))):
                raise ValueError("1st element of each tuple must be of type 'str'")
            x = dict(x)
        else:
            raise ValueError("`x` must be either a `list` of `tuple` pairs, or `dict`")
    globals().update(x)


def parse_gitignore():
    with open('.gitignore', 'r') as f:
        x = f.readlines()
    ignore = list(map(lambda s: s.split('\n')[:-1], x))
    ignore[-1] = [x[-1]]
    return ', '.join(unlist(ignore))


def which_os():
    if platform == "linux" or platform == "linux2":
        return "linux"
    elif platform == "darwin":
        return "macOS"
    elif platform == "win32":
        return "windows"
    else:
        raise ValueError("Mystery os...")

def on_windows():
    return which_os() == "windows"

def on_linux():
    return which_os() == "linux"

def on_mac():
    return which_os() == "macOS"



##############################################################################
# Prompts                                                                    #
##############################################################################
    
import sys
from datetime import datetime


class TimePS1:
    def __init__(self):
        self.count = 0

    def __str__(self):
        self.count += 1
        return f"({self.count}) {datetime.now().strftime('%H:%m %p')} 🌮 "

class TimePS2:
    def __init__(self):
        self.count = 0

    def __str__(self):
        self.count += 1
        return f"({self.count}) {datetime.now().strftime('%H:%m %p')} 🍔 "

class TimePS3:
    def __init__(self):
        self.count = 0

    def __str__(self):
        self.count += 1
        return f"({self.count}) {datetime.now().strftime('%H:%m %p')} 💩 "

sys.ps1 = TimePS1()
sys.ps2 = TimePS2()
sys.ps3 = TimePS3()

if False:
    
    sys.ps1 = "🌮"
    sys.ps2 = "💩"

    toilet = "🚽"
    tp = "🧻"
    bomb = "💣"
    tub = "🛀"
    keyboard = "⌨️"
    dagger = "🗡️"
    gun = "🔫"
    burger = "🍔"

    try:

        from prompt_toolkit.styles import style_from_pygments_dict
        from IPython.terminal.prompts import Prompts, Token
        from IPython import get_ipython

        style = style_from_pygments_dict({
            Token.User: '#f8ea6c',
            Token.Path_1: '#f08d24',
            Token.Path_2: '#67f72f',
            Token.Git_branch: '#1cafce',
            Token.Pound: '#000',
        })

        os.path.split(os.getcwd())
        path_sep = "/" if on_linux() else "\\"
        path = os.path.split(os.getcwd())
        drive = path[0]
        drive = "/" + drive.split(path_sep)[-1]
        cwd = ("/" if on_linux() else "\\") + path[1]

        class MyPrompt(Prompts):
            def in_prompt_tokens(self, cli=None):
                return [
                    (Token.User, 'stgeorge '),
                    (Token.Path_1, drive),
                    (Token.Path_2, cwd),
                    # (Token.Git_branch, ' master '),
                    (Token.Dollar, ' $ '),
                    # (Token.OutPrompt, '🌮')
                    # (Token.Prompt, "🌮")
                ]

        ipython = get_ipython()
        ipython.prompts = MyPrompt(ipython)
        ipython._style = style
    except:
        print("Error loading IPython utils...")
        print("python:", sys.version)
        pass


##############################################################################
# Math Operators                                                             #
##############################################################################

add = lambda arr: reduce(lambda x, y: x + y, arr)
sub = lambda arr: reduce(lambda x, y: x - y, arr)
mul = lambda arr: reduce(lambda x, y: x * y, arr)
div = lambda arr: reduce(lambda x, y: x / y, arr)
prod = mul


##############################################################################
# Sorting Routines                                                           #
##############################################################################

try:
    import ulid
    def get_ulid(n, as_str=False):
        return sorted(
	    [str(ulid.ULID()) if as_str else ulid.ULID() for _ in range(n)]
	)
except:
    pass

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(l, key=alphanum_key)

def sort_dict(d, by='k'):
    i = 1 if 'v' in by else 0 # v for 'val' or 'value'
    return dict(
        sorted(
            d.items(), 
            key=lambda x: x[i]
        )
    )

def sort_dict2(d, by='key', rev=False):
    if 'v' in by:
        return {k: d[k] for k in sorted(d, key=d.get, reverse=rev)}
    return {k: d[k] for k in sorted(d, reverse=rev)}

##############################################################################
# List utilities                                                             #
##############################################################################

def tf_counts(arr, x):
    arr = tf.constant(arr)
    return tf.where(arr == x).shape[0]

def counts(arr, x):
    arr = np.asarray(arr)
    return len(np.where(arr == x)[0])

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')

def is_bool(x):
    if x not in [True, False, 0, 1]:
        return False
    return True
isTrueOrFalse = is_bool

def is_strlike(x):
    if type(x) == bytes:
        return type(x.decode()) == str
    if is_numpy(x):
        try:
            return 'str' in x.astype('str').dtype.name
        except:
            return False
    return type(x) == str

def regextract(x, regex):
    matches = vmatch(x, regex)
    return np.asarray(x)[matches]

import re
def vmatch(x, regex):
    r = re.compile(regex)
    return np.vectorize(lambda x: bool(r.match(x)))(x)

def lengths(x):
    def maybe_len(e):
        if type(e) == list:
            return len(e)
        else:
            return 1
    if type(x) is not list: return [1]
    if len(x) == 1: return [1]
    return(list(map(maybe_len, x)))

def is_numpy(x):
    return x.__class__ in [
        np.ndarray,
        np.rec.recarray,
        np.char.chararray,
        np.ma.masked_array
    ]

def is_empty(x):
    if x is None: return True
    if not is_numpy(x): x = np.asarray(x)
    return np.equal(np.size(x), 0)

def next2pow(x):
    return 2**int(np.ceil(np.log(float(x))/np.log(2.0)))

def unnest(x, return_numpy=False):
    if return_numpy:
        return np.asarray([np.asarray(e).ravel() for e in x]).ravel()
    out = []
    for e in x:
        out.extend(np.asarray(e).ravel())
    return out
    
def unwrap_np(x):
    *y, = x
    return y


def unwrap_df(df):
    if len(df.values.shape) >= 2:
        return df.values.flatten()
    return df.values

def summarize(x):
    x = np.asarray(x)
    x = np.squeeze(x)
    try:
        df = pd.Series(x)        
    except:
        try:
            df = pd.DataFrame(x)
        except:
            raise TypeError("`x` cannot be coerced to a pandas type.")
    return df.describe(include='all')

def list_product(els):
  prod = els[0]
  for el in els[1:]:
    prod *= el
  return prod


def np_arr_to_py(x):
    x = np.unstack()
    return list(x)

def logical2idx(x):
    x = np.asarray(x)
    return np.arange(len(x))[x]

def get(x, f):
        return x.iloc[logical2idx(f(x))] if is_pandas(x) else x[logical2idx(f(x))]

def apply_pred(x, p):
    return list(map(lambda e: p(e), x))

def extract_mask(x, m):
    if len(x) != len(m):
        raise ValueError("Shapes of `x` and `m` must be equivalent.")
    return np.asarray(x)[logical2idx(m)]

def extract_cond(x, p):
    mask = list(map(lambda e: p(e), x))
    return extract_mask(x, mask)

def idx_of(vals_to_idx, arr_to_seek):
    if isinstance(vals_to_idx[0], str):
        return idx_of_str(vals_to_idx, arr_to_seek)
    vals_to_idx = maybe_list_up(vals_to_idx)
    nested_idx = list(map(lambda x: np.where(arr_to_seek == x), vals_to_idx))
    return list(set(unnest(nested_idx)))

def idx_of_str(vals_to_idx, arr_to_seek):
    vals_to_idx = maybe_list_up(vals_to_idx)
    arr_to_seek = np.asarray(arr_to_seek)
    nested_idx = list(map(lambda x: np.where(x == arr_to_seek), vals_to_idx))
    return list(set(unnest(nested_idx)))

mwhere = where_map = lambda elem, arr: np.asarray(list(map(lambda x: elem in x, arr)))
mindex = index_map = lambda elem, arr: logical2idx(mwhere(elem, arr))

def within(x, y, eps=1e-3):
    ymax = y + eps
    ymin = y - eps
    return x <= ymax and x >= ymin

def within1(x, y):
    return within(x, y, 1.)

def within_vec(x, y, eps=1e-3):
    vf = np.vectorize(within)
    return np.all(vf(x, y, eps=eps))

def dim(x):
    if is_numpy(x):
        return x.shape
    return np.asarray(x).shape

def comma_sep_str_to_int_list(s):
  return [int(i) for i in s.split(",") if i]

def unzip(x):
    if type(x) is not list:
        raise ValueError("`x` must be a list of tuple pairs")
    return list(zip(*x))

# Useful for printing all arguments to function call, mimicks dots `...` in R.
def printa(*argv):
    [print(i) for i in argv]

def printk(**kwargs):
    [print(k, ":\t", v) for k,v in kwargs.items()]

def pyrange(x, return_type='dict'):
    return {'min': np.min(x), 'max': np.max(x)} \
        if return_type == 'dict' \
        else (np.min(x), np.max(x))

def alleq(x):
    try:
        iter(x)
    except:
        x = [x]
    current = x[0]
    for v in x:
        if v != current:
            return False
    return True

def types(x):
    if isinstance(x, dict):
        return {k: type(v) for k,v in x.items()}
    return list(map(type, x))

# nearest element to `elem` in array `x`
def nearest1d(x, elem):
    lnx = len(x)
    if lnx % 2 == 1:
        mid = (lnx + 1) // 2
    else:
        mid = lnx // 2
    if mid == 1:
        return x[0]
    if x[mid] >= elem:
        return nearest1d(x[:mid], elem)
    elif x[mid] < elem:
        return nearest1d(x[mid:], elem)
    else:
        return x[0]

def idx_of_1d(x, elem):
    if elem not in x:
        raise ValueError("`elem` not contained in `x`")
    return dict(zip(x, range(len(x))))[elem]

def unlist(x):
    if len(x[0]) > 1:
        raise ValueError("inner dim of `x` must == 1")
    return list(map(lambda l: l[0], x))

def random_ints(length, lo=-1e4, hi=1e4):
    return [random.randint(lo, hi) for _ in range(length)]

def random_floats(length, lo=-1e4, hi=1e4):
    return [random.random() for _ in range(length)]

def dict_diff(d1, d2):
    if d1.keys() != d2.keys():
        raise ValueError("Keys don't match.")
    return {
        k: abs(np.asarray(d1[k]) - np.asarray(d2[k])) \
        for k in d1.keys()
    }

if sys.version_info.major == 3 and sys.version_info.minor == 9:
    def merge_dict(d1, d2):
        return d1 | d2
else:
    def merge_dict(d1, d2):
        return {**d1, **d2}

def dict_list(keys):
    return {k: [] for k,v in dict.fromkeys(keys).items()}
    
def flip_kv(d):
    return {v: k for k,v in d.items()}

def where(arr, elem, op='in'):
    return list(
        map(
            lambda x: elem == x or elem in x,
            arr
        )
    )

index = lambda arr, elem: unlist(
    logical2idx(
        where(
            arr, elem
        )
    )
)

def shuffle_npz(xs, ys):
    """Shuffle numpy training arrays, assuming `xs` is training data
        and `ys` is target data for classification of int valued targets.
    xs: (batch, x, x)
    ys: (batch,) integer valued targets
    """
    idx2tgt = dict(zip(list(range(len(xs))), ys))
    df = pd.DataFrame({'index': list(idx2tgt), 'target': idx2tgt.values()})
    df = shuffle(df, random_state=FLAGS['random_seed'])
    new_x = np.stack(df['index'].apply(lambda i: xs[i]).values)
    new_y = np.stack(df['index'].apply(lambda i: ys[i]).values)
    del xs, ys
    ## TODO: Add functionality to control class balance and keep even distributions!!
    return new_x, new_y

def copy_dirtree(inpath, outpath):
    def ignore_files(dir, files):
        return [f for f in files if os.path.isfile(os.path.join(dir, f))]
    shutil.copytree(inpath, outpath, ignore=ignore_files)
    print("Success copying directory structure\n {} \n -- to --\n {}".format(
        inpath, outpath)
    )

# apply callable `f()` at indices `i` on data `x`
def apply_at(f, i, x):
    """
    # usage:
    >> x = [1,2,3,4,5]
    >> apply_at(lambda x: x**2, index(2, x), x)
    >> [1,4,3,4,5]
    """
    x = np.asarray(x)
    i = np.asarray(i)
    x[i] = np.asarray(
        list(
            map(
                lambda x: f(x), x[i]
            )
        )
    )
    return x

def replace_at(x, indices, repl, colname=None):
    if is_pandas(x) and colname is None:
        if x.__class__ == pd.core.frame.DataFrame:
            raise ValueError("Must supply colname with a DataFrame object")
        return replace_at_pd(x, indices, repl)
    x = np.asarray(x)
    x[indices] = np.repeat(repl, len(indices))
    return x

def replace_at_pd(x, colname, indices, repl):
    x.loc[indices, colname] = repl
    return x

def delete_at(x, at):
    return x[~np.isin(np.arange(len(x)), at)]

def delete_at2(x, at):
    return x[[z for z in range(len(x)) if not z in at]]

def is_pandas(x):
    return x.__class__ in [
        pd.core.frame.DataFrame,
        pd.core.series.Series
    ]

def onehot_targets_pd(df, _from='class', _to='target'):
    if _from not in df.columns: raise ValueError("No column {} found.".format(_from))
    t = df[_from].unique()
    d = dict(zip(t, list(range(len(t)))))
    df[_to] = df[_from].apply(lambda x: d[x])
    df[_to] = df[_to].apply(lambda x: tf.one_hot(x, len(t)).numpy())
    return df

def find_and_replace(xs, e, r):
    return replace_at(xs, np.where(xs == e)[0], r)


from itertools import zip_longest
def groupl(iterable, n, padvalue=None):
  "groupl(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
  return list(zip_longest(*[iter(iterable)]*n, fillvalue=padvalue))

def merge_by_colname(df1, df2, colname='target', how='outer'):
    pt = df1.pop(colname)
    nt = df2.pop(colname)    
    targets = pt.append(nt).reset_index()
    df2 = df1.merge(df2, how=how)
    df2.loc[:, (colname)] = targets
    return df2

# Shuffle 0th axis, e.g. list of tuples
shuffle0 = lambda p: random.sample(p, k=len(p))

def merge_with_targets(df1, df2):
    pt = df1.pop('target')
    nt = df2.pop('target')    
    targets = pt.append(nt).reset_index()
    df2 = df1.merge(df2, how='outer')
    df2.loc[:, ('target')] = targets
    return df2

reduce_df = lambda dfs: reduce(lambda x, y: merge_with_targets(x, y), dfs)
merge_df = reduce_df

merge_dicts = lambda dicts: reduce(
    lambda x,y: merge_dict(x,y), 
    dicts
)

def _complex(real, imag):
    """ Efficiently create a complex array from 2 floating point """
    real = np.asarray(real)
    imag = np.asarray(imag)
    cplx = 1j * imag    
    return cplx + real

# Safe log10
def log10(data):
    np.seterr(divide='ignore')
    data = np.log10(data)
    np.seterr(divide='warn')
    return data
log10_safe = log10

def is_scalar(x):
    if is_numpy(x):
        return x.ndim == 0
    if isinstance(x, str) or type(x) == bytes:
        return True
    if hasattr(x, "__len__"):
        return len(x) == 1
    try:
        x = iter(x)
    except:
        return True
    return np.asarray(x).ndim == 0

def first(x):
    if is_scalar(x):
        return x
    if not is_numpy(x):
        x = np.asarray(x)
    return x.ravel()[0]

def last(x):
    if not is_numpy(x):
        x = np.asarray(x)
    return x.ravel()[-1] 

def listmap(x, f):
    return list(map(f, x))
lmap = listmap

def loadz(path, key='arr_0'):
    x = np.load(path)
    if is_scalar(key):
        return x[key]
    d = {k: v if k in keys else None for k,v in x.items()}
    return drop_none(d)

def drop_none(x):
    if isinstance(x, dict):
        i = logical2idx(np.asarray(list(x.values())) == None)
        k = np.asarray(list(x))
        for key in k[i]:
            x.pop(key)
        return x
    if isinstance(x, list) or is_numpy(x):
        x = np.asarray(x)
        i = np.where(x != None)[0]
        return x[i]
    raise TypeError("Don't know how to process {} type".format(type(x)))


##############################################################################
# Plotting Routines                                                          #
##############################################################################
import matplotlib.pyplot as plt

#Defaults for legible figures
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams["image.cmap"] = 'jet'

# ALL_COLORS = list(colors.CSS4_COLORS)
COLORS = [
    'blue', # for original signal
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
] * 2

def histogram(x, bins='auto', show=True, save=False, outpath='histogram.png'):
    x = np.asarray(x).ravel()
    hist, bins = np.histogram(x, bins=bins)
    plt.bar(bins[:-1], hist, width=1)
    plt.savefig(outpath)
    if show:
        plt.show()

def plotx(y, x=None, xlab='obs', ylab='value', 
          title='', save=False, filepath='plot.png'):
    if x is None: x = np.linspace(0, len(y), len(y))
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set(xlabel=xlab, ylabel=ylab,
        title=title)
    ax.grid()
    if save:
       fig.savefig(filepath)
    plt.show()

def poverlap(y, x=None):
    shp = np.asarray(y).shape
    if len(shp) < 2:
        raise ValueError("`y` must be at least 2D")
    if x is None: x = np.linspace(0, len(y[0]), len(y[0]))
    for i in range(shp[0]):
        plt.plot(x, y[i], COLORS[i])
    plt.show()

# 2 cols per row
def psubplot(y, x=None, figsize=[6.4, 4.8], filename=None, title=None):
    shp = np.asarray(y).shape
    if len(shp) < 2:
        raise ValueError("`y` must be at least 2D")
    if x is None: x = np.linspace(0, len(y[0]), len(y[0]))    
    i = 0
    _, ax = plt.subplots(nrows=shp[0]//2+1, ncols=2, figsize=figsize)
    for row in ax:
        for col in row:
            if i >= shp[0]: break
            col.plot(x, y[i], COLORS[i])
            i += 1
    if title:
        plt.title(title)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

# one col per row
def psub1(y, x=None, figsize=[6.4, 4.8], filename=None, title=None, xlab=None, ylab=None, hspace=0.5):
    shp = np.asarray(y).shape
    if len(shp) < 2:
        raise ValueError("`y` must be at least 2D")
    if x is None: x = np.linspace(0, len(y[0]), len(y[0]))    
    i = 0
    fig, ax = plt.subplots(nrows=shp[0], ncols=1, figsize=figsize)
    for row in ax:
        if i >= shp[0]: break
        row.plot(x, y[i], COLORS[i])
        i += 1
    fig.subplots_adjust(hspace=hspace)
    if xlab:
        plt.xlabel(xlab)
    if ylab:
        plt.ylabel(ylab)
    if title:
        ax[0].set_title(title)
    if filename:
        plt.savefig(filename)
    else:
        plt.show()

psub = psubplot
polp = poverlap



import scipy as sp
from scipy import stats as scipy_stats

def stats(x, axis=None, epsilon=1e-7):
    if not is_numpy(x):
        x = np.asarray(x)
    if np.min(x) < 0:
        _x = x + abs(np.min(x) - epsilon)
    else:
        _x = x
    gmn = scipy_stats.gmean(_x, axis=axis)
    hmn = scipy_stats.hmean(_x, axis=axis)
    mode = scipy_stats.mode(x, axis=axis).mode[0]
    mnt2, mnt3, mnt4 = scipy_stats.moment(x, [2,3,4], axis=axis)
    lq, med, uq = scipy_stats.mstats.hdquantiles(x, axis=axis)
    lq, med, uq = np.quantile(x, [0.25, 0.5, 0.75], axis=axis)
    var = scipy_stats.variation(x, axis=axis) # coefficient of variation
    sem = scipy_stats.sem(x, axis=axis) # std error of the means
    res = scipy_stats.describe(x, axis=axis)
    nms = ['nobs          ', 
           'minmax        ', 
           'mean          ', 
           'variance      ', 
           'skewness      ', 
           'kurtosis      ']
    description = dict(zip(nms, list(res)))
    description.update({
        'coeff_of_var  ': var,
        'std_err_means ': sem,
        'lower_quartile': lq,
        'median        ': med,
        'upper_quartile': uq,
        '2nd_moment    ': mnt2,
        '3rd_moment    ': mnt3,
        '4th_moment    ': mnt4,
        'mode          ': mode,
        'geometric_mean': gmn,
        'harmoinc_mean ': hmn
    })
    return description


def switch(on, pairs, default=None):
    """ Create dict switch-case from key-word pairs, mimicks R's `switch()`

        Params:
            on: key to index OR predicate returning boolean value to index into dict
            pairs: dict k,v pairs containing predicate enumeration results
        
        Returns: 
            indexed item by `on` in `pairs` dict

        Usage:
        # Predicate
            pairs = {
                True: lambda x: x**2,
                False: lambda x: x // 2
            }
            switch(
                1 == 2, # predicate
                pairs,  # dict 
                default=lambda x: x # identity
            )

        # Index on value
            key = 2
            switch(
                key, 
                values={1:"YAML", 2:"JSON", 3:"CSV"},
                default=0
            )
    """
    if type(pairs) is tuple:
        keys, vals = unzip(pairs)
        return switch2(on, keys=keys, vals=vals, default=default)
    if type(pairs) is not dict:
        raise ValueError("`pairs` must be a list of tuple pairs or a dict")
    return pairs.get(on, default)


def switch2(on, keys, vals, default=None):
    """
    Usage:
        switch(
            'a',
            keys=['a', 'b', 'c'],
            vals=[1, 2, 3],
            default=0
        )
        >>> 1

        # Can be used to select functions
        x = 10
        func = switch(
            x == 10, # predicate
            keys=[True, False],
            vals=[lambda x: x + 1, lambda x: x -1],
            default=lambda x: x # identity
        )
        func(x)
        >>> 11
    """
    if len(keys) == len(vals):
        raise ValueError("`keys` must be same length as `vals`")
    tuples = dict(zip(keys, vals))
    return tuples.get(on, default)
