import argparse
import json
import collections
from munch import Munch
import ast
import shlex


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            if '+' in v:
                # This typing is a bit hacky
                # Assumes something is in the list
                v = [type(d[k][0])(x) for x in v.split('+')]
            try:
                d[k] = type(d[k])(v)
            except (TypeError, ValueError) as e:
                raise TypeError(e)  # Types not compatible.
            except KeyError:
                d[k] = v  # No matching key in dict.
    return d


class ConfigParse(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        options_dict = {}
        for overrides in shlex.split(values):
            k, v = overrides.split('=')
            k_parts = k.split('.')
            dic = options_dict
            for key in k_parts[:-1]:
                dic = dic.setdefault(key, {})
            if v.startswith('[') and v.endswith(']'):
                v = ast.literal_eval(v)
            dic[k_parts[-1]] = v
        setattr(namespace, self.dest, options_dict)


def get_config_overrides():
    parser = argparse.ArgumentParser(description='Experiments with Lipschitz networks')
    parser.add_argument('config', help='Base config file')
    parser.add_argument('-o', action=ConfigParse,
                        help='Config option overrides. Separated like: e.g. optim.lr_init=1.0,,optim.lr_decay=0.1')
    return parser.parse_args()


def process_config(verbose=True):
    args = get_config_overrides()
    config = json.load(open(args.config))
    if args.o is not None:
        print(args.o)
        config = update(config, args.o)

    if verbose:
        import pprint
        pp = pprint.PrettyPrinter()
        print('-------- Config --------')
        pp.pprint(config)
        print('------------------------')

    # Use a munch object for ease of access. Munch is almost the same as Bunch, but better integrated with Python 3.
    config = Munch.fromDict(config)

    return config
