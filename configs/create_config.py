from argparse import ArgumentParser
import os
import json


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--n_heads', help='number of attention heads to use', required=True, type=int)
    parser.add_argument('--n_layers', help='number of transformer layers to use', required=True, type=int)
    parser.add_argument('--hidden_size', help='size of hidden state to use', required=True, type=int)
    parser.add_argument('--scales', help='if given, creates multiple configurations as scaled versions of the same configuration',
                        required=False, nargs='+', type=int, default=(1, ))
    parser.add_argument('--override_existing', help='If true, overrides existing configuration with the same name. '
                                                    'Otherwise, raises an exception on existing', action='store_true')
    return parser.parse_args()


def create_configuration(args):
    base_name = f'mbert_{args.n_layers}_{args.hidden_size}_{args.n_heads}'
    configs_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(configs_dir, base_name + '.json')
    if os.path.isfile(out_path):
        if args.override_existing:
            print('Overriding existing configurations')
        else:
            raise ValueError(f'Confugration already exists in: {out_path}')

    config = {
        "architectures": [
            "BertForMaskedLM"
        ],
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": args.hidden_size,
        "initializer_range": 0.02,
        "intermediate_size": args.hidden_size*4,
        "layer_norm_eps": 1e-12,
        "max_position_embeddings": 512,
        "model_type": "bert",
        "num_attention_heads": args.n_heads,
        "num_hidden_layers": args.n_layers,
        "pad_token_id": 0,
        "type_vocab_size": 2,
        "vocab_size": 30522
    }

    with open(out_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f'Done dumping configuration to: {out_path}')


def main():
    args = get_args()
    layers, hidden = args.n_layers, args.hidden_size
    for scale in args.scales:
        args.n_layers, args.hidden_size = scale*layers, scale*hidden
        create_configuration(args)


if __name__ == '__main__':
    main()
