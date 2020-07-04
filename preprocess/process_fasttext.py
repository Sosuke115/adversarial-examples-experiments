#usage
#python3 process_fasttext.py sample/ja.tok sample/ja.vec --dim 300

import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='source corpus filename')
    parser.add_argument('output', help='output name')
    parser.add_argument('--min_count', type=int, default=None, help='min count for embedding')
    parser.add_argument('--thread', type=int, default=12, help='(default: 12)')
    parser.add_argument('--dim', type=int, default=300, help='dim size (default: 300)')
    args = parser.parse_args()

    cmd = [
        '/home/smg/nishikawa/fastText-0.9.2/fasttext', 'skipgram',
        '-input', args.input,
        '-output', args.output,
        '-dim', str(args.dim),
    ]

    if args.min_count:
        cmd.append('-minCount')
        cmd.append(str(args.min_count))

    # 実行
    res = subprocess.run(cmd)


if __name__ == '__main__':
    main()

