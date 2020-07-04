
#usage
#python3 process_vecmap.py sample/en.vec.vec sample/ja.vec.vec sample/en.map sample/ja.map --cuda 
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src')
    parser.add_argument('trg')
    parser.add_argument('mapped_src')
    parser.add_argument('mapped_trg')
    parser.add_argument('--cuda', action='store_true', default=False)
    args = parser.parse_args()

    cmd = [
        'python', '/home/smg/nishikawa/vecmap/map_embeddings.py', '--unsupervised',
        args.src, args.trg,
        args.mapped_src, args.mapped_trg
    ]

    if args.cuda:
        cmd.append('--cuda')

    res = subprocess.run(cmd)

if __name__ == '__main__':
    main()