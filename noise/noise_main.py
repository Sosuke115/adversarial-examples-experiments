"""
テキストファイルを読み込み、ノイズファイルを返す関数

python noise_main.py generate_noise multi30k/test2016.en.tok en swap 
"""

import click
from tqdm import tqdm
from noise_methods import swap, random_middle, fully_random, key, natural

noise_method_list = [swap, random_middle, fully_random, key, natural]
noise_list = ["swap", "mid", "rand", "key", "nat"]

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())
    else:
        print("gonna invoke %s" % ctx.invoked_subcommand)

@cli.command(help="add noise to text")
@click.argument("file_path", default="multi30k/test2016.en.tok")
@click.argument("lang", default="en") # 'en fr de cs'
@click.argument("noise_type", default="swap") # 'swap mid rand key nat'
@click.option('--prob', default=1.0)
def generate_noise(file_path, lang, noise_type, prob):
    lines = open(file_path, encoding='utf-8')
    data = []
    for line in lines:
        data.append(line.split())
    processed = []

    noise_method = noise_method_list[noise_list.index(noise_type)]
    noise_dict = None
    if noise_type == "key":
        noise_dict = {}
        for line in open("noise/" + lang + ".key"):
            line = line.split()
            noise_dict[line[0]] = line[1:]
    if noise_type == "nat":
        noise_dict = {}
        for line in open("noise/" + lang + ".natural"):
            line = line.strip().split()
            noise_dict[line[0]] = line[1:]

    for line in tqdm(data):
        new_line = []
        for word in line:
            new_line.append(noise_method(word, noise_dict, prob))
        processed += [" ".join(new_line)]
    output_path = file_path + "." + noise_type
    with open(output_path, mode='w', errors="ignore") as f:
        f.write('\n'.join(processed))
    
    print("complete")



if __name__ == "__main__":
    cli()