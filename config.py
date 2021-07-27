from option import args_parser
args = args_parser()

crop_size = args.crop_size
attention_size = args.attention_size

POOLING_SIZE = 7
total_stride = 16
top_k = 5
device = "cuda"
