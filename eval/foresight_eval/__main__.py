from .runner import build_arg_parser, evaluate_from_args

def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    evaluate_from_args(args)

if __name__ == "__main__":
    main()

