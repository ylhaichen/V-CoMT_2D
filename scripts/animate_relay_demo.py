from _demo_common import build_demo_parser, run_animation_demo


if __name__ == "__main__":
    args = build_demo_parser("Run the relay animation demo.").parse_args()
    run_animation_demo("relay", backend=args.backend, model=args.model, reasoning_effort=args.reasoning_effort, output_dir=args.output_dir)
