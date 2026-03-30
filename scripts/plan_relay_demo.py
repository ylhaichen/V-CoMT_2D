from _demo_common import build_demo_parser, run_planning_demo


if __name__ == "__main__":
    args = build_demo_parser("Run the relay planning demo.").parse_args()
    run_planning_demo("relay", backend=args.backend, model=args.model, reasoning_effort=args.reasoning_effort)
