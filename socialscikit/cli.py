"""SocialSciKit CLI entry point."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="socialscikit",
        description="SocialSciKit — zero-code toolkit for social science text analysis",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Unified launcher (default)
    sub_launch = subparsers.add_parser("launch", help="Launch unified SocialSciKit web UI (default)")
    sub_launch.add_argument("--port", type=int, default=7860, help="Port number")
    sub_launch.add_argument("--share", action="store_true", help="Create public link")

    # QuantiKit UI launcher (standalone)
    sub_quanti = subparsers.add_parser("quantikit", help="Launch QuantiKit web UI only")
    sub_quanti.add_argument("--port", type=int, default=7860, help="Port number")
    sub_quanti.add_argument("--share", action="store_true", help="Create public link")

    # QualiKit UI launcher (standalone)
    sub_quali = subparsers.add_parser("qualikit", help="Launch QualiKit web UI only")
    sub_quali.add_argument("--port", type=int, default=7861, help="Port number")
    sub_quali.add_argument("--share", action="store_true", help="Create public link")

    args = parser.parse_args()

    if args.command == "quantikit":
        from socialscikit.ui.quantikit_app import launch
        launch(port=args.port, share=args.share)
    elif args.command == "qualikit":
        from socialscikit.ui.qualikit_app import launch
        launch(port=args.port, share=args.share)
    elif args.command == "launch":
        from socialscikit.ui.main_app import launch
        launch(port=args.port, share=args.share)
    else:
        # Default: launch unified app
        from socialscikit.ui.main_app import launch
        launch()


if __name__ == "__main__":
    main()
