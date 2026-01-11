"""
ikun-net - Image processing tools for color separation and analysis.
"""

import argparse
import sys

from ikunnet.cli import add_separate_colors_parser


def main():
    """Main entry point for ikun-net CLI."""
    parser = argparse.ArgumentParser(
        description="ikun-net: Image color separator and processing tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py separate-colors image.jpg
  python main.py separate-colors image.jpg --interval-size 10 --output custom_output
  python main.py separate-colors image.jpg --analyze-only
        """
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(
        dest='command',
        title='Available commands',
        description='Use one of the following commands'
    )

    # Add separate-colors subcommand
    add_separate_colors_parser(subparsers)

    # Parse arguments
    args = parser.parse_args()

    # If no command is specified, show help
    if args.command is None:
        parser.print_help()
        return 0

    # Execute the command
    if hasattr(args, 'func'):
        return args.func(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
