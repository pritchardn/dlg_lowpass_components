# __main__ is not required for DALiuGE components.
import argparse  # pragma: no cover

from dlg.drop import InMemoryDROP

from dlg_lowpass_components import LPSignalGenerator


def main() -> None:  # pragma: no cover
    """
    The main function executes on commands:
    `python -m dlg_lowpass_components` and `$ dlg_lowpass_components `.

    This is your program's entry point.

    You can change this function to do whatever you want.
    Examples:
        * Run a test suite
        * Run a server
        * Do some other stuff
        * Run a command line application (Click, Typer, ArgParse)
        * List all available tasks
        * Run an application (Flask, FastAPI, Django, etc.)
    """
    parser = argparse.ArgumentParser(
        description="dlg_lowpass_components.",
        epilog="Enjoy the dlg_lowpass_components functionality!",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Optionally adds verbosity",
    )
    args = parser.parse_args()
    print("Executing main function")
    comp = LPSignalGenerator("A", "A")
    memory = InMemoryDROP("b", "b")
    comp.addOutput(memory)
    comp.run()
    print("End of main function")


if __name__ == "__main__":  # pragma: no cover
    main()
