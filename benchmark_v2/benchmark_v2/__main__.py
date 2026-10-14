from . import get_cli_app


def main() -> None:
    app = get_cli_app()
    app()


if __name__ == "__main__":
    main()
