"""Smoke test for the scaffold (TICKET-01).

Real tests arrive with their tickets (tokenizer/dataset/model). This just proves the
package imports and the suite collects+passes green.
"""

from yegpt import __version__


def test_package_imports_and_has_version() -> None:
    assert isinstance(__version__, str)
    assert __version__ != ""
