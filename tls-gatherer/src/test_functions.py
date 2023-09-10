"""Test suite for tls-gatherer
"""

import task_scheduler


def test_resolver():
    """tests if correct ip adress is returned"""

    ipv4 = task_scheduler.dns_lookup("https://google.de")
    assert len(ipv4.split(".")) == 4
    ipv4 = task_scheduler.dns_lookup("https://www.google.de")
    assert len(ipv4.split(".")) == 4
