import socket

import requests.packages.urllib3.util.connection as urllib3_cn


def _ipv4_only():
    # Force urllib3 / requests to use IPv4 addresses
    def allowed_gai_family():
        return socket.AF_INET
    urllib3_cn.allowed_gai_family = allowed_gai_family

_ipv4_only()
