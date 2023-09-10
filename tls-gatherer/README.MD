# TLS-gatherer: a Docker-based traffic capture system
**TLS-gatherer is an automated system that captures traffic flows for a set of given web-applications automatically.**

It takes in a set of services and produces a corpus of packet traces that can be used for further analysis. Applications are captured seperately, each trace corresponding to an application.
Applications can be executed with different browsers, each contained in a Docker container.

## Table of contents
[[_TOC_]]


## Installation

### Setup
Clone Repository:
```shell
git clone https://gitlab.lrz.de/swc/tls-gatherer.git
```
Dependencies:
```shell
pip install -R requirements.txt
```
The package was developed using Python 3.8.



### Configuration

All necessary configuartion options can be set within the file '**src/config.py**'

> HOST_TRACES_DIR (string): absolute path of the directory trace files are saved in

> HOST_LOGGING_DIR (string): absolute path of the directory log files are saved in

> CALLS_PER_URL (integer): number of traces per url

```python
SQL = {
    "host": "%%%.%%%.%%%.%%%",
    "user": "%user%",
    "passwd": "%password%",
    "database": "%database_name%"
}
```
> USER (string): identifier which the captured traces are linked with inside the database

> DOCKER_IMAGE_TAG (string): tag used for local conatainer

> DOCKER_AVAILABLE_BROWSERS: list of bowsers that are used for generating traffic. Allowed options: "firefox", "wget", "chromium"


TLS gatherer is tightly integrated with the database and it's structure. 
A new database can be set up using the file "**swc_tls_gatherer.sql**". It contains the exact table structure required by tls gather to operate. 

#### VPN Configuration

> COUNTRY_CODES (list of strings), limited to a maximum of 5 country codes, e.g.: ["de", "us", "it", "fr"]

> VPN_RETRIES = 6

> VPN_PARALLEL_CONNECTIONS = 5

Test

## Usage

To start the capture process, simply run:

```shell
(sudo) python src/entrypoint.py
```

The table **swc_tls_gatherer.urls** has to contain entries with **trace_done = False**. TLS gatherer processes the urls sequentially until there are no jobs left.
